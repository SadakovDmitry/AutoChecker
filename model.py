from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .config import ValidatorConfig
from .features import (
    DenseClassifier,
    EmbeddingBackend,
    RuleFeatureBuilder,
    TextVectorClassifier,
    dense_normalize,
    leave_one_out_similarity,
    max_similarity,
)
from .text import add_role_columns


FINAL_FEATURE_COLUMNS = [
    "p_lsa",
    "p_embedding",
    "sim_pos",
    "sim_neg",
    "sim_margin",
    "client_keyword_hits",
    "full_keyword_hits",
    "operator_keyword_hits",
    "bot_keyword_hits",
    "client_required_hits",
    "operator_only_hits",
    "client_text_share",
]


def _class_counts(y: Sequence[int]) -> Tuple[int, int]:
    y_arr = np.asarray(y, dtype=int)
    return int((y_arr == 0).sum()), int((y_arr == 1).sum())


def _cv_splits(y: Sequence[int], max_splits: int = 5) -> int:
    neg, pos = _class_counts(y)
    return max(0, min(max_splits, neg, pos))


def _fit_final_classifier(x: pd.DataFrame, y: Sequence[int], random_state: int):
    y_arr = np.asarray(y, dtype=int)
    if len(np.unique(y_arr)) < 2:
        return None
    base = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=random_state,
    )
    splits = _cv_splits(y_arr, max_splits=5)
    if splits >= 3 and len(y_arr) >= 12:
        try:
            clf = CalibratedClassifierCV(estimator=base, cv=splits, method="sigmoid")
            clf.fit(x, y_arr)
            return clf
        except TypeError:
            clf = CalibratedClassifierCV(base_estimator=base, cv=splits, method="sigmoid")
            clf.fit(x, y_arr)
            return clf
    base.fit(x, y_arr)
    return base


def _predict_final_classifier(clf, x: pd.DataFrame, fallback: float = 0.0) -> np.ndarray:
    if clf is None:
        return np.full(len(x), fallback, dtype=float)
    return clf.predict_proba(x)[:, 1]


def _oof_text_predictions(texts: Sequence[str], y: Sequence[int], config: ValidatorConfig) -> np.ndarray:
    y_arr = np.asarray(y, dtype=int)
    splits = _cv_splits(y_arr)
    if splits < 2:
        model = TextVectorClassifier(config.max_lsa_components, config.random_state).fit(texts, y_arr)
        return model.predict_proba(texts)
    oof = np.zeros(len(y_arr), dtype=float)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=config.random_state)
    for train_idx, test_idx in cv.split(np.zeros(len(y_arr)), y_arr):
        model = TextVectorClassifier(config.max_lsa_components, config.random_state)
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        model.fit(train_texts, y_arr[train_idx])
        oof[test_idx] = model.predict_proba(test_texts)
    return oof


def _oof_dense_predictions(vectors: np.ndarray, y: Sequence[int], config: ValidatorConfig) -> np.ndarray:
    y_arr = np.asarray(y, dtype=int)
    splits = _cv_splits(y_arr)
    if splits < 2:
        model = DenseClassifier(config.random_state).fit(vectors, y_arr)
        return model.predict_proba(vectors)
    oof = np.zeros(len(y_arr), dtype=float)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=config.random_state)
    for train_idx, test_idx in cv.split(np.zeros(len(y_arr)), y_arr):
        model = DenseClassifier(config.random_state)
        model.fit(vectors[train_idx], y_arr[train_idx])
        oof[test_idx] = model.predict_proba(vectors[test_idx])
    return oof


def _oof_final_predictions(x: pd.DataFrame, y: Sequence[int], config: ValidatorConfig) -> np.ndarray:
    y_arr = np.asarray(y, dtype=int)
    splits = _cv_splits(y_arr)
    if splits < 2:
        clf = _fit_final_classifier(x, y_arr, config.random_state)
        return _predict_final_classifier(clf, x, fallback=float(y_arr.mean()))
    oof = np.zeros(len(y_arr), dtype=float)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=config.random_state)
    for train_idx, test_idx in cv.split(np.zeros(len(y_arr)), y_arr):
        clf = _fit_final_classifier(x.iloc[train_idx], y_arr[train_idx], config.random_state)
        oof[test_idx] = _predict_final_classifier(
            clf, x.iloc[test_idx], fallback=float(y_arr[train_idx].mean())
        )
    return oof


def choose_threshold(
    y: Sequence[int],
    probabilities: Sequence[float],
    target_precision: float,
) -> Tuple[float, float, float]:
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(probabilities, dtype=float)
    thresholds = sorted(set(float(x) for x in p_arr), reverse=True)
    best = None
    for threshold in thresholds:
        mask = p_arr >= threshold
        if not mask.any():
            continue
        precision = float(y_arr[mask].mean())
        coverage = float(mask.mean())
        if precision >= target_precision:
            if best is None or coverage > best[2]:
                best = (threshold, precision, coverage)
    if best is None:
        return 1.000001, 0.0, 0.0
    return best


def choose_low_threshold(
    y: Sequence[int],
    probabilities: Sequence[float],
    target_precision: float,
) -> Tuple[float, float, float]:
    """Choose p_correct cutoff for reliable automatic "no" decisions."""

    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(probabilities, dtype=float)
    thresholds = sorted(set(float(x) for x in p_arr))
    best = None
    for threshold in thresholds:
        mask = p_arr <= threshold
        if not mask.any():
            continue
        no_precision = float((y_arr[mask] == 0).mean())
        coverage = float(mask.mean())
        if no_precision >= target_precision:
            if best is None or coverage > best[2]:
                best = (threshold, no_precision, coverage)
    if best is None:
        return -0.000001, 0.0, 0.0
    return best


def evaluate_low_threshold(
    y: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> Tuple[float, float, float]:
    """Evaluate p_correct cutoff for automatic "no" decisions."""

    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(probabilities, dtype=float)
    mask = p_arr <= threshold
    if not mask.any():
        return -0.000001, 0.0, 0.0
    no_precision = float((y_arr[mask] == 0).mean())
    coverage = float(mask.mean())
    return float(threshold), no_precision, coverage


def choose_rate_matching_threshold(
    y: Sequence[int],
    probabilities: Sequence[float],
) -> Tuple[float, float, float, float]:
    """Choose yes/no threshold that matches historical positive rate.

    This is used by the experimental full yes/no mode. The threshold is learned
    on out-of-fold probabilities from previous manual checks, not on the latest
    evaluation labels.
    """

    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(probabilities, dtype=float)
    if len(y_arr) == 0:
        return 0.5, 0.0, 0.0, 0.0
    manual_rate = float(y_arr.mean())
    thresholds = [1.000001, *sorted(set(float(x) for x in p_arr), reverse=True), -0.000001]
    best = None
    for threshold in thresholds:
        predicted = (p_arr >= threshold).astype(int)
        predicted_rate = float(predicted.mean())
        rate_gap = abs(predicted_rate - manual_rate)
        row_accuracy = float((predicted == y_arr).mean())
        candidate = (rate_gap, -row_accuracy, -float(threshold), float(threshold), predicted_rate, row_accuracy)
        if best is None or candidate < best:
            best = candidate
    _, _, _, threshold, predicted_rate, row_accuracy = best
    return threshold, predicted_rate, row_accuracy, abs(predicted_rate - manual_rate)


def _final_features(
    *,
    p_lsa: Sequence[float],
    p_embedding: Sequence[float],
    sim_pos: Sequence[float],
    sim_neg: Sequence[float],
    rule_features: pd.DataFrame,
) -> pd.DataFrame:
    x = pd.DataFrame(
        {
            "p_lsa": np.asarray(p_lsa, dtype=float),
            "p_embedding": np.asarray(p_embedding, dtype=float),
            "sim_pos": np.asarray(sim_pos, dtype=float),
            "sim_neg": np.asarray(sim_neg, dtype=float),
        },
        index=rule_features.index,
    )
    x["sim_margin"] = x["sim_pos"] - x["sim_neg"]
    x = pd.concat([x, rule_features.reset_index(drop=True)], axis=1)
    for column in FINAL_FEATURE_COLUMNS:
        if column not in x.columns:
            x[column] = 0.0
    return x[FINAL_FEATURE_COLUMNS].fillna(0.0)


@dataclass
class ReasonValidator:
    reason_id: str
    threshold: float
    threshold_precision: float
    threshold_coverage: float
    no_threshold: float
    no_threshold_precision: float
    no_threshold_coverage: float
    n_samples: int
    n_positive: int
    n_negative: int
    yesno_threshold: float = 0.5
    yesno_train_predicted_positive_rate: float = 0.0
    yesno_train_rate_gap: float = 0.0
    yesno_train_row_label_accuracy: float = 0.0
    warnings: List[str] = field(default_factory=list)
    text_model: Optional[TextVectorClassifier] = None
    embedding_model: Optional[DenseClassifier] = None
    final_model: object = None
    positive_chat_ids: List[str] = field(default_factory=list)
    negative_chat_ids: List[str] = field(default_factory=list)
    positive_lsa_vectors: Optional[np.ndarray] = None
    negative_lsa_vectors: Optional[np.ndarray] = None
    positive_embedding_vectors: Optional[np.ndarray] = None
    negative_embedding_vectors: Optional[np.ndarray] = None

    @property
    def low_data(self) -> bool:
        return bool(self.warnings) and any("low_data" in item for item in self.warnings)


class HybridValidator:
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.reason_validators: Dict[str, ReasonValidator] = {}
        self.trained_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self.embedding_enabled = False
        self.embedding_error = ""

    @staticmethod
    def _load_embedding_backend(config: ValidatorConfig) -> Optional[EmbeddingBackend]:
        if not config.use_embeddings:
            return None
        try:
            return EmbeddingBackend.load(config.embedding_model)
        except Exception as exc:  # optional dependency / model cache can fail
            return None

    @classmethod
    def train(cls, frame: pd.DataFrame, config: ValidatorConfig) -> "HybridValidator":
        model = cls(config)
        frame = add_role_columns(frame)
        train_frame = frame[frame["human_label"].isin([0, 1])].copy()
        if train_frame.empty:
            raise ValueError("No labeled rows found. Need да/нет or yes/no values for training.")

        embedding_backend = cls._load_embedding_backend(config)
        model.embedding_enabled = embedding_backend is not None
        if config.use_embeddings and embedding_backend is None:
            model.embedding_error = (
                "embedding_unavailable: sentence-transformers or local model is unavailable; "
                "using TF-IDF/LSA fallback"
            )

        rule_builder = RuleFeatureBuilder(config.rules)
        for reason_id, reason_frame in train_frame.groupby("reason_id", sort=True):
            validator = model._train_reason(reason_id, reason_frame.reset_index(drop=True), rule_builder, embedding_backend)
            model.reason_validators[str(reason_id)] = validator
        return model

    def _train_reason(
        self,
        reason_id: str,
        frame: pd.DataFrame,
        rule_builder: RuleFeatureBuilder,
        embedding_backend: Optional[EmbeddingBackend],
    ) -> ReasonValidator:
        y = frame["human_label"].astype(int).to_numpy()
        texts = frame["model_text"].fillna("").astype(str).tolist()
        chat_ids = frame["chat_id"].fillna("").astype(str).tolist()
        n_negative, n_positive = _class_counts(y)
        warnings: List[str] = []

        if len(frame) < self.config.min_reason_samples or min(n_negative, n_positive) < self.config.min_class_samples:
            warnings.append(
                f"low_data: samples={len(frame)}, positive={n_positive}, negative={n_negative}"
            )

        text_model = TextVectorClassifier(self.config.max_lsa_components, self.config.random_state)
        text_model.fit(texts, y)
        p_lsa_oof = _oof_text_predictions(texts, y, self.config)
        lsa_vectors = dense_normalize(text_model.transform_vectors(texts))

        if embedding_backend is not None:
            embedding_vectors = embedding_backend.encode(texts)
            embedding_model = DenseClassifier(self.config.random_state).fit(embedding_vectors, y)
            p_embedding_oof = _oof_dense_predictions(embedding_vectors, y, self.config)
        else:
            embedding_vectors = None
            embedding_model = None
            p_embedding_oof = p_lsa_oof.copy()

        retrieval_vectors = embedding_vectors if embedding_vectors is not None else lsa_vectors
        sim_pos_oof, sim_neg_oof = leave_one_out_similarity(retrieval_vectors, y)
        rule_features = rule_builder.transform(frame).reset_index(drop=True)
        final_x = _final_features(
            p_lsa=p_lsa_oof,
            p_embedding=p_embedding_oof,
            sim_pos=sim_pos_oof,
            sim_neg=sim_neg_oof,
            rule_features=rule_features,
        )
        final_oof = _oof_final_predictions(final_x, y, self.config)
        final_model = _fit_final_classifier(final_x, y, self.config.random_state)
        (
            yesno_threshold,
            yesno_train_predicted_positive_rate,
            yesno_train_row_label_accuracy,
            yesno_train_rate_gap,
        ) = choose_rate_matching_threshold(y, final_oof)

        enable_auto_no = bool(getattr(self.config, "enable_auto_no", False))

        if any("low_data" in item for item in warnings):
            threshold, threshold_precision, threshold_coverage = 1.000001, 0.0, 0.0
            no_threshold, no_threshold_precision, no_threshold_coverage = -0.000001, 0.0, 0.0
        else:
            threshold, threshold_precision, threshold_coverage = choose_threshold(
                y, final_oof, self.config.target_precision
            )
            if enable_auto_no:
                no_threshold, no_threshold_precision, no_threshold_coverage = choose_low_threshold(
                    y, final_oof, self.config.target_no_precision
                )
            else:
                no_threshold, no_threshold_precision, no_threshold_coverage = -0.000001, 0.0, 0.0
            if threshold > 1:
                warnings.append(
                    f"no_auto_yes_threshold_for_target_precision: target={self.config.target_precision}"
                )
            if enable_auto_no and no_threshold < 0:
                warnings.append(
                    f"no_auto_no_threshold_for_target_precision: target={self.config.target_no_precision}"
                )

        pos_mask = y == 1
        neg_mask = y == 0
        return ReasonValidator(
            reason_id=str(reason_id),
            threshold=float(threshold),
            threshold_precision=float(threshold_precision),
            threshold_coverage=float(threshold_coverage),
            no_threshold=float(no_threshold),
            no_threshold_precision=float(no_threshold_precision),
            no_threshold_coverage=float(no_threshold_coverage),
            yesno_threshold=float(yesno_threshold),
            yesno_train_predicted_positive_rate=float(yesno_train_predicted_positive_rate),
            yesno_train_rate_gap=float(yesno_train_rate_gap),
            yesno_train_row_label_accuracy=float(yesno_train_row_label_accuracy),
            n_samples=int(len(frame)),
            n_positive=int(n_positive),
            n_negative=int(n_negative),
            warnings=warnings,
            text_model=text_model,
            embedding_model=embedding_model,
            final_model=final_model,
            positive_chat_ids=[chat_ids[i] for i in np.where(pos_mask)[0]],
            negative_chat_ids=[chat_ids[i] for i in np.where(neg_mask)[0]],
            positive_lsa_vectors=lsa_vectors[pos_mask],
            negative_lsa_vectors=lsa_vectors[neg_mask],
            positive_embedding_vectors=embedding_vectors[pos_mask] if embedding_vectors is not None else None,
            negative_embedding_vectors=embedding_vectors[neg_mask] if embedding_vectors is not None else None,
        )

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = add_role_columns(frame)
        embedding_backend = self._load_embedding_backend(self.config) if self.embedding_enabled else None
        rule_builder = RuleFeatureBuilder(self.config.rules)
        outputs = []
        for reason_id, reason_frame in frame.groupby("reason_id", sort=False):
            validator = self.reason_validators.get(str(reason_id))
            if validator is None:
                unknown = reason_frame.copy()
                unknown["p_correct"] = 0.0
                unknown["threshold"] = 1.000001
                unknown["yes_threshold"] = 1.000001
                unknown["no_threshold"] = -0.000001
                unknown["decision"] = "review"
                unknown["auto_answer"] = "review"
                unknown["p_lsa"] = 0.0
                unknown["p_embedding"] = 0.0
                unknown["nearest_positive_chat_id"] = ""
                unknown["nearest_positive_score"] = 0.0
                unknown["nearest_negative_chat_id"] = ""
                unknown["nearest_negative_score"] = 0.0
                unknown["rule_flags"] = "unknown_reason"
                outputs.append(unknown)
                continue
            outputs.append(self._predict_reason(reason_frame.copy(), validator, rule_builder, embedding_backend))
        if not outputs:
            return frame.copy()
        return pd.concat(outputs, ignore_index=True)

    def _predict_reason(
        self,
        frame: pd.DataFrame,
        validator: ReasonValidator,
        rule_builder: RuleFeatureBuilder,
        embedding_backend: Optional[EmbeddingBackend],
    ) -> pd.DataFrame:
        texts = frame["model_text"].fillna("").astype(str).tolist()
        p_lsa = validator.text_model.predict_proba(texts) if validator.text_model else np.zeros(len(frame))
        lsa_vectors = dense_normalize(validator.text_model.transform_vectors(texts)) if validator.text_model else np.zeros((len(frame), 1))

        if embedding_backend is not None and validator.embedding_model is not None:
            embedding_vectors = embedding_backend.encode(texts)
            p_embedding = validator.embedding_model.predict_proba(embedding_vectors)
            pos_ref = validator.positive_embedding_vectors
            neg_ref = validator.negative_embedding_vectors
            retrieval_vectors = embedding_vectors
        else:
            p_embedding = p_lsa.copy()
            pos_ref = validator.positive_lsa_vectors
            neg_ref = validator.negative_lsa_vectors
            retrieval_vectors = lsa_vectors

        sim_pos, pos_idx = max_similarity(retrieval_vectors, pos_ref)
        sim_neg, neg_idx = max_similarity(retrieval_vectors, neg_ref)
        rule_features = rule_builder.transform(frame).reset_index(drop=True)
        final_x = _final_features(
            p_lsa=p_lsa,
            p_embedding=p_embedding,
            sim_pos=sim_pos,
            sim_neg=sim_neg,
            rule_features=rule_features,
        )
        p_correct = _predict_final_classifier(
            validator.final_model,
            final_x,
            fallback=float(validator.n_positive / max(1, validator.n_samples)),
        )
        yes_threshold = float(getattr(validator, "threshold", 1.000001))
        no_threshold = float(getattr(validator, "no_threshold", -0.000001))
        if no_threshold >= 0 and self.config.max_auto_no_p_correct is not None:
            no_threshold = min(no_threshold, float(self.config.max_auto_no_p_correct))
        decision = np.full(len(p_correct), "review", dtype=object)
        decision[p_correct >= yes_threshold] = "auto_yes"
        decision[p_correct <= no_threshold] = "auto_no"
        auto_answer = np.full(len(p_correct), "review", dtype=object)
        auto_answer[decision == "auto_yes"] = "да"
        auto_answer[decision == "auto_no"] = "нет"

        nearest_pos_ids = []
        for idx in pos_idx:
            nearest_pos_ids.append(validator.positive_chat_ids[int(idx)] if idx >= 0 and validator.positive_chat_ids else "")
        nearest_neg_ids = []
        for idx in neg_idx:
            nearest_neg_ids.append(validator.negative_chat_ids[int(idx)] if idx >= 0 and validator.negative_chat_ids else "")

        flags = []
        for _, row in rule_features.iterrows():
            row_flags = []
            if row.get("operator_only_hits", 0) >= 1:
                row_flags.append("operator_only_keywords")
            if row.get("client_text_share", 0) == 0:
                row_flags.append("no_client_text")
            flags.append(",".join(row_flags))

        out = frame.copy()
        out["p_correct"] = p_correct
        out["threshold"] = yes_threshold
        out["yes_threshold"] = yes_threshold
        out["no_threshold"] = no_threshold
        out["decision"] = decision
        out["auto_answer"] = auto_answer
        out["p_lsa"] = p_lsa
        out["p_embedding"] = p_embedding
        out["nearest_positive_chat_id"] = nearest_pos_ids
        out["nearest_positive_score"] = sim_pos
        out["nearest_negative_chat_id"] = nearest_neg_ids
        out["nearest_negative_score"] = sim_neg
        out["rule_flags"] = flags
        out["validator_warnings"] = "; ".join(validator.warnings)
        return out

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for reason_id, validator in sorted(self.reason_validators.items(), key=lambda item: item[0]):
            rows.append(
                {
                    "reason_id": reason_id,
                    "auto_no_enabled": bool(getattr(self.config, "enable_auto_no", False)),
                    "target_yes_precision": self.config.target_precision,
                    "target_no_precision": self.config.target_no_precision,
                    "max_auto_no_p_correct": self.config.max_auto_no_p_correct,
                    "yes_threshold": validator.threshold,
                    "yes_threshold_precision": validator.threshold_precision,
                    "yes_threshold_coverage": validator.threshold_coverage,
                    "no_threshold": getattr(validator, "no_threshold", -0.000001),
                    "no_threshold_precision": getattr(validator, "no_threshold_precision", 0.0),
                    "no_threshold_coverage": getattr(validator, "no_threshold_coverage", 0.0),
                    "yesno_threshold": getattr(validator, "yesno_threshold", 0.5),
                    "yesno_train_predicted_positive_rate": getattr(
                        validator, "yesno_train_predicted_positive_rate", 0.0
                    ),
                    "yesno_train_rate_gap": getattr(validator, "yesno_train_rate_gap", 0.0),
                    "yesno_train_row_label_accuracy": getattr(
                        validator, "yesno_train_row_label_accuracy", 0.0
                    ),
                    "n_samples": validator.n_samples,
                    "n_positive": validator.n_positive,
                    "n_negative": validator.n_negative,
                    "warnings": "; ".join(validator.warnings),
                }
            )
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out / "model.joblib")
        self.summary_frame().to_csv(out / "training_summary.csv", index=False)

    @staticmethod
    def load(path: str) -> "HybridValidator":
        model_path = Path(path) / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)
