from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import re

import numpy as np
import pandas as pd

from .config import ValidatorConfig
from .iteration_compare import split_train_calibration
from .model import HybridValidator
from .reports import write_table


@dataclass
class QuotaYesNoResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    priors: pd.DataFrame
    training_summary: pd.DataFrame


@dataclass
class GuardedEstimateResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    offset_summary: pd.DataFrame
    training_summary: pd.DataFrame


@dataclass
class HybridRouterResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    risk_summary: pd.DataFrame
    offset_summary: pd.DataFrame
    training_summary: pd.DataFrame


def _sheet_sort_key(value: object) -> tuple:
    text = str(value)
    numbers = tuple(float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[\.,]\d+)?", text))
    return (numbers, text)


def _latest_reason_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "_source_sheet" not in frame.columns:
        return frame.copy()
    parts = []
    for _, group in frame.groupby("reason_id", sort=False):
        sheets = list(group["_source_sheet"].fillna("").astype(str).unique())
        if not sheets:
            parts.append(group)
            continue
        latest_sheet = max(sheets, key=_sheet_sort_key)
        parts.append(group[group["_source_sheet"].fillna("").astype(str) == latest_sheet])
    if not parts:
        return frame.copy()
    return pd.concat(parts, ignore_index=True)


def build_reason_priors(
    train_frame: pd.DataFrame,
    *,
    source: str = "all",
) -> Tuple[pd.DataFrame, Dict[str, float], float]:
    labeled = train_frame[train_frame["human_label"].isin([0, 1])].copy()
    if labeled.empty:
        raise ValueError("Quota yes/no mode needs labeled train rows.")
    if source == "latest":
        labeled_for_priors = _latest_reason_rows(labeled)
    elif source == "all":
        labeled_for_priors = labeled
    else:
        raise ValueError("source must be 'all' or 'latest'")

    rows = []
    priors: Dict[str, float] = {}
    for reason_id, group in labeled_for_priors.groupby("reason_id", sort=True):
        rate = float(group["human_label"].astype(int).mean())
        reason_key = str(reason_id)
        priors[reason_key] = rate
        rows.append(
            {
                "reason_id": reason_key,
                "prior_source": source,
                "prior_source_sheet": (
                    ",".join(sorted(group["_source_sheet"].fillna("").astype(str).unique()))
                    if "_source_sheet" in group.columns
                    else ""
                ),
                "train_rows": int(len(group)),
                "train_yes": int(group["human_label"].astype(int).sum()),
                "train_no": int((group["human_label"].astype(int) == 0).sum()),
                "train_positive_rate": rate,
            }
        )
    global_rate = float(labeled_for_priors["human_label"].astype(int).mean())
    return pd.DataFrame(rows), priors, global_rate


def apply_quota_yesno(
    predictions: pd.DataFrame,
    priors: Dict[str, float],
    global_positive_rate: float,
) -> pd.DataFrame:
    """Force every row into да/нет using historical positive-rate quota per reason.

    The latest labels are not used for threshold selection. For every reason, the
    historical positive rate from previous manual checks defines how many rows in
    the new batch should become "да"; rows with highest p_correct get "да".
    """

    outputs = []
    for reason_id, group in predictions.groupby("reason_id", sort=False):
        group = group.copy()
        reason_key = str(reason_id)
        expected_rate = float(priors.get(reason_key, global_positive_rate))
        expected_rate = max(0.0, min(1.0, expected_rate))
        n_rows = len(group)
        yes_count = int(np.floor(expected_rate * n_rows + 0.5))
        yes_count = max(0, min(n_rows, yes_count))

        ordered = group.sort_values(
            ["p_correct", "nearest_positive_score", "nearest_negative_score"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        yes_index = set(ordered.head(yes_count).index)
        is_yes = group.index.to_series().map(lambda idx: idx in yes_index).to_numpy()

        if yes_count <= 0:
            quota_threshold = 1.000001
        elif yes_count >= n_rows:
            quota_threshold = -0.000001
        else:
            quota_threshold = float(ordered.iloc[yes_count - 1]["p_correct"])

        group["quota_expected_positive_rate"] = expected_rate
        group["quota_yes_count"] = yes_count
        group["quota_threshold"] = quota_threshold
        group["quota_unknown_reason"] = reason_key not in priors
        group["decision"] = np.where(is_yes, "quota_yes", "quota_no")
        group["auto_answer"] = np.where(is_yes, "да", "нет")
        group["auto_label"] = np.where(is_yes, 1, 0)
        outputs.append(group)

    if not outputs:
        return predictions.copy()
    return pd.concat(outputs, ignore_index=True)


def _choose_previous_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[float, float, float, float]:
    """Pick p_correct threshold on previous labels.

    Objective order:
    1. make predicted yes/no ratio as close as possible to previous true ratio;
    2. maximize row-level agreement on previous labels;
    3. prefer a higher threshold if still tied.
    """

    if len(y) == 0:
        return 0.5, 0.0, 0.0, 0.0
    manual_rate = float(y.mean())
    candidates = sorted(set(float(x) for x in p), reverse=True)
    candidates = [1.000001, *candidates, -0.000001]
    best = None
    for threshold in candidates:
        auto = (p >= threshold).astype(int)
        predicted_rate = float(auto.mean())
        rate_gap = abs(predicted_rate - manual_rate)
        row_accuracy = float((auto == y).mean())
        candidate = (rate_gap, -row_accuracy, -float(threshold), float(threshold), predicted_rate, row_accuracy)
        if best is None or candidate < best:
            best = candidate
    _, _, _, threshold, predicted_rate, row_accuracy = best
    return threshold, predicted_rate, row_accuracy, abs(predicted_rate - manual_rate)


def learn_previous_thresholds(train_predictions: pd.DataFrame) -> pd.DataFrame:
    labeled = train_predictions[train_predictions["human_label"].isin([0, 1])].copy()
    rows = []
    for reason_id, group in labeled.groupby("reason_id", sort=True):
        y = group["human_label"].astype(int).to_numpy()
        p = group["p_correct"].astype(float).to_numpy()
        threshold, predicted_rate, row_accuracy, rate_gap = _choose_previous_threshold(y, p)
        rows.append(
            {
                "reason_id": str(reason_id),
                "train_rows": int(len(group)),
                "train_yes": int(y.sum()),
                "train_no": int((y == 0).sum()),
                "train_positive_rate": float(y.mean()) if len(y) else 0.0,
                "learned_yesno_threshold": threshold,
                "train_predicted_positive_rate": predicted_rate,
                "train_rate_gap_pp": rate_gap * 100,
                "train_row_label_accuracy": row_accuracy,
            }
        )
    return pd.DataFrame(rows)


def learned_thresholds_from_model(model: HybridValidator) -> pd.DataFrame:
    summary = model.summary_frame().copy()
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for row in summary.to_dict(orient="records"):
        n_samples = int(row.get("n_samples", 0) or 0)
        n_positive = int(row.get("n_positive", 0) or 0)
        n_negative = int(row.get("n_negative", 0) or 0)
        rows.append(
            {
                "reason_id": str(row.get("reason_id", "")),
                "train_rows": n_samples,
                "train_yes": n_positive,
                "train_no": n_negative,
                "train_positive_rate": n_positive / n_samples if n_samples else 0.0,
                "learned_yesno_threshold": float(row.get("yesno_threshold", 0.5) or 0.5),
                "train_predicted_positive_rate": float(
                    row.get("yesno_train_predicted_positive_rate", 0.0) or 0.0
                ),
                "train_rate_gap_pp": float(row.get("yesno_train_rate_gap", 0.0) or 0.0) * 100,
                "train_row_label_accuracy": float(
                    row.get("yesno_train_row_label_accuracy", 0.0) or 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def apply_threshold_yesno(
    predictions: pd.DataFrame,
    thresholds: pd.DataFrame,
    global_threshold: float = 0.5,
) -> pd.DataFrame:
    threshold_map = (
        thresholds.set_index("reason_id")["learned_yesno_threshold"].astype(float).to_dict()
        if not thresholds.empty
        else {}
    )
    outputs = []
    for reason_id, group in predictions.groupby("reason_id", sort=False):
        group = group.copy()
        reason_key = str(reason_id)
        threshold = float(threshold_map.get(reason_key, global_threshold))
        is_yes = group["p_correct"].astype(float).to_numpy() >= threshold
        group["quota_expected_positive_rate"] = np.nan
        group["quota_yes_count"] = int(is_yes.sum())
        group["quota_threshold"] = threshold
        group["quota_unknown_reason"] = reason_key not in threshold_map
        group["decision"] = np.where(is_yes, "threshold_yes", "threshold_no")
        group["auto_answer"] = np.where(is_yes, "да", "нет")
        group["auto_label"] = np.where(is_yes, 1, 0)
        outputs.append(group)
    if not outputs:
        return predictions.copy()
    return pd.concat(outputs, ignore_index=True)


def build_quota_summary(predictions: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    labeled = predictions[predictions["human_label"].isin([0, 1])].copy()
    rows = []
    prior_lookup = priors.set_index("reason_id").to_dict(orient="index") if not priors.empty else {}

    for reason_id, group in labeled.groupby("reason_id", sort=True):
        human = group["human_label"].astype(int)
        auto = group["auto_label"].astype(int)
        reason_key = str(reason_id)
        prior = prior_lookup.get(reason_key, {})
        manual_accuracy = float(human.mean()) if len(group) else 0.0
        auto_estimated_accuracy = float(auto.mean()) if len(group) else 0.0
        rows.append(
            {
                "reason_id": reason_key,
                "rows": int(len(group)),
                "manual_yes": int(human.sum()),
                "manual_no": int((human == 0).sum()),
                "manual_prompt_accuracy": manual_accuracy,
                "auto_yes": int(auto.sum()),
                "auto_no": int((auto == 0).sum()),
                "auto_estimated_prompt_accuracy": auto_estimated_accuracy,
                "accuracy_gap_pp": (auto_estimated_accuracy - manual_accuracy) * 100,
                "abs_accuracy_gap_pp": abs(auto_estimated_accuracy - manual_accuracy) * 100,
                "row_label_accuracy": float((auto == human).mean()) if len(group) else 0.0,
                "train_rows": int(prior.get("train_rows", 0) or 0),
                "train_positive_rate": float(prior.get("train_positive_rate", np.nan)),
                "yesno_threshold": float(group["quota_threshold"].iloc[0]),
                "quota_unknown_reason": bool(group["quota_unknown_reason"].iloc[0]),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    total_rows = int(summary["rows"].sum())
    manual_total = int(summary["manual_yes"].sum())
    auto_total = int(summary["auto_yes"].sum())
    agreement_total = int(
        (labeled["auto_label"].astype(int) == labeled["human_label"].astype(int)).sum()
    )
    overall_manual = manual_total / total_rows if total_rows else 0.0
    overall_auto = auto_total / total_rows if total_rows else 0.0
    overall = {
        "reason_id": "__overall_weighted__",
        "rows": total_rows,
        "manual_yes": manual_total,
        "manual_no": int(total_rows - manual_total),
        "manual_prompt_accuracy": overall_manual,
        "auto_yes": auto_total,
        "auto_no": int(total_rows - auto_total),
        "auto_estimated_prompt_accuracy": overall_auto,
        "accuracy_gap_pp": (overall_auto - overall_manual) * 100,
        "abs_accuracy_gap_pp": abs(overall_auto - overall_manual) * 100,
        "row_label_accuracy": agreement_total / total_rows if total_rows else 0.0,
        "train_rows": int(summary["train_rows"].sum()),
        "train_positive_rate": np.nan,
        "yesno_threshold": np.nan,
        "quota_unknown_reason": bool(summary["quota_unknown_reason"].any()),
    }
    return pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)


def _normalize_reason(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return text


def _history_rates(frame: pd.DataFrame, *, latest: bool = False) -> tuple[Dict[str, float], Dict[str, int]]:
    labeled = frame[frame["human_label"].isin([0, 1])].copy()
    if latest:
        labeled = _latest_reason_rows(labeled)
    rates: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for reason_id, group in labeled.groupby("reason_id", sort=True):
        key = _normalize_reason(reason_id)
        values = group["human_label"].astype(int)
        rates[key] = float(values.mean())
        counts[key] = int(len(group))
    return rates, counts


def _topic_rate(frame: pd.DataFrame) -> float:
    labeled = frame[frame["human_label"].isin([0, 1])].copy()
    if labeled.empty:
        return 0.0
    return float(labeled["human_label"].astype(int).mean())


def _history_rate_std(frame: pd.DataFrame) -> Dict[str, float]:
    if "_source_sheet" not in frame.columns:
        return {}
    labeled = frame[frame["human_label"].isin([0, 1])].copy()
    rows = {}
    for reason_id, group in labeled.groupby("reason_id", sort=True):
        rates = []
        for _, sheet_group in group.groupby("_source_sheet", sort=False):
            if len(sheet_group):
                rates.append(float(sheet_group["human_label"].astype(int).mean()))
        rows[_normalize_reason(reason_id)] = float(np.std(rates)) if len(rates) >= 2 else 0.0
    return rows


def _guarded_bayes_summary(
    predictions: pd.DataFrame,
    train_frame: pd.DataFrame,
    *,
    offset: float = 0.0,
    k: float = 40.0,
    guard_gap: float = 0.10,
) -> pd.DataFrame:
    history_latest, history_counts = _history_rates(train_frame, latest=True)
    topic_rate = _topic_rate(train_frame)
    labeled = predictions.copy()
    rows = []
    for reason_id, group in labeled.groupby("reason_id", sort=True):
        key = _normalize_reason(reason_id)
        n_rows = int(len(group))
        mean_p = float(group["p_correct"].astype(float).mean()) if n_rows else 0.0
        hist = float(history_latest.get(key, topic_rate))
        bayes = (mean_p * n_rows + hist * k) / (n_rows + k) if n_rows else hist
        guarded = max(bayes, topic_rate - guard_gap)
        estimated = max(0.0, min(1.0, guarded + offset))
        item = {
            "reason_id": key,
            "rows": n_rows,
            "mean_p_correct": mean_p,
            "history_latest_rate": hist,
            "history_rows": int(history_counts.get(key, 0)),
            "topic_history_rate": topic_rate,
            "guarded_bayes_k40": max(0.0, min(1.0, guarded)),
            "guarded_offset": float(offset),
            "estimated_prompt_accuracy": estimated,
            "estimated_yes": int(np.floor(estimated * n_rows + 0.5)),
            "estimated_no": n_rows - int(np.floor(estimated * n_rows + 0.5)),
        }
        if "human_label" in group.columns and group["human_label"].isin([0, 1]).any():
            human = group[group["human_label"].isin([0, 1])]["human_label"].astype(int)
            manual = float(human.mean())
            item.update(
                {
                    "manual_prompt_accuracy": manual,
                    "accuracy_gap_pp": (estimated - manual) * 100,
                    "abs_accuracy_gap_pp": abs(estimated - manual) * 100,
                    "manual_yes": int(human.sum()),
                    "manual_no": int((human == 0).sum()),
                }
            )
        rows.append(item)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    if "manual_prompt_accuracy" in summary.columns:
        valid = summary[summary["manual_prompt_accuracy"].notna()].copy()
        if not valid.empty:
            total_rows = int(valid["rows"].sum())
            manual = float((valid["manual_prompt_accuracy"] * valid["rows"]).sum() / total_rows)
            estimated = float((valid["estimated_prompt_accuracy"] * valid["rows"]).sum() / total_rows)
            abs_gap = float((valid["abs_accuracy_gap_pp"] * valid["rows"]).sum() / total_rows)
            overall = {
                "reason_id": "__overall_weighted__",
                "rows": total_rows,
                "mean_p_correct": float((valid["mean_p_correct"] * valid["rows"]).sum() / total_rows),
                "history_latest_rate": np.nan,
                "history_rows": int(valid["history_rows"].sum()),
                "topic_history_rate": valid["topic_history_rate"].iloc[0],
                "guarded_bayes_k40": float((valid["guarded_bayes_k40"] * valid["rows"]).sum() / total_rows),
                "guarded_offset": float(offset),
                "estimated_prompt_accuracy": estimated,
                "estimated_yes": int(valid["estimated_yes"].sum()),
                "estimated_no": int(valid["estimated_no"].sum()),
                "manual_prompt_accuracy": manual,
                "accuracy_gap_pp": (estimated - manual) * 100,
                "abs_accuracy_gap_pp": abs_gap,
                "manual_yes": int(valid["manual_yes"].sum()),
                "manual_no": int(valid["manual_no"].sum()),
            }
            summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)
    return summary


def learn_guarded_bayes_offset(
    train_frame: pd.DataFrame,
    *,
    config: ValidatorConfig,
    k: float = 40.0,
    guard_gap: float = 0.10,
    min_offset: float = 0.0,
    max_offset: float = 0.15,
) -> tuple[float, pd.DataFrame]:
    early_train, calibration, calibration_sheet = split_train_calibration(train_frame)
    if early_train.empty or calibration.empty:
        return 0.0, pd.DataFrame(
            [
                {
                    "offset": 0.0,
                    "source": "not_enough_iterations",
                    "calibration_sheet": calibration_sheet,
                    "calibration_rows": int(len(calibration)),
                }
            ]
        )

    model = HybridValidator.train(early_train, config)
    calibration_predictions = model.predict(calibration)
    calibration_summary = _guarded_bayes_summary(
        calibration_predictions,
        early_train,
        offset=0.0,
        k=k,
        guard_gap=guard_gap,
    )
    valid = calibration_summary[
        (calibration_summary["reason_id"] != "__overall_weighted__")
        & calibration_summary["manual_prompt_accuracy"].notna()
    ].copy()
    if valid.empty:
        offset = 0.0
    else:
        residual = valid["manual_prompt_accuracy"] - valid["guarded_bayes_k40"]
        offset = float((residual * valid["rows"]).sum() / valid["rows"].sum())
    offset = max(float(min_offset), min(float(max_offset), offset))
    offset_summary = pd.DataFrame(
        [
            {
                "offset": offset,
                "source": "n_minus_1_calibration",
                "calibration_sheet": calibration_sheet,
                "calibration_rows": int(len(calibration)),
                "calibration_subreasons": int(valid["reason_id"].nunique()) if not valid.empty else 0,
                "calibration_manual_accuracy": (
                    float((valid["manual_prompt_accuracy"] * valid["rows"]).sum() / valid["rows"].sum())
                    if not valid.empty
                    else np.nan
                ),
                "calibration_guarded_accuracy": (
                    float((valid["guarded_bayes_k40"] * valid["rows"]).sum() / valid["rows"].sum())
                    if not valid.empty
                    else np.nan
                ),
            }
        ]
    )
    return offset, offset_summary


def apply_estimated_rate_yesno(predictions: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    estimate_map = (
        summary[summary["reason_id"] != "__overall_weighted__"]
        .set_index("reason_id")["estimated_prompt_accuracy"]
        .astype(float)
        .to_dict()
    )
    outputs = []
    for reason_id, group in predictions.groupby("reason_id", sort=False):
        group = group.copy()
        key = _normalize_reason(reason_id)
        estimated = max(0.0, min(1.0, float(estimate_map.get(key, 0.0))))
        yes_count = int(np.floor(estimated * len(group) + 0.5))
        yes_count = max(0, min(len(group), yes_count))
        ordered = group.sort_values(
            ["p_correct", "nearest_positive_score", "nearest_negative_score"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        yes_index = set(ordered.head(yes_count).index)
        is_yes = group.index.to_series().map(lambda idx: idx in yes_index).to_numpy()
        group["estimated_prompt_accuracy"] = estimated
        group["estimated_yes_count"] = yes_count
        group["decision"] = np.where(is_yes, "estimated_yes", "estimated_no")
        group["auto_answer"] = np.where(is_yes, "да", "нет")
        group["auto_label"] = np.where(is_yes, 1, 0)
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True) if outputs else predictions.copy()


def build_hybrid_risk_summary(
    guarded_summary: pd.DataFrame,
    train_frame: pd.DataFrame,
    *,
    min_history_rows: int = 30,
    max_history_std: float = 0.18,
    max_model_history_gap: float = 0.25,
    min_estimated_accuracy: float = 0.65,
) -> pd.DataFrame:
    std_map = _history_rate_std(train_frame)
    rows = []
    detail = guarded_summary[guarded_summary["reason_id"] != "__overall_weighted__"].copy()
    for _, row in detail.iterrows():
        reason_id = _normalize_reason(row["reason_id"])
        history_rows = int(row.get("history_rows", 0) or 0)
        mean_p = float(row.get("mean_p_correct", 0.0) or 0.0)
        history_rate = float(row.get("history_latest_rate", np.nan))
        estimated = float(row.get("estimated_prompt_accuracy", 0.0) or 0.0)
        history_std = float(std_map.get(reason_id, 0.0))
        gap = abs(mean_p - history_rate) if not np.isnan(history_rate) else 1.0
        flags = []
        if reason_id.startswith("unmapped::"):
            flags.append("unmapped_subreason")
        if history_rows < min_history_rows:
            flags.append("low_history")
        if history_std > max_history_std:
            flags.append("unstable_history")
        if gap > max_model_history_gap:
            flags.append("model_history_disagreement")
        if estimated < min_estimated_accuracy:
            flags.append("low_estimated_accuracy")
        mode = "safe_auto_yes" if flags else "full_yesno"
        rows.append(
            {
                "reason_id": reason_id,
                "rows": int(row.get("rows", 0) or 0),
                "mode": mode,
                "risk_flags": ",".join(flags),
                "history_rows": history_rows,
                "history_rate_std": history_std,
                "mean_p_correct": mean_p,
                "history_latest_rate": history_rate,
                "model_history_gap": gap,
                "estimated_prompt_accuracy": estimated,
            }
        )
    return pd.DataFrame(rows)


def apply_hybrid_router(
    raw_predictions: pd.DataFrame,
    guarded_summary: pd.DataFrame,
    risk_summary: pd.DataFrame,
) -> pd.DataFrame:
    risk_map = risk_summary.set_index("reason_id").to_dict(orient="index") if not risk_summary.empty else {}
    estimate_map = (
        guarded_summary[guarded_summary["reason_id"] != "__overall_weighted__"]
        .set_index("reason_id")["estimated_prompt_accuracy"]
        .astype(float)
        .to_dict()
        if not guarded_summary.empty
        else {}
    )
    outputs = []
    for reason_id, group in raw_predictions.groupby("reason_id", sort=False):
        group = group.copy()
        key = _normalize_reason(reason_id)
        risk = risk_map.get(key, {})
        mode = str(risk.get("mode", "safe_auto_yes"))
        group["hybrid_mode"] = mode
        group["risk_flags"] = str(risk.get("risk_flags", "unknown_reason"))
        if mode == "full_yesno":
            estimated = max(0.0, min(1.0, float(estimate_map.get(key, 0.0))))
            yes_count = int(np.floor(estimated * len(group) + 0.5))
            yes_count = max(0, min(len(group), yes_count))
            ordered = group.sort_values(
                ["p_correct", "nearest_positive_score", "nearest_negative_score"],
                ascending=[False, False, True],
                kind="mergesort",
            )
            yes_index = set(ordered.head(yes_count).index)
            is_yes = group.index.to_series().map(lambda idx: idx in yes_index).to_numpy()
            group["estimated_prompt_accuracy"] = estimated
            group["decision"] = np.where(is_yes, "hybrid_yes", "hybrid_no")
            group["auto_answer"] = np.where(is_yes, "да", "нет")
            group["auto_label"] = np.where(is_yes, 1, 0)
        else:
            is_yes = group["decision"].eq("auto_yes").to_numpy()
            group["estimated_prompt_accuracy"] = np.nan
            group["decision"] = np.where(is_yes, "hybrid_safe_yes", "review")
            group["auto_answer"] = np.where(is_yes, "да", "review")
            group["auto_label"] = np.where(is_yes, 1, np.nan)
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True) if outputs else raw_predictions.copy()


def build_hybrid_summary(predictions: pd.DataFrame, risk_summary: pd.DataFrame) -> pd.DataFrame:
    labeled = predictions[predictions["human_label"].isin([0, 1])].copy()
    if labeled.empty:
        return pd.DataFrame()

    risk_lookup = risk_summary.set_index("reason_id").to_dict(orient="index") if not risk_summary.empty else {}
    rows = []
    for reason_id, group in labeled.groupby("reason_id", sort=True):
        key = _normalize_reason(reason_id)
        auto = group[group["auto_answer"].isin(["да", "нет"])].copy()
        correct = (
            auto["auto_label"].astype(int).to_numpy() == auto["human_label"].astype(int).to_numpy()
            if len(auto)
            else np.array([], dtype=bool)
        )
        risk = risk_lookup.get(key, {})
        rows.append(
            {
                "reason_id": key,
                "rows": int(len(group)),
                "hybrid_mode": str(risk.get("mode", "")),
                "risk_flags": str(risk.get("risk_flags", "")),
                "auto_rows": int(len(auto)),
                "review_rows": int(len(group) - len(auto)),
                "coverage": float(len(auto) / len(group)) if len(group) else 0.0,
                "precision": float(correct.mean()) if len(correct) else np.nan,
                "errors": int((~correct).sum()) if len(correct) else 0,
                "auto_yes": int((auto["auto_answer"] == "да").sum()) if len(auto) else 0,
                "auto_no": int((auto["auto_answer"] == "нет").sum()) if len(auto) else 0,
                "manual_yes_rate": float(group["human_label"].astype(int).mean()) if len(group) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    auto_all = labeled[labeled["auto_answer"].isin(["да", "нет"])].copy()
    if len(auto_all):
        correct_all = auto_all["auto_label"].astype(int).to_numpy() == auto_all["human_label"].astype(int).to_numpy()
        precision = float(correct_all.mean())
        errors = int((~correct_all).sum())
    else:
        precision = np.nan
        errors = 0
    overall = {
        "reason_id": "__overall_weighted__",
        "rows": int(len(labeled)),
        "hybrid_mode": "",
        "risk_flags": "",
        "auto_rows": int(len(auto_all)),
        "review_rows": int(len(labeled) - len(auto_all)),
        "coverage": float(len(auto_all) / len(labeled)) if len(labeled) else 0.0,
        "precision": precision,
        "errors": errors,
        "auto_yes": int((auto_all["auto_answer"] == "да").sum()) if len(auto_all) else 0,
        "auto_no": int((auto_all["auto_answer"] == "нет").sum()) if len(auto_all) else 0,
        "manual_yes_rate": float(labeled["human_label"].astype(int).mean()),
    }
    return pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)


def run_guarded_bayes_yesno_experiment(
    *,
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    config: ValidatorConfig,
    offset: Optional[float] = None,
    k: float = 40.0,
    guard_gap: float = 0.10,
    min_offset: float = 0.0,
    max_offset: float = 0.15,
) -> GuardedEstimateResult:
    learned_offset, offset_summary = learn_guarded_bayes_offset(
        train_frame,
        config=config,
        k=k,
        guard_gap=guard_gap,
        min_offset=min_offset,
        max_offset=max_offset,
    )
    if offset is None:
        offset = learned_offset
    else:
        offset_summary = offset_summary.copy()
        offset_summary["offset"] = float(offset)
        offset_summary["source"] = "manual_cli_offset"
    model = HybridValidator.train(train_frame, config)
    raw_predictions = model.predict(evaluation_frame)
    summary = _guarded_bayes_summary(
        raw_predictions,
        train_frame,
        offset=float(offset),
        k=k,
        guard_gap=guard_gap,
    )
    predictions = apply_estimated_rate_yesno(raw_predictions, summary)
    if "human_label" in predictions.columns:
        labeled = predictions[predictions["human_label"].isin([0, 1])].copy()
        if not labeled.empty:
            row_acc = (
                labeled["auto_label"].astype(int) == labeled["human_label"].astype(int)
            ).mean()
            summary["row_label_accuracy"] = np.nan
            summary.loc[summary["reason_id"] == "__overall_weighted__", "row_label_accuracy"] = float(row_acc)
    return GuardedEstimateResult(
        predictions=predictions,
        summary=summary,
        offset_summary=offset_summary,
        training_summary=model.summary_frame(),
    )


def run_hybrid_router_experiment(
    *,
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    config: ValidatorConfig,
    offset: Optional[float] = None,
    k: float = 40.0,
    guard_gap: float = 0.10,
    min_offset: float = 0.0,
    max_offset: float = 0.15,
    min_history_rows: int = 30,
    max_history_std: float = 0.18,
    max_model_history_gap: float = 0.25,
    min_estimated_accuracy: float = 0.65,
) -> HybridRouterResult:
    learned_offset, offset_summary = learn_guarded_bayes_offset(
        train_frame,
        config=config,
        k=k,
        guard_gap=guard_gap,
        min_offset=min_offset,
        max_offset=max_offset,
    )
    if offset is None:
        offset = learned_offset
    else:
        offset_summary = offset_summary.copy()
        offset_summary["offset"] = float(offset)
        offset_summary["source"] = "manual_cli_offset"
    model = HybridValidator.train(train_frame, config)
    raw_predictions = model.predict(evaluation_frame)
    guarded_summary = _guarded_bayes_summary(
        raw_predictions,
        train_frame,
        offset=float(offset),
        k=k,
        guard_gap=guard_gap,
    )
    risk_summary = build_hybrid_risk_summary(
        guarded_summary,
        train_frame,
        min_history_rows=min_history_rows,
        max_history_std=max_history_std,
        max_model_history_gap=max_model_history_gap,
        min_estimated_accuracy=min_estimated_accuracy,
    )
    predictions = apply_hybrid_router(raw_predictions, guarded_summary, risk_summary)
    summary = build_hybrid_summary(predictions, risk_summary)
    return HybridRouterResult(
        predictions=predictions,
        summary=summary,
        risk_summary=risk_summary,
        offset_summary=offset_summary,
        training_summary=model.summary_frame(),
    )


def write_hybrid_router_report(result: HybridRouterResult, output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        try:
            with pd.ExcelWriter(path) as writer:
                result.summary.to_excel(writer, sheet_name="summary", index=False)
                result.risk_summary.to_excel(writer, sheet_name="risk_summary", index=False)
                result.predictions.to_excel(writer, sheet_name="predictions", index=False)
                result.offset_summary.to_excel(writer, sheet_name="offset", index=False)
                result.training_summary.to_excel(writer, sheet_name="model_training", index=False)
        except ImportError as exc:
            raise RuntimeError(
                "openpyxl is required to write .xlsx reports. Install requirements or use .csv."
            ) from exc
    else:
        stem = path.with_suffix("")
        write_table(result.summary, f"{stem}_summary.csv")
        write_table(result.risk_summary, f"{stem}_risk_summary.csv")
        write_table(result.predictions, f"{stem}_predictions.csv")
        write_table(result.offset_summary, f"{stem}_offset.csv")
        write_table(result.training_summary, f"{stem}_model_training.csv")


def write_guarded_estimate_report(result: GuardedEstimateResult, output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        try:
            with pd.ExcelWriter(path) as writer:
                result.summary.to_excel(writer, sheet_name="summary", index=False)
                result.predictions.to_excel(writer, sheet_name="predictions", index=False)
                result.offset_summary.to_excel(writer, sheet_name="offset", index=False)
                result.training_summary.to_excel(writer, sheet_name="model_training", index=False)
        except ImportError as exc:
            raise RuntimeError(
                "openpyxl is required to write .xlsx reports. Install requirements or use .csv."
            ) from exc
    else:
        stem = path.with_suffix("")
        write_table(result.summary, f"{stem}_summary.csv")
        write_table(result.predictions, f"{stem}_predictions.csv")
        write_table(result.offset_summary, f"{stem}_offset.csv")
        write_table(result.training_summary, f"{stem}_model_training.csv")


def run_quota_yesno_experiment(
    *,
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    model: HybridValidator,
    strategy: str = "threshold",
) -> QuotaYesNoResult:
    raw_predictions = model.predict(evaluation_frame)
    if strategy == "threshold":
        priors = learned_thresholds_from_model(model)
        predictions = apply_threshold_yesno(raw_predictions, priors)
    elif strategy == "quota":
        priors, prior_map, global_rate = build_reason_priors(train_frame, source="all")
        predictions = apply_quota_yesno(raw_predictions, prior_map, global_rate)
    elif strategy == "latest-prior":
        priors, prior_map, global_rate = build_reason_priors(train_frame, source="latest")
        predictions = apply_quota_yesno(raw_predictions, prior_map, global_rate)
    else:
        raise ValueError("strategy must be 'threshold', 'quota' or 'latest-prior'")
    predictions["yesno_strategy"] = strategy
    summary = build_quota_summary(predictions, priors)
    if not summary.empty:
        summary["yesno_strategy"] = strategy
    return QuotaYesNoResult(
        predictions=predictions,
        summary=summary,
        priors=priors,
        training_summary=model.summary_frame(),
    )


def write_quota_yesno_report(result: QuotaYesNoResult, output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        try:
            with pd.ExcelWriter(path) as writer:
                result.summary.to_excel(writer, sheet_name="summary", index=False)
                result.predictions.to_excel(writer, sheet_name="predictions", index=False)
                result.priors.to_excel(writer, sheet_name="train_priors", index=False)
                result.training_summary.to_excel(writer, sheet_name="model_training", index=False)
        except ImportError as exc:
            raise RuntimeError(
                "openpyxl is required to write .xlsx reports. Install requirements or use .csv."
            ) from exc
    else:
        stem = path.with_suffix("")
        write_table(result.summary, f"{stem}_summary.csv")
        write_table(result.predictions, f"{stem}_predictions.csv")
        write_table(result.priors, f"{stem}_train_priors.csv")
        write_table(result.training_summary, f"{stem}_model_training.csv")
