from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

from .config import ReasonRule


def _safe_texts(texts: Iterable[object]) -> List[str]:
    return ["" if value is None else str(value) for value in texts]


class TextVectorClassifier:
    """TF-IDF + optional LSA + LogisticRegression binary classifier."""

    def __init__(self, max_lsa_components: int = 200, random_state: int = 42):
        self.max_lsa_components = max_lsa_components
        self.random_state = random_state
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.classifier: Optional[LogisticRegression] = None
        self.constant_probability: Optional[float] = None

    def fit(self, texts: Sequence[str], y: Sequence[int]) -> "TextVectorClassifier":
        texts = _safe_texts(texts)
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            self.constant_probability = float(y_arr.mean()) if len(y_arr) else 0.0
            return self

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            max_features=50000,
            sublinear_tf=True,
        )
        x_tfidf = self.vectorizer.fit_transform(texts)

        n_features = x_tfidf.shape[1]
        max_components = min(self.max_lsa_components, max(2, len(texts) - 1), max(2, n_features - 1))
        if len(texts) >= 20 and n_features > 3 and max_components >= 2:
            n_components = min(max_components, n_features - 1)
            self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath")
                x_model = self.svd.fit_transform(x_tfidf)
            x_model = np.nan_to_num(x_model, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            self.svd = None
            x_model = x_tfidf

        self.classifier = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=self.random_state,
        )
        self.classifier.fit(x_model, y_arr)
        return self

    def transform_vectors(self, texts: Sequence[str]):
        if self.constant_probability is not None and self.vectorizer is None:
            return np.zeros((len(texts), 1), dtype=float)
        if self.vectorizer is None:
            raise RuntimeError("TextVectorClassifier is not fitted")
        x_tfidf = self.vectorizer.transform(_safe_texts(texts))
        if self.svd is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath")
                vectors = self.svd.transform(x_tfidf)
            return np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
        return x_tfidf

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.constant_probability is not None and self.classifier is None:
            return np.full(len(texts), self.constant_probability, dtype=float)
        if self.classifier is None:
            raise RuntimeError("TextVectorClassifier is not fitted")
        x_model = self.transform_vectors(texts)
        return self.classifier.predict_proba(x_model)[:, 1]


class DenseClassifier:
    """LogisticRegression on dense vectors such as sentence embeddings."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.classifier: Optional[LogisticRegression] = None
        self.constant_probability: Optional[float] = None

    def fit(self, vectors: np.ndarray, y: Sequence[int]) -> "DenseClassifier":
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            self.constant_probability = float(y_arr.mean()) if len(y_arr) else 0.0
            return self
        self.classifier = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=self.random_state,
        )
        self.classifier.fit(vectors, y_arr)
        return self

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        if self.constant_probability is not None and self.classifier is None:
            return np.full(vectors.shape[0], self.constant_probability, dtype=float)
        if self.classifier is None:
            raise RuntimeError("DenseClassifier is not fitted")
        return self.classifier.predict_proba(vectors)[:, 1]


@dataclass
class EmbeddingBackend:
    model_name: str
    model: object

    @classmethod
    def load(cls, model_name: str) -> "EmbeddingBackend":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        return cls(model_name=model_name, model=SentenceTransformer(model_name))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            _safe_texts(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=float)


def dense_normalize(vectors) -> np.ndarray:
    if sparse.issparse(vectors):
        vectors = vectors.toarray()
    vectors = np.asarray(vectors, dtype=float)
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    normalized = normalize(vectors)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def max_similarity(query_vectors, reference_vectors) -> Tuple[np.ndarray, np.ndarray]:
    query = dense_normalize(query_vectors)
    if reference_vectors is None or len(reference_vectors) == 0:
        return np.zeros(query.shape[0], dtype=float), np.full(query.shape[0], -1, dtype=int)
    refs = dense_normalize(reference_vectors)
    sims = cosine_similarity(query, refs)
    idx = sims.argmax(axis=1)
    score = sims[np.arange(sims.shape[0]), idx]
    return score, idx


def leave_one_out_similarity(vectors, y: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    vectors_norm = dense_normalize(vectors)
    y_arr = np.asarray(y, dtype=int)
    n = len(y_arr)
    sim_pos = np.zeros(n, dtype=float)
    sim_neg = np.zeros(n, dtype=float)
    all_sims = cosine_similarity(vectors_norm, vectors_norm)
    np.fill_diagonal(all_sims, -1.0)
    for i in range(n):
        pos_mask = y_arr == 1
        neg_mask = y_arr == 0
        pos_mask[i] = False
        neg_mask[i] = False
        sim_pos[i] = all_sims[i, pos_mask].max() if pos_mask.any() else 0.0
        sim_neg[i] = all_sims[i, neg_mask].max() if neg_mask.any() else 0.0
    return sim_pos, sim_neg


class RuleFeatureBuilder:
    def __init__(self, rules: Dict[str, ReasonRule]):
        self.rules = rules

    @staticmethod
    def _count_hits(text: object, keywords: Sequence[str]) -> int:
        lowered = ("" if text is None else str(text)).lower()
        return sum(1 for keyword in keywords if keyword and keyword.lower() in lowered)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in frame.iterrows():
            reason_id = str(row.get("reason_id", ""))
            rule = self.rules.get(reason_id, ReasonRule())
            keywords = sorted(set(rule.keywords + rule.client_required_keywords + rule.operator_only_keywords))
            client_hits = self._count_hits(row.get("client_text", ""), keywords)
            full_hits = self._count_hits(row.get("full_text", row.get("chat_text", "")), keywords)
            operator_hits = self._count_hits(row.get("operator_text", ""), keywords)
            bot_hits = self._count_hits(row.get("bot_text", ""), keywords)
            required_hits = self._count_hits(row.get("client_text", ""), rule.client_required_keywords)
            operator_only_config_hits = self._count_hits(
                row.get("operator_text", ""), rule.operator_only_keywords
            )
            client_len = len(str(row.get("client_text", "")).split())
            full_len = max(1, len(str(row.get("full_text", row.get("chat_text", ""))).split()))
            operator_only = int((operator_hits + bot_hits + operator_only_config_hits) > 0 and client_hits == 0)
            rows.append(
                {
                    "client_keyword_hits": float(client_hits),
                    "full_keyword_hits": float(full_hits),
                    "operator_keyword_hits": float(operator_hits),
                    "bot_keyword_hits": float(bot_hits),
                    "client_required_hits": float(required_hits),
                    "operator_only_hits": float(operator_only),
                    "client_text_share": float(client_len / full_len),
                }
            )
        return pd.DataFrame(rows, index=frame.index)
