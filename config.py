from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional


@dataclass
class ReasonRule:
    """Lightweight lexical hints for one predicted reason."""

    keywords: List[str] = field(default_factory=list)
    client_required_keywords: List[str] = field(default_factory=list)
    operator_only_keywords: List[str] = field(default_factory=list)


@dataclass
class ValidatorConfig:
    target_precision: float = 0.95
    target_no_precision: float = 0.97
    min_reason_samples: int = 8
    min_class_samples: int = 2
    max_lsa_components: int = 200
    top_k: int = 5
    random_state: int = 42
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    use_embeddings: bool = True
    enable_auto_no: bool = False
    max_auto_no_p_correct: Optional[float] = None
    rules: Dict[str, ReasonRule] = field(default_factory=dict)


def load_rules(path: Optional[str]) -> Dict[str, ReasonRule]:
    if not path:
        return {}

    rule_path = Path(path)
    if not rule_path.exists():
        raise FileNotFoundError(f"Rules file not found: {rule_path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "pyyaml is required to read rules files. Install requirements.txt or omit --rules."
        ) from exc

    raw = yaml.safe_load(rule_path.read_text(encoding="utf-8")) or {}
    reasons = raw.get("reasons", raw)
    parsed: Dict[str, ReasonRule] = {}

    for reason_id, value in reasons.items():
        if value is None:
            value = {}
        if isinstance(value, list):
            value = {"keywords": value}
        if not isinstance(value, Mapping):
            raise ValueError(f"Invalid rule block for reason {reason_id!r}: {value!r}")
        parsed[str(reason_id)] = ReasonRule(
            keywords=[str(x).lower() for x in value.get("keywords", [])],
            client_required_keywords=[
                str(x).lower() for x in value.get("client_required_keywords", [])
            ],
            operator_only_keywords=[
                str(x).lower() for x in value.get("operator_only_keywords", [])
            ],
        )

    return parsed
