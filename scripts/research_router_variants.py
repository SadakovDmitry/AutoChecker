from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import pandas as pd

from auto_classifier.config import ValidatorConfig, load_rules
from auto_classifier.data import load_tables
from auto_classifier.quota_yesno import run_hybrid_router_experiment
from auto_classifier.reports import write_table
from auto_classifier.subreason_mapping import (
    apply_subreason_mapping,
    load_subreason_mapping,
    use_subreason_key_as_reason_id,
)


warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

BASE = Path("auto_classifier/local_data/prepared")
OUT = Path("auto_classifier/local_data/reports/router_variant_research")
MAPPING_PATH = "auto_classifier/configs/subreason_versions.yaml"


@dataclass(frozen=True)
class TestCase:
    key: str
    product: str
    topic: str
    train: str
    validation: str


@dataclass(frozen=True)
class Variant:
    name: str
    target_precision: float
    min_history_rows: int
    max_history_std: float
    max_model_history_gap: float
    min_estimated_accuracy: float
    description: str
    full_yesno_strategy: str = "threshold"
    estimate_strategy: str = "max_history_latest"
    min_full_mean_p_correct: float = 0.65
    min_full_train_row_accuracy: float = 0.65
    max_full_train_rate_gap: float = 0.05


TESTS = [
    TestCase("kasko_oformlenie", "КАСКО", "Оформление", "kasko_oformlenie_auto_train.csv", "kasko_oformlenie_auto_val.csv"),
    TestCase("kasko_prolongacii", "КАСКО", "Пролонгации", "kasko_prolongacii_auto_train.csv", "kasko_prolongacii_auto_val.csv"),
    TestCase("kasko_rastorzhenie", "КАСКО", "Расторжение", "kasko_rastorzhenie_auto_train.csv", "kasko_rastorzhenie_auto_val.csv"),
    TestCase("kasko_uregulirovanie", "КАСКО", "Урегулирование", "kasko_uregulirovanie_auto_train.csv", "kasko_uregulirovanie_auto_val.csv"),
    TestCase("vzr_izmenenia", "ВЗР", "Изменения", "vzr_izmenenia_auto_train.csv", "vzr_izmenenia_auto_val.csv"),
    TestCase("vzr_oformlenie", "ВЗР", "Оформление", "vzr_oformlenie_auto_train.csv", "vzr_oformlenie_auto_val.csv"),
    TestCase("vzr_prolongacii", "ВЗР", "Пролонгации", "vzr_prolongacii_auto_train.csv", "vzr_prolongacii_auto_val.csv"),
    TestCase("vzr_rastorzhenia", "ВЗР", "Расторжения", "vzr_rastorzhenia_auto_train.csv", "vzr_rastorzhenia_auto_val.csv"),
    TestCase("vzr_uregulirovanie", "ВЗР", "Урегулирование", "vzr_uregulirovanie_auto_train.csv", "vzr_uregulirovanie_auto_val.csv"),
]


VARIANTS = [
    Variant(
        name="safe_only_p80_reference",
        target_precision=0.80,
        min_history_rows=10_000,
        max_history_std=0.0,
        max_model_history_gap=0.0,
        min_estimated_accuracy=1.0,
        description="Reference: all subreasons go to safe auto_yes/review.",
    ),
    Variant(
        name="hybrid_current",
        target_precision=0.80,
        min_history_rows=10,
        max_history_std=0.05,
        max_model_history_gap=0.35,
        min_estimated_accuracy=0.70,
        description="Current hybrid default: max(history, p_correct) routing + personal p_correct threshold.",
    ),
    Variant(
        name="hybrid_current_legacy_topn",
        target_precision=0.80,
        min_history_rows=10,
        max_history_std=0.05,
        max_model_history_gap=0.35,
        min_estimated_accuracy=0.70,
        description="Legacy hybrid router: low-risk full yes/no uses old top-N quota behavior.",
        full_yesno_strategy="legacy_quota",
        estimate_strategy="guarded_bayes",
        min_full_mean_p_correct=0.0,
        min_full_train_row_accuracy=0.0,
        max_full_train_rate_gap=1.0,
    ),
    Variant(
        name="hybrid_selective_p85",
        target_precision=0.85,
        min_history_rows=10,
        max_history_std=0.05,
        max_model_history_gap=0.35,
        min_estimated_accuracy=0.70,
        description="Selective-classification variant: stricter safe part, same router.",
    ),
    Variant(
        name="hybrid_selective_p90",
        target_precision=0.90,
        min_history_rows=10,
        max_history_std=0.05,
        max_model_history_gap=0.35,
        min_estimated_accuracy=0.70,
        description="More conservative selective-classification variant.",
    ),
    Variant(
        name="hybrid_more_coverage",
        target_precision=0.80,
        min_history_rows=10,
        max_history_std=0.10,
        max_model_history_gap=0.35,
        min_estimated_accuracy=0.70,
        description="Router allows more full yes/no by relaxing historical stability.",
    ),
    Variant(
        name="hybrid_precision_guard",
        target_precision=0.85,
        min_history_rows=10,
        max_history_std=0.05,
        max_model_history_gap=0.25,
        min_estimated_accuracy=0.75,
        description="Risk-control variant: fewer full yes/no subreasons, stricter safe threshold.",
    ),
]


def _load_stable(path: Path, mapping) -> pd.DataFrame:
    frame = load_tables([str(path)], require_text=True, require_answer=True)
    frame = apply_subreason_mapping(frame, mapping)
    return use_subreason_key_as_reason_id(frame)


def _overall(summary: pd.DataFrame) -> dict:
    row = summary[summary["reason_id"].eq("__overall_weighted__")]
    return row.iloc[0].to_dict() if not row.empty else {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = load_subreason_mapping(MAPPING_PATH)
    rules = load_rules(None)
    loaded = {}
    for test in TESTS:
        train_path = BASE / test.train
        validation_path = BASE / test.validation
        if not train_path.exists() or not validation_path.exists():
            continue
        loaded[test.key] = (
            test,
            _load_stable(train_path, mapping),
            _load_stable(validation_path, mapping),
        )

    summary_rows = []
    topic_rows = []
    for variant in VARIANTS:
        for test, train, validation in loaded.values():
            config = ValidatorConfig(
                target_precision=variant.target_precision,
                use_embeddings=False,
                rules=rules,
            )
            result = run_hybrid_router_experiment(
                train_frame=train,
                evaluation_frame=validation,
                config=config,
                offset=None,
                k=40.0,
                guard_gap=0.10,
                min_offset=0.0,
                max_offset=0.15,
                min_history_rows=variant.min_history_rows,
                max_history_std=variant.max_history_std,
                max_model_history_gap=variant.max_model_history_gap,
                min_estimated_accuracy=variant.min_estimated_accuracy,
                min_full_mean_p_correct=variant.min_full_mean_p_correct,
                min_full_train_row_accuracy=variant.min_full_train_row_accuracy,
                max_full_train_rate_gap=variant.max_full_train_rate_gap,
                full_yesno_strategy=variant.full_yesno_strategy,
                estimate_strategy=variant.estimate_strategy,
            )
            row = _overall(result.summary)
            if not row:
                continue
            item = {
                "variant": variant.name,
                "description": variant.description,
                "test": test.key,
                "product": test.product,
                "topic": test.topic,
                "rows": int(row["rows"]),
                "auto_rows": int(row["auto_rows"]),
                "review_rows": int(row["review_rows"]),
                "coverage_pct": round(float(row["coverage"]) * 100, 2),
                "precision_pct": round(float(row["precision"]) * 100, 2),
                "errors": int(row["errors"]),
                "auto_yes": int(row["auto_yes"]),
                "auto_no": int(row["auto_no"]),
            }
            topic_rows.append(item)
        variant_rows = [row for row in topic_rows if row["variant"] == variant.name]
        rows = sum(row["rows"] for row in variant_rows)
        auto_rows = sum(row["auto_rows"] for row in variant_rows)
        errors = sum(row["errors"] for row in variant_rows)
        summary_rows.append(
            {
                "variant": variant.name,
                "description": variant.description,
                "rows": rows,
                "auto_rows": auto_rows,
                "review_rows": rows - auto_rows,
                "coverage_pct": round(auto_rows / rows * 100, 2) if rows else 0.0,
                "precision_pct": round((auto_rows - errors) / auto_rows * 100, 2) if auto_rows else 0.0,
                "errors": errors,
                "target_precision": variant.target_precision,
                "min_history_rows": variant.min_history_rows,
                "max_history_std": variant.max_history_std,
                "max_model_history_gap": variant.max_model_history_gap,
                "min_estimated_accuracy": variant.min_estimated_accuracy,
                "min_full_mean_p_correct": variant.min_full_mean_p_correct,
                "min_full_train_row_accuracy": variant.min_full_train_row_accuracy,
                "max_full_train_rate_gap": variant.max_full_train_rate_gap,
                "full_yesno_strategy": variant.full_yesno_strategy,
                "estimate_strategy": variant.estimate_strategy,
            }
        )

    overall = pd.DataFrame(summary_rows)
    by_topic = pd.DataFrame(topic_rows)
    write_table(overall, str(OUT / "router_variant_summary.csv"))
    write_table(by_topic, str(OUT / "router_variant_by_topic.csv"))
    with pd.ExcelWriter(OUT / "router_variant_research.xlsx", engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="summary", index=False)
        by_topic.to_excel(writer, sheet_name="by_topic", index=False)

    print("Wrote router variant research to", OUT)
    print(overall.sort_values(["precision_pct", "coverage_pct"], ascending=[False, False]).to_string(index=False))


if __name__ == "__main__":
    main()
