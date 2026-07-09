from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import pandas as pd

from auto_classifier.config import ValidatorConfig, load_rules
from auto_classifier.data import load_tables
from auto_classifier.model import HybridValidator
from auto_classifier.quota_yesno import (
    _max_history_latest_summary,
    apply_hybrid_router,
    build_hybrid_risk_summary,
    build_hybrid_summary,
    learned_thresholds_from_model,
)
from auto_classifier.reports import write_table
from auto_classifier.subreason_mapping import (
    apply_subreason_mapping,
    load_subreason_mapping,
    use_subreason_key_as_reason_id,
)


warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

BASE = Path("auto_classifier/local_data/prepared")
OUT = Path("auto_classifier/local_data/reports/hybrid_fullness_sweep")
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
    min_history_rows: int
    max_history_std: float
    max_model_history_gap: float
    min_estimated_accuracy: float
    description: str
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
    Variant("current", 10, 0.05, 0.35, 0.70, "Текущий рабочий роутер."),
    Variant("std_055", 10, 0.055, 0.35, 0.70, "Минимально мягче по historical std."),
    Variant("std_060", 10, 0.06, 0.35, 0.70, "Немного мягче по historical std."),
    Variant("std_070", 10, 0.07, 0.35, 0.70, "Мягче по historical std."),
    Variant("std_080", 10, 0.08, 0.35, 0.70, "Еще мягче по historical std."),
    Variant("std_010", 10, 0.10, 0.35, 0.70, "Больше допускаем скачки истории."),
    Variant("std_015", 10, 0.15, 0.35, 0.70, "Еще мягче по historical std."),
    Variant("estimate_065", 10, 0.05, 0.35, 0.65, "Чуть мягче по ожидаемой точности подпричины."),
    Variant("estimate_060", 10, 0.05, 0.35, 0.60, "Мягче по ожидаемой точности подпричины."),
    Variant("history_8", 8, 0.05, 0.35, 0.70, "Немного меньше исторических строк."),
    Variant("gap_050", 10, 0.05, 0.50, 0.70, "Мягче по разрыву model-vs-history."),
    Variant("history_5", 5, 0.05, 0.35, 0.70, "Разрешаем меньше исторических строк."),
    Variant("std060_est065", 10, 0.06, 0.35, 0.65, "Комбо: слегка мягче по std и estimated accuracy."),
    Variant("balanced_loose", 5, 0.10, 0.50, 0.60, "Умеренно ослабляем все фильтры."),
    Variant("aggressive", 5, 0.20, 0.60, 0.55, "Сильно ослабляем фильтры."),
    Variant("very_aggressive", 1, 0.30, 0.80, 0.50, "Почти все известные подпричины пускаем в full yes/no."),
    Variant(
        "all_known_full",
        0,
        999.0,
        999.0,
        0.0,
        "Все замапленные подпричины идут в full yes/no.",
        min_full_mean_p_correct=0.0,
        min_full_train_row_accuracy=0.0,
        max_full_train_rate_gap=1.0,
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
    config = ValidatorConfig(target_precision=0.80, use_embeddings=False, rules=load_rules(None))

    summary_rows = []
    topic_rows = []
    risk_rows = []

    for test in TESTS:
        train_path = BASE / test.train
        validation_path = BASE / test.validation
        if not train_path.exists() or not validation_path.exists():
            continue

        train = _load_stable(train_path, mapping)
        validation = _load_stable(validation_path, mapping)

        model = HybridValidator.train(train, config)
        raw_predictions = model.predict(validation)
        guarded_summary = _max_history_latest_summary(raw_predictions, train)
        thresholds = learned_thresholds_from_model(model)

        for variant in VARIANTS:
            risk_summary = build_hybrid_risk_summary(
                guarded_summary,
                train,
                min_history_rows=variant.min_history_rows,
                max_history_std=variant.max_history_std,
                max_model_history_gap=variant.max_model_history_gap,
                min_estimated_accuracy=variant.min_estimated_accuracy,
                min_full_mean_p_correct=variant.min_full_mean_p_correct,
                yesno_thresholds=thresholds,
                min_full_train_row_accuracy=variant.min_full_train_row_accuracy,
                max_full_train_rate_gap=variant.max_full_train_rate_gap,
            )
            predictions = apply_hybrid_router(
                raw_predictions,
                guarded_summary,
                risk_summary,
                full_yesno_strategy="threshold",
                yesno_thresholds=thresholds,
            )
            summary = build_hybrid_summary(predictions, risk_summary)
            row = _overall(summary)
            if not row:
                continue

            labeled_predictions = predictions[predictions["human_label"].isin([0, 1])].copy()
            full_rows = int(labeled_predictions["hybrid_mode"].eq("full_yesno").sum())
            full_subreasons = int(risk_summary["mode"].eq("full_yesno").sum())
            safe_rows = int(labeled_predictions["hybrid_mode"].ne("full_yesno").sum())
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
                "full_rows": full_rows,
                "full_rows_pct": round(full_rows / max(1, int(row["rows"])) * 100, 2),
                "safe_rows": safe_rows,
                "full_subreasons": full_subreasons,
                "min_history_rows": variant.min_history_rows,
                "max_history_std": variant.max_history_std,
                "max_model_history_gap": variant.max_model_history_gap,
                "min_estimated_accuracy": variant.min_estimated_accuracy,
                "min_full_mean_p_correct": variant.min_full_mean_p_correct,
                "min_full_train_row_accuracy": variant.min_full_train_row_accuracy,
                "max_full_train_rate_gap": variant.max_full_train_rate_gap,
            }
            topic_rows.append(item)

            risk = risk_summary.copy()
            risk.insert(0, "variant", variant.name)
            risk.insert(1, "test", test.key)
            risk.insert(2, "product", test.product)
            risk.insert(3, "topic", test.topic)
            risk_rows.append(risk)

    by_topic = pd.DataFrame(topic_rows)
    for variant in VARIANTS:
        part = by_topic[by_topic["variant"].eq(variant.name)]
        if part.empty:
            continue
        rows = int(part["rows"].sum())
        auto_rows = int(part["auto_rows"].sum())
        errors = int(part["errors"].sum())
        full_rows = int(part["full_rows"].sum())
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
                "auto_yes": int(part["auto_yes"].sum()),
                "auto_no": int(part["auto_no"].sum()),
                "full_rows": full_rows,
                "full_rows_pct": round(full_rows / rows * 100, 2) if rows else 0.0,
                "full_subreasons": int(part["full_subreasons"].sum()),
                "min_history_rows": variant.min_history_rows,
                "max_history_std": variant.max_history_std,
                "max_model_history_gap": variant.max_model_history_gap,
                "min_estimated_accuracy": variant.min_estimated_accuracy,
                "min_full_mean_p_correct": variant.min_full_mean_p_correct,
                "min_full_train_row_accuracy": variant.min_full_train_row_accuracy,
                "max_full_train_rate_gap": variant.max_full_train_rate_gap,
            }
        )

    overall = pd.DataFrame(summary_rows)
    risks = pd.concat(risk_rows, ignore_index=True) if risk_rows else pd.DataFrame()
    write_table(overall, str(OUT / "hybrid_fullness_summary.csv"))
    write_table(by_topic, str(OUT / "hybrid_fullness_by_topic.csv"))
    write_table(risks, str(OUT / "hybrid_fullness_risk_summary.csv"))
    with pd.ExcelWriter(OUT / "hybrid_fullness_sweep.xlsx", engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="summary", index=False)
        by_topic.to_excel(writer, sheet_name="by_topic", index=False)
        risks.to_excel(writer, sheet_name="risk_summary", index=False)

    print("Wrote hybrid fullness sweep to", OUT)
    print(overall.sort_values(["coverage_pct", "precision_pct"], ascending=[False, False]).to_string(index=False))


if __name__ == "__main__":
    main()
