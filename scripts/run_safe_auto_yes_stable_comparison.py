from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import pandas as pd

from auto_classifier.config import ValidatorConfig, load_rules
from auto_classifier.data import load_tables
from auto_classifier.model import HybridValidator
from auto_classifier.reports import build_evaluation_report, write_table
from auto_classifier.subreason_mapping import (
    apply_subreason_mapping,
    load_subreason_mapping,
    use_subreason_key_as_reason_id,
)


warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

BASE = Path("auto_classifier/local_data/prepared")
OUT = Path("auto_classifier/local_data/reports/safe_auto_yes_stable_comparison")
MAPPING_PATH = "auto_classifier/configs/subreason_versions.yaml"


@dataclass(frozen=True)
class TestCase:
    key: str
    product: str
    topic: str
    train: str
    validation: str


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


def _load(path: Path, mapping=None, stable: bool = False) -> pd.DataFrame:
    frame = load_tables([str(path)], require_text=True, require_answer=True)
    if stable:
        frame = apply_subreason_mapping(frame, mapping)
        frame = use_subreason_key_as_reason_id(frame)
    return frame


def _summarize_predictions(test: TestCase, mode: str, predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    labeled = predictions[predictions["human_label"].isin([0, 1])].copy()
    auto_yes = labeled[labeled["decision"].eq("auto_yes")]
    false_yes = auto_yes[auto_yes["human_label"].eq(0)]
    precision = float(auto_yes["human_label"].mean()) if len(auto_yes) else 0.0
    coverage = float(len(auto_yes) / len(labeled)) if len(labeled) else 0.0
    summary, errors = build_evaluation_report(predictions)
    summary = summary.copy()
    summary.insert(0, "mode", mode)
    summary.insert(0, "topic", test.topic)
    summary.insert(0, "product", test.product)
    summary.insert(0, "test", test.key)
    row = {
        "test": test.key,
        "product": test.product,
        "topic": test.topic,
        "mode": mode,
        "rows": int(len(labeled)),
        "auto_yes": int(len(auto_yes)),
        "review": int(len(labeled) - len(auto_yes)),
        "auto_yes_precision_pct": round(precision * 100, 2),
        "auto_yes_coverage_pct": round(coverage * 100, 2),
        "false_auto_yes": int(len(false_yes)),
        "reasons_in_validation": int(labeled["reason_id"].nunique()),
        "reasons_with_auto_yes": int(auto_yes["reason_id"].nunique()) if len(auto_yes) else 0,
    }
    return row, summary


def _overall(rows: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for mode, group in rows.groupby("mode", sort=True):
        total = int(group["rows"].sum())
        auto = int(group["auto_yes"].sum())
        false = int(group["false_auto_yes"].sum())
        precision = (auto - false) / auto if auto else 0.0
        coverage = auto / total if total else 0.0
        out_rows.append(
            {
                "mode": mode,
                "rows": total,
                "auto_yes": auto,
                "review": int(group["review"].sum()),
                "auto_yes_precision_pct": round(precision * 100, 2),
                "auto_yes_coverage_pct": round(coverage * 100, 2),
                "false_auto_yes": false,
                "tests": int(group["test"].nunique()),
            }
        )
    return pd.DataFrame(out_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = load_subreason_mapping(MAPPING_PATH)
    config = ValidatorConfig(
        target_precision=0.90,
        use_embeddings=False,
        enable_auto_no=False,
        rules=load_rules(None),
    )

    topic_rows: list[dict] = []
    reason_summaries = []
    all_predictions = []

    for test in TESTS:
        for mode, stable in [("raw_reason_id", False), ("stable_subreason_key", True)]:
            train = _load(BASE / test.train, mapping=mapping, stable=stable)
            validation = _load(BASE / test.validation, mapping=mapping, stable=stable)
            model = HybridValidator.train(train, config)
            predictions = model.predict(validation)
            predictions.insert(0, "mode", mode)
            predictions.insert(0, "topic", test.topic)
            predictions.insert(0, "product", test.product)
            predictions.insert(0, "test", test.key)
            topic_row, reason_summary = _summarize_predictions(test, mode, predictions)
            topic_rows.append(topic_row)
            reason_summaries.append(reason_summary)
            all_predictions.append(predictions)

    by_topic = pd.DataFrame(topic_rows)
    by_reason = pd.concat(reason_summaries, ignore_index=True) if reason_summaries else pd.DataFrame()
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    overall = _overall(by_topic)

    write_table(overall, str(OUT / "safe_auto_yes_overall.csv"))
    write_table(by_topic, str(OUT / "safe_auto_yes_by_topic.csv"))
    write_table(by_reason, str(OUT / "safe_auto_yes_by_reason.csv"))
    write_table(predictions, str(OUT / "safe_auto_yes_predictions.csv"))

    with pd.ExcelWriter(OUT / "safe_auto_yes_comparison.xlsx", engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="overall", index=False)
        by_topic.to_excel(writer, sheet_name="by_topic", index=False)
        by_reason.to_excel(writer, sheet_name="by_reason", index=False)
        predictions.to_excel(writer, sheet_name="predictions", index=False)

    print("Wrote safe auto-yes comparison to", OUT)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
