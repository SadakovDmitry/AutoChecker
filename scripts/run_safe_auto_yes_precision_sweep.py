from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import pandas as pd

from auto_classifier.config import ValidatorConfig, load_rules
from auto_classifier.data import load_tables
from auto_classifier.model import HybridValidator
from auto_classifier.reports import write_table
from auto_classifier.subreason_mapping import (
    apply_subreason_mapping,
    load_subreason_mapping,
    use_subreason_key_as_reason_id,
)


warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

BASE = Path("auto_classifier/local_data/prepared")
OUT = Path("auto_classifier/local_data/reports/safe_auto_yes_precision_sweep")
MAPPING_PATH = "auto_classifier/configs/subreason_versions.yaml"
TARGETS = [0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80, 0.78, 0.75, 0.72, 0.70, 0.65, 0.60]


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


def _load_stable(path: Path, mapping) -> pd.DataFrame:
    frame = load_tables([str(path)], require_text=True, require_answer=True)
    frame = apply_subreason_mapping(frame, mapping)
    return use_subreason_key_as_reason_id(frame)


def _metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    labeled = predictions[predictions["human_label"].isin([0, 1])].copy()
    auto_yes = labeled[labeled["decision"].eq("auto_yes")]
    false_yes = auto_yes[auto_yes["human_label"].eq(0)]
    true_yes = auto_yes[auto_yes["human_label"].eq(1)]
    precision = len(true_yes) / len(auto_yes) if len(auto_yes) else 0.0
    coverage = len(auto_yes) / len(labeled) if len(labeled) else 0.0
    return {
        "rows": int(len(labeled)),
        "auto_yes": int(len(auto_yes)),
        "review": int(len(labeled) - len(auto_yes)),
        "false_auto_yes": int(len(false_yes)),
        "auto_yes_precision_pct": round(precision * 100, 2),
        "auto_yes_coverage_pct": round(coverage * 100, 2),
        "error_rate_among_all_rows_pct": round(len(false_yes) / len(labeled) * 100, 2) if len(labeled) else 0.0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = load_subreason_mapping(MAPPING_PATH)
    rules = load_rules(None)

    loaded = {}
    for test in TESTS:
        loaded[test.key] = (
            _load_stable(BASE / test.train, mapping),
            _load_stable(BASE / test.validation, mapping),
        )

    rows = []
    topic_rows = []
    for target in TARGETS:
        all_predictions = []
        for test in TESTS:
            train, validation = loaded[test.key]
            config = ValidatorConfig(
                target_precision=target,
                use_embeddings=False,
                enable_auto_no=False,
                rules=rules,
            )
            model = HybridValidator.train(train, config)
            predictions = model.predict(validation)
            predictions.insert(0, "target_precision", target)
            predictions.insert(0, "topic", test.topic)
            predictions.insert(0, "product", test.product)
            predictions.insert(0, "test", test.key)
            all_predictions.append(predictions)

            item = {
                "target_precision": target,
                "test": test.key,
                "product": test.product,
                "topic": test.topic,
            }
            item.update(_metrics(predictions))
            topic_rows.append(item)

        joined = pd.concat(all_predictions, ignore_index=True)
        item = {"target_precision": target}
        item.update(_metrics(joined))
        rows.append(item)

    overall = pd.DataFrame(rows)
    by_topic = pd.DataFrame(topic_rows)
    write_table(overall, str(OUT / "safe_auto_yes_precision_sweep_overall.csv"))
    write_table(by_topic, str(OUT / "safe_auto_yes_precision_sweep_by_topic.csv"))

    with pd.ExcelWriter(OUT / "safe_auto_yes_precision_sweep.xlsx", engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="overall", index=False)
        by_topic.to_excel(writer, sheet_name="by_topic", index=False)

    print("Wrote precision sweep to", OUT)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
