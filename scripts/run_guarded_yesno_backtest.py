from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import pandas as pd

from auto_classifier.config import ValidatorConfig, load_rules
from auto_classifier.data import load_tables
from auto_classifier.quota_yesno import run_guarded_bayes_yesno_experiment
from auto_classifier.reports import write_table
from auto_classifier.subreason_mapping import (
    apply_subreason_mapping,
    load_subreason_mapping,
    use_subreason_key_as_reason_id,
)


warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

BASE = Path("auto_classifier/local_data/prepared")
OUT = Path("auto_classifier/local_data/reports/guarded_yesno_backtest")
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


def _load_stable(path: Path, mapping) -> pd.DataFrame:
    frame = load_tables([str(path)], require_text=True, require_answer=True)
    frame = apply_subreason_mapping(frame, mapping)
    return use_subreason_key_as_reason_id(frame)


def _add_names(summary: pd.DataFrame, *frames: pd.DataFrame) -> pd.DataFrame:
    names: dict[str, str] = {}
    for frame in frames:
        if "subreason_key" not in frame.columns or "subreason_name" not in frame.columns:
            continue
        for _, row in frame[["subreason_key", "subreason_name"]].dropna().iterrows():
            key = str(row["subreason_key"])
            name = str(row["subreason_name"] or "").strip()
            if name:
                names[key] = name
    out = summary.copy()
    out["subreason_name"] = out["reason_id"].astype(str).map(names).fillna("")
    return out


def _overall_row(summary: pd.DataFrame) -> dict:
    overall = summary[summary["reason_id"].eq("__overall_weighted__")]
    return overall.iloc[0].to_dict() if not overall.empty else {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = load_subreason_mapping(MAPPING_PATH)
    config = ValidatorConfig(use_embeddings=False, rules=load_rules(None))

    all_summary = []
    all_details = []
    all_offsets = []
    for test in TESTS:
        train_path = BASE / test.train
        validation_path = BASE / test.validation
        if not train_path.exists() or not validation_path.exists():
            continue
        train = _load_stable(train_path, mapping)
        validation = _load_stable(validation_path, mapping)
        result = run_guarded_bayes_yesno_experiment(
            train_frame=train,
            evaluation_frame=validation,
            config=config,
            offset=None,
            k=40.0,
            guard_gap=0.10,
        )

        summary = _add_names(result.summary, train, validation)
        summary.insert(0, "topic", test.topic)
        summary.insert(0, "product", test.product)
        summary.insert(0, "test", test.key)
        all_details.append(summary)

        offset = result.offset_summary.copy()
        offset.insert(0, "topic", test.topic)
        offset.insert(0, "product", test.product)
        offset.insert(0, "test", test.key)
        all_offsets.append(offset)

        row = _overall_row(summary)
        if row:
            all_summary.append(
                {
                    "test": test.key,
                    "product": test.product,
                    "topic": test.topic,
                    "rows": int(row["rows"]),
                    "manual_accuracy_pct": round(float(row["manual_prompt_accuracy"]) * 100, 2),
                    "estimated_accuracy_pct": round(float(row["estimated_prompt_accuracy"]) * 100, 2),
                    "gap_pp": round(float(row["accuracy_gap_pp"]), 2),
                    "abs_gap_pp": round(float(row["abs_accuracy_gap_pp"]), 2),
                    "row_label_accuracy_pct": round(float(row.get("row_label_accuracy", 0.0)) * 100, 2),
                    "offset": round(float(offset.iloc[0]["offset"]), 4) if not offset.empty else 0.0,
                    "offset_source": str(offset.iloc[0]["source"]) if not offset.empty else "",
                }
            )

    details = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    offsets = pd.concat(all_offsets, ignore_index=True) if all_offsets else pd.DataFrame()
    summary = pd.DataFrame(all_summary)
    if not summary.empty:
        total_rows = int(summary["rows"].sum())
        weighted_abs = float((summary["abs_gap_pp"] * summary["rows"]).sum() / total_rows)
        weighted_manual = float((summary["manual_accuracy_pct"] * summary["rows"]).sum() / total_rows)
        weighted_estimated = float((summary["estimated_accuracy_pct"] * summary["rows"]).sum() / total_rows)
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    [
                        {
                            "test": "__overall_weighted__",
                            "product": "",
                            "topic": "",
                            "rows": total_rows,
                            "manual_accuracy_pct": round(weighted_manual, 2),
                            "estimated_accuracy_pct": round(weighted_estimated, 2),
                            "gap_pp": round(weighted_estimated - weighted_manual, 2),
                            "abs_gap_pp": round(weighted_abs, 2),
                            "row_label_accuracy_pct": "",
                            "offset": "",
                            "offset_source": "",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    write_table(summary, str(OUT / "guarded_yesno_summary_by_topic.csv"))
    write_table(details, str(OUT / "guarded_yesno_subreason_details.csv"))
    write_table(offsets, str(OUT / "guarded_yesno_offsets.csv"))
    with pd.ExcelWriter(OUT / "guarded_yesno_backtest.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary_by_topic", index=False)
        details.to_excel(writer, sheet_name="subreason_details", index=False)
        offsets.to_excel(writer, sheet_name="offsets", index=False)

    print("Wrote guarded yes/no backtest to", OUT)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
