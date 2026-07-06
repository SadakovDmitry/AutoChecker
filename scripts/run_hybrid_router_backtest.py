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
OUT = Path("auto_classifier/local_data/reports/hybrid_router_backtest")
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


def _name_lookup(*frames: pd.DataFrame) -> dict[str, str]:
    names: dict[str, str] = {}
    for frame in frames:
        if "subreason_key" not in frame.columns or "subreason_name" not in frame.columns:
            continue
        for _, row in frame[["subreason_key", "subreason_name"]].dropna().iterrows():
            key = str(row["subreason_key"])
            name = str(row["subreason_name"] or "").strip()
            if name:
                names[key] = name
    return names


def _add_context(frame: pd.DataFrame, test: TestCase, names: dict[str, str]) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "topic", test.topic)
    out.insert(0, "product", test.product)
    out.insert(0, "test", test.key)
    if "reason_id" in out.columns:
        out["subreason_name"] = out["reason_id"].astype(str).map(names).fillna("")
    return out


def _overall(summary: pd.DataFrame) -> dict:
    row = summary[summary["reason_id"].eq("__overall_weighted__")]
    return row.iloc[0].to_dict() if not row.empty else {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = load_subreason_mapping(MAPPING_PATH)
    config = ValidatorConfig(
        target_precision=0.80,
        use_embeddings=False,
        rules=load_rules(None),
    )

    topic_rows = []
    detail_frames = []
    risk_frames = []
    offset_frames = []
    prediction_frames = []

    for test in TESTS:
        train_path = BASE / test.train
        validation_path = BASE / test.validation
        if not train_path.exists() or not validation_path.exists():
            continue
        train = _load_stable(train_path, mapping)
        validation = _load_stable(validation_path, mapping)
        names = _name_lookup(train, validation)
        result = run_hybrid_router_experiment(
            train_frame=train,
            evaluation_frame=validation,
            config=config,
            offset=None,
            k=40.0,
            guard_gap=0.10,
            min_offset=0.0,
            max_offset=0.15,
            min_history_rows=10,
            max_history_std=0.05,
            max_model_history_gap=0.35,
            min_estimated_accuracy=0.70,
        )
        detail_frames.append(_add_context(result.summary, test, names))
        risk_frames.append(_add_context(result.risk_summary, test, names))
        offset_frames.append(_add_context(result.offset_summary, test, names))
        predictions = result.predictions.copy()
        predictions.insert(0, "topic", test.topic)
        predictions.insert(0, "product", test.product)
        predictions.insert(0, "test", test.key)
        prediction_frames.append(predictions)

        row = _overall(result.summary)
        if row:
            topic_rows.append(
                {
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
            )

    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    risks = pd.concat(risk_frames, ignore_index=True) if risk_frames else pd.DataFrame()
    offsets = pd.concat(offset_frames, ignore_index=True) if offset_frames else pd.DataFrame()
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    summary = pd.DataFrame(topic_rows)
    if not summary.empty:
        rows = int(summary["rows"].sum())
        auto_rows = int(summary["auto_rows"].sum())
        errors = int(summary["errors"].sum())
        overall = {
            "test": "__overall_weighted__",
            "product": "",
            "topic": "",
            "rows": rows,
            "auto_rows": auto_rows,
            "review_rows": int(summary["review_rows"].sum()),
            "coverage_pct": round(auto_rows / rows * 100, 2) if rows else 0.0,
            "precision_pct": round((auto_rows - errors) / auto_rows * 100, 2) if auto_rows else 0.0,
            "errors": errors,
            "auto_yes": int(summary["auto_yes"].sum()),
            "auto_no": int(summary["auto_no"].sum()),
        }
        summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)

    write_table(summary, str(OUT / "hybrid_router_summary_by_topic.csv"))
    write_table(details, str(OUT / "hybrid_router_subreason_details.csv"))
    write_table(risks, str(OUT / "hybrid_router_risk_summary.csv"))
    write_table(offsets, str(OUT / "hybrid_router_offsets.csv"))
    write_table(predictions, str(OUT / "hybrid_router_predictions.csv"))
    with pd.ExcelWriter(OUT / "hybrid_router_backtest.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary_by_topic", index=False)
        details.to_excel(writer, sheet_name="subreason_details", index=False)
        risks.to_excel(writer, sheet_name="risk_summary", index=False)
        offsets.to_excel(writer, sheet_name="offsets", index=False)
        predictions.to_excel(writer, sheet_name="predictions", index=False)

    print("Wrote hybrid router backtest to", OUT)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
