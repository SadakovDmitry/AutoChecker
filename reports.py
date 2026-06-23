from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def write_table(frame: pd.DataFrame, output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        try:
            frame.to_excel(path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "openpyxl is required to write .xlsx files. Install auto_classifier/requirements.txt "
                "or use a .csv output path."
            ) from exc
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def build_evaluation_report(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "human_label" not in predictions.columns:
        raise ValueError("Evaluation requires human_label column.")
    labeled = predictions[predictions["human_label"].isin([0, 1])].copy()
    rows = []
    for reason_id, group in labeled.groupby("reason_id", sort=True):
        auto_yes = group[group["decision"].isin(["auto_yes", "accept"])]
        auto_no = group[group["decision"] == "auto_no"]
        auto_total = len(auto_yes) + len(auto_no)
        review = len(group) - auto_total
        rows.append(
            {
                "reason_id": reason_id,
                "rows": len(group),
                "auto_yes": len(auto_yes),
                "auto_no": len(auto_no),
                "review": review,
                "auto_coverage": auto_total / len(group) if len(group) else 0.0,
                "auto_yes_precision": auto_yes["human_label"].mean() if len(auto_yes) else 0.0,
                "auto_no_precision": (auto_no["human_label"] == 0).mean() if len(auto_no) else 0.0,
                "overall_positive_rate": group["human_label"].mean() if len(group) else 0.0,
                "yes_threshold": group["yes_threshold"].iloc[0] if "yes_threshold" in group and len(group) else None,
                "no_threshold": group["no_threshold"].iloc[0] if "no_threshold" in group and len(group) else None,
            }
        )
    summary = pd.DataFrame(rows)
    false_yes = labeled[
        (labeled["decision"].isin(["auto_yes", "accept"])) & (labeled["human_label"] == 0)
    ].copy()
    false_yes["error_type"] = "false_auto_yes"
    false_no = labeled[
        (labeled["decision"] == "auto_no") & (labeled["human_label"] == 1)
    ].copy()
    false_no["error_type"] = "false_auto_no"
    errors = pd.concat([false_yes, false_no], ignore_index=True)
    errors = errors.sort_values(["reason_id", "p_correct"], ascending=[True, False])
    return summary, errors


def write_evaluation(predictions: pd.DataFrame, output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary, accepted_errors = build_evaluation_report(predictions)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        try:
            with pd.ExcelWriter(path) as writer:
                summary.to_excel(writer, sheet_name="summary", index=False)
                predictions.to_excel(writer, sheet_name="predictions", index=False)
                accepted_errors.to_excel(writer, sheet_name="accepted_errors", index=False)
        except ImportError as exc:
            raise RuntimeError(
                "openpyxl is required to write .xlsx reports. Install requirements or use .csv."
            ) from exc
    else:
        stem = path.with_suffix("")
        summary.to_csv(f"{stem}_summary.csv", index=False, encoding="utf-8-sig")
        predictions.to_csv(f"{stem}_predictions.csv", index=False, encoding="utf-8-sig")
        accepted_errors.to_csv(f"{stem}_accepted_errors.csv", index=False, encoding="utf-8-sig")
