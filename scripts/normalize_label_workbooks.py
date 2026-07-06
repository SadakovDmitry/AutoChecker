from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from auto_classifier.data import DataFormatError, normalize_human_answer, normalize_table, read_input_table
from auto_classifier.prepare import _repair_positional_label_columns
from auto_classifier.reports import write_table
from auto_classifier.subreason_mapping import repair_swapped_reason_and_rn
from auto_classifier.text import clean_text


OUTPUT_COLUMNS = ["comm_id", "link", "reason_number", "rn", "да/нет", "комментарий"]


def _text(value: object) -> str:
    return clean_text(value)


def _normalized_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    repaired = _repair_positional_label_columns(frame)
    normalized = normalize_table(repaired, require_text=False, require_answer=True)
    if normalized.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.DataFrame()
    out["comm_id"] = normalized["chat_id"].map(_text)
    out["link"] = normalized["link"].map(_text) if "link" in normalized.columns else ""
    out["reason_number"] = normalized["reason_id"].map(_text)
    if "rn" in normalized.columns:
        out["rn"] = normalized["rn"].map(_text)
    else:
        out["rn"] = normalized.groupby("reason_id").cumcount().add(1).astype(str)
    out["да/нет"] = normalized["human_answer_raw"].map(_text)
    out["комментарий"] = normalized["comment"].map(_text) if "comment" in normalized.columns else ""
    out = repair_swapped_reason_and_rn(out)

    # Keep only rows that contain enough information to be useful for training
    # or evaluation. Blank answers are preserved only if the row has a chat id,
    # because they may be useful for manual follow-up, but rows without chat id
    # cannot be matched to text exports.
    out = out[(out["comm_id"] != "") & (out["reason_number"] != "")].copy()
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def normalize_workbook(path: Path, output_dir: Path) -> list[dict[str, object]]:
    raw = read_input_table(path)
    output_path = output_dir / path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    wrote_any_sheet = False
    try:
        sheet_names = [str(x) for x in pd.ExcelFile(path).sheet_names]
    except Exception:
        sheet_names = []
    raw_sheet_names = set()
    if "_source_sheet" in raw.columns:
        raw_sheet_names = set(str(x) for x in raw["_source_sheet"].dropna().unique())
    if not sheet_names:
        sheet_names = sorted(raw_sheet_names)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name in sheet_names:
            if sheet_name not in raw_sheet_names:
                rows.append(
                    {
                        "file": path.name,
                        "sheet": sheet_name,
                        "status": "empty_source_sheet",
                        "raw_rows": 0,
                        "normalized_rows": 0,
                        "nonempty_answer_rows": 0,
                        "labeled_rows": 0,
                        "output_file": "",
                    }
                )
                continue
            sheet_raw = raw[raw["_source_sheet"].astype(str) == sheet_name].copy()
            try:
                sheet_out = _normalized_sheet(sheet_raw)
                status = "ok" if not sheet_out.empty else "empty_after_normalization"
                nonempty_answer_rows = int(sheet_out["да/нет"].str.strip().ne("").sum()) if not sheet_out.empty else 0
                labeled_rows = (
                    int(sheet_out["да/нет"].map(normalize_human_answer).isin([0, 1]).sum())
                    if not sheet_out.empty
                    else 0
                )
                if not sheet_out.empty:
                    safe_sheet = str(sheet_name)[:31]
                    sheet_out.to_excel(writer, sheet_name=safe_sheet, index=False)
                    wrote_any_sheet = True
                rows.append(
                    {
                        "file": path.name,
                        "sheet": sheet_name,
                        "status": status,
                        "raw_rows": int(len(sheet_raw)),
                        "normalized_rows": int(len(sheet_out)),
                        "nonempty_answer_rows": nonempty_answer_rows,
                        "labeled_rows": labeled_rows,
                        "output_file": str(output_path) if not sheet_out.empty else "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "file": path.name,
                        "sheet": sheet_name,
                        "status": f"error: {type(exc).__name__}",
                        "raw_rows": int(len(sheet_raw)),
                        "normalized_rows": 0,
                        "nonempty_answer_rows": 0,
                        "labeled_rows": 0,
                        "output_file": "",
                        "error": str(exc),
                    }
                )

        if not wrote_any_sheet:
            pd.DataFrame(columns=OUTPUT_COLUMNS).to_excel(writer, sheet_name="empty", index=False)
            rows.append(
                {
                    "file": path.name,
                    "sheet": "__workbook__",
                    "status": "no_recoverable_sheets",
                    "raw_rows": int(len(raw)),
                    "normalized_rows": 0,
                    "nonempty_answer_rows": 0,
                    "labeled_rows": 0,
                    "output_file": str(output_path),
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize manual label workbooks into a stable label format.")
    parser.add_argument("--input-dir", default="auto_classifier/local_data/labels")
    parser.add_argument("--output-dir", default="auto_classifier/local_data/labels_normalized")
    parser.add_argument("--report", default="auto_classifier/local_data/reports/labels_normalization_summary.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)

    if not input_dir.exists():
        raise DataFormatError(f"Input directory does not exist: {input_dir}")

    all_rows: list[dict[str, object]] = []
    for path in sorted(input_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        all_rows.extend(normalize_workbook(path, output_dir))

    report = pd.DataFrame(all_rows)
    write_table(report, str(report_path))
    print(f"Wrote normalized workbooks to {output_dir}")
    print(f"Wrote report to {report_path}")
    if not report.empty:
        print(
            report.groupby("status", dropna=False)
            .agg(sheets=("sheet", "count"), rows=("normalized_rows", "sum"), labeled=("labeled_rows", "sum"))
            .reset_index()
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
