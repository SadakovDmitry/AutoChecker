from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from auto_classifier.data import normalize_reason_id, read_input_table
from auto_classifier.reports import write_table
from auto_classifier.subreason_mapping import apply_subreason_mapping, load_subreason_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Add stable subreason columns to normalized label workbooks.")
    parser.add_argument("--input-dir", default="auto_classifier/local_data/labels_normalized")
    parser.add_argument("--output-dir", default="auto_classifier/local_data/labels_mapped")
    parser.add_argument("--mapping", default="auto_classifier/configs/subreason_versions.yaml")
    parser.add_argument("--report", default="auto_classifier/local_data/reports/subreason_mapping_coverage.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_subreason_mapping(args.mapping)
    if mapping is None:
        raise ValueError("Mapping file is required.")

    report_rows = []
    for path in sorted(input_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        normalized = read_input_table(path)
        if "reason_id" not in normalized.columns and "reason_number" in normalized.columns:
            normalized = normalized.copy()
            normalized["reason_id"] = normalized["reason_number"].map(normalize_reason_id)
        mapped = apply_subreason_mapping(normalized, mapping)
        output_path = output_dir / path.name
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name, group in mapped.groupby("_source_sheet", sort=False):
                out = group.drop(columns=["_source_file", "_source_sheet"], errors="ignore")
                out.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)
                report_rows.append(
                    {
                        "file": path.name,
                        "sheet": sheet_name,
                        "rows": int(len(group)),
                        "mapped_rows": int(group["subreason_mapping_status"].astype(str).str.startswith("mapped").sum()),
                        "unmapped_rows": int(group["subreason_mapping_status"].astype(str).str.startswith("unmapped").sum()),
                        "fallback_rows": int(group["subreason_mapping_status"].astype(str).str.contains("fallback").sum()),
                        "missing_iteration_rows": int((group["subreason_mapping_status"] == "missing_iteration").sum()),
                        "unique_subreason_keys": int(group["subreason_key"].nunique()),
                        "output_file": str(output_path),
                    }
                )

    report = pd.DataFrame(report_rows)
    write_table(report, args.report)
    print(f"Wrote mapped workbooks to {output_dir}")
    print(f"Wrote report to {args.report}")
    if not report.empty:
        print(
            report[["file", "sheet", "rows", "mapped_rows", "fallback_rows", "unmapped_rows", "missing_iteration_rows"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
