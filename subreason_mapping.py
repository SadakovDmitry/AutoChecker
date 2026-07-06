from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from .data import normalize_reason_id
from .iteration_compare import sheet_sort_key


@dataclass
class SubreasonMapping:
    raw: Mapping[str, Any]
    datasets: Mapping[str, Any]
    file_to_dataset: Mapping[str, str]


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "pyyaml is required to read subreason mapping files. Install requirements.txt."
        ) from exc
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Invalid subreason mapping file: {path}")
    return data


def _file_keys(value: object) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    path = Path(text)
    return {text, path.name, path.stem}


def load_subreason_mapping(path: Optional[str]) -> Optional[SubreasonMapping]:
    if not path:
        return None
    raw = _load_yaml(path)
    datasets = raw.get("datasets", raw)
    if not isinstance(datasets, Mapping):
        raise ValueError("Subreason mapping must contain a mapping under 'datasets'.")

    file_to_dataset: dict[str, str] = {}
    for dataset_key, block in datasets.items():
        if block is None:
            block = {}
        if not isinstance(block, Mapping):
            raise ValueError(f"Invalid mapping block for dataset {dataset_key!r}.")
        aliases = [dataset_key, *block.get("files", []), *block.get("aliases", [])]
        for alias in aliases:
            for key in _file_keys(alias):
                file_to_dataset[key] = str(dataset_key)

    return SubreasonMapping(raw=raw, datasets=datasets, file_to_dataset=file_to_dataset)


def _dataset_for_row(row: pd.Series, mapping: SubreasonMapping) -> str:
    explicit = str(row.get("_dataset", "") or "").strip()
    if explicit and explicit in mapping.datasets:
        return explicit
    source_file = row.get("_source_file", "")
    for key in _file_keys(source_file):
        dataset = mapping.file_to_dataset.get(key)
        if dataset:
            return dataset
    return ""


def _iteration_for_row(row: pd.Series) -> str:
    return str(row.get("_source_sheet", "") or "").strip()


def _reason_map_for_iteration(dataset_block: Mapping[str, Any], iteration: str) -> tuple[Mapping[str, Any], str, str]:
    iterations = dataset_block.get("iterations", {})
    if not isinstance(iterations, Mapping):
        return {}, "", "no_iterations"
    block = iterations.get(iteration)
    if block is None:
        block = iterations.get(str(iteration))
    if block is not None:
        source_iteration = iteration
        status = "exact"
    else:
        candidates = [str(key) for key in iterations.keys()]
        previous = [
            key
            for key in candidates
            if sheet_sort_key(key) <= sheet_sort_key(iteration)
        ]
        if previous:
            source_iteration = max(previous, key=sheet_sort_key)
            block = iterations.get(source_iteration)
            status = "fallback_previous_iteration"
        else:
            source_iteration = ""
            status = "missing_iteration"
    if isinstance(block, Mapping) and "reasons" in block:
        block = block.get("reasons", {})
    return (block if isinstance(block, Mapping) else {}), source_iteration, status


def _fallback_key(dataset: str, iteration: str, reason_id: str) -> str:
    dataset_part = dataset or "unknown_dataset"
    iteration_part = iteration or "unknown_iteration"
    return f"unmapped::{dataset_part}::{iteration_part}::{reason_id}"


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def repair_swapped_reason_and_rn(frame: pd.DataFrame) -> pd.DataFrame:
    """Fix sheets where reason number and row number were parsed backwards.

    Some manually repaired sheets have rows like reason_id=10, rn=1 while the
    real meaning is reason_id=1, rn=10. The signal is stable: reason_id contains
    many values above 9, while rn contains a small set of repeated low reason
    numbers. We repair per source sheet so normal sheets are left unchanged.
    """

    if "rn" not in frame.columns:
        return frame
    reason_col = "reason_id" if "reason_id" in frame.columns else "reason_number" if "reason_number" in frame.columns else ""
    if not reason_col:
        return frame

    out = frame.copy()
    group_cols = [col for col in ["_source_file", "_source_sheet"] if col in out.columns]
    groups = out.groupby(group_cols, dropna=False).groups if group_cols else {None: out.index}
    for _, index in groups.items():
        idx = list(index)
        reason_values = _numeric_series(out.loc[idx, reason_col])
        rn_values = _numeric_series(out.loc[idx, "rn"])
        valid = reason_values.notna() & rn_values.notna()
        if not valid.any():
            continue
        reason_valid = reason_values[valid]
        rn_valid = rn_values[valid]
        high_reason_share = float((reason_valid > 9).mean())
        low_rn_share = float(rn_valid.between(1, 9).mean())
        rn_unique = int(rn_valid.nunique())
        reason_unique = int(reason_valid.nunique())
        repeated_low_rn = rn_unique <= 12 and reason_unique > rn_unique
        if high_reason_share >= 0.20 and low_rn_share >= 0.90 and repeated_low_rn:
            old_reason = out.loc[idx, reason_col].copy()
            old_rn = out.loc[idx, "rn"].copy()
            out.loc[idx, reason_col] = old_rn.map(normalize_reason_id)
            out.loc[idx, "rn"] = old_reason.map(normalize_reason_id)
            if reason_col != "reason_id" and "reason_id" in out.columns:
                out.loc[idx, "reason_id"] = old_rn.map(normalize_reason_id)
            if "reason_id_raw" in out.columns:
                out.loc[idx, "reason_id_raw"] = old_rn.map(normalize_reason_id)
            if "reason_number" in out.columns:
                out.loc[idx, "reason_number"] = old_rn.map(normalize_reason_id)
            out.loc[idx, "reason_rn_repair_status"] = "swapped_reason_and_rn"
    if "reason_rn_repair_status" not in out.columns:
        out["reason_rn_repair_status"] = ""
    else:
        out["reason_rn_repair_status"] = out["reason_rn_repair_status"].fillna("")
    return out


def apply_subreason_mapping(frame: pd.DataFrame, mapping: Optional[SubreasonMapping]) -> pd.DataFrame:
    """Add stable subreason_key based on dataset + iteration + reason_id.

    Rows without an explicit mapping get a conservative fallback key that keeps
    them separate per dataset/iteration/reason_id. That avoids accidental mixing
    of different meanings when prompt versions changed.
    """

    out = repair_swapped_reason_and_rn(frame)
    if "reason_id" not in out.columns:
        return out
    out["original_reason_id"] = out["reason_id"].map(normalize_reason_id)
    if mapping is None:
        out["subreason_key"] = out["original_reason_id"]
        out["subreason_name"] = out["original_reason_id"]
        out["subreason_mapping_status"] = "no_mapping"
        return out

    keys: list[str] = []
    names: list[str] = []
    statuses: list[str] = []
    datasets: list[str] = []
    iterations: list[str] = []
    source_iterations: list[str] = []

    for _, row in out.iterrows():
        reason_id = normalize_reason_id(row.get("reason_id", ""))
        dataset = _dataset_for_row(row, mapping)
        iteration = _iteration_for_row(row)
        dataset_block = mapping.datasets.get(dataset, {}) if dataset else {}
        if isinstance(dataset_block, Mapping):
            reason_map, source_iteration, iteration_status = _reason_map_for_iteration(dataset_block, iteration)
        else:
            reason_map, source_iteration, iteration_status = {}, "", "missing_dataset"
        mapped = reason_map.get(reason_id)
        if mapped is None:
            mapped = reason_map.get(str(reason_id))

        if mapped is None:
            key = _fallback_key(dataset, iteration, reason_id)
            keys.append(key)
            names.append("")
            statuses.append("unmapped" if iteration_status == "exact" else f"unmapped_{iteration_status}")
        elif isinstance(mapped, Mapping):
            key = str(mapped.get("key") or mapped.get("subreason_key") or "").strip()
            name = str(mapped.get("name") or mapped.get("title") or "").strip()
            if not key:
                key = _fallback_key(dataset, iteration, reason_id)
                statuses.append("mapped_without_key")
            else:
                statuses.append("mapped" if iteration_status == "exact" else f"mapped_{iteration_status}")
            keys.append(key)
            names.append(name or key)
        else:
            key = str(mapped).strip()
            keys.append(key)
            names.append(key)
            statuses.append("mapped" if iteration_status == "exact" else f"mapped_{iteration_status}")
        datasets.append(dataset)
        iterations.append(iteration)
        source_iterations.append(source_iteration)

    out["subreason_key"] = keys
    out["subreason_name"] = names
    out["subreason_mapping_status"] = statuses
    out["subreason_dataset"] = datasets
    out["subreason_iteration"] = iterations
    out["subreason_source_iteration"] = source_iterations
    return out


def use_subreason_key_as_reason_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where model grouping uses stable subreason keys."""

    out = frame.copy()
    if "subreason_key" not in out.columns:
        return out
    if "original_reason_id" not in out.columns:
        out["original_reason_id"] = out["reason_id"].map(normalize_reason_id)
    out["reason_id"] = out["subreason_key"].astype(str)
    return out
