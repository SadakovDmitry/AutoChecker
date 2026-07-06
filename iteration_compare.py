from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ValidatorConfig, load_rules
from .data import load_tables, normalize_reason_id
from .model import HybridValidator
from .reports import write_table


def sheet_sort_key(value: object) -> tuple:
    text = str(value)
    numbers = tuple(float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[\.,]\d+)?", text))
    return (numbers, text)


def split_train_calibration(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if "_source_sheet" not in frame.columns:
        return frame.iloc[0:0].copy(), frame.copy(), ""
    sheets = [str(x) for x in frame["_source_sheet"].fillna("").unique()]
    if len(sheets) < 2:
        return frame.iloc[0:0].copy(), frame.copy(), sheets[0] if sheets else ""
    calibration_sheet = max(sheets, key=sheet_sort_key)
    sheet_values = frame["_source_sheet"].fillna("").astype(str)
    train = frame[sheet_values != calibration_sheet].copy()
    calibration = frame[sheet_values == calibration_sheet].copy()
    return train, calibration, calibration_sheet


def ordered_sheets(frame: pd.DataFrame) -> list[str]:
    if "_source_sheet" not in frame.columns:
        return []
    sheets = [str(x) for x in frame["_source_sheet"].fillna("").unique()]
    return sorted(sheets, key=sheet_sort_key)


def choose_threshold_for_rate(y: Iterable[int], p: Iterable[float]) -> tuple[float, float, float]:
    y_arr = np.asarray(list(y), dtype=int)
    p_arr = np.asarray(list(p), dtype=float)
    if len(y_arr) == 0:
        return 0.5, 0.0, 0.0
    true_rate = float(y_arr.mean())
    candidates = [1.000001, *sorted(set(float(x) for x in p_arr), reverse=True), -0.000001]
    best = None
    for threshold in candidates:
        predicted = (p_arr > threshold).astype(int)
        predicted_rate = float(predicted.mean())
        rate_gap = abs(predicted_rate - true_rate)
        row_accuracy = float((predicted == y_arr).mean())
        candidate = (rate_gap, -row_accuracy, -float(threshold), float(threshold), predicted_rate)
        if best is None or candidate < best:
            best = candidate
    _, negative_row_accuracy, _, threshold, predicted_rate = best
    return threshold, predicted_rate, -negative_row_accuracy


def learn_backtest_thresholds(calibration_predictions: pd.DataFrame) -> pd.DataFrame:
    labeled = calibration_predictions[calibration_predictions["human_label"].isin([0, 1])].copy()
    rows = []
    for reason_id, group in labeled.groupby("reason_id", sort=True):
        reason_key = normalize_reason_id(reason_id)
        threshold, predicted_rate, row_accuracy = choose_threshold_for_rate(
            group["human_label"].astype(int),
            group["p_correct"].astype(float),
        )
        rows.append(
            {
                "reason_id": reason_key,
                "calibration_rows": int(len(group)),
                "calibration_true_accuracy": float(group["human_label"].astype(int).mean()),
                "backtest_threshold": threshold,
                "calibration_predicted_accuracy": predicted_rate,
                "calibration_gap_pp": (predicted_rate - float(group["human_label"].astype(int).mean())) * 100,
                "calibration_row_label_accuracy": row_accuracy,
            }
        )
    return pd.DataFrame(rows)


def _thresholds_from_predictions(
    predictions: pd.DataFrame,
    *,
    source: str,
    source_sheet: str = "",
) -> pd.DataFrame:
    if predictions.empty or "human_label" not in predictions.columns:
        return pd.DataFrame()
    thresholds = learn_backtest_thresholds(predictions)
    if thresholds.empty:
        return thresholds
    thresholds = thresholds.copy()
    thresholds["threshold_source"] = source
    thresholds["threshold_source_sheet"] = source_sheet
    return thresholds


def collect_walk_forward_predictions(
    frame: pd.DataFrame,
    *,
    config: ValidatorConfig,
) -> pd.DataFrame:
    """Predict each historical iteration using only earlier iterations.

    This creates out-of-time predictions for threshold calibration. It avoids
    using probabilities produced by a model trained on the same rows.
    """

    sheets = ordered_sheets(frame)
    if len(sheets) < 2:
        return pd.DataFrame()

    predictions = []
    sheet_values = frame["_source_sheet"].fillna("").astype(str)
    for idx in range(1, len(sheets)):
        train_sheets = set(sheets[:idx])
        calibration_sheet = sheets[idx]
        train_part = frame[sheet_values.isin(train_sheets)].copy()
        calibration_part = frame[sheet_values == calibration_sheet].copy()
        if train_part.empty or calibration_part.empty:
            continue
        model = HybridValidator.train(train_part, config)
        predicted = model.predict(calibration_part)
        predicted["_calibration_sheet"] = calibration_sheet
        predictions.append(predicted)
    if not predictions:
        return pd.DataFrame()
    return pd.concat(predictions, ignore_index=True)


def learn_latest_available_thresholds(walk_forward_predictions: pd.DataFrame) -> pd.DataFrame:
    if walk_forward_predictions.empty or "_calibration_sheet" not in walk_forward_predictions.columns:
        return pd.DataFrame()
    rows = []
    for reason_id, group in walk_forward_predictions.groupby("reason_id", sort=True):
        sheets = sorted(
            [str(x) for x in group["_calibration_sheet"].fillna("").unique()],
            key=sheet_sort_key,
            reverse=True,
        )
        for sheet in sheets:
            sheet_group = group[group["_calibration_sheet"].fillna("").astype(str) == sheet]
            thresholds = _thresholds_from_predictions(
                sheet_group,
                source="history_latest_available",
                source_sheet=sheet,
            )
            if not thresholds.empty:
                thresholds = thresholds.copy()
                thresholds["reason_id"] = normalize_reason_id(reason_id)
                rows.append(thresholds)
                break
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def merge_hierarchical_thresholds(
    primary: pd.DataFrame,
    fallback: pd.DataFrame,
) -> pd.DataFrame:
    """Use primary per-reason thresholds first, then historical thresholds."""

    frames = []
    seen = set()
    if not primary.empty:
        primary = primary.copy()
        primary["threshold_source"] = primary.get("threshold_source", "previous_iteration")
        primary["threshold_source_sheet"] = primary.get("threshold_source_sheet", "")
        frames.append(primary)
        primary["reason_id"] = primary["reason_id"].map(normalize_reason_id)
        seen.update(primary["reason_id"].tolist())
    if not fallback.empty:
        fallback = fallback.copy()
        fallback["reason_id"] = fallback["reason_id"].map(normalize_reason_id)
        extra = fallback[~fallback["reason_id"].isin(seen)].copy()
        if not extra.empty:
            frames.append(extra)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _mean_probability_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    labeled = predictions[predictions["human_label"].isin([0, 1])].copy()
    rows = []
    for reason_id, group in labeled.groupby("reason_id", sort=True):
        reason_key = normalize_reason_id(reason_id)
        manual = float(group["human_label"].astype(int).mean())
        estimated = float(group["p_correct"].astype(float).mean())
        rows.append(
            {
                "reason_id": reason_key,
                "rows": int(len(group)),
                "manual_prompt_accuracy": manual,
                "mean_p_correct_accuracy": estimated,
                "mean_p_correct_gap_pp": (estimated - manual) * 100,
                "mean_p_correct_abs_gap_pp": abs(estimated - manual) * 100,
            }
        )
    return pd.DataFrame(rows)


def _backtest_threshold_summary(
    predictions: pd.DataFrame,
    thresholds: pd.DataFrame,
    global_threshold: Optional[float] = None,
) -> pd.DataFrame:
    labeled = predictions[predictions["human_label"].isin([0, 1])].copy()
    if thresholds.empty:
        threshold_map: Dict[str, float] = {}
        source_map: Dict[str, str] = {}
        sheet_map: Dict[str, str] = {}
    else:
        indexed = thresholds.copy()
        indexed["reason_id"] = indexed["reason_id"].map(normalize_reason_id)
        threshold_map = indexed.set_index("reason_id")["backtest_threshold"].astype(float).to_dict()
        source_map = (
            indexed.set_index("reason_id")["threshold_source"].astype(str).to_dict()
            if "threshold_source" in indexed.columns
            else {}
        )
        sheet_map = (
            indexed.set_index("reason_id")["threshold_source_sheet"].astype(str).to_dict()
            if "threshold_source_sheet" in indexed.columns
            else {}
        )
    rows = []
    for reason_id, group in labeled.groupby("reason_id", sort=True):
        reason_key = normalize_reason_id(reason_id)
        human = group["human_label"].astype(int).to_numpy()
        manual = float(human.mean())
        threshold_from_reason = reason_key in threshold_map
        if threshold_from_reason:
            threshold = float(threshold_map[reason_key])
            auto = (group["p_correct"].astype(float).to_numpy() > threshold).astype(int)
            estimated = float(auto.mean())
            gap = (estimated - manual) * 100
            abs_gap = abs(estimated - manual) * 100
            row_accuracy = float((auto == human).mean())
        elif global_threshold is not None:
            threshold = float(global_threshold)
            auto = (group["p_correct"].astype(float).to_numpy() > threshold).astype(int)
            estimated = float(auto.mean())
            gap = (estimated - manual) * 100
            abs_gap = abs(estimated - manual) * 100
            row_accuracy = float((auto == human).mean())
        else:
            threshold = np.nan
            estimated = np.nan
            gap = np.nan
            abs_gap = np.nan
            row_accuracy = np.nan
        rows.append(
            {
                "reason_id": reason_key,
                "rows": int(len(group)),
                "manual_prompt_accuracy": manual,
                "backtest_threshold_accuracy": estimated,
                "backtest_threshold_gap_pp": gap,
                "backtest_threshold_abs_gap_pp": abs_gap,
                "backtest_row_label_accuracy": row_accuracy,
                "backtest_threshold": threshold,
                "threshold_from_reason": threshold_from_reason,
                "threshold_source": source_map.get(
                    reason_key,
                    "global" if global_threshold is not None else "no_threshold",
                ),
                "threshold_source_sheet": sheet_map.get(reason_key, ""),
            }
        )
    return pd.DataFrame(rows)


def learn_probability_offsets(calibration_predictions: pd.DataFrame) -> pd.DataFrame:
    labeled = calibration_predictions[calibration_predictions["human_label"].isin([0, 1])].copy()
    if labeled.empty:
        return pd.DataFrame()

    rows = []
    global_true = float(labeled["human_label"].astype(int).mean())
    global_raw = float(labeled["p_correct"].astype(float).mean())
    rows.append(
        {
            "reason_id": "__global__",
            "calibration_rows": int(len(labeled)),
            "calibration_true_accuracy": global_true,
            "calibration_raw_mean_p_correct": global_raw,
            "calibration_offset": global_true - global_raw,
        }
    )

    for reason_id, group in labeled.groupby("reason_id", sort=True):
        reason_key = normalize_reason_id(reason_id)
        true_rate = float(group["human_label"].astype(int).mean())
        raw_mean = float(group["p_correct"].astype(float).mean())
        rows.append(
            {
                "reason_id": reason_key,
                "calibration_rows": int(len(group)),
                "calibration_true_accuracy": true_rate,
                "calibration_raw_mean_p_correct": raw_mean,
                "calibration_offset": true_rate - raw_mean,
            }
        )
    return pd.DataFrame(rows)


def apply_probability_offsets(predictions: pd.DataFrame, calibration_table: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    if calibration_table.empty:
        out["calibrated_p_correct"] = out["p_correct"].astype(float)
        out["calibration_offset"] = 0.0
        out["calibration_source"] = "none"
        return out

    table = calibration_table.set_index("reason_id").to_dict(orient="index")
    global_offset = float(table.get("__global__", {}).get("calibration_offset", 0.0))
    values = []
    offsets = []
    sources = []
    for _, row in out.iterrows():
        reason_key = normalize_reason_id(row["reason_id"])
        if reason_key in table:
            offset = float(table[reason_key]["calibration_offset"])
            source = "reason"
        else:
            offset = global_offset
            source = "global"
        raw = float(row.get("p_correct", 0.0))
        values.append(max(0.0, min(1.0, raw + offset)))
        offsets.append(offset)
        sources.append(source)
    out["calibrated_p_correct"] = values
    out["calibration_offset"] = offsets
    out["calibration_source"] = sources
    return out


def _calibrated_mean_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    labeled = predictions[predictions["human_label"].isin([0, 1])].copy()
    rows = []
    for reason_id, group in labeled.groupby("reason_id", sort=True):
        reason_key = normalize_reason_id(reason_id)
        manual = float(group["human_label"].astype(int).mean())
        estimated = float(group["calibrated_p_correct"].astype(float).mean())
        raw_estimated = float(group["p_correct"].astype(float).mean())
        rows.append(
            {
                "reason_id": reason_key,
                "rows": int(len(group)),
                "manual_prompt_accuracy": manual,
                "calibrated_mean_p_correct_accuracy": estimated,
                "calibrated_mean_p_correct_gap_pp": (estimated - manual) * 100,
                "calibrated_mean_p_correct_abs_gap_pp": abs(estimated - manual) * 100,
                "raw_mean_p_correct_accuracy": raw_estimated,
                "calibration_source": ",".join(sorted(group["calibration_source"].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows)


def add_weighted_overall(summary: pd.DataFrame, estimate_col: str, gap_col: str, abs_gap_col: str) -> pd.DataFrame:
    if summary.empty:
        return summary
    rows = int(summary["rows"].sum())
    manual = float((summary["manual_prompt_accuracy"] * summary["rows"]).sum() / rows)
    estimated = float((summary[estimate_col] * summary["rows"]).sum() / rows)
    overall = {
        "reason_id": "__overall_weighted__",
        "rows": rows,
        "manual_prompt_accuracy": manual,
        estimate_col: estimated,
        gap_col: (estimated - manual) * 100,
        abs_gap_col: abs(estimated - manual) * 100,
    }
    row_acc_cols = [c for c in summary.columns if c.endswith("row_label_accuracy")]
    for col in row_acc_cols:
        overall[col] = float((summary[col] * summary["rows"]).sum() / rows)
    return pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)


@dataclass
class IterationComparisonResult:
    test_name: str
    mean_summary: pd.DataFrame
    calibrated_mean_summary: pd.DataFrame
    backtest_summary: pd.DataFrame
    calibration_table: pd.DataFrame
    thresholds: pd.DataFrame
    mean_predictions: pd.DataFrame
    calibrated_latest_predictions: pd.DataFrame
    backtest_calibration_predictions: pd.DataFrame
    backtest_latest_predictions: pd.DataFrame
    calibration_sheet: str
    backtest_applicable: bool


def compare_iteration_methods(
    *,
    test_name: str,
    train_path: str,
    validation_path: str,
    use_embeddings: bool = False,
) -> IterationComparisonResult:
    train_all = load_tables([train_path], require_text=True, require_answer=True)
    latest = load_tables([validation_path], require_text=True, require_answer=True)

    rules = load_rules(None)

    # Variant 1: train on every previous iteration, estimate prompt accuracy by mean p_correct.
    mean_model = HybridValidator.train(
        train_all,
        ValidatorConfig(use_embeddings=use_embeddings, rules=rules),
    )
    mean_predictions = mean_model.predict(latest)
    mean_summary = add_weighted_overall(
        _mean_probability_summary(mean_predictions),
        "mean_p_correct_accuracy",
        "mean_p_correct_gap_pp",
        "mean_p_correct_abs_gap_pp",
    )

    # Variant 2: train on <= n-2, calibrate per-reason threshold on n-1, apply to n.
    early_train, calibration, calibration_sheet = split_train_calibration(train_all)
    backtest_applicable = not early_train.empty and not calibration.empty
    if not backtest_applicable:
        empty = pd.DataFrame()
        return IterationComparisonResult(
            test_name=test_name,
            mean_summary=mean_summary,
            calibrated_mean_summary=empty,
            backtest_summary=empty,
            calibration_table=empty,
            thresholds=empty,
            mean_predictions=mean_predictions,
            calibrated_latest_predictions=empty,
            backtest_calibration_predictions=empty,
            backtest_latest_predictions=empty,
            calibration_sheet=calibration_sheet,
            backtest_applicable=False,
        )

    backtest_model = HybridValidator.train(
        early_train,
        ValidatorConfig(use_embeddings=use_embeddings, rules=rules),
    )
    calibration_predictions = backtest_model.predict(calibration)
    calibration_table = learn_probability_offsets(calibration_predictions)
    calibrated_latest_predictions = apply_probability_offsets(
        backtest_model.predict(latest),
        calibration_table,
    )
    calibrated_mean_summary = add_weighted_overall(
        _calibrated_mean_summary(calibrated_latest_predictions),
        "calibrated_mean_p_correct_accuracy",
        "calibrated_mean_p_correct_gap_pp",
        "calibrated_mean_p_correct_abs_gap_pp",
    )
    thresholds = learn_backtest_thresholds(calibration_predictions)
    if thresholds.empty:
        global_threshold = 0.5
    else:
        global_threshold, _, _ = choose_threshold_for_rate(
            calibration_predictions[
                calibration_predictions["human_label"].isin([0, 1])
            ]["human_label"].astype(int),
            calibration_predictions[
                calibration_predictions["human_label"].isin([0, 1])
            ]["p_correct"].astype(float),
        )
    latest_predictions = backtest_model.predict(latest)
    backtest_summary = add_weighted_overall(
        _backtest_threshold_summary(latest_predictions, thresholds, global_threshold),
        "backtest_threshold_accuracy",
        "backtest_threshold_gap_pp",
        "backtest_threshold_abs_gap_pp",
    )
    return IterationComparisonResult(
        test_name=test_name,
        mean_summary=mean_summary,
        calibrated_mean_summary=calibrated_mean_summary,
        backtest_summary=backtest_summary,
        calibration_table=calibration_table,
        thresholds=thresholds,
        mean_predictions=mean_predictions,
        calibrated_latest_predictions=calibrated_latest_predictions,
        backtest_calibration_predictions=calibration_predictions,
        backtest_latest_predictions=latest_predictions,
        calibration_sheet=calibration_sheet,
        backtest_applicable=True,
    )


def write_iteration_comparison(result: IterationComparisonResult, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / result.test_name
    write_table(result.mean_summary, str(prefix.with_name(prefix.name + "_mean_p_correct_summary.csv")))
    write_table(result.mean_predictions, str(prefix.with_name(prefix.name + "_mean_p_correct_predictions.csv")))
    if result.backtest_applicable:
        write_table(
            result.calibrated_mean_summary,
            str(prefix.with_name(prefix.name + "_calibrated_mean_p_correct_summary.csv")),
        )
        write_table(
            result.calibration_table,
            str(prefix.with_name(prefix.name + "_probability_calibration.csv")),
        )
        write_table(
            result.calibrated_latest_predictions,
            str(prefix.with_name(prefix.name + "_calibrated_latest_predictions.csv")),
        )
        write_table(result.backtest_summary, str(prefix.with_name(prefix.name + "_backtest_threshold_summary.csv")))
        write_table(result.thresholds, str(prefix.with_name(prefix.name + "_backtest_thresholds.csv")))
        write_table(
            result.backtest_calibration_predictions,
            str(prefix.with_name(prefix.name + "_backtest_calibration_predictions.csv")),
        )
        write_table(
            result.backtest_latest_predictions,
            str(prefix.with_name(prefix.name + "_backtest_latest_predictions.csv")),
        )
