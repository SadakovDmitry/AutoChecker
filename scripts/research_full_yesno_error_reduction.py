from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from auto_classifier.config import ValidatorConfig, load_rules
from auto_classifier.data import load_tables, normalize_reason_id
from auto_classifier.iteration_compare import (
    _backtest_threshold_summary,
    _thresholds_from_predictions,
    collect_walk_forward_predictions,
    learn_latest_available_thresholds,
    sheet_sort_key,
)
from auto_classifier.model import HybridValidator
from auto_classifier.reports import write_table
from auto_classifier.subreason_mapping import (
    apply_subreason_mapping,
    load_subreason_mapping,
    use_subreason_key_as_reason_id,
)


warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

BASE = Path("auto_classifier/local_data/prepared")
OUT = Path("auto_classifier/local_data/reports/full_yesno_error_reduction")
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


ESTIMATOR_NAMES = {
    "hard_history_all": "Hard да/нет: порог по всей истории",
    "mean_p_correct": "Среднее p_correct",
    "calibrated_mean_p_correct": "Среднее p_correct + offset по истории",
    "history_all_rate": "Историческая точность подпричины",
    "history_latest_rate": "Последняя историческая точность подпричины",
    "blend_75p_25hist": "75% p_correct + 25% история",
    "blend_50p_50hist": "50% p_correct + 50% история",
    "blend_25p_75hist": "25% p_correct + 75% история",
    "bayes_p_k10": "Байес-сглаживание p_correct, k=10",
    "bayes_p_k20": "Байес-сглаживание p_correct, k=20",
    "bayes_p_k40": "Байес-сглаживание p_correct, k=40",
    "topic_history_rate": "Историческая точность всей причины",
    "max_p_history_latest": "max(p_correct, последняя история)",
    "max_p_history_all": "max(p_correct, вся история)",
    "max_p_topic_history": "max(p_correct, история причины)",
    "bayes_p_topic_k20": "Байес-сглаживание p_correct историей причины, k=20",
    "guarded_bayes_k40": "Байес k=40 с защитой от сильного занижения",
    "max_p_history_latest_oof_offset": "max(p_correct, последняя история) + OOF offset",
    "max_p_topic_history_oof_offset": "max(p_correct, история причины) + OOF offset",
    "guarded_bayes_k40_oof_offset": "Байес k=40 с защитой + OOF offset",
    "meta_global_ridge": "Мета-калибровка Ridge по всем прошлым итерациям",
    "meta_local_or_global_ridge": "Мета-калибровка Ridge по своей теме, иначе global",
}


def _load_stable(path: Path, mapping) -> pd.DataFrame:
    frame = load_tables([str(path)], require_text=True, require_answer=True)
    frame = apply_subreason_mapping(frame, mapping)
    return use_subreason_key_as_reason_id(frame)


def _name_map(*frames: pd.DataFrame) -> dict[str, str]:
    names: dict[str, str] = {}
    for frame in frames:
        if "subreason_key" not in frame.columns or "subreason_name" not in frame.columns:
            continue
        for _, row in frame[["subreason_key", "subreason_name"]].dropna().iterrows():
            key = normalize_reason_id(row["subreason_key"])
            name = str(row["subreason_name"] or "").strip()
            if name:
                names[key] = name
    return names


def _latest_history_rates(train: pd.DataFrame) -> dict[str, float]:
    labeled = train[train["human_label"].isin([0, 1])].copy()
    rates: dict[str, float] = {}
    if "_source_sheet" not in labeled.columns:
        return {
            normalize_reason_id(reason_id): float(group["human_label"].astype(int).mean())
            for reason_id, group in labeled.groupby("reason_id", sort=True)
        }
    for reason_id, group in labeled.groupby("reason_id", sort=True):
        sheets = sorted(group["_source_sheet"].fillna("").astype(str).unique(), key=sheet_sort_key)
        if not sheets:
            continue
        latest_sheet = sheets[-1]
        latest = group[group["_source_sheet"].fillna("").astype(str) == latest_sheet]
        if latest.empty:
            continue
        rates[normalize_reason_id(reason_id)] = float(latest["human_label"].astype(int).mean())
    return rates


def _all_history_rates(train: pd.DataFrame) -> dict[str, float]:
    labeled = train[train["human_label"].isin([0, 1])].copy()
    return {
        normalize_reason_id(reason_id): float(group["human_label"].astype(int).mean())
        for reason_id, group in labeled.groupby("reason_id", sort=True)
    }


def _history_rows(train: pd.DataFrame) -> dict[str, int]:
    labeled = train[train["human_label"].isin([0, 1])].copy()
    return {
        normalize_reason_id(reason_id): int(len(group))
        for reason_id, group in labeled.groupby("reason_id", sort=True)
    }


def _topic_history_rate(train: pd.DataFrame) -> float:
    labeled = train[train["human_label"].isin([0, 1])].copy()
    if labeled.empty:
        return np.nan
    return float(labeled["human_label"].astype(int).mean())


def _offsets_from_walk_forward(walk_forward: pd.DataFrame) -> tuple[dict[str, float], float]:
    if walk_forward.empty or "human_label" not in walk_forward.columns:
        return {}, 0.0
    labeled = walk_forward[walk_forward["human_label"].isin([0, 1])].copy()
    if labeled.empty:
        return {}, 0.0
    global_offset = float(labeled["human_label"].astype(int).mean() - labeled["p_correct"].astype(float).mean())
    offsets: dict[str, float] = {}
    for reason_id, group in labeled.groupby("reason_id", sort=True):
        offsets[normalize_reason_id(reason_id)] = float(
            group["human_label"].astype(int).mean() - group["p_correct"].astype(float).mean()
        )
    return offsets, global_offset


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _weighted_summary(details: pd.DataFrame, estimator_col: str) -> dict[str, float | int]:
    estimated = details[details[estimator_col].notna()].copy()
    if estimated.empty:
        return {
            "subreasons_estimated": 0,
            "rows_estimated": 0,
            "manual_accuracy_pct": np.nan,
            "auto_accuracy_pct": np.nan,
            "gap_pp": np.nan,
            "weighted_abs_gap_pp": np.nan,
        }
    rows = int(estimated["rows"].sum())
    manual = float((estimated["manual_prompt_accuracy"] * estimated["rows"]).sum() / rows)
    auto = float((estimated[estimator_col] * estimated["rows"]).sum() / rows)
    abs_gap = float((abs(estimated[estimator_col] - estimated["manual_prompt_accuracy"]) * estimated["rows"]).sum() / rows)
    return {
        "subreasons_estimated": int(estimated["reason_id"].nunique()),
        "rows_estimated": rows,
        "manual_accuracy_pct": round(manual * 100, 2),
        "auto_accuracy_pct": round(auto * 100, 2),
        "gap_pp": round((auto - manual) * 100, 2),
        "weighted_abs_gap_pp": round(abs_gap * 100, 2),
    }


def _rows_for_estimator(
    *,
    test: TestCase,
    predictions: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    history_all: dict[str, float],
    history_latest: dict[str, float],
    history_n: dict[str, int],
    topic_history_rate: float,
    offsets: dict[str, float],
    global_offset: float,
    name_by_key: dict[str, str],
) -> pd.DataFrame:
    labeled = predictions[predictions["human_label"].isin([0, 1])].copy()
    threshold_lookup = threshold_summary.set_index("reason_id").to_dict(orient="index") if not threshold_summary.empty else {}
    rows = []
    for reason_id, group in labeled.groupby("reason_id", sort=True):
        reason_key = normalize_reason_id(reason_id)
        human = group["human_label"].astype(int).to_numpy()
        p = group["p_correct"].astype(float).to_numpy()
        n = int(len(group))
        manual = float(human.mean())
        mean_p = float(p.mean())
        hist_all = history_all.get(reason_key, np.nan)
        hist_latest = history_latest.get(reason_key, hist_all)
        hist_for_blend = hist_latest if not pd.isna(hist_latest) else hist_all
        topic_rate = topic_history_rate
        offset = offsets.get(reason_key, global_offset)
        calibrated_mean = _clip(mean_p + offset)

        threshold_data = threshold_lookup.get(reason_key, {})
        hard_estimate = threshold_data.get("backtest_threshold_accuracy", np.nan)
        row_label_accuracy = threshold_data.get("backtest_row_label_accuracy", np.nan)
        hard_threshold = threshold_data.get("backtest_threshold", np.nan)

        item = {
            "test": test.key,
            "product": test.product,
            "topic": test.topic,
            "reason_id": reason_key,
            "subreason_name": name_by_key.get(reason_key, ""),
            "rows": n,
            "manual_prompt_accuracy": manual,
            "manual_yes": int(human.sum()),
            "manual_no": int((human == 0).sum()),
            "history_rows": int(history_n.get(reason_key, 0)),
            "hard_history_all": hard_estimate,
            "hard_threshold": hard_threshold,
            "hard_row_label_accuracy": row_label_accuracy,
            "mean_p_correct": mean_p,
            "calibrated_mean_p_correct": calibrated_mean,
            "history_all_rate": hist_all,
            "history_latest_rate": hist_latest,
            "topic_history_rate": topic_rate,
            "blend_75p_25hist": _clip(0.75 * mean_p + 0.25 * hist_for_blend) if not pd.isna(hist_for_blend) else np.nan,
            "blend_50p_50hist": _clip(0.50 * mean_p + 0.50 * hist_for_blend) if not pd.isna(hist_for_blend) else np.nan,
            "blend_25p_75hist": _clip(0.25 * mean_p + 0.75 * hist_for_blend) if not pd.isna(hist_for_blend) else np.nan,
            "max_p_history_latest": _clip(max(mean_p, hist_latest)) if not pd.isna(hist_latest) else np.nan,
            "max_p_history_all": _clip(max(mean_p, hist_all)) if not pd.isna(hist_all) else np.nan,
            "max_p_topic_history": _clip(max(mean_p, topic_rate)) if not pd.isna(topic_rate) else np.nan,
            "bayes_p_topic_k20": (
                _clip((mean_p * n + topic_rate * 20) / (n + 20))
                if not pd.isna(topic_rate)
                else np.nan
            ),
        }
        for k in [10, 20, 40]:
            item[f"bayes_p_k{k}"] = (
                _clip((mean_p * n + hist_for_blend * k) / (n + k))
                if not pd.isna(hist_for_blend)
                else np.nan
            )
        guarded = item["bayes_p_k40"]
        if not pd.isna(guarded) and not pd.isna(topic_rate):
            # If both the model and the latest subreason history are pessimistic,
            # keep the estimate from falling too far below the whole-topic prior.
            guarded = max(guarded, topic_rate - 0.10)
        item["guarded_bayes_k40"] = _clip(guarded) if not pd.isna(guarded) else np.nan
        for estimator in ESTIMATOR_NAMES:
            value = item.get(estimator, np.nan)
            item[f"{estimator}_gap_pp"] = (value - manual) * 100 if not pd.isna(value) else np.nan
            item[f"{estimator}_abs_gap_pp"] = abs(value - manual) * 100 if not pd.isna(value) else np.nan
        rows.append(item)
    return pd.DataFrame(rows)


def _learn_oof_offsets(
    *,
    train: pd.DataFrame,
    walk_forward: pd.DataFrame,
    estimators: list[str],
) -> dict[str, float]:
    if walk_forward.empty or "_calibration_sheet" not in walk_forward.columns:
        return {estimator: 0.0 for estimator in estimators}
    if "_source_sheet" not in train.columns:
        return {estimator: 0.0 for estimator in estimators}

    sheets = sorted(train["_source_sheet"].fillna("").astype(str).unique(), key=sheet_sort_key)
    sheet_rank = {sheet: idx for idx, sheet in enumerate(sheets)}
    offset_parts = []
    for calibration_sheet, predictions in walk_forward.groupby("_calibration_sheet", sort=False):
        calibration_sheet = str(calibration_sheet)
        rank = sheet_rank.get(calibration_sheet)
        if rank is None or rank <= 0:
            continue
        previous_sheets = set(sheets[:rank])
        previous = train[train["_source_sheet"].fillna("").astype(str).isin(previous_sheets)].copy()
        if previous.empty:
            continue
        details = _rows_for_estimator(
            test=TestCase("__oof__", "", "", "", ""),
            predictions=predictions,
            threshold_summary=pd.DataFrame(),
            history_all=_all_history_rates(previous),
            history_latest=_latest_history_rates(previous),
            history_n=_history_rows(previous),
            topic_history_rate=_topic_history_rate(previous),
            offsets={},
            global_offset=0.0,
            name_by_key={},
        )
        if not details.empty:
            offset_parts.append(details)
    if not offset_parts:
        return {estimator: 0.0 for estimator in estimators}

    oof = pd.concat(offset_parts, ignore_index=True)
    offsets: dict[str, float] = {}
    for estimator in estimators:
        if estimator not in oof.columns:
            offsets[estimator] = 0.0
            continue
        valid = oof[oof[estimator].notna()].copy()
        if valid.empty:
            offsets[estimator] = 0.0
            continue
        residual = valid["manual_prompt_accuracy"].astype(float) - valid[estimator].astype(float)
        offsets[estimator] = float((residual * valid["rows"]).sum() / valid["rows"].sum())
    return offsets


def _collect_oof_details(*, train: pd.DataFrame, walk_forward: pd.DataFrame, test: TestCase) -> pd.DataFrame:
    if walk_forward.empty or "_calibration_sheet" not in walk_forward.columns:
        return pd.DataFrame()
    if "_source_sheet" not in train.columns:
        return pd.DataFrame()

    sheets = sorted(train["_source_sheet"].fillna("").astype(str).unique(), key=sheet_sort_key)
    sheet_rank = {sheet: idx for idx, sheet in enumerate(sheets)}
    parts = []
    for calibration_sheet, predictions in walk_forward.groupby("_calibration_sheet", sort=False):
        calibration_sheet = str(calibration_sheet)
        rank = sheet_rank.get(calibration_sheet)
        if rank is None or rank <= 0:
            continue
        previous_sheets = set(sheets[:rank])
        previous = train[train["_source_sheet"].fillna("").astype(str).isin(previous_sheets)].copy()
        if previous.empty:
            continue
        detail = _rows_for_estimator(
            test=test,
            predictions=predictions,
            threshold_summary=pd.DataFrame(),
            history_all=_all_history_rates(previous),
            history_latest=_latest_history_rates(previous),
            history_n=_history_rows(previous),
            topic_history_rate=_topic_history_rate(previous),
            offsets={},
            global_offset=0.0,
            name_by_key={},
        )
        if not detail.empty:
            detail["_calibration_sheet"] = calibration_sheet
            parts.append(detail)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


META_FEATURES = [
    "rows",
    "history_rows",
    "mean_p_correct",
    "calibrated_mean_p_correct",
    "history_all_rate",
    "history_latest_rate",
    "topic_history_rate",
    "max_p_history_latest",
    "max_p_topic_history",
    "guarded_bayes_k40",
]


def _fit_meta_ridge(oof: pd.DataFrame) -> tuple[Ridge | None, dict[str, float]]:
    needed = [*META_FEATURES, "manual_prompt_accuracy", "rows"]
    if oof.empty or any(col not in oof.columns for col in needed):
        return None, {}
    train = oof[oof["manual_prompt_accuracy"].notna()].copy()
    if len(train) < 8:
        return None, {}
    fill_values = {}
    for col in META_FEATURES:
        value = float(train[col].mean()) if train[col].notna().any() else 0.0
        fill_values[col] = value
        train[col] = train[col].fillna(value)
    model = Ridge(alpha=1.0)
    model.fit(
        train[META_FEATURES].astype(float).to_numpy(),
        train["manual_prompt_accuracy"].astype(float).to_numpy(),
        sample_weight=train["rows"].astype(float).to_numpy(),
    )
    return model, fill_values


def _predict_meta_ridge(model: Ridge | None, fill_values: dict[str, float], frame: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.full(len(frame), np.nan)
    features = frame[META_FEATURES].copy()
    for col in META_FEATURES:
        features[col] = features[col].fillna(fill_values.get(col, 0.0))
    return np.clip(model.predict(features.astype(float).to_numpy()), 0.0, 1.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = load_subreason_mapping(MAPPING_PATH)
    config = ValidatorConfig(use_embeddings=False, rules=load_rules(None))

    details = []
    for test in TESTS:
        train_path = BASE / test.train
        validation_path = BASE / test.validation
        if not train_path.exists() or not validation_path.exists():
            continue

        train = _load_stable(train_path, mapping)
        validation = _load_stable(validation_path, mapping)
        name_by_key = _name_map(train, validation)

        walk_forward = collect_walk_forward_predictions(train, config=config)
        history_thresholds = _thresholds_from_predictions(
            walk_forward,
            source="history_all",
            source_sheet="all_history",
        )
        # Keep this call around as an explicit comparison hook in the output folder.
        latest_thresholds = learn_latest_available_thresholds(walk_forward)
        if not latest_thresholds.empty:
            write_table(latest_thresholds, str(OUT / f"{test.key}_latest_available_thresholds.csv"))

        model = HybridValidator.train(train, config)
        predictions = model.predict(validation)
        threshold_summary = _backtest_threshold_summary(predictions, history_thresholds, global_threshold=None)
        offsets, global_offset = _offsets_from_walk_forward(walk_forward)
        detail_part = _rows_for_estimator(
            test=test,
            predictions=predictions,
            threshold_summary=threshold_summary,
            history_all=_all_history_rates(train),
            history_latest=_latest_history_rates(train),
            history_n=_history_rows(train),
            topic_history_rate=_topic_history_rate(train),
            offsets=offsets,
            global_offset=global_offset,
            name_by_key=name_by_key,
        )
        oof_offsets = _learn_oof_offsets(
            train=train,
            walk_forward=walk_forward,
            estimators=[
                "max_p_history_latest",
                "max_p_topic_history",
                "guarded_bayes_k40",
            ],
        )
        for base_estimator, target_estimator in [
            ("max_p_history_latest", "max_p_history_latest_oof_offset"),
            ("max_p_topic_history", "max_p_topic_history_oof_offset"),
            ("guarded_bayes_k40", "guarded_bayes_k40_oof_offset"),
        ]:
            offset_value = oof_offsets.get(base_estimator, 0.0)
            detail_part[target_estimator] = detail_part[base_estimator].map(
                lambda value: _clip(float(value) + offset_value) if not pd.isna(value) else np.nan
            )
            detail_part[f"{target_estimator}_oof_offset"] = offset_value
            detail_part[f"{target_estimator}_gap_pp"] = (
                detail_part[target_estimator] - detail_part["manual_prompt_accuracy"]
            ) * 100
            detail_part[f"{target_estimator}_abs_gap_pp"] = (
                detail_part[target_estimator] - detail_part["manual_prompt_accuracy"]
            ).abs() * 100
        details.append(detail_part)

    if not details:
        print("No details generated.")
        return

    detail = pd.concat(details, ignore_index=True)
    oof_parts = []
    # Rebuild OOF details once for meta calibration. Kept separate from the main
    # loop so every estimator column already exists in one consistent format.
    for test in TESTS:
        train_path = BASE / test.train
        validation_path = BASE / test.validation
        if not train_path.exists() or not validation_path.exists():
            continue
        train = _load_stable(train_path, mapping)
        walk_forward = collect_walk_forward_predictions(train, config=config)
        oof = _collect_oof_details(train=train, walk_forward=walk_forward, test=test)
        if not oof.empty:
            oof_parts.append(oof)
    oof_detail = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()

    global_meta, global_fill = _fit_meta_ridge(oof_detail)
    detail["meta_global_ridge"] = _predict_meta_ridge(global_meta, global_fill, detail)
    detail["meta_global_ridge_gap_pp"] = (detail["meta_global_ridge"] - detail["manual_prompt_accuracy"]) * 100
    detail["meta_global_ridge_abs_gap_pp"] = (
        detail["meta_global_ridge"] - detail["manual_prompt_accuracy"]
    ).abs() * 100

    local_values = np.full(len(detail), np.nan)
    for test_key, group in detail.groupby("test", sort=False):
        local_oof = oof_detail[oof_detail["test"].eq(test_key)] if not oof_detail.empty else pd.DataFrame()
        model, fill = _fit_meta_ridge(local_oof)
        if model is None:
            model, fill = global_meta, global_fill
        local_values[group.index.to_numpy()] = _predict_meta_ridge(model, fill, group)
    detail["meta_local_or_global_ridge"] = local_values
    detail["meta_local_or_global_ridge_gap_pp"] = (
        detail["meta_local_or_global_ridge"] - detail["manual_prompt_accuracy"]
    ) * 100
    detail["meta_local_or_global_ridge_abs_gap_pp"] = (
        detail["meta_local_or_global_ridge"] - detail["manual_prompt_accuracy"]
    ).abs() * 100

    write_table(detail, str(OUT / "estimator_subreason_details.csv"))
    if not oof_detail.empty:
        write_table(oof_detail, str(OUT / "oof_meta_training_details.csv"))

    summary_rows = []
    topic_rows = []
    for estimator, estimator_name in ESTIMATOR_NAMES.items():
        overall = _weighted_summary(detail, estimator)
        overall.update({"estimator": estimator, "estimator_name": estimator_name})
        summary_rows.append(overall)

        for (test_key, product, topic), group in detail.groupby(["test", "product", "topic"], sort=True):
            item = _weighted_summary(group, estimator)
            item.update(
                {
                    "test": test_key,
                    "product": product,
                    "topic": topic,
                    "estimator": estimator,
                    "estimator_name": estimator_name,
                }
            )
            topic_rows.append(item)

    summary = pd.DataFrame(summary_rows)
    by_topic = pd.DataFrame(topic_rows)
    summary = summary.sort_values(["weighted_abs_gap_pp", "rows_estimated"], ascending=[True, False])
    by_topic = by_topic.sort_values(["estimator", "test"])
    write_table(summary, str(OUT / "estimator_summary_overall.csv"))
    write_table(by_topic, str(OUT / "estimator_summary_by_topic.csv"))

    readable = detail.copy()
    percent_cols = [
        "manual_prompt_accuracy",
        *ESTIMATOR_NAMES.keys(),
        "hard_row_label_accuracy",
    ]
    for col in percent_cols:
        if col in readable.columns:
            readable[col] = (readable[col].astype(float) * 100).round(2)
    for col in [c for c in readable.columns if c.endswith("_gap_pp") or c.endswith("_abs_gap_pp")]:
        readable[col] = readable[col].astype(float).round(2)
    write_table(readable, str(OUT / "estimator_subreason_details_readable.csv"))

    with pd.ExcelWriter(OUT / "full_yesno_error_reduction.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary_overall", index=False)
        by_topic.to_excel(writer, sheet_name="summary_by_topic", index=False)
        readable.to_excel(writer, sheet_name="subreason_details", index=False)

    best = summary.iloc[0].to_dict()
    readme = f"""# Исследование снижения ошибки full yes/no

Цель: понять, как приблизить автооценку точности промпта к ручной точности по каждой подпричине, когда все строки должны получить `да` или `нет`.

## Лучший вариант в этом прогоне

- Метод: `{best['estimator_name']}`
- Ручная точность по оцененным строкам: `{best['manual_accuracy_pct']}%`
- Автооценка точности: `{best['auto_accuracy_pct']}%`
- Средняя абсолютная ошибка: `{best['weighted_abs_gap_pp']} п.п.`
- Оценено строк: `{best['rows_estimated']}`

## Что сравнивалось

1. `hard_history_all` - текущая hard-логика: подобрать `p_correct`-порог на прошлых итерациях и поставить каждой строке `да/нет`.
2. `mean_p_correct` - не считать hard-ответы, а брать среднее `p_correct` как ожидаемую точность подпричины.
3. `calibrated_mean_p_correct` - поправить среднее `p_correct` историческим offset по прошлым walk-forward итерациям.
4. `history_all_rate` / `history_latest_rate` - переносить историческую точность подпричины.
5. `blend_*` и `bayes_*` - смешивать текущий сигнал модели с историей подпричины.

## Файлы

- `estimator_summary_overall.csv` - общая таблица по методам.
- `estimator_summary_by_topic.csv` - метод x продукт x причина.
- `estimator_subreason_details.csv` - подробная таблица по каждой подпричине.
- `full_yesno_error_reduction.xlsx` - те же данные в Excel.

## Важная интерпретация

Hard-разметка нужна, если нужно физически проставить `да/нет` каждой строке. Но если главная метрика - точность промпта по подпричине, то среднее/калиброванное `p_correct` может быть стабильнее, потому что оно не теряет информацию о неуверенности модели.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print("Wrote full yes/no error reduction research to", OUT)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
