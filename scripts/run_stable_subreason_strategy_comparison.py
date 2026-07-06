from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from auto_classifier.config import ValidatorConfig, load_rules
from auto_classifier.data import load_tables, normalize_reason_id
from auto_classifier.iteration_compare import (
    _backtest_threshold_summary,
    _thresholds_from_predictions,
    collect_walk_forward_predictions,
    learn_latest_available_thresholds,
    merge_hierarchical_thresholds,
    split_train_calibration,
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
OUT = Path("auto_classifier/local_data/reports/threshold_strategy_comparison_stable")
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


STRATEGY_NAMES = {
    "previous_iteration_only": "Только n-1",
    "history_all": "Вся история",
    "history_latest_available": "Последняя доступная история",
    "hierarchy_prev_else_history_all": "Иерархия: n-1, иначе вся история",
    "hierarchy_prev_else_history_latest": "Иерархия: n-1, иначе последняя история",
}


def _load_mapped(path: Path, mapping) -> pd.DataFrame:
    frame = load_tables([str(path)], require_text=True, require_answer=True)
    frame = apply_subreason_mapping(frame, mapping)
    frame = use_subreason_key_as_reason_id(frame)
    return frame


def _name_map(*frames: pd.DataFrame) -> dict[str, str]:
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


def _coverage_rows(test: TestCase, train: pd.DataFrame, validation: pd.DataFrame) -> list[dict]:
    rows = []
    for split_name, frame in [("train", train), ("validation", validation)]:
        for status, group in frame.groupby("subreason_mapping_status", dropna=False):
            rows.append(
                {
                    "test": test.key,
                    "product": test.product,
                    "topic": test.topic,
                    "split": split_name,
                    "mapping_status": str(status),
                    "rows": int(len(group)),
                    "unique_subreason_keys": int(group["subreason_key"].nunique()),
                }
            )
    return rows


def _strategy_details(
    *,
    test: TestCase,
    strategy_code: str,
    summary: pd.DataFrame,
    name_by_key: dict[str, str],
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    out = summary.copy()
    out = out[out["reason_id"] != "__overall_weighted__"].copy()
    out["test"] = test.key
    out["product"] = test.product
    out["topic"] = test.topic
    out["strategy_code"] = strategy_code
    out["strategy"] = STRATEGY_NAMES.get(strategy_code, strategy_code)
    out["subreason_key"] = out["reason_id"].map(normalize_reason_id)
    out["subreason_name"] = out["subreason_key"].map(name_by_key).fillna("")
    return out


def _summarize(details: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    estimated = details[details["backtest_threshold_accuracy"].notna()].copy()
    if estimated.empty:
        return pd.DataFrame()
    rows = []
    for group_key, group in estimated.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        total_rows = int(group["rows"].sum())
        manual = float((group["manual_prompt_accuracy"] * group["rows"]).sum() / total_rows)
        auto = float((group["backtest_threshold_accuracy"] * group["rows"]).sum() / total_rows)
        abs_gap = float((group["backtest_threshold_abs_gap_pp"] * group["rows"]).sum() / total_rows)
        row_accuracy = float((group["backtest_row_label_accuracy"] * group["rows"]).sum() / total_rows)
        item = dict(zip(group_cols, group_key))
        item.update(
            {
                "subreasons_estimated": int(group["reason_id"].nunique()),
                "rows_estimated": total_rows,
                "manual_accuracy_pct": round(manual * 100, 2),
                "auto_accuracy_pct": round(auto * 100, 2),
                "gap_pp": round((auto - manual) * 100, 2),
                "weighted_abs_gap_pp": round(abs_gap, 2),
                "row_label_accuracy_pct": round(row_accuracy * 100, 2),
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def _to_readable(details: pd.DataFrame) -> pd.DataFrame:
    readable = details.copy()
    for col in ["manual_prompt_accuracy", "backtest_threshold_accuracy", "backtest_row_label_accuracy"]:
        readable[col] = (readable[col].astype(float) * 100).round(2)
    for col in ["backtest_threshold_gap_pp", "backtest_threshold_abs_gap_pp"]:
        readable[col] = readable[col].astype(float).round(2)
    readable["backtest_threshold"] = readable["backtest_threshold"].astype(float).round(6)
    return readable[
        [
            "test",
            "strategy",
            "product",
            "topic",
            "strategy_code",
            "subreason_key",
            "subreason_name",
            "rows",
            "manual_prompt_accuracy",
            "backtest_threshold_accuracy",
            "backtest_threshold_gap_pp",
            "backtest_threshold_abs_gap_pp",
            "backtest_row_label_accuracy",
            "backtest_threshold",
            "threshold_from_reason",
            "threshold_source",
            "threshold_source_sheet",
        ]
    ].rename(
        columns={
            "test": "тест",
            "strategy": "стратегия",
            "product": "продукт",
            "topic": "причина",
            "subreason_key": "ключ подпричины",
            "subreason_name": "название подпричины",
            "rows": "строк в подпричине",
            "manual_prompt_accuracy": "ручная точность, %",
            "backtest_threshold_accuracy": "автооценка точности, %",
            "backtest_threshold_gap_pp": "разница авто-ручная, п.п.",
            "backtest_threshold_abs_gap_pp": "абсолютная ошибка, п.п.",
            "backtest_row_label_accuracy": "точность да/нет по строкам, %",
            "backtest_threshold": "p_correct порог",
            "threshold_from_reason": "есть собственный порог",
            "threshold_source": "источник порога",
            "threshold_source_sheet": "итерация источника порога",
        }
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = load_subreason_mapping(MAPPING_PATH)
    config = ValidatorConfig(use_embeddings=False, rules=load_rules(None))

    all_details = []
    coverage = []

    for test in TESTS:
        train_path = BASE / test.train
        validation_path = BASE / test.validation
        if not train_path.exists() or not validation_path.exists():
            continue
        train = _load_mapped(train_path, mapping)
        validation = _load_mapped(validation_path, mapping)
        coverage.extend(_coverage_rows(test, train, validation))
        name_by_key = _name_map(train, validation)

        early_train, calibration, calibration_sheet = split_train_calibration(train)
        if early_train.empty or calibration.empty:
            continue

        early_model = HybridValidator.train(early_train, config)
        calibration_predictions = early_model.predict(calibration)
        previous_thresholds = _thresholds_from_predictions(
            calibration_predictions,
            source="previous_iteration",
            source_sheet=calibration_sheet,
        )

        walk_forward_predictions = collect_walk_forward_predictions(train, config=config)
        history_all_thresholds = _thresholds_from_predictions(
            walk_forward_predictions,
            source="history_all",
            source_sheet="all_history",
        )
        history_latest_thresholds = learn_latest_available_thresholds(walk_forward_predictions)
        hierarchy_all = merge_hierarchical_thresholds(previous_thresholds, history_all_thresholds)
        hierarchy_latest = merge_hierarchical_thresholds(previous_thresholds, history_latest_thresholds)

        latest_model = HybridValidator.train(train, config)
        latest_predictions = latest_model.predict(validation)

        strategies = {
            "previous_iteration_only": previous_thresholds,
            "history_all": history_all_thresholds,
            "history_latest_available": history_latest_thresholds,
            "hierarchy_prev_else_history_all": hierarchy_all,
            "hierarchy_prev_else_history_latest": hierarchy_latest,
        }

        for strategy_code, thresholds in strategies.items():
            summary = _backtest_threshold_summary(latest_predictions, thresholds, global_threshold=None)
            all_details.append(
                _strategy_details(
                    test=test,
                    strategy_code=strategy_code,
                    summary=summary,
                    name_by_key=name_by_key,
                )
            )

    details = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    write_table(pd.DataFrame(coverage), str(OUT / "subreason_mapping_coverage_in_prepared.csv"))
    if details.empty:
        print("No details generated.")
        return

    write_table(details, str(OUT / "strategy_subreason_details.csv"))
    readable = _to_readable(details)
    write_table(readable, str(OUT / "strategy_subreason_details_readable.csv"))

    summary_by_topic = _summarize(details, ["test", "strategy_code", "strategy", "product", "topic"])
    summary_overall = _summarize(details, ["strategy_code", "strategy"])
    write_table(summary_by_topic, str(OUT / "strategy_summary_by_topic.csv"))
    write_table(summary_overall, str(OUT / "strategy_summary_overall.csv"))

    with pd.ExcelWriter(OUT / "strategy_comparison_readable.xlsx", engine="openpyxl") as writer:
        summary_overall.to_excel(writer, sheet_name="summary_overall", index=False)
        summary_by_topic.to_excel(writer, sheet_name="summary_by_topic", index=False)
        readable.to_excel(writer, sheet_name="subreason_details", index=False)
        pd.DataFrame(coverage).to_excel(writer, sheet_name="mapping_coverage", index=False)

    print("Wrote stable subreason strategy comparison to", OUT)
    print(summary_overall.to_string(index=False))


if __name__ == "__main__":
    main()
