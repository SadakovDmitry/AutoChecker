from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import ValidatorConfig, load_rules
from .data import DataFormatError
from .prepare import prepare_training_data
from .quota_yesno import HybridRouterResult, run_hybrid_router_experiment
from .subreason_mapping import (
    SubreasonMapping,
    apply_subreason_mapping,
    load_subreason_mapping,
    use_subreason_key_as_reason_id,
)


DEFAULT_SUBREASON_MAP = Path(__file__).resolve().parent / "configs" / "subreason_versions.yaml"


@dataclass
class AutolabelResult:
    output_path: Path
    latest_sheet: str
    train_rows: int
    latest_rows: int
    auto_rows: int
    review_rows: int
    auto_yes: int
    auto_no: int
    coverage: float
    missing_text_rows: int
    predictions: pd.DataFrame
    operational_summary: pd.DataFrame
    hybrid_result: HybridRouterResult


def available_dataset_keys(mapping_path: str | Path | None = DEFAULT_SUBREASON_MAP) -> list[str]:
    mapping = load_subreason_mapping(str(mapping_path)) if mapping_path else None
    if mapping is None:
        return []
    return sorted(str(key) for key in mapping.datasets.keys())


def excel_sheet_names(path: str | Path) -> list[str]:
    file_path = Path(path)
    if file_path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
        return []
    try:
        return [str(name) for name in pd.ExcelFile(file_path).sheet_names]
    except Exception as exc:
        raise DataFormatError(f"Не удалось прочитать листы Excel-файла: {file_path}") from exc


def detect_latest_sheet(path: str | Path, latest_sheet: str | None = None) -> str:
    if latest_sheet:
        return str(latest_sheet)
    sheets = excel_sheet_names(path)
    if not sheets:
        raise DataFormatError(
            "Для автоматического выбора новой итерации нужен Excel-файл с листами "
            "или явно передайте --latest-sheet."
        )
    return sheets[-1]


def _mapping_with_dataset(frame: pd.DataFrame, mapping: Optional[SubreasonMapping], dataset_key: str | None) -> pd.DataFrame:
    out = frame.copy()
    if dataset_key:
        out["_dataset"] = dataset_key
    if mapping is None:
        return out
    out = apply_subreason_mapping(out, mapping)
    return use_subreason_key_as_reason_id(out)


def _validate_dataset_key(mapping: Optional[SubreasonMapping], dataset_key: str | None) -> None:
    if not dataset_key or mapping is None:
        return
    if dataset_key not in mapping.datasets:
        available = ", ".join(sorted(str(key) for key in mapping.datasets.keys()))
        raise DataFormatError(
            f"Неизвестная тема/датасет: {dataset_key}. Доступные значения: {available}"
        )


def _labeled_train_rows(frame: pd.DataFrame, latest_sheet: str) -> pd.DataFrame:
    if "_source_sheet" not in frame.columns:
        raise DataFormatError("В файле разметки не найдены листы итераций (_source_sheet).")
    train = frame[frame["_source_sheet"].fillna("").astype(str) != str(latest_sheet)].copy()
    train = train[train["human_label"].isin([0, 1])].copy()
    train = train[train["has_chat_text"].fillna(False).astype(bool)].copy()
    if train.empty:
        raise DataFormatError(
            "Не нашлось обучающих строк: на предыдущих листах должны быть chat_id, reason_id и да/нет."
        )
    return train


def _latest_rows(frame: pd.DataFrame, latest_sheet: str) -> pd.DataFrame:
    latest = frame[frame["_source_sheet"].fillna("").astype(str) == str(latest_sheet)].copy()
    if latest.empty:
        raise DataFormatError(f"Не найдены строки последней итерации на листе: {latest_sheet}")
    return latest


def build_operational_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["reason_id"]
    for reason_id, group in predictions.groupby(group_columns, sort=True):
        if isinstance(reason_id, tuple):
            reason_id = reason_id[0]
        auto = group[group["auto_answer"].isin(["да", "нет"])]
        labeled_auto = auto[auto["human_label"].isin([0, 1])] if "human_label" in auto.columns else pd.DataFrame()
        if len(labeled_auto):
            precision = float((labeled_auto["auto_label"].astype(int) == labeled_auto["human_label"].astype(int)).mean())
            errors = int((labeled_auto["auto_label"].astype(int) != labeled_auto["human_label"].astype(int)).sum())
        else:
            precision = None
            errors = None
        rows.append(
            {
                "reason_id": str(reason_id),
                "subreason_name": str(group["subreason_name"].dropna().iloc[0]) if "subreason_name" in group and group["subreason_name"].notna().any() else "",
                "rows": int(len(group)),
                "auto_rows": int(len(auto)),
                "review_rows": int(len(group) - len(auto)),
                "coverage": float(len(auto) / len(group)) if len(group) else 0.0,
                "auto_yes": int((auto["auto_answer"] == "да").sum()) if len(auto) else 0,
                "auto_no": int((auto["auto_answer"] == "нет").sum()) if len(auto) else 0,
                "mean_p_correct": float(group["p_correct"].astype(float).mean()) if "p_correct" in group else None,
                "hybrid_mode": str(group["hybrid_mode"].dropna().iloc[0]) if "hybrid_mode" in group and group["hybrid_mode"].notna().any() else "",
                "risk_flags": str(group["risk_flags"].dropna().iloc[0]) if "risk_flags" in group and group["risk_flags"].notna().any() else "",
                "precision_if_labeled": precision,
                "errors_if_labeled": errors,
            }
        )
    summary = pd.DataFrame(rows)
    auto_all = predictions[predictions["auto_answer"].isin(["да", "нет"])]
    overall = {
        "reason_id": "__overall__",
        "subreason_name": "",
        "rows": int(len(predictions)),
        "auto_rows": int(len(auto_all)),
        "review_rows": int(len(predictions) - len(auto_all)),
        "coverage": float(len(auto_all) / len(predictions)) if len(predictions) else 0.0,
        "auto_yes": int((auto_all["auto_answer"] == "да").sum()) if len(auto_all) else 0,
        "auto_no": int((auto_all["auto_answer"] == "нет").sum()) if len(auto_all) else 0,
        "mean_p_correct": float(predictions["p_correct"].astype(float).mean()) if "p_correct" in predictions else None,
        "hybrid_mode": "",
        "risk_flags": "",
        "precision_if_labeled": None,
        "errors_if_labeled": None,
    }
    return pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)


def _add_analyst_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    out.insert(0, "авто_ответ", out["auto_answer"])
    out.insert(1, "нужно_проверить_вручную", out["auto_answer"].eq("review"))
    out.insert(2, "уверенность_p_correct", out["p_correct"])
    out.insert(3, "режим_разметки", out.get("hybrid_mode", ""))
    out.insert(4, "причина_review", out.get("risk_flags", ""))
    return out


def write_autolabel_workbook(
    *,
    output_path: str | Path,
    predictions: pd.DataFrame,
    operational_summary: pd.DataFrame,
    result: HybridRouterResult,
    metadata: pd.DataFrame,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    marked = _add_analyst_columns(predictions)
    try:
        with pd.ExcelWriter(path) as writer:
            marked.to_excel(writer, sheet_name="marked_latest", index=False)
            operational_summary.to_excel(writer, sheet_name="summary", index=False)
            result.risk_summary.to_excel(writer, sheet_name="risk_summary", index=False)
            result.training_summary.to_excel(writer, sheet_name="model_training", index=False)
            result.offset_summary.to_excel(writer, sheet_name="offset", index=False)
            metadata.to_excel(writer, sheet_name="metadata", index=False)
    except ImportError as exc:
        raise RuntimeError("openpyxl is required to write .xlsx output.") from exc
    return path


def autolabel_latest_iteration(
    *,
    labels_path: str | Path,
    messages_path: str | Path,
    output_path: str | Path,
    dataset_key: str | None = None,
    latest_sheet: str | None = None,
    messages_sheet: str | None = None,
    subreason_map: str | Path | None = DEFAULT_SUBREASON_MAP,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    use_embeddings: bool = False,
    target_precision: float = 0.80,
) -> AutolabelResult:
    latest_sheet_name = detect_latest_sheet(labels_path, latest_sheet)
    mapping = load_subreason_mapping(str(subreason_map)) if subreason_map else None
    _validate_dataset_key(mapping, dataset_key)

    prepared, stats = prepare_training_data(
        labels_paths=[str(labels_path)],
        messages_paths=[str(messages_path)],
        output=None,
        require_answer=False,
        labels_sheet=None,
        messages_sheet=messages_sheet,
    )
    if prepared.empty:
        raise DataFormatError("После подготовки данных не осталось строк.")

    prepared = _mapping_with_dataset(prepared, mapping, dataset_key)
    train_frame = _labeled_train_rows(prepared, latest_sheet_name)
    latest_frame = _latest_rows(prepared, latest_sheet_name)
    missing_latest_text = int((~latest_frame["has_chat_text"].fillna(False).astype(bool)).sum())
    if missing_latest_text:
        raise DataFormatError(
            f"Для {missing_latest_text} строк последней итерации не найден текст чата. "
            "Проверьте файл текстовок и chat_id."
        )

    config = ValidatorConfig(
        target_precision=float(target_precision),
        min_reason_samples=8,
        min_class_samples=2,
        embedding_model=embedding_model,
        use_embeddings=bool(use_embeddings),
        rules=load_rules(None),
    )
    hybrid = run_hybrid_router_experiment(
        train_frame=train_frame,
        evaluation_frame=latest_frame,
        config=config,
        min_history_rows=10,
        max_history_std=0.05,
        max_model_history_gap=0.35,
        min_estimated_accuracy=0.70,
        min_full_mean_p_correct=0.65,
        min_full_train_row_accuracy=0.65,
        max_full_train_rate_gap=0.05,
        full_yesno_strategy="threshold",
        estimate_strategy="max_history_latest",
    )
    predictions = hybrid.predictions
    operational_summary = build_operational_summary(predictions)
    auto = predictions[predictions["auto_answer"].isin(["да", "нет"])]
    metadata = pd.DataFrame(
        [
            {"key": "labels_path", "value": str(labels_path)},
            {"key": "messages_path", "value": str(messages_path)},
            {"key": "dataset_key", "value": dataset_key or ""},
            {"key": "latest_sheet", "value": latest_sheet_name},
            {"key": "train_rows", "value": int(len(train_frame))},
            {"key": "latest_rows", "value": int(len(latest_frame))},
            {"key": "prepared_rows", "value": int(len(prepared))},
            {"key": "matched_text_rows", "value": int(stats.matched_rows)},
            {"key": "missing_text_rows", "value": int(stats.missing_text_rows)},
            {"key": "use_embeddings", "value": bool(use_embeddings)},
            {"key": "target_precision", "value": float(target_precision)},
            {"key": "mode", "value": "hybrid_router_threshold_max_history_latest"},
        ]
    )
    saved_path = write_autolabel_workbook(
        output_path=output_path,
        predictions=predictions,
        operational_summary=operational_summary,
        result=hybrid,
        metadata=metadata,
    )
    return AutolabelResult(
        output_path=saved_path,
        latest_sheet=latest_sheet_name,
        train_rows=int(len(train_frame)),
        latest_rows=int(len(latest_frame)),
        auto_rows=int(len(auto)),
        review_rows=int(len(predictions) - len(auto)),
        auto_yes=int((auto["auto_answer"] == "да").sum()) if len(auto) else 0,
        auto_no=int((auto["auto_answer"] == "нет").sum()) if len(auto) else 0,
        coverage=float(len(auto) / len(predictions)) if len(predictions) else 0.0,
        missing_text_rows=int(stats.missing_text_rows),
        predictions=predictions,
        operational_summary=operational_summary,
        hybrid_result=hybrid,
    )
