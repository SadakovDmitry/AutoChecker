from __future__ import annotations

import glob
import re
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import xml.etree.ElementTree as ET


CHAT_ID_ALIASES = (
    "chat_id",
    "comm_id",
    "thread_id",
    "id",
    "dialog_id",
    "id_diologa",
    "id_dialoga",
    "id диалога",
    "ид диалога",
)
TEXT_ALIASES = ("chat_text", "text", "communication", "dialog", "chat", "текст", "чат")
REASON_ALIASES = (
    "reason_id",
    "reason_number",
    "reason_numb",
    "reason_num",
    "reason_no",
    "predicted_reason",
    "reason",
    "label",
    "причина",
    "номер причины",
)
ANSWER_ALIASES = ("human_answer", "да/нет", "да", "yes/no", "answer", "is_correct")
COMMENT_ALIASES = ("comment", "комментарий", "коментарий", "комментарии")
LINK_ALIASES = ("link", "url", "ссылка")


class DataFormatError(ValueError):
    """Raised when input tables cannot be normalized."""


def expand_paths(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(x) for x in matches)
        else:
            paths.append(Path(pattern))
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def normalize_reason_id(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def split_reason_ids(value: object) -> List[str]:
    text = normalize_reason_id(value)
    if not text:
        return []
    parts = re.split(r"[,;]\s*", text)
    return [normalize_reason_id(part) for part in parts if normalize_reason_id(part)]


def normalize_human_answer(value: object) -> Optional[int]:
    """Return 1 for yes, 0 for no, None for missing/ambiguous values."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if re.fullmatch(r"да\s*\?+", text):
        return 1
    if re.fullmatch(r"нет\s*\?+", text):
        return 0
    if "?" in text:
        return None
    if text in {"неь", "нкт"}:
        return 0
    if re.fullmatch(r"ошибка\s*[-—–]\s*да", text):
        return 1
    if re.fullmatch(r"ошибка\s*[-—–]\s*нет", text):
        return 0
    if text in {"1", "true", "yes", "y", "да", "дa"}:
        return 1
    if text in {"0", "false", "no", "n", "нет"}:
        return 0
    if text.startswith("да") and "?" not in text:
        return 1
    if text.startswith("нет") and "?" not in text:
        return 0
    return None


def _canonical_column(name: object) -> str:
    text = str(name).strip()
    lowered = re.sub(r"\s+", " ", text.lower())
    return lowered


def _find_column(columns: Iterable[str], aliases: Sequence[str]) -> Optional[str]:
    normalized = {_canonical_column(column): column for column in columns}
    for alias in aliases:
        found = normalized.get(_canonical_column(alias))
        if found is not None:
            return found
    return None


def _find_columns(columns: Iterable[str], aliases: Sequence[str]) -> List[str]:
    normalized = {_canonical_column(column): column for column in columns}
    found_columns: List[str] = []
    seen = set()
    for alias in aliases:
        found = normalized.get(_canonical_column(alias))
        if found is not None and found not in seen:
            seen.add(found)
            found_columns.append(found)
    return found_columns


def _coalesce_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    if not columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=object)
    values = frame[list(columns)].copy()
    values = values.where(~values.isna(), None)
    for column in values.columns:
        values[column] = values[column].map(
            lambda value: None if value is None or str(value).strip() == "" else value
        )
    return values.bfill(axis=1).iloc[:, 0].fillna("")


XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref)
    if not letters:
        return 0
    value = 0
    for char in letters.group(0):
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def _read_xlsx_without_openpyxl(path: Path) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(path) as archive:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{XML_NS}si"):
                shared_strings.append("".join(t.text or "" for t in item.iter(f"{XML_NS}t")))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet_names = [sheet.attrib.get("name", f"sheet{i+1}") for i, sheet in enumerate(workbook.findall(f".//{XML_NS}sheet"))]
        for index, sheet_name in enumerate(sheet_names, start=1):
            sheet_path = f"xl/worksheets/sheet{index}.xml"
            if sheet_path not in archive.namelist():
                continue
            sheet = ET.fromstring(archive.read(sheet_path))
            rows = []
            for row in sheet.find(f"{XML_NS}sheetData").findall(f"{XML_NS}row"):
                values = {}
                for cell in row.findall(f"{XML_NS}c"):
                    ref = cell.attrib.get("r", "A1")
                    col_idx = _column_index(ref)
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find(f"{XML_NS}v")
                    if cell_type == "s" and value_node is not None:
                        value = shared_strings[int(value_node.text)]
                    elif cell_type == "inlineStr":
                        inline = cell.find(f"{XML_NS}is")
                        value = "".join(t.text or "" for t in inline.iter(f"{XML_NS}t")) if inline is not None else ""
                    elif value_node is not None:
                        value = value_node.text or ""
                    else:
                        value = ""
                    values[col_idx] = value
                if values:
                    rows.append(values)
            if not rows:
                continue
            max_col = max(max(row) for row in rows) + 1
            matrix = [[row.get(i, "") for i in range(max_col)] for row in rows]
            header = [str(x).strip() or f"column_{i+1}" for i, x in enumerate(matrix[0])]
            data = matrix[1:]
            frame = pd.DataFrame(data, columns=header)
            frame["_source_sheet"] = sheet_name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_excel_all_sheets(path: Path) -> pd.DataFrame:
    try:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    except ImportError as exc:
        return _read_xlsx_without_openpyxl(path)
    frames = []
    for sheet_name, frame in sheets.items():
        if frame.empty:
            continue
        frame = frame.copy()
        frame["_source_sheet"] = sheet_name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_input_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        frame = _read_excel_all_sheets(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path, dtype=str)
    elif suffix in {".jsonl", ".json"}:
        frame = pd.read_json(path, lines=suffix == ".jsonl", dtype=False)
    else:
        raise DataFormatError(f"Unsupported input format: {path}")
    frame["_source_file"] = str(path)
    return frame


def normalize_table(
    frame: pd.DataFrame,
    *,
    text_column: str = "chat_text",
    require_text: bool = True,
    require_answer: bool = False,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    columns = list(frame.columns)
    chat_cols = _find_columns(columns, CHAT_ID_ALIASES)
    text_cols = _find_columns(columns, (text_column, *TEXT_ALIASES))
    reason_cols = _find_columns(columns, REASON_ALIASES)
    answer_cols = _find_columns(columns, ANSWER_ALIASES)
    comment_cols = _find_columns(columns, COMMENT_ALIASES)
    link_cols = _find_columns(columns, LINK_ALIASES)

    if require_text and not text_cols:
        raise DataFormatError(
            "Для валидатора нужен полный текст чата: добавьте колонку chat_text "
            "или передайте --text-column с названием колонки."
        )
    if not reason_cols:
        raise DataFormatError("Input table must contain reason_id / reason_number / predicted_reason.")
    if require_answer and not answer_cols:
        raise DataFormatError("Training/evaluation data must contain да/нет / yes/no / human_answer.")

    out = frame.copy()
    out["chat_id"] = (
        _coalesce_columns(out, chat_cols).astype(str)
        if chat_cols
        else [f"row_{i}" for i in range(len(out))]
    )
    out["chat_text"] = _coalesce_columns(out, text_cols).astype(str) if text_cols else ""
    out["reason_id_raw"] = _coalesce_columns(out, reason_cols).astype(str)
    out["comment"] = _coalesce_columns(out, comment_cols).astype(str) if comment_cols else ""
    out["link"] = _coalesce_columns(out, link_cols).astype(str) if link_cols else ""
    if answer_cols:
        answer_values = _coalesce_columns(out, answer_cols)
        out["human_answer_raw"] = answer_values
        out["human_label"] = answer_values.map(normalize_human_answer)
    else:
        out["human_answer_raw"] = ""
        out["human_label"] = None

    exploded = []
    for _, row in out.iterrows():
        reason_ids = split_reason_ids(row["reason_id_raw"])
        if not reason_ids:
            continue
        for reason_id in reason_ids:
            item = row.copy()
            item["reason_id"] = reason_id
            exploded.append(item)
    if not exploded:
        return pd.DataFrame(columns=list(out.columns) + ["reason_id"])
    result = pd.DataFrame(exploded).reset_index(drop=True)
    return result


def load_tables(
    patterns: Sequence[str],
    *,
    text_column: str = "chat_text",
    require_text: bool = True,
    require_answer: bool = False,
) -> pd.DataFrame:
    frames = []
    for path in expand_paths(patterns):
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        frame = read_input_table(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    return normalize_table(
        raw,
        text_column=text_column,
        require_text=require_text,
        require_answer=require_answer,
    )
