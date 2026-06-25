from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from .data import (
    CHAT_ID_ALIASES,
    DataFormatError,
    _find_column,
    expand_paths,
    normalize_table,
    read_input_table,
)
from .reports import write_table
from .text import clean_text


TIME_ALIASES = (
    "vremya",
    "time",
    "timestamp",
    "created_at",
    "created",
    "datetime",
    "date",
    "время",
    "дата",
)
SENDER_ALIASES = (
    "kto",
    "sender",
    "role",
    "author",
    "from",
    "кто",
    "отправитель",
    "автор",
)
MESSAGE_ALIASES = (
    "soobschenie",
    "message",
    "text",
    "content",
    "body",
    "сообщение",
    "текст",
)
FULL_DIALOG_ALIASES = (
    "dialog_polnostyu",
    "full_dialog",
    "full_chat",
    "chat_text",
    "dialog",
    "communication",
    "диалог полностью",
    "полный диалог",
    "текст диалога",
)
MESSAGE_COUNT_ALIASES = (
    "kolichestvo_soobscheniy",
    "message_count",
    "messages_count",
    "количество сообщений",
)
FIRST_MESSAGE_ALIASES = (
    "pervoe_soobschenie",
    "first_message_at",
    "first_message",
    "первое сообщение",
)
LAST_MESSAGE_ALIASES = (
    "poslednee_soobschenie",
    "last_message_at",
    "last_message",
    "последнее сообщение",
)


@dataclass
class PrepareStats:
    labels_rows: int
    prepared_rows: int
    message_rows: int
    message_chats: int
    matched_rows: int
    missing_text_rows: int


def _load_raw_tables(
    patterns: Sequence[str],
    *,
    source_sheet: Optional[str] = None,
) -> pd.DataFrame:
    frames = []
    for path in expand_paths(patterns):
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        frames.append(read_input_table(path))
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    if source_sheet:
        if "_source_sheet" not in frame.columns:
            raise DataFormatError(
                f"Cannot filter by sheet {source_sheet!r}: input has no sheet metadata."
            )
        available = sorted(str(x) for x in frame["_source_sheet"].dropna().unique())
        frame = frame[frame["_source_sheet"].astype(str) == source_sheet].copy()
        if frame.empty:
            raise DataFormatError(
                f"No rows found for sheet {source_sheet!r}. Available sheets: {available}"
            )
    return frame


def _series_has_chat_ids(values: pd.Series) -> bool:
    return values.fillna("").astype(str).str.match(r"^T\d+").any()


def _copy_if_missing(frame: pd.DataFrame, target: str, source: str) -> None:
    if source not in frame.columns:
        return
    source_values = frame[source]
    if target not in frame.columns:
        frame[target] = source_values
        return
    target_values = frame[target].fillna("").astype(str).str.strip()
    frame[target] = frame[target].where(target_values != "", source_values)


def _column_position(name: object) -> Optional[int]:
    text = str(name)
    if text == "rn":
        return 0
    if text.startswith("Unnamed: "):
        try:
            return int(text.split(":", 1)[1].strip())
        except ValueError:
            return None
    return None


def _non_empty(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def _column_has_links(values: pd.Series) -> bool:
    return _non_empty(values).str.contains(r"https?://", regex=True).any()


def _column_has_answers(values: pd.Series) -> bool:
    text = _non_empty(values)
    non_empty = text[text != ""]
    if non_empty.empty or _column_has_reason_numbers(values):
        return False
    answer_like = non_empty.str.lower().str.fullmatch(r"(да|дa|нет|yes|no|y|n)\s*\?*")
    return answer_like.mean() >= 0.5


def _column_has_reason_numbers(values: pd.Series) -> bool:
    text = _non_empty(values)
    non_empty = text[text != ""]
    if non_empty.empty:
        return False
    return non_empty.str.fullmatch(r"\d+(\.\d+)?(,\s*\d+(\.\d+)?)*").mean() >= 0.7


def _closest_column(columns: list[str], target_position: Optional[int]) -> Optional[str]:
    if not columns:
        return None
    if target_position is None:
        return columns[0]
    return min(
        columns,
        key=lambda column: abs((_column_position(column) or 10_000) - target_position),
    )


def _repair_positional_label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover label columns from sheets exported without headers."""

    if frame.empty:
        return frame
    repaired = frame.copy()
    columns = list(repaired.columns)

    has_chat_column = _find_column(columns, CHAT_ID_ALIASES)
    if has_chat_column and _series_has_chat_ids(repaired[has_chat_column]):
        return repaired

    positional_columns = [
        column for column in columns if _column_position(column) is not None
    ]
    chat_columns = [
        column for column in positional_columns if _series_has_chat_ids(repaired[column])
    ]
    if not chat_columns:
        return repaired

    chat_column = chat_columns[0]
    link_columns = [
        column for column in positional_columns if _column_has_links(repaired[column])
    ]
    answer_columns = [
        column for column in positional_columns if _column_has_answers(repaired[column])
    ]
    reason_columns = [
        column
        for column in positional_columns
        if column not in {chat_column, "rn", *link_columns, *answer_columns}
        and _column_has_reason_numbers(repaired[column])
    ]

    link_column = link_columns[0] if link_columns else None
    link_position = _column_position(link_column) if link_column else None
    answer_column = answer_columns[0] if answer_columns else None
    reason_column = _closest_column(reason_columns, link_position)

    _copy_if_missing(repaired, "comm_id", chat_column)
    if link_column:
        _copy_if_missing(repaired, "link", link_column)
    if reason_column:
        _copy_if_missing(repaired, "reason_number", reason_column)
    if answer_column:
        _copy_if_missing(repaired, "да/нет", answer_column)
        answer_position = _column_position(answer_column)
        comment_column = next(
            (
                column
                for column in positional_columns
                if _column_position(column) == (answer_position or -10) + 1
            ),
            None,
        )
        if comment_column:
            _copy_if_missing(repaired, "comment", comment_column)

    return repaired


def _normalize_role(value: object) -> str:
    text = clean_text(value).lower()
    if text in {"client", "customer", "user", "клиент"}:
        return "client"
    if text in {"manager", "operator", "agent", "support", "сотрудник", "оператор"}:
        return "manager"
    if text in {"bot", "robot", "бот"}:
        return "bot"
    return "unknown"


def _one_line(value: object) -> str:
    return clean_text(value).replace("\n", " ")


def _dialog_line_to_role_text(line: object) -> tuple[str, str]:
    text = clean_text(line)
    if not text:
        return "unknown", ""

    if "|" in text:
        parts = [part.strip() for part in text.split("|", 2)]
        if len(parts) == 3:
            role = _normalize_role(parts[1])
            return role, parts[2]
        if len(parts) == 2:
            role = _normalize_role(parts[0])
            return role, parts[1]

    for separator in (":", "-", "–", "—"):
        if separator in text:
            raw_role, message = text.split(separator, 1)
            role = _normalize_role(raw_role)
            if role != "unknown":
                return role, message.strip()

    return "unknown", text


def _format_full_dialog(value: object) -> tuple[str, dict[str, int]]:
    lines = []
    counts = {"client": 0, "manager": 0, "bot": 0, "unknown": 0}
    last_role: Optional[str] = None
    for raw_line in clean_text(value).splitlines():
        role, message = _dialog_line_to_role_text(raw_line)
        if not message:
            continue
        if role == "unknown" and last_role is not None and lines:
            lines[-1] = f"{lines[-1]} {_one_line(message)}"
            continue
        counts[role if role in counts else "unknown"] += 1
        lines.append(f"{role}: {_one_line(message)}")
        last_role = role if role != "unknown" else last_role
    return "\n".join(lines), counts


def _normalize_full_dialog_table(
    frame: pd.DataFrame,
    *,
    chat_col: str,
    dialog_col: str,
) -> pd.DataFrame:
    count_col = _find_column(frame.columns, MESSAGE_COUNT_ALIASES)
    first_col = _find_column(frame.columns, FIRST_MESSAGE_ALIASES)
    last_col = _find_column(frame.columns, LAST_MESSAGE_ALIASES)

    rows = []
    for _, row in frame.iterrows():
        chat_id = clean_text(row.get(chat_col, ""))
        if not chat_id:
            continue
        chat_text, role_counts = _format_full_dialog(row.get(dialog_col, ""))
        if not chat_text:
            continue
        parsed_message_count = sum(role_counts.values())
        raw_count = row.get(count_col, "") if count_col else ""
        try:
            message_count = int(float(str(raw_count).replace(",", "."))) if str(raw_count).strip() else parsed_message_count
        except ValueError:
            message_count = parsed_message_count
        rows.append(
            {
                "chat_id": chat_id,
                "chat_text": chat_text,
                "message_count": message_count,
                "client_message_count": int(role_counts.get("client", 0)),
                "manager_message_count": int(role_counts.get("manager", 0)),
                "bot_message_count": int(role_counts.get("bot", 0)),
                "unknown_message_count": int(role_counts.get("unknown", 0)),
                "first_message_at": clean_text(row.get(first_col, "")) if first_col else "",
                "last_message_at": clean_text(row.get(last_col, "")) if last_col else "",
            }
        )
    if not rows:
        raise DataFormatError("Full-dialog table has no non-empty dialogs after normalization.")
    result = pd.DataFrame(rows)
    return (
        result.sort_values(["chat_id", "message_count"], ascending=[True, False])
        .drop_duplicates("chat_id", keep="first")
        .reset_index(drop=True)
    )


def normalize_messages_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn message exports into one-chat-per-row chat_text.

    Supports both formats:
    - one message per row: ID_diologa / Vremya / Kto / Soobschenie
    - one dialog per row: ID_diologa / Dialog_polnostyu
    """

    if frame.empty:
        raise DataFormatError("Messages table is empty.")

    columns = list(frame.columns)
    chat_col = _find_column(columns, CHAT_ID_ALIASES)
    time_col = _find_column(columns, TIME_ALIASES)
    sender_col = _find_column(columns, SENDER_ALIASES)
    message_col = _find_column(columns, MESSAGE_ALIASES)
    full_dialog_col = _find_column(columns, FULL_DIALOG_ALIASES)

    if chat_col is not None and full_dialog_col is not None and (
        sender_col is None or message_col is None
    ):
        return _normalize_full_dialog_table(
            frame,
            chat_col=chat_col,
            dialog_col=full_dialog_col,
        )

    missing = []
    if chat_col is None:
        missing.append("chat_id / ID_diologa")
    if sender_col is None:
        missing.append("sender / Kto")
    if message_col is None:
        missing.append("message / Soobschenie")
    if missing:
        raise DataFormatError(
            "Messages table must contain columns: " + ", ".join(missing)
        )

    messages = frame.copy()
    messages["_message_order"] = range(len(messages))
    messages["chat_id"] = messages[chat_col].fillna("").astype(str).map(clean_text)
    messages["sender_role"] = messages[sender_col].map(_normalize_role)
    messages["message_text"] = messages[message_col].map(_one_line)
    if time_col is not None:
        messages["message_time"] = messages[time_col].fillna("").astype(str).map(clean_text)
        messages["_parsed_time"] = pd.to_datetime(
            messages["message_time"], errors="coerce", dayfirst=False
        )
    else:
        messages["message_time"] = ""
        messages["_parsed_time"] = pd.NaT

    messages = messages[
        (messages["chat_id"] != "") & (messages["message_text"] != "")
    ].copy()
    if messages.empty:
        raise DataFormatError("Messages table has no non-empty messages after normalization.")

    messages = messages.sort_values(
        ["chat_id", "_parsed_time", "_message_order"],
        na_position="last",
        kind="mergesort",
    )

    rows = []
    for chat_id, group in messages.groupby("chat_id", sort=False):
        lines = [
            f"{row.sender_role}: {row.message_text}"
            for row in group.itertuples(index=False)
            if row.message_text
        ]
        role_counts = group["sender_role"].value_counts()
        rows.append(
            {
                "chat_id": chat_id,
                "chat_text": "\n".join(lines),
                "message_count": int(len(group)),
                "client_message_count": int(role_counts.get("client", 0)),
                "manager_message_count": int(role_counts.get("manager", 0)),
                "bot_message_count": int(role_counts.get("bot", 0)),
                "unknown_message_count": int(role_counts.get("unknown", 0)),
                "first_message_at": group["message_time"].iloc[0],
                "last_message_at": group["message_time"].iloc[-1],
            }
        )
    return pd.DataFrame(rows)


def prepare_training_data(
    *,
    labels_paths: Sequence[str],
    messages_paths: Sequence[str],
    output: Optional[str] = None,
    require_answer: bool = True,
    labels_sheet: Optional[str] = None,
    messages_sheet: Optional[str] = None,
) -> tuple[pd.DataFrame, PrepareStats]:
    labels_raw = _load_raw_tables(labels_paths, source_sheet=labels_sheet)
    messages_raw = _load_raw_tables(messages_paths, source_sheet=messages_sheet)
    labels_raw = _repair_positional_label_columns(labels_raw)

    labels = normalize_table(
        labels_raw,
        require_text=False,
        require_answer=require_answer,
    )
    if labels.empty:
        raise DataFormatError("Labels table has no rows after normalization.")

    messages = normalize_messages_table(messages_raw)
    prepared = labels.merge(
        messages,
        on="chat_id",
        how="left",
        suffixes=("", "_from_messages"),
    )
    if "chat_text_from_messages" in prepared.columns:
        label_text = prepared["chat_text"].fillna("").astype(str)
        message_text = prepared["chat_text_from_messages"].fillna("").astype(str)
        prepared["chat_text"] = label_text.where(label_text.str.len() > 0, message_text)
        prepared = prepared.drop(columns=["chat_text_from_messages"])
    prepared["has_chat_text"] = prepared["chat_text"].fillna("").astype(str).str.len() > 0

    ordered_columns = [
        "chat_id",
        "reason_id",
        "reason_id_raw",
        "human_answer_raw",
        "human_label",
        "comment",
        "link",
        "chat_text",
        "has_chat_text",
        "message_count",
        "client_message_count",
        "manager_message_count",
        "bot_message_count",
        "unknown_message_count",
        "first_message_at",
        "last_message_at",
    ]
    remaining = [column for column in prepared.columns if column not in ordered_columns]
    prepared = prepared[[column for column in ordered_columns if column in prepared.columns] + remaining]

    stats = PrepareStats(
        labels_rows=int(len(labels)),
        prepared_rows=int(len(prepared)),
        message_rows=int(len(messages_raw)),
        message_chats=int(messages["chat_id"].nunique()),
        matched_rows=int(prepared["has_chat_text"].sum()),
        missing_text_rows=int((~prepared["has_chat_text"]).sum()),
    )

    if output:
        write_table(prepared, output)
    return prepared, stats
