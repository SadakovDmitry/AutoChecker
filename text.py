from __future__ import annotations

import re
from dataclasses import dataclass


CLIENT_ROLES = {"client", "клиент", "customer", "user"}
MANAGER_ROLES = {"manager", "operator", "оператор", "сотрудник", "agent", "support"}
BOT_ROLES = {"bot", "бот", "robot"}


@dataclass
class RoleText:
    full_text: str
    client_text: str
    operator_text: str
    bot_text: str
    model_text: str


ROLE_RE = re.compile(
    r"^\s*(client|клиент|customer|user|manager|operator|оператор|сотрудник|agent|support|bot|бот|robot)\s*[:\\-–—]\s*(.*)$",
    flags=re.IGNORECASE,
)


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_roles(text: object) -> RoleText:
    full_text = clean_text(text)
    client_parts = []
    operator_parts = []
    bot_parts = []
    unknown_parts = []

    for line in full_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = ROLE_RE.match(stripped)
        if not match:
            unknown_parts.append(stripped)
            continue
        role = match.group(1).lower()
        content = match.group(2).strip()
        if not content:
            continue
        if role in CLIENT_ROLES:
            client_parts.append(content)
        elif role in MANAGER_ROLES:
            operator_parts.append(content)
        elif role in BOT_ROLES:
            bot_parts.append(content)
        else:
            unknown_parts.append(content)

    client_text = clean_text("\n".join(client_parts))
    operator_text = clean_text("\n".join(operator_parts))
    bot_text = clean_text("\n".join(bot_parts))
    model_text = client_text or full_text
    return RoleText(
        full_text=full_text,
        client_text=client_text,
        operator_text=operator_text,
        bot_text=bot_text,
        model_text=model_text,
    )


def add_role_columns(frame):
    roles = frame["chat_text"].map(split_roles)
    out = frame.copy()
    out["full_text"] = roles.map(lambda x: x.full_text)
    out["client_text"] = roles.map(lambda x: x.client_text)
    out["operator_text"] = roles.map(lambda x: x.operator_text)
    out["bot_text"] = roles.map(lambda x: x.bot_text)
    out["model_text"] = roles.map(lambda x: x.model_text)
    return out
