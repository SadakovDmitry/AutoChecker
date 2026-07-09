from __future__ import annotations

import argparse
import cgi
import html
import shutil
import sys
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from .analyst_workflow import (
    DEFAULT_SUBREASON_MAP,
    autolabel_latest_iteration,
    available_dataset_keys,
)
from .data import DataFormatError


APP_ROOT = Path(__file__).resolve().parent
LOCAL_DATA = APP_ROOT / "local_data"
UPLOAD_DIR = LOCAL_DATA / "uploads"
OUTPUT_DIR = LOCAL_DATA / "ui_outputs"


def _safe_filename(name: str, fallback: str) -> str:
    cleaned = Path(name or fallback).name.strip().replace("/", "_").replace("\\", "_")
    return cleaned or fallback


def _read_field(form: cgi.FieldStorage, name: str, default: str = "") -> str:
    value = form.getfirst(name, default)
    return str(value or "").strip()


def _save_upload(form: cgi.FieldStorage, field_name: str, prefix: str) -> Path:
    item = form[field_name] if field_name in form else None
    if item is None or not getattr(item, "filename", ""):
        raise DataFormatError(f"Не загружен файл: {field_name}")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(item.filename, f"{prefix}.xlsx")
    target = UPLOAD_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{filename}"
    with target.open("wb") as file_obj:
        shutil.copyfileobj(item.file, file_obj)
    return target


def _html_page(body: str, *, title: str = "AutoChecker") -> bytes:
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      background: #f5f7fb;
      color: #1f2937;
    }}
    main {{
      width: min(980px, calc(100% - 40px));
      margin: 34px auto;
      background: #fff;
      border: 1px solid #dfe5ef;
      border-radius: 10px;
      padding: 28px;
      box-shadow: 0 14px 42px rgba(15, 23, 42, 0.08);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p {{ line-height: 1.5; }}
    label {{ display: block; margin: 18px 0 6px; font-weight: 650; }}
    input, select {{
      box-sizing: border-box;
      width: 100%;
      padding: 11px 12px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      font-size: 15px;
      background: #fff;
    }}
    input[type="checkbox"] {{ width: auto; margin-right: 8px; }}
    .hint {{ color: #64748b; font-size: 14px; margin-top: 4px; }}
    button {{
      margin-top: 24px;
      padding: 12px 18px;
      background: #ffd84d;
      border: 0;
      border-radius: 8px;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
    }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 9px; text-align: left; }}
    th {{ background: #f8fafc; }}
    .ok {{ color: #166534; font-weight: 700; }}
    .error {{ color: #b91c1c; font-weight: 700; }}
    .download {{
      display: inline-block;
      margin-top: 18px;
      padding: 11px 14px;
      background: #2563eb;
      color: #fff;
      border-radius: 8px;
      text-decoration: none;
      font-weight: 700;
    }}
    .note {{
      margin-top: 16px;
      padding: 12px 14px;
      border-radius: 8px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
    }}
  </style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>""".encode("utf-8")


def _form_html() -> str:
    try:
        datasets = available_dataset_keys(DEFAULT_SUBREASON_MAP)
    except Exception:
        datasets = []
    options = ['<option value="">Не использовать mapping</option>']
    for key in datasets:
        selected = " selected" if key == "kasko_oformlenie" else ""
        options.append(f'<option value="{html.escape(key)}"{selected}>{html.escape(key)}</option>')
    return f"""
<h1>AutoChecker: разметка последней итерации</h1>
<p>
  Загрузите Excel, где каждый лист - отдельная итерация разметки, а последний лист -
  новая неразмеченная итерация. Вторым файлом загрузите текстовки всех диалогов.
</p>
<form action="/run" method="post" enctype="multipart/form-data">
  <label>Файл с итерациями разметки</label>
  <input type="file" name="labels_file" accept=".xlsx,.xlsm,.xls" required>
  <div class="hint">На предыдущих листах должны быть разметки `да/нет`, последний лист будет размечен автоматически.</div>

  <label>Файл с текстовками диалогов</label>
  <input type="file" name="messages_file" accept=".xlsx,.xlsm,.xls,.csv" required>
  <div class="hint">Поддерживается формат “одна строка = сообщение” и “одна строка = полный диалог”.</div>

  <label>Тема / набор подпричин</label>
  <select name="dataset_key">
    {''.join(options)}
  </select>
  <div class="hint">Нужно для стабильных subreason_key, если номера подпричин менялись между итерациями.</div>

  <label>Лист новой итерации, если это не последний лист</label>
  <input type="text" name="latest_sheet" placeholder="оставьте пустым, чтобы взять последний лист">

  <label>Лист с текстовками, если нужен конкретный лист</label>
  <input type="text" name="messages_sheet" placeholder="оставьте пустым, чтобы прочитать все листы">

  <label>
    <input type="checkbox" name="use_embeddings" value="1">
    Использовать embeddings, если локальная модель уже скачана
  </label>
  <div class="hint">По умолчанию используется LSA/fallback без внешних загрузок.</div>

  <button type="submit">Разметить последнюю итерацию</button>
</form>
<div class="note">
  Рабочий режим: гибридный роутер `threshold + max(history_latest, mean p_correct)`.
  Для стабильных подпричин ставит `да/нет`, для рискованных - только уверенные `да`, остальное `review`.
</div>
"""


class AutoCheckerHandler(BaseHTTPRequestHandler):
    server_version = "AutoCheckerHTTP/1.0"

    def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = _html_page(body)
        self.send_response(status.value)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            self._send_html(_form_html())
            return
        if self.path.startswith("/download/"):
            filename = _safe_filename(unquote(self.path[len("/download/") :]), "output.xlsx")
            file_path = OUTPUT_DIR / filename
            if not file_path.exists() or file_path.parent != OUTPUT_DIR:
                self._send_html("<h1 class='error'>Файл не найден</h1><p><a href='/'>Назад</a></p>", HTTPStatus.NOT_FOUND)
                return
            payload = file_path.read_bytes()
            self.send_response(HTTPStatus.OK.value)
            self.send_header(
                "Content-Type",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._send_html("<h1 class='error'>Страница не найдена</h1><p><a href='/'>Назад</a></p>", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/run":
            self._send_html("<h1 class='error'>Страница не найдена</h1><p><a href='/'>Назад</a></p>", HTTPStatus.NOT_FOUND)
            return
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            labels_path = _save_upload(form, "labels_file", "labels.xlsx")
            messages_path = _save_upload(form, "messages_file", "messages.xlsx")
            dataset_key = _read_field(form, "dataset_key") or None
            latest_sheet = _read_field(form, "latest_sheet") or None
            messages_sheet = _read_field(form, "messages_sheet") or None
            use_embeddings = _read_field(form, "use_embeddings") == "1"

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_name = f"autochecker_marked_{int(time.time())}_{uuid.uuid4().hex[:8]}.xlsx"
            output_path = OUTPUT_DIR / output_name
            result = autolabel_latest_iteration(
                labels_path=labels_path,
                messages_path=messages_path,
                output_path=output_path,
                dataset_key=dataset_key,
                latest_sheet=latest_sheet,
                messages_sheet=messages_sheet,
                use_embeddings=use_embeddings,
            )
            coverage = f"{result.coverage:.2%}"
            body = f"""
<h1 class="ok">Готово</h1>
<p>Последняя итерация размечена. Строки с `review` нужно проверить руками.</p>
<table>
  <tr><th>Показатель</th><th>Значение</th></tr>
  <tr><td>Лист новой итерации</td><td>{html.escape(result.latest_sheet)}</td></tr>
  <tr><td>Строк в обучении</td><td>{result.train_rows}</td></tr>
  <tr><td>Строк в новой итерации</td><td>{result.latest_rows}</td></tr>
  <tr><td>Авторазмечено</td><td>{result.auto_rows} ({coverage})</td></tr>
  <tr><td>Авто `да`</td><td>{result.auto_yes}</td></tr>
  <tr><td>Авто `нет`</td><td>{result.auto_no}</td></tr>
  <tr><td>Review</td><td>{result.review_rows}</td></tr>
</table>
<a class="download" href="/download/{html.escape(output_name)}">Скачать размеченный Excel</a>
<p><a href="/">Разметить другой файл</a></p>
"""
            self._send_html(body)
        except Exception as exc:
            message = html.escape(str(exc))
            self._send_html(
                f"<h1 class='error'>Ошибка</h1><p>{message}</p><p><a href='/'>Назад</a></p>",
                HTTPStatus.BAD_REQUEST,
            )

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local AutoChecker upload UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args(argv)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), AutoCheckerHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"AutoChecker UI: {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
