from __future__ import annotations

import argparse
import re
import zlib
from pathlib import Path


def _streams(data: bytes):
    for match in re.finditer(rb"(\d+)\s+0\s+obj(.*?)endobj", data, re.S):
        obj_id = int(match.group(1))
        obj = match.group(2)
        stream_match = re.search(rb"stream\r?\n(.*?)\r?\nendstream", obj, re.S)
        if not stream_match:
            continue
        raw = stream_match.group(1)
        if b"FlateDecode" in obj[: stream_match.start()]:
            try:
                raw = zlib.decompress(raw)
            except Exception:
                continue
        yield obj_id, obj, raw


def _parse_cmap(raw: bytes) -> dict[int, str]:
    text = raw.decode("latin1", errors="ignore")
    mapping: dict[int, str] = {}
    for start, end, dst in re.findall(
        r"<([0-9A-Fa-f]{4})><([0-9A-Fa-f]{4})><([0-9A-Fa-f]+)>",
        text,
    ):
        start_i = int(start, 16)
        end_i = int(end, 16)
        dst_i = int(dst, 16)
        for code in range(start_i, end_i + 1):
            mapping[code] = chr(dst_i + (code - start_i))
    return mapping


def _unescape_pdf_literal(value: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(value):
        char = value[i]
        if char == 0x5C and i + 1 < len(value):
            nxt = value[i + 1]
            if nxt in b"nrtbf":
                out.append(
                    {
                        ord("n"): 10,
                        ord("r"): 13,
                        ord("t"): 9,
                        ord("b"): 8,
                        ord("f"): 12,
                    }[nxt]
                )
                i += 2
            elif nxt in b"()\\":
                out.append(nxt)
                i += 2
            elif 48 <= nxt <= 55:
                j = i + 1
                digits = []
                while j < len(value) and len(digits) < 3 and 48 <= value[j] <= 55:
                    digits.append(chr(value[j]))
                    j += 1
                out.append(int("".join(digits), 8))
                i = j
            elif nxt in b"\r\n":
                i += 2
                if nxt == 13 and i < len(value) and value[i] == 10:
                    i += 1
            else:
                out.append(nxt)
                i += 2
        else:
            out.append(char)
            i += 1
    return bytes(out)


def _decode_cids(value: bytes, cmap: dict[int, str]) -> str:
    chars = []
    i = 0
    while i + 1 < len(value):
        cid = (value[i] << 8) + value[i + 1]
        chars.append(cmap.get(cid, ""))
        i += 2
    return "".join(chars)


def _extract_text_chunks(raw: bytes, cmap: dict[int, str]) -> list[tuple[float, float, str]]:
    chunks: list[tuple[float, float, str]] = []
    for text_object in re.finditer(rb"BT(.*?)ET", raw, re.S):
        block = text_object.group(1)
        coord = (0.0, 0.0)
        tm = list(
            re.finditer(
                rb"([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+Tm",
                block,
            )
        )
        if tm:
            coord = (float(tm[-1].group(5)), float(tm[-1].group(6)))

        for match in re.finditer(rb"\((?:\\.|[^\\()])*\)\s*Tj", block, re.S):
            literal = match.group(0)
            content = literal[1 : literal.rfind(b")")]
            text = _decode_cids(_unescape_pdf_literal(content), cmap)
            if text.strip():
                chunks.append((coord[0], coord[1], text))

        for array in re.finditer(rb"\[(.*?)\]\s*TJ", block, re.S):
            text = ""
            for match in re.finditer(rb"\((?:\\.|[^\\()])*\)", array.group(1), re.S):
                text += _decode_cids(_unescape_pdf_literal(match.group(0)[1:-1]), cmap)
            if text.strip():
                chunks.append((coord[0], coord[1], text))
    return chunks


def extract_pdf_text(path: Path) -> str:
    data = path.read_bytes()
    cmap: dict[int, str] = {}
    stream_data = []
    for obj_id, obj, raw in _streams(data):
        stream_data.append((obj_id, obj, raw))
        if b"begincmap" in raw:
            cmap.update(_parse_cmap(raw))

    chunks = []
    for _, _, raw in stream_data:
        if b"Tj" in raw or b"TJ" in raw:
            chunks.extend(_extract_text_chunks(raw, cmap))
    return "\n".join(text for _, _, text in chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from simple ToUnicode PDFs.")
    parser.add_argument("pdf", nargs="+")
    parser.add_argument("--out-dir", default="auto_classifier/local_data/reports/pdf_text_extracted")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pdf in args.pdf:
        path = Path(pdf)
        text = extract_pdf_text(path)
        out = out_dir / f"{path.stem}.txt"
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
