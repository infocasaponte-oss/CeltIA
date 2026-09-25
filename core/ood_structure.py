# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
"""Deterministic structural OOD detector for the CDE routing pipeline.

Some inputs are not user tasks by *form*: bare JSON/XML control blobs, HTTP-style header
blocks, key=value log lines, ANSI escape soup and bracket markers. A semantic classifier
tends to score these as in-domain because they contain no hostile prose. This module flags
them without a model call.

The detector is intentionally conservative: an input is structural only when it carries
(almost) no natural-language content outside the structure, so a real request that merely
*contains* JSON, XML, logs or headers ("why does this fail?", "fix this HTML") is untouched.
It never looks at route or label names.
"""
from __future__ import annotations

import json
import re

_TAG = re.compile(r"</?[A-Za-z][^<>]*>|<\?[^<>]*\?>|<!--.*?-->", re.S)
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_KV = re.compile(r"[A-Za-z_][\w.\-]*=(?:\"[^\"]*\"|'[^']*'|\S+)")
_MARKER = re.compile(r"\[\[[^\]]*\]\]")
_HEADER_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,40}:[ \t]*\S.{0,60}$")
_CONTROL_HEADER_KEY = re.compile(r"^(?:x-|authorization|content-|accept|cookie|host|user-agent)", re.I)
_WORD = re.compile(r"[^\W\d_]{3,}", re.U)

MAX_STRUCTURAL_CHARS = 4000


def _prose_words(text: str) -> int:
    return len(_WORD.findall(text))


def _json_strings(value, out: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            out.append(str(key))
            _json_strings(item, out)
    elif isinstance(value, list):
        for item in value:
            _json_strings(item, out)
    elif isinstance(value, str):
        out.append(value)


def structural_ood_reason(text: str) -> str | None:
    """Return a short reason when ``text`` is a non-task structural blob, else ``None``."""
    if not isinstance(text, str):
        return None
    body = text.strip()
    if not body or len(body) > MAX_STRUCTURAL_CHARS or "?" in body:
        return None

    if body[0] in "{[":
        try:
            parsed = json.loads(body)
        except (ValueError, RecursionError):
            parsed = None
        if isinstance(parsed, (dict, list)):
            strings: list[str] = []
            _json_strings(parsed, strings)
            if all(len(s.split()) <= 3 for s in strings):
                return "json_control_blob"

    without_ansi = _ANSI.sub(" ", body)
    if without_ansi != body and _prose_words(_KV.sub(" ", without_ansi)) <= 3:
        return "terminal_escape_blob"

    if body[0] == "<" and body[-1] == ">":
        if _prose_words(_TAG.sub(" ", body)) <= 3:
            return "markup_control_blob"

    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if lines and all(_HEADER_LINE.match(line) for line in lines):
        if len(lines) >= 2 or _CONTROL_HEADER_KEY.match(lines[0]):
            values = [line.split(":", 1)[1].strip() for line in lines]
            if all(len(value.split()) <= 3 and not re.search(r"[.!]$", value) for value in values):
                return "header_block"

    pairs = _KV.findall(body)
    if len(pairs) >= 2:
        remainder = _KV.sub(" ", body)
        if _prose_words(remainder) <= 2 and len(pairs) >= 3 or _prose_words(remainder) == 0:
            return "key_value_log"

    markers = _MARKER.findall(body)
    if len(markers) >= 2 and not _MARKER.sub("", body).strip():
        return "marker_blob"

    return None
