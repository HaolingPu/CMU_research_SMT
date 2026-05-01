"""Text helpers: spacing-aware concatenation of source chunks and units."""
from __future__ import annotations

import ast
from typing import Any, Dict, List

import pandas as pd


def parse_trajectory(raw: str) -> List[str]:
    return ast.literal_eval(raw)


def join_source_chunks(chunks: List[str]) -> str:
    text = ""
    for raw_piece in chunks:
        piece = str(raw_piece or "")
        if not piece:
            continue
        if not text:
            text = piece
            continue
        if text[-1].isspace() or piece[0].isspace():
            text += piece
        elif piece[0] in ",.!?;:)]}\"'":
            text += piece
        elif text[-1] in "([{\"'":
            text += piece
        else:
            text += " " + piece
    return text.strip()


def build_source_observed(chunks: List[str], t: int) -> str:
    return join_source_chunks(chunks[: t + 1])


def parse_source_units(raw: Any) -> List[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [str(x or "") for x in parsed]
    return [str(parsed)]


def append_text_continuation(prefix: str, continuation: str) -> str:
    if not prefix:
        return continuation
    if not continuation:
        return prefix
    if prefix[-1].isspace() or continuation[0].isspace():
        return prefix + continuation
    if continuation[0] in ",.!?;:)]}\"'":
        return prefix + continuation
    return prefix + " " + continuation


def build_source_observed_recent_units(
    source_units: List[str],
    observed_full: str,
    num_units: int,
) -> str:
    if not observed_full:
        return ""
    if not source_units or num_units <= 0:
        return observed_full

    prev_full = ""
    for unit_idx, unit in enumerate(source_units):
        full_through_unit = append_text_continuation(prev_full, unit)
        if observed_full == full_through_unit or full_through_unit.startswith(observed_full):
            start_idx = max(0, unit_idx - num_units + 1)
            prefix = ""
            for prior_unit in source_units[start_idx:unit_idx]:
                prefix = append_text_continuation(prefix, prior_unit)
            partial_current = observed_full[len(prev_full):] if observed_full.startswith(prev_full) else observed_full
            return append_text_continuation(prefix, partial_current)
        prev_full = full_through_unit

    return observed_full


def get_full_source_text(row: Dict[str, Any]) -> str:
    raw = row.get("src_text")
    if raw is None or pd.isna(raw):
        raise ValueError("src_text is missing from input row")
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        raise ValueError("src_text is empty in input row")
    return text


def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
