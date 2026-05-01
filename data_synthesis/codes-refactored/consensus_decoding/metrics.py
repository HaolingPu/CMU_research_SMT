"""LAAL, char-level BLEU, and length-ratio metrics for the streaming output."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd


def _extract_reference_text_from_row(row: Dict[str, Any], target_lang: str = "Chinese") -> Optional[str]:
    lang_suffix_map = {"Japanese": "ja", "German": "de", "French": "fr", "Spanish": "es"}
    lang_suffix = lang_suffix_map.get(target_lang, "")
    keys = []
    if lang_suffix:
        keys.extend([f"target_full_{lang_suffix}", f"tgt_text_full_{lang_suffix}", f"llm_reference_text_{lang_suffix}"])
    keys.extend(["llm_reference_text", "tgt_text_full", "tgt_text", "target_text", "translation", "ref_text", "reference"])
    for key in keys:
        raw = row.get(key)
        if raw is None or pd.isna(raw):
            continue
        text = str(raw).strip()
        if text and text.lower() != "nan":
            return text
    return None


def compute_laal(
    source_chunks: List[str], target_deltas: List[str], actions: List[str], reference: str,
) -> float:
    timeline: List[int] = []
    source_read = 0
    for chunk, delta, action in zip(source_chunks, target_deltas, actions):
        source_read += len(str(chunk).strip().split()) if str(chunk).strip() else 0
        if action == "WRITE" and str(delta).strip():
            for _ in str(delta).strip():
                timeline.append(source_read)
    y_len = len("".join(d for d in target_deltas if d))
    yref_len = len(str(reference).replace(" ", ""))
    x_len = sum(len(str(c).strip().split()) for c in source_chunks if str(c).strip())
    if y_len == 0 or x_len == 0 or yref_len == 0:
        return float("nan")
    denom = max(y_len, yref_len)
    if denom <= 0 or not timeline:
        return float("nan")
    total = 0.0
    for i in range(1, denom + 1):
        d_i = timeline[i - 1] if i <= len(timeline) else x_len
        total += d_i - (i - 1) * x_len / denom
    return total / denom


def compute_bleu_char(hypothesis: str, reference: str, max_order: int = 4, smooth: bool = True) -> float:
    hyp = [c for c in str(hypothesis) if not c.isspace()]
    ref = [c for c in str(reference) if not c.isspace()]
    hyp_len, ref_len = len(hyp), len(ref)
    if hyp_len == 0 or ref_len == 0:
        return float("nan")
    eff_order = min(max_order, hyp_len, ref_len)
    if eff_order <= 0:
        return float("nan")
    precisions: List[float] = []
    for n in range(1, eff_order + 1):
        hyp_ngrams = Counter(tuple(hyp[i:i + n]) for i in range(hyp_len - n + 1))
        ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(ref_len - n + 1))
        total = sum(hyp_ngrams.values())
        if total <= 0:
            return float("nan")
        clipped = sum(min(cnt, ref_ngrams.get(ng, 0)) for ng, cnt in hyp_ngrams.items())
        if smooth:
            precisions.append((clipped + 1.0) / (total + 1.0))
        else:
            if clipped == 0:
                return 0.0
            precisions.append(clipped / total)
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len))
    return bp * math.exp(sum(math.log(p) for p in precisions) / eff_order) * 100.0


def _nonspace_char_count(text: str) -> int:
    return sum(1 for c in str(text or "") if not c.isspace())


def compute_length_ratio_ref(hypothesis: str, reference: str) -> float:
    hyp_len = _nonspace_char_count(hypothesis)
    ref_len = _nonspace_char_count(reference)
    if hyp_len == 0 or ref_len == 0:
        return float("nan")
    return hyp_len / ref_len


def compute_length_ratio_src(hypothesis: str, source: str) -> float:
    hyp_len = _nonspace_char_count(hypothesis)
    src_word_count = len(str(source or "").split())
    if hyp_len == 0 or src_word_count == 0:
        return float("nan")
    return hyp_len / src_word_count
