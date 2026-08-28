#!/usr/bin/env python3
"""
Word alignment via simalign (drop-in replacement for awesome-align).

【Simalign 整体逻辑】
1. 用 simalign（基于 LaBSE 等句子向量的词对齐）做 源文(如英文) ↔ 译文(如中文) 的词级对齐。
2. 源文按空格分词，译文用 jieba 分词后，调用 SentenceAligner.get_word_aligns() 得到 (src_word_idx, tgt_word_idx)。
3. 返回给上层的是 (src_word_idx, tgt_char_idx)：tgt_char_idx 是「归一化译文」里该目标词结尾字符的索引，
   这样 core 用 translation[:len(result)] 截断时，和归一化字符串的字符一一对应。
4. 可选：环境变量 SIMALIGN_USE_MONOTONIC=1 时，会对齐结果做单调化（源索引随目标索引非降），便于轨迹/边界一致。

Default behavior follows the "minimal simalign path":
  - split source by whitespace
  - split Chinese target by jieba
  - call SentenceAligner.get_word_aligns()
  - use returned (src_word_idx, tgt_word_idx) pairs directly

Optional monotonic post-processing can be enabled via env var for A/B testing.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

# 零宽/不可见字符，对齐前从译文里去掉，避免 jieba 和字符偏移错位
_ZERO_WIDTH_AND_INVISIBLE = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u2060\u2061\u2062\u2063\u2064\u180e\u034f]+"
)

_DEBUG_PRINT_COUNT = 0


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# Optional: normalize translation before alignment (strip, first line)
def _normalize_tgt_for_jieba(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    # First line only, like clean_translation_for_alignment
    if "\n" in t:
        t = t.split("\n")[0].strip()
    t = _ZERO_WIDTH_AND_INVISIBLE.sub("", t)
    return t


def _translation_prefix_for_norm_char_count(translation: str, norm_char_count: int) -> str:
    """Return prefix of translation (first line, stripped) that has exactly norm_char_count
    non-space chars. So core's translation[:len(result)] aligns with norm[:norm_char_count].
    """
    if norm_char_count <= 0:
        return ""
    trans_work = _normalize_tgt_for_jieba(translation)
    n = 0
    for i, c in enumerate(trans_work):
        if not c.isspace():
            n += 1
            if n >= norm_char_count:
                return trans_work[: i + 1]
    return trans_work


def load_align_model(cache_dir: Optional[str] = None, device: Optional[str] = None):
    """Load simalign SentenceAligner. Returns (aligner, None) for drop-in with core.

    Must be called before loading vLLM if align shares GPU with base.
    If cache_dir is set (e.g. from HF_HOME), model is loaded from that directory.
    """
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
    from simalign import SentenceAligner
    if device is None:
        device = "cuda"
    elif isinstance(device, str) and device.startswith("cuda"):
        device = "cuda"  # simalign uses "cuda" or "cpu"
    align_model_name = os.environ.get("SIMALIGN_MODEL", "sentence-transformers/LaBSE")
    matching_methods = os.environ.get("SIMALIGN_MATCHING_METHODS", "a")
    use_monotonic = _env_flag("SIMALIGN_USE_MONOTONIC", default=False)
    debug_enabled = _env_flag("SIMALIGN_DEBUG", default=False)
    aligner = SentenceAligner(
        model=align_model_name,
        token_type="bpe",
        matching_methods=matching_methods,
        device=device,
    )
    cfg = {
        "matching_methods": matching_methods,
        "use_monotonic": use_monotonic,
        "debug": debug_enabled,
        "debug_max_cases": int(os.environ.get("SIMALIGN_DEBUG_MAX_CASES", "20")),
    }
    print(
        f"[Align v2] simalign loaded on {device} | model={align_model_name} "
        f"| methods={matching_methods} | monotonic={use_monotonic} | debug={debug_enabled}"
    )
    return aligner, cfg # 后面 get_word_alignments()用到


def _to_monotonic_alignments(
    alignments_inter: List[Tuple[int, int]],
    n_src: int,
    n_tgt: int,
) -> List[Tuple[int, int]]:
    """Convert crossing alignments to monotonic (src non-decreasing along tgt).

    Follows GigaSpeech build_trajectory_full_mfa.py:
    - sort by (tgt_idx, src_idx)
    - append (len(src_words)-1, len(tgt_words)-1)
    - merge same tgt: keep last (larger src)
    - ensure src non-decreasing: alignments_r[i] = (max(a[0], alignments_r[i-1][0]), a[1])
    - prepend (-1, -1) for trajectory use; we return without (-1,-1) for truncation.
    """
    if not alignments_inter:
        return []
    alignments = sorted(alignments_inter, key=lambda x: (x[1], x[0]))
    alignments.append((n_src - 1, n_tgt - 1))
    alignments_r = []
    for a in alignments:
        if len(alignments_r) > 0 and alignments_r[-1][1] == a[1]:
            alignments_r[-1] = a  # replace element (same tgt idx -> keep later src)
        else:
            alignments_r.append(a)
    for i in range(1, len(alignments_r)):
        s, t = alignments_r[i]
        s_prev = alignments_r[i - 1][0]
        alignments_r[i] = (max(s, s_prev), t)
    # Return without (-1,-1); caller uses (src_idx, tgt_idx) for truncation
    return alignments_r


def _pick_alignment_pairs(
    alignments_dict: Dict[str, Any],
    matching_methods: str,
) -> List[Tuple[int, int]]:
    methods = set((matching_methods or "").lower())
    key_priority: List[str] = []
    if "a" in methods:
        key_priority.extend(["mwmf", "fwd", "forward", "argmax"])
    if "m" in methods:
        key_priority.extend(["inter", "intersection", "mwmf"])
    if "i" in methods:
        key_priority.extend(["itermax", "iter"])
    key_priority.extend(["mwmf", "inter", "itermax", "fwd", "forward"])

    seen = set()
    for k in key_priority:
        if k in seen:
            continue
        seen.add(k)
        v = alignments_dict.get(k)
        if isinstance(v, list) and v:
            return [(int(s), int(t)) for s, t in v]

    for v in alignments_dict.values():
        if isinstance(v, list) and v:
            try:
                return [(int(s), int(t)) for s, t in v]
            except Exception:
                continue
    return []


def get_word_alignments(
    src_text: str,
    tgt_text: str,
    align_model: Any,
    align_tokenizer: Any,
) -> List[Tuple[int, int]]:
    """Word-level alignment via simalign.

    Returns (src_word_idx, tgt_char_idx) so core's len(translation) checks work.
    tgt_char_idx = end character index (inclusive) of that tgt word in normalized text.
    - src: split by whitespace (English words).
    - tgt: jieba (Chinese words). Translation is normalized (first line, strip).
    """
    import jieba
    src_words = [w for w in src_text.strip().split() if w and not w.isspace()]
    tgt_raw = _normalize_tgt_for_jieba(tgt_text)
    tgt_words = [w for w in jieba.cut(tgt_raw) if w and not w.isspace()] if tgt_raw else []

    if not src_words or not tgt_words:
        return []

    # 
    cfg = align_tokenizer if isinstance(align_tokenizer, dict) else {} # align_tokenizer 实际上不是 tokenizer，而是 cfg 配置 dict
    matching_methods = str(cfg.get("matching_methods", "a"))
    use_monotonic = bool(cfg.get("use_monotonic", False))
    debug_enabled = bool(cfg.get("debug", False))
    debug_max_cases = int(cfg.get("debug_max_cases", 20))

    try:
        alignments_dict = align_model.get_word_aligns(src_words, tgt_words)
    except Exception:
        return []

    raw_pairs = _pick_alignment_pairs(alignments_dict, matching_methods)
    if not raw_pairs:
        return []

    out_pairs = raw_pairs
    if use_monotonic:
        out_pairs = _to_monotonic_alignments(raw_pairs, len(src_words), len(tgt_words))

    global _DEBUG_PRINT_COUNT
    if debug_enabled and _DEBUG_PRINT_COUNT < debug_max_cases:
        _DEBUG_PRINT_COUNT += 1
        size_info = {
            k: (len(v) if isinstance(v, list) else -1)
            for k, v in alignments_dict.items()
        }
        print(
            f"[Align v2 DEBUG #{_DEBUG_PRINT_COUNT}] src_words={src_words} "
            f"| tgt_words={tgt_words} | keys={size_info} "
            f"| picked={raw_pairs} | monotonic={use_monotonic} -> {out_pairs}",
            flush=True,
        )

    # Map tgt_word_idx -> end char index (inclusive) in normalized translation
    norm = "".join(tgt_words)
    cum = 0
    char_end = []
    for w in tgt_words:
        cum += len(w)
        char_end.append(cum - 1)
    return [(s, char_end[t]) for s, t in out_pairs if t < len(char_end)]


def truncate_by_alignment(
    full_src: str,
    observed_src: str,
    translation: str,
    alignments: List[Tuple[int, int]],
) -> str:
    """Truncate by scanning target-side alignments left-to-right.

    We keep consuming target content while aligned source indices stay inside
    the observed range [0, obs_n). The first aligned target position whose
    source index enters the future range [obs_n, ...) stops the prefix.

    alignments: (src_word_idx, tgt_char_idx), where tgt_char_idx is the end
    character index (inclusive) of the aligned target word in normalized text.
    Unmatched target characters are ignored naturally because only aligned
    positions participate in the scan.
    """
    obs_n = len(observed_src.strip().split())
    if obs_n <= 0 or not translation:
        return ""

    norm = _normalize_tgt_for_jieba(translation).replace(" ", "")
    n_norm = len(norm)
    if not n_norm or not alignments:
        return ""

    safe_char_idx = -1
    future_seen = False

    for s, t in sorted(alignments, key=lambda x: (x[1], x[0])):
        if t < 0:
            continue
        t = min(t, n_norm - 1)
        if s < obs_n:
            safe_char_idx = max(safe_char_idx, t)
            continue
        future_seen = True
        break

    if safe_char_idx >= 0:
        return _translation_prefix_for_norm_char_count(translation, safe_char_idx + 1)
    if future_seen:
        return ""
    return ""


def build_local_alignment_windows(
    full_src: str,
    observed_src: str,
    translation: str,
    committed: str,
    src_left_words: int = 40,
    src_right_words: int = 24,
    tgt_left_chars: int = 120,
) -> Tuple[str, str, str, int]:
    """Build shorter alignment inputs around the observed/future boundary.

    Returns:
      local_full_src, local_observed_src, local_translation, tgt_offset
    """
    src_words = full_src.strip().split()
    observed_count = min(len(observed_src.strip().split()), len(src_words))
    if not src_words or observed_count <= 0:
        return full_src, observed_src, translation, 0

    left = max(1, src_left_words)
    right = max(1, src_right_words)
    start = max(0, observed_count - left)
    end = min(len(src_words), observed_count + right)
    if start >= end:
        return full_src, " ".join(src_words[:observed_count]), translation, 0

    local_full_src = " ".join(src_words[start:end])
    local_observed_src = " ".join(src_words[start:observed_count])

    offset = max(0, len(committed) - max(0, tgt_left_chars))
    local_translation = translation[offset:] if translation else ""
    if len(local_translation) < 8:
        offset = 0
        local_translation = translation

    return local_full_src, local_observed_src, local_translation, offset


def get_word_alignments_batch(
    pairs: List[Tuple[str, str]],
    align_model: Any,
    align_tokenizer: Any,
) -> List[List[Tuple[int, int]]]:
    """Batch: call get_word_alignments for each pair (simalign has no native batch)."""
    return [
        get_word_alignments(src_text, tgt_text, align_model, align_tokenizer) # align_tokenizer 实际上不是 tokenizer，而是 cfg 配置 dict
        for src_text, tgt_text in pairs
    ]
