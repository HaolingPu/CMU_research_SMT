#!/usr/bin/env python3
"""
Future Source Sampling (Final) -- Dual-model architecture + word alignment.

Base model   (e.g. Qwen3-4B):       pure text continuation via llm.generate() (no reasoning needed)
Instruct model (vllm serve):        translation (completion from committed) + LLM judge via OpenAI API
Align model  (awesome-align, GPU):  word-level alignment for safe truncation (can share GPU with base)

Algorithm (per 960ms chunk):
  1. Accumulate observed source words.
  2. Base model generates M diverse future source continuations.
  3. Instruct model translates (observed + each future) -> M candidates (concurrent).
  4. Instruct model selects the best candidate translation.
  5. awesome-align computes word alignment between source and best translation.
  6. Truncate translation to cover only the observed source portion.
  7. Commit safe prefix or READ.
  8. At utterance end, force-commit remainder.

Usage:
  # Start instruct serve on GPU 1 first:
  CUDA_VISIBLE_DEVICES=1 vllm serve MODEL --served-model-name qwen3-instruct --port 8100

  # Then run this script on GPU 0 (base + align share the card when using 4B):
  CUDA_VISIBLE_DEVICES=0 python llm_future_sampling_core.py \\
    --input-tsv MANIFEST.tsv --output-root OUT --test-one
  # With 4B base + align on same GPU, use e.g. --gpu-memory-utilization 0.85.
"""

import argparse
import ast
import asyncio
import bisect
import contextlib
import csv
import itertools
import json
import math
import os
import queue as queue_module
import re
import sys
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from openai import OpenAI, AsyncOpenAI

# Lock that serialises base_llm.generate() when running parallel utterances.
# None when running single-threaded (no overhead).
_base_llm_lock: Optional[threading.Lock] = None

# Lock that serialises align_model (awesome-align / simalign) when parallel_utterances > 1.
_align_model_lock: Optional[threading.Lock] = None

# Batch future sampling: when set, sample_source_futures() submits to this queue
# and a dedicated worker runs base_llm.generate([src1, src2, ...]) in batch (GPU0 并行).
_future_sampling_request_queue: Optional[queue_module.Queue] = None
_future_sampling_worker_thread: Optional[threading.Thread] = None


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Future source sampling (final) -- dual model."
    )
    p.add_argument("--input-tsv", required=True)
    p.add_argument("--output-root", required=True)

    p.add_argument("--base-model-path",
                   default="/data/user_data/haolingp/models/Qwen3-4B-Base")
    p.add_argument("--instruct-api-base", default="http://localhost:8100/v1")
    p.add_argument("--instruct-model-name", default="qwen3-instruct")
    p.add_argument("--instruct-tokenizer-path",
                   default="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8",
                   help="Local tokenizer path used to build manual chat-template prompts for instruct generate.")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                   help="vLLM GPU memory fraction for base model. Use 0.80-0.85 when align model shares the same GPU.")
    p.add_argument("--align-device", default="cuda:0",
                   help="Device for align model (e.g. cuda:0 to share with base; use cpu if OOM).")
    p.add_argument("--align-method", choices=["awesome_align", "simalign"], default="awesome_align",
                   help="Word alignment: awesome_align (default) or simalign+monotonic (core_v2).")

    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)

    p.add_argument("--num-candidates", type=int, default=6,
                   help="M: number of future source samples.")
    p.add_argument("--future-tokens", type=int, default=12,
                   help="Max tokens per continuation.")
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--min-commit-chars", type=int, default=1)
    p.add_argument("--min-observed-words", type=int, default=2)
    p.add_argument("--score-threshold", type=int, default=80,
                   help="Unused.")
    p.add_argument("--consensus-ratio", type=float, default=0.6,
                   help="Fraction of candidates that must share the committed delta (K/M). Default 0.6.")

    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-details", action="store_true")
    p.add_argument("--id-column", default="id")

    p.add_argument("--test-one", action="store_true")
    p.add_argument("--utt-id", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--translation-cache-dir", default=None,
                   help="Directory of pre-computed full translations (task_*.jsonl). "
                        "If provided, used as reference instead of calling translate_final().")
    p.add_argument("--final-commit-backend", choices=["instruct", "base"], default="instruct",
                   help="Backend for end-of-utterance final commit. Use instruct to avoid base-model thinking leakage.")
    p.add_argument("--parallel-utterances", type=int, default=1,
                   help="Number of utterances to process concurrently.  "
                        "HTTP (translate/judge) and alignment calls overlap freely.  "
                        "Recommended: 2-4 on a single-node setup.")
    p.add_argument("--future-sampling-batch-size", type=int, default=4,
                   help="When parallel-utterances>1: batch N future-sampling requests into one "
                        "base_llm.generate([src1,..,srcN]) to parallelize GPU0. Default 4. Set 0 to use serial (lock) instead.")
    p.add_argument("--future-sampling-batch-wait", type=float, default=0.05,
                   help="Max seconds to wait for more requests before running a batch (default 0.05).")
    p.add_argument("--judge-prompt-version", choices=["full", "short"], default="short",
                   help="Unused.")
    p.add_argument(
        "--selection-mode",
        choices=["lcp_code", "lcp70_code", "lcp70_llm", "majority_vote"],
        default="majority_vote",
        help=(
            "Delta selection mode. "
            "lcp_code: pure code LCP (100%); "
            "lcp70_code: pure code quorum-LCP (K/M from consensus_ratio); "
            "lcp70_llm: LLM quorum-LCP prompt with boundary rule; "
            "majority_vote: LLM semantic safe-prefix synthesis with K-vote verification."
        ),
    )
    p.add_argument("--no-tee", action="store_true",
                   help="With --test-one/--verbose: write verbose log only to file, not to stdout (avoids duplicate output).")
    p.add_argument(
        "--disable-sentence-path",
        action="store_true",
        help=(
            "Disable sentence-scoped fast path and always use windowed alignment + Step 4 selection. "
            "Useful when evaluating LLM merge behavior directly."
        ),
    )
    p.add_argument(
        "--majority-vote-disable-backoff",
        action="store_true",
        dest="majority_vote_disable_backoff",
        help=(
            "For selection_mode=majority_vote, disable literal quorum backoff and "
            "measure pure LLM synthesis + verifier behavior."
        ),
    )
    p.add_argument(
        "--semantic-merge-disable-backoff",
        action="store_true",
        dest="majority_vote_disable_backoff",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--majority-vote-prompt-version",
        choices=["simple_v1", "v2", "v3_cot"],
        default="v2",
        help=(
            "Prompt variant for selection_mode=majority_vote. "
            "simple_v1: minimal baseline with 2 examples; "
            "v2: stronger anti-hallucination / divergence-aware prompt; "
            "v3_cot: v2 plus explicit internal step-by-step reasoning instructions."
        ),
    )
    p.add_argument(
        "--majority-vote-use-reasoning",
        action="store_true",
        help=(
            "For selection_mode=majority_vote, use a separate thinking/reasoning model "
            "for Step 4 semantic prefix synthesis."
        ),
    )
    p.add_argument(
        "--majority-vote-reasoning-api-base",
        default="",
        help=(
            "Optional API base for the Step-4 reasoning model. "
            "If empty, reuse --instruct-api-base."
        ),
    )
    p.add_argument(
        "--majority-vote-reasoning-model-name",
        default="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        help="Model name used for majority_vote reasoning mode.",
    )
    p.add_argument(
        "--majority-vote-reasoning-tokenizer-path",
        default="/data/user_data/haolingp/models/Qwen3-30B-A3B-Thinking-2507-FP8",
        help="Local tokenizer path used to build chat prompts for the majority_vote reasoning model.",
    )
    p.add_argument(
        "--majority-vote-reasoning-temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for majority_vote reasoning mode.",
    )
    p.add_argument(
        "--majority-vote-reasoning-max-tokens",
        type=int,
        default=512,
        help="Max generation tokens for majority_vote reasoning mode.",
    )

    return p.parse_args()


# ===================================================================
# Environment
# ===================================================================

def setup_env() -> None:
    os.environ["HF_HOME"] = "/data/user_data/haolingp/hf_cache"
    os.environ["HF_HUB_CACHE"] = "/data/user_data/haolingp/hf_cache/hub"
    os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/haolingp/hf_cache/transformers"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===================================================================
# Text Utilities
# ===================================================================

def parse_list_column(raw: Any) -> List[str]:
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return [raw] if raw else []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)] if str(parsed).strip() else []


# Zero-width and other invisible chars to strip so alignment and slicing use the same string space.
_ZERO_WIDTH_AND_INVISIBLE = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u2060\u2061\u2062\u2063\u2064\u180e\u034f]+"
)


def normalize_zh(text: str) -> str:
    text = unicodedata.normalize("NFC", text.strip())
    text = re.sub(r"\s+", "", text)
    return text


def clean_translation_for_alignment(raw: str) -> str:
    """Take first line, strip zero-width/invisible chars, then normalize. Keeps alignment and slicing in same space."""
    if not raw:
        return ""
    first_line = raw.strip().split("\n")[0].strip()
    first_line = _ZERO_WIDTH_AND_INVISIBLE.sub("", first_line)
    return normalize_zh(first_line)


def clean_llm_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Handle truncated outputs like "<think>..." without a closing tag.
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("</think>", "")
    text = text.strip().strip('"').strip("'")
    text = text.strip("\u201c\u201d\u2018\u2019")
    text = text.strip('"').strip("'")
    return text


def truncate_translation_repetition(text: str, min_period: int = 8, max_period: int = 60) -> str:
    """若译文末尾出现同一短语多次重复（模型陷入重复），截断到只保留第一段，避免整段重复污染 committed。"""
    if not text or len(text) < 2 * min_period:
        return text
    out = text
    for p in range(min_period, min(max_period, len(out) // 2) + 1):
        while len(out) > 2 * p and out[-p:] == out[-2 * p : -p]:
            out = out[:-p]
    return out


def clean_continuation(observed: str, raw_output: str, max_words: int = 15) -> str:
    text = raw_output.strip()
    if "\n" in text:
        text = text.split("\n")[0].strip()
    obs_lower = observed.lower().strip()
    text_lower = text.lower()
    if text_lower.startswith(obs_lower):
        text = text[len(obs_lower):].strip()
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text.strip()


_FIRST_SENTENCE_END_RE = re.compile(r'[.!?](?:["\')\]]+)?(?=\s|$)')


def truncate_future_to_first_sentence(text: str) -> str:
    """Keep at most the first generated English sentence.

    This prevents future sampling from spilling into the next sentence and
    contaminating downstream translation/alignment.
    """
    text = str(text or "").strip()
    if not text:
        return ""
    m = _FIRST_SENTENCE_END_RE.search(text)
    if not m:
        return text
    return text[: m.end()].strip()


def _common_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _estimate_sentence_local_target_offset(
    translation: str,
    prefix_before: str,
    committed_norm: str,
    min_anchor_len: int = 6,
    extra_search_chars: int = 64,
) -> int:
    """Best-effort estimate of where the current sentence starts in candidate target text.

    We first use exact prefix overlap, then fall back to suffix-anchor matching
    against the expected previous-sentence prefix. This is more robust than
    blindly trusting precomputed sentence lengths when the candidate rephrases
    earlier clauses.
    """
    translation = str(translation or "")
    prefix_before = str(prefix_before or "")
    committed_norm = str(committed_norm or "")
    if not translation or not prefix_before:
        return 0

    candidates = [_common_prefix_len(translation, prefix_before)]
    if committed_norm:
        candidates.append(min(len(translation), len(prefix_before), len(committed_norm)))

    search_limit = min(len(translation), len(prefix_before) + max(0, extra_search_chars))
    max_overlap = min(len(prefix_before), search_limit)
    for overlap in range(max_overlap, max(0, min_anchor_len) - 1, -1):
        suffix = prefix_before[-overlap:]
        pos = translation.find(suffix, 0, search_limit)
        if pos != -1:
            candidates.append(pos + overlap)
            break

    return max(0, min(len(translation), max(candidates)))


def _build_sentence_local_source_view(
    full_src_for_candidate: str,
    observed_source: str,
    sentence_start_word: int,
) -> Tuple[str, str]:
    """Build candidate-specific local source strings starting at the current sentence.

    The alignment source must match the actual candidate being translated, not
    the gold `input_sentences[i]` string. We still use sentence boundaries from
    `input_sentences` to drop earlier sentences.
    """
    full_words = full_src_for_candidate.strip().split()
    if not full_words:
        return "", ""
    observed_count = min(len(observed_source.strip().split()), len(full_words))
    start = max(0, min(sentence_start_word, len(full_words)))
    local_full = " ".join(full_words[start:]).strip()
    local_observed = " ".join(full_words[start:observed_count]).strip()
    return local_full, local_observed


def load_instruct_tokenizer(tokenizer_path: str, cache_dir: Optional[str] = None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=cache_dir)


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe[:200] if safe else "unknown"


_WORD_HEAD_CHARS = set("一这那其某各该此每两几何")
def _ends_on_word_head(text: str) -> bool:
    """Return True if text ends on a character that typically starts a Chinese word."""
    return bool(text) and text[-1] in _WORD_HEAD_CHARS


def longest_common_prefix(strings: List[str]) -> str:
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        limit = min(len(prefix), len(s))
        i = 0
        while i < limit and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


def strip_committed_suffix_from_delta(committed: str, delta: str) -> str:
    """If delta starts with a suffix of committed (phrase repeat), strip it so we don't double-commit."""
    if not committed or not delta:
        return delta
    for k in range(min(len(committed), len(delta)), 0, -1):
        suffix = committed[-k:]
        if delta.startswith(suffix):
            return normalize_zh(delta[len(suffix):].strip())
    return delta


def _count_prefix_support(prefix: str, fragments: List[str], normalize_leading: bool = False) -> int:
    if not prefix:
        return 0
    if normalize_leading:
        prefix = _strip_majority_vote_leading_variants(prefix)
        items = [_strip_majority_vote_leading_variants(f) for f in fragments if f]
    else:
        prefix = normalize_zh(prefix)
        items = [normalize_zh(f) for f in fragments if f]
    if not prefix:
        return 0
    return sum(1 for f in items if f.startswith(prefix))


def _has_repeated_substring_pattern(text: str, min_unit: int = 3, max_unit: int = 8) -> bool:
    t = normalize_zh(text or "")
    if len(t) < min_unit * 2:
        return False
    for unit in range(min_unit, min(max_unit, len(t) // 2) + 1):
        for i in range(0, len(t) - (2 * unit) + 1):
            if t[i:i + unit] == t[i + unit:i + 2 * unit]:
                return True
    return False


_LEADING_PUNCT = re.compile(r"^[，。、；：！？\s]+")
_NOISE_PREFIXES = ("一们", "一为", "一名", "一位", "们", "为", "名", "位")


def _semantic_normalize(delta: str, key_len: int = 5) -> str:
    d = _LEADING_PUNCT.sub("", delta.strip())
    for p in _NOISE_PREFIXES:
        if d.startswith(p):
            d = d[len(p):]
            break
    return d[:key_len] if d else ""


def check_direction(
    deltas: List[str],
    n: int = 3,
    min_ratio: float = 0.5,
) -> Tuple[bool, Dict[str, Any]]:
    """Check whether candidate deltas point in a consistent direction."""
    keys = []
    for d in deltas:
        d = (d or "").strip()
        if not d:
            continue
        sem_key = _semantic_normalize(d, key_len=n)
        key = sem_key if sem_key else d[: min(n, len(d))]
        keys.append(key)

    if not keys:
        return False, {
            "prefix_n": n,
            "min_ratio": min_ratio,
            "keys": [],
            "top_key": "",
            "top_count": 0,
            "ratio": 0.0,
        }

    counts = Counter(keys)
    top_key, top_count = counts.most_common(1)[0]
    ratio = top_count / len(keys)
    return ratio >= min_ratio, {
        "prefix_n": n,
        "min_ratio": min_ratio,
        "keys": keys,
        "top_key": top_key,
        "top_count": top_count,
        "ratio": ratio,
        "counts": dict(counts),
    }


def _vlog(log_file: Optional[Any], msg: str) -> None:
    if log_file is not None:
        log_file.write(msg)
        if not msg.endswith("\n"):
            log_file.write("\n")
        log_file.flush()


def _emit_simalign_alignment_debug(
    log_file: Optional[Any],
    label: str,
    full_src: str,
    observed_src: str,
    translation: str,
    alignments: List[Tuple[int, int]],
    tgt_offset: Optional[int] = None,
) -> None:
    if log_file is None:
        return

    try:
        import jieba
    except Exception:
        jieba = None

    src_words = [w for w in full_src.strip().split() if w and not w.isspace()]
    obs_words = [w for w in observed_src.strip().split() if w and not w.isspace()]
    obs_n = len(obs_words)
    tgt_norm = clean_translation_for_alignment(translation)
    tgt_word_meta: List[Tuple[int, int, int, str]] = []
    char_to_word_idx: Dict[int, int] = {}

    _vlog(log_file, f"{label} full_src_words={[f'{i}:{w}' for i, w in enumerate(src_words)]}")
    _vlog(log_file, f"{label} observed_src_words={[f'{i}:{w}' for i, w in enumerate(obs_words)]} obs_n={obs_n}")
    if tgt_offset is not None:
        _vlog(log_file, f"{label} tgt_offset={tgt_offset}")
    _vlog(log_file, f"{label} target_chars={[f'{i}:{ch}' for i, ch in enumerate(tgt_norm)]}")

    if jieba is not None and tgt_norm:
        tgt_words = [w for w in jieba.cut(tgt_norm) if w and not w.isspace()]
        spans = []
        start = 0
        for wi, word in enumerate(tgt_words):
            end = start + len(word) - 1
            tgt_word_meta.append((wi, start, end, word))
            for ci in range(start, end + 1):
                char_to_word_idx[ci] = wi
            spans.append(f"{wi}:{start}-{end}:{word}")
            start = end + 1
        _vlog(log_file, f"{label} target_word_spans={spans}")

    if not alignments:
        _vlog(log_file, f"{label} pairs=[]")
        _vlog(log_file, f"{label} truncation_scan: no alignments -> cut_idx=-1 cut_prefix=''")
        return

    sorted_pairs = sorted(alignments, key=lambda x: (x[1], x[0]))
    safe_char_idx = -1
    first_future_pair: Optional[Tuple[int, int]] = None

    for s, t in sorted_pairs:
        src_tok = src_words[s] if 0 <= s < len(src_words) else "<OUT_OF_RANGE>"
        tgt_char = tgt_norm[t] if 0 <= t < len(tgt_norm) else "<OUT_OF_RANGE>"
        prefix = tgt_norm[: t + 1] if 0 <= t < len(tgt_norm) else ""
        status = "OBS" if s < obs_n else "FUTURE"
        tgt_word_idx = char_to_word_idx.get(t)
        if tgt_word_idx is not None and 0 <= tgt_word_idx < len(tgt_word_meta):
            wi, start, end, word = tgt_word_meta[tgt_word_idx]
            tgt_word_desc = f"{wi}:{start}-{end}:{word}"
        else:
            tgt_word_desc = "<NO_WORD_SPAN>"
        _vlog(
            log_file,
            f"{label} pair src_idx={s} src_tok={src_tok!r} "
            f"-> tgt_word={tgt_word_desc} -> tgt_idx={t} tgt_char={tgt_char!r} "
            f"status={status} prefix={prefix!r}",
        )
        if s < obs_n:
            safe_char_idx = max(safe_char_idx, t)
        elif first_future_pair is None:
            first_future_pair = (s, t)
            break

    cut_prefix = tgt_norm[: safe_char_idx + 1] if safe_char_idx >= 0 else ""
    _vlog(
        log_file,
        f"{label} truncation_scan safe_char_idx={safe_char_idx} "
        f"first_future_pair={first_future_pair} cut_prefix={cut_prefix!r}",
    )


_VERBOSE_ALIGNMENT_DEBUG = os.environ.get("VERBOSE_ALIGNMENT_DEBUG", "").strip().lower() in {
    "1", "true", "yes", "on"
}


def _extract_reference_text_from_row(row: Dict[str, str]) -> Optional[str]:
    """Best-effort extraction of reference translation text from a manifest row."""
    candidate_keys = [
        "tgt_text_full",
        "tgt_text",
        "target_text",
        "translation",
        "ref_text",
        "reference",
    ]
    for k in candidate_keys:
        if k not in row:
            continue
        raw = row.get(k)
        if raw is None:
            continue
        vals = parse_list_column(raw)
        if vals:
            text = "".join(str(v).strip() for v in vals if str(v).strip())
        else:
            text = str(raw).strip()
        if text:
            return text
    return None


def compute_laal(
    source_chunks: List[str],
    target_deltas: List[str],
    actions: List[str],
    reference: str,
) -> float:
    """Text LAAL using source-word count as source-time surrogate."""
    timeline: List[int] = []
    source_read = 0

    for chunk, delta, action in zip(source_chunks, target_deltas, actions):
        words_in_chunk = len(str(chunk).strip().split()) if str(chunk).strip() else 0
        source_read += words_in_chunk
        if action == "WRITE" and str(delta).strip():
            for _ in str(delta).strip():
                timeline.append(source_read)

    y = "".join(d for d in target_deltas if d)
    y_len = len(y)
    yref_len = len(str(reference).replace(" ", ""))
    x_len = sum(
        len(str(c).strip().split())
        for c in source_chunks
        if str(c).strip()
    )

    if y_len == 0 or x_len == 0 or yref_len == 0:
        return float("nan")

    denom = max(y_len, yref_len)
    if denom <= 0 or len(timeline) == 0:
        return float("nan")

    total_lagging = 0.0
    for i in range(1, denom + 1):
        d_i = timeline[i - 1] if i <= len(timeline) else x_len
        d_star_i = (i - 1) * x_len / denom
        total_lagging += (d_i - d_star_i)

    return total_lagging / denom


def _char_tokens_zh(text: str) -> List[str]:
    """Simple char-level tokenization for Chinese BLEU (remove whitespace)."""
    return [c for c in str(text) if not c.isspace()]


def compute_bleu_char(
    hypothesis: str,
    reference: str,
    max_order: int = 4,
    smooth: bool = True,
) -> float:
    """Sentence-level BLEU on character tokens (Chinese-friendly, no external deps).

    Returns BLEU in [0, 100].
    """
    hyp = _char_tokens_zh(hypothesis)
    ref = _char_tokens_zh(reference)
    hyp_len = len(hyp)
    ref_len = len(ref)

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
        clipped = 0
        for ng, cnt in hyp_ngrams.items():
            clipped += min(cnt, ref_ngrams.get(ng, 0))
        if smooth:
            p_n = (clipped + 1.0) / (total + 1.0)
        else:
            if clipped == 0:
                return 0.0
            p_n = clipped / total
        precisions.append(p_n)

    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_len / hyp_len))

    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / eff_order)
    return bleu * 100.0


class _TeeWriter:
    def __init__(self, file_obj):
        self._f = file_obj

    def write(self, msg):
        self._f.write(msg)
        sys.stdout.write(msg)

    def flush(self):
        self._f.flush()
        sys.stdout.flush()

    def close(self):
        self._f.close()


# ===================================================================
# Word Alignment  (awesome-align + monotonic post-processing)
# ===================================================================

def load_align_model(cache_dir: Optional[str] = None, device: Optional[str] = None):
    """Load awesome-align BERT model.

    Defaults to CUDA if available so BERT inference runs on GPU alongside the
    base model.  Must be called BEFORE loading vLLM so the ~400 MB footprint is
    already accounted for when vLLM profiles free GPU memory.
    """
    from transformers import AutoModel, AutoTokenizer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "aneuraz/awesome-align-with-co"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    model = model.to(device).eval()
    print(f"[Align] Model loaded on {device}.")
    return model, tokenizer


# ===========================================================
# NEW: monotonic post-processing (ported from simalign path)
# ===========================================================
def _make_monotonic(
    alignments: List[Tuple[int, int]],
    n_src: int,
    n_tgt: int,
) -> List[Tuple[int, int]]:
    """Convert crossing alignments to monotonic (src non-decreasing along tgt axis).

    Follows GigaSpeech build_trajectory_full_mfa.py logic:
      1. Sort by (tgt_idx, src_idx) ascending.
      2. Append anchor (n_src-1, n_tgt-1) so the last src/tgt positions are always covered.
      3. Merge entries sharing the same tgt_idx: keep the one with the larger src_idx
         (last after sort = largest src for that tgt).
      4. Enforce src non-decreasing: alignments_r[i].src = max(alignments_r[i].src,
         alignments_r[i-1].src).

    Applied to awesome-align char-level output:
      - n_src = number of source words
      - n_tgt = number of target characters (len of tgt_chars list)
      - Eliminates crossing alignments that cause truncate_by_alignment to
        produce incorrect safe prefixes when the last observed word maps to
        a tgt position earlier than a previous word's tgt position.
    """
    if not alignments:
        return []
    # Step 1: sort by (tgt, src)
    result: List[Tuple[int, int]] = sorted(alignments, key=lambda x: (x[1], x[0]))
    # Step 2: append anchor so last src/tgt are always represented
    result.append((n_src - 1, n_tgt - 1))
    # Step 3: merge same tgt – keep last entry (largest src after sort)
    merged: List[Tuple[int, int]] = []
    for pair in result:
        if merged and merged[-1][1] == pair[1]:
            merged[-1] = pair  # replace with later (larger src) entry
        else:
            merged.append(pair)
    # Step 4: enforce src non-decreasing
    for i in range(1, len(merged)):
        s, t = merged[i]
        s_prev = merged[i - 1][0]
        if s < s_prev:
            merged[i] = (s_prev, t)
    return merged
# ===========================================================
# END NEW
# ===========================================================


def get_word_alignments(
    src_text: str,
    tgt_text: str,
    align_model,
    align_tokenizer,
) -> List[Tuple[int, int]]:
    """Extract word-level alignment pairs.

    src_text is split on whitespace (English words).
    tgt_text is split per character (Chinese characters).
    Returns list of (src_word_idx, tgt_char_idx) pairs.

    Pipeline:
      1. bidirectional argmax intersection (awesome-align style)
      2. _make_monotonic() post-processing to eliminate crossing alignments
         -> src is non-decreasing along the tgt axis, matching GigaSpeech convention.

    Fallback (when intersection is empty): forward-only argmax, then monotonic.
    """
    src_words = src_text.strip().split()
    tgt_chars = list(tgt_text.strip().replace(" ", ""))

    if not src_words or not tgt_chars:
        return []

    src_subwords = [align_tokenizer.tokenize(w) for w in src_words]
    tgt_subwords = [align_tokenizer.tokenize(c) for c in tgt_chars]
    src_subwords = [sw if sw else [align_tokenizer.unk_token] for sw in src_subwords]
    tgt_subwords = [sw if sw else [align_tokenizer.unk_token] for sw in tgt_subwords]

    max_pos = int(getattr(getattr(align_model, "config", None), "max_position_embeddings", 512) or 512)
    if max_pos <= 3:
        max_pos = 512
    per_pass_budget = max_pos - 3
    max_joint_subwords = 1024

    def _align_single(
        src_sw: List[List[str]],
        tgt_sw: List[List[str]],
        src_offset: int = 0,
        tgt_offset: int = 0,
    ) -> List[Tuple[int, int]]:
        """Run one BERT forward pass and return monotonic (src_idx, tgt_idx) pairs.

        src_offset / tgt_offset are added to convert local indices back to global
        when called from the two-pass split path.
        """
        src_ids = [align_tokenizer.convert_tokens_to_ids(sw) for sw in src_sw]
        tgt_ids = [align_tokenizer.convert_tokens_to_ids(sw) for sw in tgt_sw]
        src_flat = list(itertools.chain.from_iterable(src_ids))
        tgt_flat = list(itertools.chain.from_iterable(tgt_ids))
        if len(src_flat) + len(tgt_flat) > per_pass_budget:
            return []

        input_ids = (
            [align_tokenizer.cls_token_id]
            + src_flat
            + [align_tokenizer.sep_token_id]
            + tgt_flat
            + [align_tokenizer.sep_token_id]
        )
        token_type_ids = (
            [0] * (1 + len(src_flat) + 1)
            + [1] * (len(tgt_flat) + 1)
        )

        _dev = next(align_model.parameters()).device
        ids_t = torch.tensor([input_ids]).to(_dev)
        tt_t = torch.tensor([token_type_ids]).to(_dev)
        am_t = torch.ones_like(ids_t)

        with torch.no_grad():
            outputs = align_model(
                input_ids=ids_t,
                token_type_ids=tt_t,
                attention_mask=am_t,
                output_hidden_states=True,
            )

        hidden = outputs.hidden_states[8][0]  # layer 8, awesome-align default

        src_embeds: List[torch.Tensor] = []
        offset = 1  # skip [CLS]
        for sw in src_sw:
            n = len(sw)
            src_embeds.append(hidden[offset:offset + n].mean(dim=0))
            offset += n

        tgt_embeds: List[torch.Tensor] = []
        offset = 1 + len(src_flat) + 1  # skip [CLS] src [SEP]
        for sw in tgt_sw:
            n = len(sw)
            tgt_embeds.append(hidden[offset:offset + n].mean(dim=0))
            offset += n

        src_mat = torch.nn.functional.normalize(torch.stack(src_embeds), dim=-1)
        tgt_mat = torch.nn.functional.normalize(torch.stack(tgt_embeds), dim=-1)
        sim = src_mat @ tgt_mat.t()

        fwd = sim.argmax(dim=1)
        bwd = sim.argmax(dim=0)

        inter = [
            (s + src_offset, int(fwd[s].item()) + tgt_offset)
            for s in range(len(src_sw))
            if int(bwd[fwd[s].item()].item()) == s
        ]
        raw = inter if inter else [
            (s + src_offset, int(fwd[s].item()) + tgt_offset)
            for s in range(len(src_sw))
        ]
        # Apply monotonic post-processing at the local level.
        # Global monotonic pass is done again after merging halves (two-pass path).
        return _make_monotonic(raw, len(src_sw) + src_offset, len(tgt_sw) + tgt_offset)

    src_token_lens = [len(sw) for sw in src_subwords]
    tgt_token_lens = [len(sw) for sw in tgt_subwords]
    src_total = sum(src_token_lens)
    tgt_total = sum(tgt_token_lens)
    joint_total = src_total + tgt_total

    def _run_alignment() -> List[Tuple[int, int]]:
        # Fast path: fits in one pass.
        if joint_total <= per_pass_budget:
            return _align_single(src_subwords, tgt_subwords)

        # Hard guard: do not attempt alignment beyond 1024 subwords.
        if joint_total > max_joint_subwords:
            return []

        # 513..1024: two-pass alignment with global index offsets.
        src_n = len(src_subwords)
        tgt_n = len(tgt_subwords)
        if src_n < 2 or tgt_n < 2:
            return []

        src_ps = [0]
        for v in src_token_lens:
            src_ps.append(src_ps[-1] + v)
        tgt_ps = [0]
        for v in tgt_token_lens:
            tgt_ps.append(tgt_ps[-1] + v)

        src_mid = src_total / 2.0
        tgt_mid = tgt_total / 2.0
        split_pair: Optional[Tuple[int, int]] = None
        best_score = float("inf")

        for s_split in range(1, src_n):
            left_src = src_ps[s_split]
            right_src = src_total - left_src
            if left_src > per_pass_budget or right_src > per_pass_budget:
                continue

            lower = tgt_total - (per_pass_budget - right_src)
            upper = per_pass_budget - left_src
            if lower > upper:
                continue

            lo_idx = bisect.bisect_left(tgt_ps, lower, 1, tgt_n)
            hi_idx = bisect.bisect_right(tgt_ps, upper, 1, tgt_n) - 1
            if lo_idx > hi_idx:
                continue

            target = tgt_mid
            cand = bisect.bisect_left(tgt_ps, target, lo_idx, hi_idx + 1)
            for t_split in (cand - 1, cand):
                if t_split < lo_idx or t_split > hi_idx:
                    continue
                if t_split <= 0 or t_split >= tgt_n:
                    continue
                score = abs(left_src - src_mid) + abs(tgt_ps[t_split] - tgt_mid)
                if score < best_score:
                    best_score = score
                    split_pair = (s_split, t_split)

        if split_pair is None:
            return []

        s_split, t_split = split_pair
        # Each half gets its own monotonic pass inside _align_single (via src/tgt offsets).
        left = _align_single(src_subwords[:s_split], tgt_subwords[:t_split],
                             src_offset=0, tgt_offset=0)
        right = _align_single(src_subwords[s_split:], tgt_subwords[t_split:],
                              src_offset=s_split, tgt_offset=t_split)

        merged = left + right
        # Final global monotonic pass over the combined halves to eliminate any
        # boundary crossings introduced by merging the two independent passes.
        return _make_monotonic(merged, src_n, tgt_n)

    if _align_model_lock is not None:
        with _align_model_lock:
            return _run_alignment()
    return _run_alignment()


# Max target chars per observed source word when aligning few words to many chars.
_MAX_TGT_CHARS_PER_OBS_WORD = 5
# last_t too dispersed (max-min > this) → do not trust, use fallback.
_ALIGNMENT_SPREAD_THRESHOLD = 12
# Truncation at very end (t_idx > len(translation)-this) → likely sentence-end attraction, use fallback.
_ALIGNMENT_VERY_END_MARGIN = 3
# Last word aligned to very early position (t_idx < this) → likely wrong, use fallback.
_ALIGNMENT_TOO_EARLY_THRESHOLD = 2

# Sentence-scoped alignment gates
_SENTENCE_COVERAGE_GATE = 0.75
_BAD_LAST_WORDS = frozenset({"am", "is", "are", "was", "were", "be", "been", "being"})


def truncate_by_alignment(full_src: str, observed_src: str, translation: str, alignments):
    obs_n = len(observed_src.strip().split())
    if obs_n <= 0 or not translation:
        return ""

    last = obs_n - 1
    last_t = [t for s, t in alignments if s == last]

    def _cap_by_obs_words(raw_len: int) -> int:
        if obs_n <= 0:
            return raw_len
        cap = obs_n * _MAX_TGT_CHARS_PER_OBS_WORD
        return min(raw_len, max(2, cap), len(translation))

    def _fallback_safe_tgt_idx():
        safe_tgt_idx = -1
        for s, t in alignments:
            if s < obs_n:
                safe_tgt_idx = max(safe_tgt_idx, t)
        if safe_tgt_idx >= 0:
            raw_len = safe_tgt_idx + 1
            capped = _cap_by_obs_words(raw_len)
            return translation[:capped]
        return ""

    def _fallback_ratio():
        src_total = len(full_src.strip().split())
        if src_total == 0:
            return ""
        ratio = obs_n / src_total
        safe_chars = int(len(translation) * ratio * 0.8)
        return translation[:safe_chars] if safe_chars >= 2 else ""

    if last_t:
        spread = max(last_t) - min(last_t)
        if spread > _ALIGNMENT_SPREAD_THRESHOLD:
            out = _fallback_safe_tgt_idx() or _fallback_ratio()
            return out
        t_idx = sorted(last_t)[len(last_t) // 2]
        if t_idx > len(translation) - _ALIGNMENT_VERY_END_MARGIN:
            out = _fallback_safe_tgt_idx() or _fallback_ratio()
            return out
        if t_idx < _ALIGNMENT_TOO_EARLY_THRESHOLD:
            out = _fallback_safe_tgt_idx() or _fallback_ratio()
            return out
        raw_len = t_idx + 1
        capped = _cap_by_obs_words(raw_len)
        return translation[:capped]

    out = _fallback_safe_tgt_idx() or _fallback_ratio()
    return out


def build_local_alignment_windows(
    full_src: str,
    observed_src: str,
    translation: str,
    committed: str,
    src_left_words: int = 40,
    src_right_words: int = 24,
    tgt_left_chars: int = 120,
) -> Tuple[str, str, str, int]:
    """Build shorter alignment inputs around the observed/future boundary."""
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
    align_model,
    align_tokenizer,
) -> List[List[Tuple[int, int]]]:
    """Batch BERT forward for M (src, tgt) pairs -> one GPU kernel instead of M.

    Pairs whose combined token length fits in the single 512-position budget are
    packed into a padded batch and processed in one forward pass.  Overlong pairs
    fall back to the sequential get_word_alignments() path.

    Monotonic post-processing (_make_monotonic) is applied to every result.
    """
    if not pairs:
        return []

    max_pos = int(
        getattr(getattr(align_model, "config", None), "max_position_embeddings", 512) or 512
    )
    if max_pos <= 3:
        max_pos = 512
    per_pass_budget = max_pos - 3
    _dev = next(align_model.parameters()).device

    # ---- Phase 1: tokenize all pairs ----
    tok_data = []
    for src_text, tgt_text in pairs:
        src_words = src_text.strip().split()
        tgt_chars = list(tgt_text.strip().replace(" ", ""))
        if not src_words or not tgt_chars:
            tok_data.append(None)
            continue
        src_sw = [align_tokenizer.tokenize(w) or [align_tokenizer.unk_token] for w in src_words]
        tgt_sw = [align_tokenizer.tokenize(c) or [align_tokenizer.unk_token] for c in tgt_chars]
        src_ids = [align_tokenizer.convert_tokens_to_ids(sw) for sw in src_sw]
        tgt_ids = [align_tokenizer.convert_tokens_to_ids(sw) for sw in tgt_sw]
        src_flat = list(itertools.chain.from_iterable(src_ids))
        tgt_flat = list(itertools.chain.from_iterable(tgt_ids))
        tok_data.append((src_sw, tgt_sw, src_flat, tgt_flat))

    results: List[Optional[List[Tuple[int, int]]]] = [None] * len(pairs)

    # ---- Phase 2: pack fits-in-budget pairs into one padded batch ----
    batch_orig_idx: List[int] = []
    batch_input_ids: List[List[int]] = []
    batch_tt_ids: List[List[int]] = []
    batch_meta: List[Tuple] = []  # (src_sw, tgt_sw, src_flat_len)

    for i, td in enumerate(tok_data):
        if td is None:
            results[i] = []
            continue
        src_sw, tgt_sw, src_flat, tgt_flat = td
        if len(src_flat) + len(tgt_flat) > per_pass_budget:
            continue  # handled in fallback phase
        ids = (
            [align_tokenizer.cls_token_id]
            + src_flat
            + [align_tokenizer.sep_token_id]
            + tgt_flat
            + [align_tokenizer.sep_token_id]
        )
        tt = [0] * (1 + len(src_flat) + 1) + [1] * (len(tgt_flat) + 1)
        batch_orig_idx.append(i)
        batch_input_ids.append(ids)
        batch_tt_ids.append(tt)
        batch_meta.append((src_sw, tgt_sw, len(src_flat)))

    if batch_orig_idx:
        max_len = max(len(ids) for ids in batch_input_ids)
        pad_id = align_tokenizer.pad_token_id or 0
        padded_ids, padded_tt, attn_masks = [], [], []
        for ids, tt in zip(batch_input_ids, batch_tt_ids):
            pad = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad)
            padded_tt.append(tt + [0] * pad)
            attn_masks.append([1] * len(ids) + [0] * pad)

        ids_t = torch.tensor(padded_ids, dtype=torch.long, device=_dev)
        tt_t  = torch.tensor(padded_tt,  dtype=torch.long, device=_dev)
        am_t  = torch.tensor(attn_masks, dtype=torch.long, device=_dev)

        if _align_model_lock is not None:
            with _align_model_lock:
                with torch.no_grad():
                    out = align_model(
                        input_ids=ids_t,
                        token_type_ids=tt_t,
                        attention_mask=am_t,
                        output_hidden_states=True,
                    )
                hidden_batch = out.hidden_states[8]  # [B, max_len, hidden_dim]
        else:
            with torch.no_grad():
                out = align_model(
                    input_ids=ids_t,
                    token_type_ids=tt_t,
                    attention_mask=am_t,
                    output_hidden_states=True,
                )
            hidden_batch = out.hidden_states[8]  # [B, max_len, hidden_dim]

        for b, (orig_i, (src_sw, tgt_sw, src_len)) in enumerate(
            zip(batch_orig_idx, batch_meta)
        ):
            hidden = hidden_batch[b]

            src_embeds: List[torch.Tensor] = []
            offset = 1  # skip [CLS]
            for sw in src_sw:
                src_embeds.append(hidden[offset:offset + len(sw)].mean(dim=0))
                offset += len(sw)

            tgt_embeds: List[torch.Tensor] = []
            offset = 1 + src_len + 1  # skip [CLS] src [SEP]
            for sw in tgt_sw:
                tgt_embeds.append(hidden[offset:offset + len(sw)].mean(dim=0))
                offset += len(sw)

            src_mat = torch.nn.functional.normalize(torch.stack(src_embeds), dim=-1)
            tgt_mat = torch.nn.functional.normalize(torch.stack(tgt_embeds), dim=-1)
            sim = src_mat @ tgt_mat.t()

            fwd = sim.argmax(dim=1)
            bwd = sim.argmax(dim=0)
            inter = [
                (s, int(fwd[s].item()))
                for s in range(len(src_sw))
                if int(bwd[fwd[s].item()].item()) == s
            ]
            raw = inter if inter else [
                (s, int(fwd[s].item())) for s in range(len(src_sw))
            ]
            # Apply monotonic post-processing to each batch result.
            results[orig_i] = _make_monotonic(raw, len(src_sw), len(tgt_sw))

    # ---- Phase 3: fallback for overlong pairs not included in the batch ----
    for i, (src_text, tgt_text) in enumerate(pairs):
        if results[i] is None:
            results[i] = get_word_alignments(src_text, tgt_text, align_model, align_tokenizer)

    return [r for r in results]  # type: ignore[return-value]


# ===================================================================
# Prompt Templates
# ===================================================================

def build_translate_prompt(full_source: str, prev_context: str = "") -> str:
    ctx = f"Previous context: {prev_context}\n\n" if prev_context else ""
    return (
        f"{ctx}"
        f"Translate the following English to Chinese. "
        f"Output ONLY the Chinese translation, no explanation.\n\n"
        f'English: "{full_source}"\n'
        f"Chinese:"
    )


def build_complete_prompt(full_source: str, committed: str) -> str:
    return (
        f"Complete the Chinese translation of this English text.\n\n"
        f'English: "{full_source}"\n'
        f'Chinese so far: "{committed}"\n\n'
        f"Continue the translation from where it left off. "
        f"Output ONLY the remaining Chinese text:"
    )


def build_select_prompt(
    observed_source: str,
    committed: str,
    candidate_translations: List[str],
) -> str:
    candidates_str = "\n".join(
        f"  {i+1}. \"{t}\"" for i, t in enumerate(candidate_translations)
    )
    return (
        f"You are evaluating simultaneous translation candidates.\n\n"
        f"Observed English so far: \"{observed_source}\"\n"
        f"Chinese committed so far: \"{committed}\"\n\n"
        f"Candidate translations (each from a different predicted future):\n"
        f"{candidates_str}\n\n"
        f"Select the BEST candidate that:\n"
        f"1. Most accurately translates the observed English\n"
        f"2. Is most natural and fluent in Chinese\n"
        f"3. Best continues from the committed Chinese\n\n"
        f"Output ONLY the number (1-{len(candidate_translations)}) "
        f"of the best candidate."
    )


def build_score_prompt(
    observed_source: str,
    committed: str,
    candidate_items: List[Dict[str, Any]],
) -> str:
    candidates_str = "\n".join(
        (
            f'Candidate {item["candidate_id"]}:\n'
            f'  full_prefix_so_far: "{item["safe_prefix"]}"\n'
            f'  new_delta_only: "{item["delta"]}"'
        )
        for item in candidate_items
    )
    return (
        "You are evaluating incremental simultaneous-translation candidates.\n\n"
        f'Observed English so far: "{observed_source}"\n'
        f'Chinese already committed: "{committed}"\n\n'
        "Each candidate provides:\n"
        "- full_prefix_so_far: full aligned/truncated Chinese prefix after this step\n"
        "- new_delta_only: the incremental delta beyond committed\n\n"
        "Score the quality of the NEW DELTA as the immediate next continuation,\n"
        "but use full_prefix_so_far to avoid over-penalizing truncation artifacts.\n"
        "Do NOT re-evaluate already committed content by itself.\n"
        "Judge whether the candidate is a correct next incremental continuation for the observed English,\n"
        "given the committed Chinese context.\n\n"
        "## Definition of Over-translation (IMPORTANT)\n"
        "- Over-translation means translating content that is NOT YET present in the observed English.\n"
        "- If the delta translates words that ARE already present in observed English, that is CORRECT and should NOT be penalized.\n"
        "- If delta starts mid-phrase because of truncation, evaluate it using full_prefix_so_far and committed Chinese as context.\n\n"
        "## Hanging / incomplete English sources\n"
        "- The observed English is often an INCOMPLETE sentence that ends mid-clause (e.g. ends with an adverb or conjunction).\n"
        "- A delta that faithfully translates ALL observed words and then STOPS is CORRECT even if the Chinese also hangs mid-air.\n"
        "- Do NOT penalize a delta just because it ends without a continuation particle.\n"
        "- Adding a continuation particle (地会, 地是, 地都是, 地将, 的是, etc.) that goes BEYOND the observed English words\n"
        "  is MILD over-translation: deduct 15-25 points.\n\n"
        "Scoring criteria:\n"
        "- 90-100: Perfectly faithful, natural, no over-translation\n"
        "- 70-89: Mostly correct, minor issues or mild over-translation (e.g. a continuation particle)\n"
        "- 50-69: Partially correct or moderately over-translated\n"
        "- 0-49: Wrong translation, hallucination, or significant over-translation\n\n"
        "Deduction rules:\n"
        "- NEVER deduct: delta faithfully covers ALL words in observed English (even if Chinese sentence hangs)\n"
        "- DO NOT deduct: new_delta_only starts mid-word/phrase due to truncation (use full_prefix_so_far as context)\n"
        "- Deduct 15-25: delta adds a Chinese continuation particle (地会/地是/地将/地都是/的是 etc.) beyond observed English\n"
        "- Deduct heavily: delta includes semantic content (nouns, verbs, clauses) not in observed English\n"
        "- Deduct for mistranslation, hallucination, semantic drift, awkward continuation\n"
        "- Deduct if the new delta contradicts or degrades the committed context\n"
        "- Do NOT reward a candidate just because committed content (already fixed) is good\n\n"
        "Few-shot examples:\n"
        'Example 1 (good - full faithful translation):\n'
        '  Observed: "monotonous and unavailing"\n'
        '  Committed: ""\n'
        '  Delta: "显得单调乏味且毫无成效"\n'
        '  Score: 92\n\n'
        'Example 2 (partial/incomplete - under-translation):\n'
        '  Observed: "monotonous and unavailing"\n'
        '  Committed: ""\n'
        '  Delta: "显得单调"\n'
        '  Score: 68\n\n'
        'Example 3 (good - faithful hanging translation, sentence left open):\n'
        '  Observed: "And these introductions are inevitably"\n'
        '  Committed: ""\n'
        '  FullPrefix: "而这些介绍不可避免"\n'
        '  Delta: "而这些介绍不可避免"\n'
        '  Score: 88\n'
        '  Reason: faithfully translates ALL observed words; "不可避免"=inevitably; '
        'sentence hangs like the English, which is correct for simultaneous translation.\n\n'
        'Example 4 (mild over-translation - adds continuation particle beyond observed source):\n'
        '  Observed: "And these introductions are inevitably"\n'
        '  Committed: ""\n'
        '  FullPrefix: "而这些介绍不可避免地会"\n'
        '  Delta: "而这些介绍不可避免地会"\n'
        '  Score: 73\n'
        '  Reason: "地会" (will) commits to future verb structure not yet observed; mild over-translation.\n\n'
        'Example 5 (bad over-translation - fabricated future content):\n'
        '  Observed: "And these introductions are"\n'
        '  Committed: "而且这"\n'
        '  FullPrefix: "而且这些介绍出于文学上的诚实感"\n'
        '  Delta: "些介绍出于文学上的诚实感"\n'
        '  Score: 40\n'
        '  Reason: "出于文学上的诚实感" is fabricated semantic content completely absent from observed English.\n\n'
        f"{candidates_str}\n\n"
        "Output STRICT JSON only in this format:\n"
        '{"scores":[{"candidate_id":1,"score":87,"tags":["ok"]},{"candidate_id":2,"score":41,"tags":["mistranslation","drift"]}]}\n'
        "Do not include any extra text."
    )


def build_score_prompt_short(
    observed_source: str,
    committed: str,
    candidate_items: List[Dict[str, Any]],
) -> str:
    """Shortened judge prompt (~half length) for faster inference. Use with max_tokens=64."""
    candidates_str = "\n".join(
        (
            f'Candidate {item["candidate_id"]}:\n'
            f'  full_prefix_so_far: "{item["safe_prefix"]}"\n'
            f'  new_delta_only: "{item["delta"]}"'
        )
        for item in candidate_items
    )
    return (
        "Score incremental simultaneous-translation candidates (new_delta only).\n\n"
        f'Observed English: "{observed_source}"\n'
        f'Committed Chinese: "{committed}"\n\n'
        "Over-translation = translating content NOT in observed English. "
        "Faithful delta for observed words = correct. "
        "90-100: faithful, natural; 70-89: minor issues; 50-69: partial; 0-49: wrong/over-translation.\n\n"
        "Example good: Observed \"monotonous and unavailing\", Delta \"显得单调乏味且毫无成效\" -> 92\n"
        "Example bad:  Observed \"And these introductions are\", Delta \"些介绍出于文学上的诚实感\" -> 40 (fabricated)\n\n"
        f"{candidates_str}\n\n"
        "Output JSON only:\n"
        '{"scores":[{"candidate_id":1,"score":87},{"candidate_id":2,"score":41}]}'
    )


# ===================================================================
# Base Model -- Source Continuation (llm.generate, no chat template)
# ===================================================================

def _run_batch_future_sampling_worker(
    base_llm: LLM,
    num_candidates: int,
    future_tokens: int,
    temperature: float,
    batch_size: int,
    batch_wait_sec: float,
    request_queue: queue_module.Queue,
) -> None:
    """Worker: collect (observed_source, result_queue) from request_queue, run batched generate, put results."""
    params = SamplingParams(
        temperature=temperature,
        max_tokens=future_tokens,
        n=num_candidates,
        top_p=0.95,
        top_k=50,
        presence_penalty=0.6,
        stop=["\n"],
    )
    while True:
        batch: List[Tuple[str, queue_module.Queue]] = []
        try:
            item = request_queue.get(timeout=batch_wait_sec)
            if item[0] is None:
                return
            batch.append(item)
            while len(batch) < batch_size:
                try:
                    item = request_queue.get_nowait()
                    if item[0] is None:
                        request_queue.put(item)
                        break
                    batch.append(item)
                except queue_module.Empty:
                    break
        except queue_module.Empty:
            continue
        if not batch:
            continue
        observed_sources = [b[0] for b in batch]
        try:
            outputs = base_llm.generate(observed_sources, params)
        except Exception as e:
            for _, result_q in batch:
                result_q.put([])
            continue
        for i, (observed_source, result_q) in enumerate(batch):
            futures: List[str] = []
            for out in outputs[i].outputs:
                cleaned = clean_continuation(observed_source, out.text)
                if cleaned:
                    futures.append(cleaned)
            result_q.put(futures)


def sample_source_futures(
    base_llm: LLM,
    observed_source: str,
    num_candidates: int,
    future_tokens: int,
    temperature: float,
) -> List[str]:
    """Pure text continuation using base model."""
    global _future_sampling_request_queue
    if _future_sampling_request_queue is not None:
        result_q: queue_module.Queue = queue_module.Queue(1)
        _future_sampling_request_queue.put((observed_source, result_q))
        return result_q.get()
    params = SamplingParams(
        temperature=temperature,
        max_tokens=future_tokens,
        n=num_candidates,
        top_p=0.95,
        top_k=50,
        presence_penalty=0.6,
        stop=["\n"],
    )
    with (_base_llm_lock if _base_llm_lock is not None else contextlib.nullcontext()):
        outputs = base_llm.generate([observed_source], params)

    futures: List[str] = []
    for out in outputs[0].outputs:
        cleaned = clean_continuation(observed_source, out.text)
        if cleaned:
            futures.append(cleaned)
    return futures


# ===================================================================
# Instruct Model -- Translation + Selection (OpenAI API)
# ===================================================================

def _make_sync_client(api_base: str) -> OpenAI:
    return OpenAI(base_url=api_base, api_key="dummy")


def _make_async_client(api_base: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=api_base, api_key="dummy")


def _build_instruct_generate_prompt(
    instruct_tokenizer,
    user_prompt: str,
    assistant_prefix: str = "",
) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    text = instruct_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if assistant_prefix:
        text += assistant_prefix
    return text


def _uses_thinking_model(model_name_or_path: str) -> bool:
    return "thinking" in (model_name_or_path or "").lower()


def _build_translation_prompt_text(
    tokenizer: Any,
    observed_source: str,
    committed: str,
    force_close_think: bool = False,
) -> str:
    committed_instruction = ""
    if committed:
        committed_instruction = (
            "\n\n[IMPORTANT] A partial Chinese translation is already committed and will appear at the start of the assistant reply. "
            "You MUST output ONLY the new characters that immediately follow (the continuation). "
            "Do NOT repeat, copy, or output the committed part again."
        )
    messages = [{
        "role": "user",
        "content": (
            "[TASK]\n"
            "Translate the [INPUT] text into Chinese.\n\n"
            f"[INPUT]\n{observed_source}"
            f"{committed_instruction}"
        ),
    }]
    if force_close_think:
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        text += "</think>\n"
    else:
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        text += "<|im_start|>assistant\n"
    if committed:
        text += normalize_zh(committed)
    return text


def _build_sentence_translation_prompt_text(
    tokenizer: Any,
    observed_source: str,
    force_close_think: bool = False,
) -> str:
    messages = [{
        "role": "user",
        "content": (
            "[TASK]\n"
            "Translate the [INPUT] sentence into Chinese. "
            "Output ONLY the Chinese translation, no explanation. "
            "Keep terminology consistent across sentences.\n\n"
            f"[INPUT]\n{observed_source}"
        ),
    }]
    if force_close_think:
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        text += "</think>\n"
    else:
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        text += "<|im_start|>assistant\n"
    return text


def _build_continue_prompt(full_source: str, committed: str) -> str:
    if not committed:
        return build_translate_prompt(full_source)
    return (
        f"Translate the following English to Chinese. "
        f"A partial Chinese translation is already committed; output ONLY what comes next.\n\n"
        f'English: "{full_source}"\n\n'
        f'Chinese already committed (do NOT reproduce this in your output): "{committed}"\n\n'
        f"Output ONLY the new Chinese characters that immediately follow the committed portion. "
        f"Do NOT start from the beginning. Do NOT reproduce the committed text."
    )


def _is_chinese_output(text: str, min_ratio: float = 0.5) -> bool:
    if not text:
        return True
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return chinese / len(text) >= min_ratio


async def _translate_batch_async(
    client: AsyncOpenAI,
    model: str,
    instruct_tokenizer,
    sources: List[str],
    committed: str = "",
    use_sentence_prompt: bool = False,
    force_close_think: bool = False,
) -> List[str]:
    async def _one(src: str) -> str:
        if use_sentence_prompt:
            prompt_text = _build_sentence_translation_prompt_text(
                instruct_tokenizer,
                observed_source=src,
                force_close_think=force_close_think,
            )
        else:
            prompt_text = _build_translation_prompt_text(
                instruct_tokenizer,
                observed_source=src,
                committed=committed,
                force_close_think=force_close_think,
            )
        resp = await client.completions.create(
            model=model,
            prompt=prompt_text,
            temperature=0.0,
            max_tokens=512,
        )
        raw = clean_llm_output((resp.choices[0].text or "").strip())
        raw = truncate_translation_repetition(raw)
        return raw if _is_chinese_output(raw) else ""

    results = await asyncio.gather(*[_one(s) for s in sources])
    return list(results)


async def _translate_batch_with_client_async(
    api_base: str,
    model: str,
    instruct_tokenizer,
    sources: List[str],
    committed: str = "",
    use_sentence_prompt: bool = False,
    force_close_think: bool = False,
) -> List[str]:
    client = _make_async_client(api_base)
    try:
        return await _translate_batch_async(
            client, model, instruct_tokenizer, sources, committed,
            use_sentence_prompt=use_sentence_prompt,
            force_close_think=force_close_think,
        )
    finally:
        await client.close()


def translate_candidates(
    api_base: str,
    model: str,
    instruct_tokenizer,
    observed_source: str,
    futures: List[str],
    committed: str = "",
) -> List[str]:
    sources = []
    for future in futures:
        full = (observed_source + " " + future).strip() if future else observed_source
        sources.append(full)

    extensions = asyncio.run(
        _translate_batch_with_client_async(
            api_base, model, instruct_tokenizer, sources, committed,
            force_close_think=_uses_thinking_model(model),
        )
    )
    committed_norm = normalize_zh(committed) if committed else ""
    out = []
    for ext in extensions:
        ext_norm = clean_translation_for_alignment(ext)
        while committed_norm and len(ext_norm) > len(committed_norm) and ext_norm.startswith(committed_norm):
            ext_norm = normalize_zh(ext_norm[len(committed_norm):].strip())
        if committed_norm and ext_norm.startswith(committed_norm):
            out.append(ext_norm)
        else:
            out.append(committed_norm + ext_norm)
    return out


def translate_final(
    api_base: str, model: str,
    instruct_tokenizer,
    full_source: str, committed: str,
) -> str:
    client = _make_sync_client(api_base)
    prompt_text = _build_translation_prompt_text(
        instruct_tokenizer,
        observed_source=full_source,
        committed=committed,
        force_close_think=_uses_thinking_model(model),
    )
    resp = client.completions.create(
        model=model,
        prompt=prompt_text,
        temperature=0.0,
        max_tokens=512,
    )
    raw = (resp.choices[0].text or "").strip()
    cleaned = normalize_zh(truncate_translation_repetition(clean_llm_output(raw)))
    if committed:
        committed_norm = normalize_zh(committed)
        while committed_norm and len(cleaned) > len(committed_norm) and cleaned.startswith(committed_norm):
            cleaned = normalize_zh(cleaned[len(committed_norm):].strip())
        if cleaned.startswith(committed_norm):
            return cleaned
        return committed_norm + cleaned
    return cleaned


def translate_final_base(
    base_llm: LLM,
    instruct_tokenizer,
    full_source: str,
    committed: str,
    force_close_think: bool = False,
) -> str:
    prompt_text = _build_translation_prompt_text(
        instruct_tokenizer,
        observed_source=full_source,
        committed=committed,
        force_close_think=force_close_think,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["<|im_end|>"],
    )
    with (_base_llm_lock if _base_llm_lock is not None else contextlib.nullcontext()):
        outputs = base_llm.generate([prompt_text], params)
    continuation = (outputs[0].outputs[0].text or "").strip()
    continuation = normalize_zh(truncate_translation_repetition(clean_llm_output(continuation)))

    committed_norm = normalize_zh(committed) if committed else ""
    while committed_norm and len(continuation) > len(committed_norm) and continuation.startswith(committed_norm):
        continuation = normalize_zh(continuation[len(committed_norm):].strip())
    return committed_norm + continuation


def select_best_candidate(
    api_base: str, model: str,
    observed_source: str,
    committed: str,
    candidate_translations: List[str],
) -> int:
    client = _make_sync_client(api_base)
    prompt = build_select_prompt(
        observed_source, committed, candidate_translations,
    )
    resp = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=0.0,
        max_tokens=16,
    )
    raw = (resp.choices[0].text or "").strip()
    text = clean_llm_output(raw).strip()
    match = re.search(r"\d+", text)
    if match:
        idx = int(match.group()) - 1
        if 0 <= idx < len(candidate_translations):
            return idx
    return 0


def _extract_json_block(text: str) -> Optional[str]:
    text = text.strip()
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
    return None


def score_candidate_prefixes(
    api_base: str,
    model: str,
    instruct_tokenizer,
    observed_source: str,
    committed: str,
    candidate_items: List[Dict[str, Any]],
    prompt_version: str = "full",
) -> List[Dict[str, Any]]:
    if not candidate_items:
        return []

    use_short = prompt_version == "short"
    user_prompt = (
        build_score_prompt_short(observed_source, committed, candidate_items)
        if use_short
        else build_score_prompt(observed_source, committed, candidate_items)
    )
    max_tokens = 64 if use_short else 256

    client = _make_sync_client(api_base)
    prompt_text = _build_instruct_generate_prompt(instruct_tokenizer, user_prompt)
    resp = client.completions.create(
        model=model,
        prompt=prompt_text,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    raw = (resp.choices[0].text or "").strip()
    text = clean_llm_output(raw).strip()

    parsed_items: List[Dict[str, Any]] = []
    json_block = _extract_json_block(text)
    if json_block:
        try:
            parsed = json.loads(json_block)
            if isinstance(parsed, dict):
                parsed_items = list(parsed.get("scores", []))
            elif isinstance(parsed, list):
                parsed_items = list(parsed)
        except Exception:
            parsed_items = []

    score_map: Dict[int, Dict[str, Any]] = {}
    for item in parsed_items:
        if not isinstance(item, dict):
            continue
        cid = item.get("candidate_id", item.get("id"))
        try:
            cid_int = int(cid)
        except Exception:
            continue
        try:
            score = int(round(float(item.get("score", 0))))
        except Exception:
            score = 0
        score = max(0, min(100, score))
        tags_raw = item.get("tags", [])
        if isinstance(tags_raw, list):
            tags = [str(x) for x in tags_raw][:8]
        elif tags_raw:
            tags = [str(tags_raw)]
        else:
            tags = []
        score_map[cid_int] = {"score": score, "tags": tags}

    if not score_map:
        for line in text.splitlines():
            m = re.search(r"(\d+)\D+?(\d{1,3})", line)
            if not m:
                continue
            cid_int = int(m.group(1))
            score = max(0, min(100, int(m.group(2))))
            score_map[cid_int] = {"score": score, "tags": ["fallback_parse"]}

    results: List[Dict[str, Any]] = []
    for item in candidate_items:
        cid = int(item["candidate_id"])
        scored = score_map.get(cid, {"score": 0, "tags": ["parse_failed"]})
        results.append({
            "candidate_id": cid,
            "score": scored["score"],
            "tags": scored["tags"],
        })
    return results


# ===================================================================
# LLM LCP70 Delta
# ===================================================================

def build_lcp70_llm_prompt(
    observed_source: str,
    committed_norm: str,
    candidate_safe_prefixes: List[str],
    K: int,
    with_boundary_rule: bool = True,
) -> str:
    M = len(candidate_safe_prefixes)
    if committed_norm:
        new_parts = [
            normalize_zh(sp)[len(committed_norm):]
            if normalize_zh(sp).startswith(committed_norm) else ""
            for sp in candidate_safe_prefixes
        ]
    else:
        new_parts = [normalize_zh(sp) for sp in candidate_safe_prefixes]
    new_parts_str = "\n".join(
        f'  [{i + 1}] "{p}"' for i, p in enumerate(new_parts)
    )
    boundary_rule = (
        f"2. S must end at a complete Chinese word or morpheme boundary.\n"
        if with_boundary_rule
        else f"2. Ignore word-boundary constraints; return the literal shared prefix.\n"
    )
    return (
        f'Observed English so far: "{observed_source}"\n'
        f'Chinese already committed (for context only): "{committed_norm}"\n\n'
        f"Below are {M} new-text fragments to be appended after the committed text:\n"
        f"{new_parts_str}\n\n"
        f"Task: Find the longest string S such that S appears at the VERY START "
        f"(character position 0) of at least {K} of the {M} fragments above.\n\n"
        f"Rules:\n"
        f"1. S must LITERALLY match the beginning of ≥{K} fragments — copy characters "
        f"from the START of the fragments, do NOT invent new text.\n"
        f"{boundary_rule}"
        f"3. Output ONLY S. No explanation, no quotes, no punctuation outside S.\n"
        f"4. If no common starting prefix of length ≥1 exists, output: EMPTY\n\n"
        f"Example:\n"
        f"  fragments: [\"民间故事总是\", \"民间故事是\", \"民间故\", \"民间\", \"民众\"]\n"
        f"  K=3 → answer: \"民间故事\" (starts fragments 1&2 fully, but need {K};"
        f" try shorter: \"民间故\" starts fragments 1,2,3 = 3 ≥ {K} → \"民间故\" if word boundary; "
        f"or \"民间\" if 故 splits a word)\n"
        f"Output S or EMPTY:"
    )


def _extract_deltas_from_safe_prefixes(
    committed: str,
    candidate_safe_prefixes: List[str],
) -> Tuple[str, List[str]]:
    committed_norm = normalize_zh(committed) if committed else ""
    deltas: List[str] = []
    for sp in candidate_safe_prefixes:
        sp_norm = normalize_zh(sp)
        if committed_norm:
            if sp_norm.startswith(committed_norm) and len(sp_norm) > len(committed_norm):
                deltas.append(sp_norm[len(committed_norm):])
            else:
                deltas.append("")
        else:
            deltas.append(sp_norm)
    return committed_norm, deltas


def longest_prefix_with_quorum(deltas: List[str], K: int) -> str:
    """Longest prefix shared by at least K deltas (character-level)."""
    if not deltas or K <= 0:
        return ""

    prefix_count: Dict[str, int] = {}
    for d in deltas:
        p = ""
        for ch in d:
            p += ch
            prefix_count[p] = prefix_count.get(p, 0) + 1

    best = ""
    for p, c in prefix_count.items():
        if c >= K and len(p) > len(best):
            best = p
    return best

# lcp_code_entry
def get_quorum_lcp_delta_code( 
    committed: str,
    candidate_safe_prefixes: List[str],
    consensus_ratio: float = 1.0,
) -> str:
    """Pure code baseline: no LLM, longest quorum-prefix."""
    if not candidate_safe_prefixes:
        return ""

    _committed_norm, deltas = _extract_deltas_from_safe_prefixes(
        committed, candidate_safe_prefixes
    )
    M = len(deltas)
    K = max(1, math.ceil(consensus_ratio * M))
    if sum(1 for d in deltas if d) < K:
        return ""
    return longest_prefix_with_quorum([d for d in deltas if d], K)


def get_lcp70_llm_delta_via_llm(
    api_base: str,
    model: str,
    instruct_tokenizer,
    observed_source: str,
    committed: str,
    candidate_safe_prefixes: List[str],
    consensus_ratio: float = 0.6,
    with_boundary_rule: bool = True,
) -> str:
    if not candidate_safe_prefixes:
        return ""

    committed_norm, deltas = _extract_deltas_from_safe_prefixes(
        committed, candidate_safe_prefixes
    )
    M = len(deltas)
    K = max(1, math.ceil(consensus_ratio * M))

    if sum(1 for d in deltas if d) < K:
        return ""

    user_prompt = build_lcp70_llm_prompt(
        observed_source, committed_norm, candidate_safe_prefixes, K,
        with_boundary_rule=with_boundary_rule,
    )
    client = _make_sync_client(api_base)
    prompt_text = _build_instruct_generate_prompt(
        instruct_tokenizer, user_prompt, assistant_prefix="</think>\n"
    )
    try:
        resp = client.completions.create(
            model=model,
            prompt=prompt_text,
            temperature=0.0,
            max_tokens=128,
            stop=["<|im_end|>"],
        )
        raw = (resp.choices[0].text or "").strip()
        first_line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
        output = normalize_zh(clean_llm_output(first_line))
        print(f"[LCP70LLM DEBUG] raw={repr(raw[:120])} | first_line={repr(first_line[:80])} | output={repr(output[:80])}", flush=True)
    except Exception as e:
        print(f"[LCP70LLM] API exception: {e}", flush=True)
        return ""

    if not output or output.upper() == "EMPTY":
        print(f"[LCP70LLM DEBUG] -> empty/EMPTY, returning ''", flush=True)
        return ""

    if committed_norm and output.startswith(committed_norm):
        output = output[len(committed_norm):]
    if not output:
        return ""

    # hard validation: 从 LLM 输出开始往短截，找到一个真正被至少 K 个 delta 以开头匹配的最长前缀，再返回
    valid_output = ""
    for end in range(len(output), 0, -1):
        # 从长到短枚举 output 的所有前缀
        prefix = output[:end]
        # 对每个前缀 prefix，统计有多少个 delta 以它开头
        count = sum(1 for d in deltas if d and d.startswith(prefix))
        if count >= K:
            # 找到第一个 count >= K 的最长前缀，作为 valid_output
            valid_output = prefix
            break
    print(
        f"[LCP70LLM DEBUG] validation: output={repr(output[:60])} "
        f"→ valid={repr(valid_output[:60])} K={K} boundary={with_boundary_rule}",
        flush=True,
    )
    return valid_output


#
# Prompt diff summary: CURRENT V2 vs SIMPLE V1
# 1. V2 adds explicit "stop before divergence" and "do not invent content" rules;
#    V1 only says "stay close to fragments", which is weaker against hallucination.
# 2. V2 includes boundary-case examples (divergence, subject conflict, hallucination risk);
#    V1 mostly demonstrates synonym merging plus one noun-conflict counter-example.
# 3. V2 makes K-of-M support and EMPTY conditions more explicit;
#    V1 leaves more of that reasoning implicit in the examples.
# 4. V2 uses a single canonical answer for the divergence example ("这些介绍");
#    V1-style prompting had looser guidance, which can encourage unstable outputs.
# 5. Both avoid numeric labels in live prompts, but V2's output rules more strongly
#    forbid copying labels/brackets and are paired with output sanitization in code.
#
def build_majority_vote_prompt_v2(
    observed_source: str,
    committed_norm: str,
    fragments: List[str],
    K: int,
    start_anchor: str = "",
    max_chars: int = 20,
) -> str:
    M = len(fragments)
    frag_str = "\n".join(f'候选{chr(65 + i)}: "{f}"' for i, f in enumerate(fragments))
    anchor_block = (
        f'Literal start anchor shared by many fragments: "{start_anchor}"\n'
        "- Your output must start with this anchor exactly.\n"
        "- Do NOT skip this anchor and jump to a later content word.\n\n"
        if start_anchor
        else ""
    )
    return (
        "You are finding the Semantic Common Prefix (SCP) of Chinese translation fragments.\n\n"
        "What is Semantic LCP:\n"
        "- Literal LCP = exact same starting characters.\n"
        "- Semantic LCP = same starting meaning, even if wording differs.\n"
        "- Synonyms count as same meaning.\n"
        "- Ignore leading connectives or punctuation differences such as 而且 / 并且 / 而 / ，.\n\n"
        "Common synonym families:\n"
        "- 编辑 ↔ 编者\n"
        "- 任务 ↔ 职责 ↔ 工作\n"
        "- 寻找 ↔ 搜寻 ↔ 去找\n"
        "- 道歉 ↔ 表示歉意\n"
        "- 重要 ↔ 关键 ↔ 至关重要\n\n"
        "Your task:\n"
        f"- Read the {M} fragments below.\n"
        f"- Find the longest short prefix S (at most {max_chars} Chinese characters) such that at least K={K} fragments begin with the SAME MEANING as S.\n"
        "- S must stay aligned to the START of the fragments.\n"
        "- Do NOT skip earlier words and jump directly to a later content word.\n"
        "- Stop S before any point where fragments disagree on a key fact, noun, role, subject, object, or predicate.\n"
        "- Do NOT invent content that is not grounded in at least K fragments.\n"
        "- If fewer than K fragments support a safe S, output EMPTY.\n\n"
        f'English observed so far: "{observed_source}"\n'
        f'Committed Chinese (already output, do NOT repeat it): "{committed_norm}"\n\n'
        f"{anchor_block}"
        f"Fragments (M={M}, need K={K} support):\n{frag_str}\n\n"
        "Example 1 — synonyms merge:\n"
        "Committed: \"\"\n"
        "K=3\n"
        "候选A: \"编辑的职责是寻找故事\"\n"
        "候选B: \"编者的任务就是搜寻故事\"\n"
        "候选C: \"编辑的工作是找故事\"\n"
        "SCP: 编辑的任务是寻找故事\n\n"
        "Example 2 — stop before divergence:\n"
        "Committed: \"\"\n"
        "K=3\n"
        "候选A: \"这些介绍既长又无用\"\n"
        "候选B: \"这些介绍冗长且令人厌倦\"\n"
        "候选C: \"这些介绍简单又无效\"\n"
        "SCP: 这些介绍\n\n"
        "Example 3 — EMPTY due to key noun conflict:\n"
        "Committed: \"\"\n"
        "K=3\n"
        "候选A: \"编辑只是编辑\"\n"
        "候选B: \"人只是人\"\n"
        "候选C: \"出版商只是出版商\"\n"
        "SCP: EMPTY\n\n"
        "Example 4 — EMPTY: no K-way agreement (subject/predicate differ):\n"
        "Committed: \"\"\n"
        "K=3\n"
        "候选A: \"他感到高兴\"\n"
        "候选B: \"她感到悲伤\"\n"
        "候选C: \"他们感到困惑\"\n"
        "SCP: EMPTY\n\n"
        "Example 5 — do not skip the beginning:\n"
        "Committed: \"\"\n"
        "K=3\n"
        "候选A: \"迫使编辑承认\"\n"
        "候选B: \"迫使编辑声明\"\n"
        "候选C: \"迫使编辑解释\"\n"
        "SCP: 迫使编辑\n"
        "Bad: 编辑\n\n"
        "Example 6 — committed non-empty, output only the new delta:\n"
        "Committed: \"编辑的\"\n"
        "K=3\n"
        "候选A: \"任务是寻找好故事\"\n"
        "候选B: \"工作是搜寻好故事\"\n"
        "候选C: \"职责是找到好故事\"\n"
        "SCP: 任务是寻找好故事\n"
        "Bad: 编辑的任务是寻找好故事  ← wrong, repeats committed\n\n"
        "Example 7 — committed non-empty, EMPTY because delta diverges immediately:\n"
        "Committed: \"这是一个\"\n"
        "K=3\n"
        "候选A: \"重要的科学突破\"\n"
        "候选B: \"关键的政治决定\"\n"
        "候选C: \"深刻的历史时刻\"\n"
        "SCP: EMPTY (重要/关键/深刻 and 科学突破/政治决定/历史时刻 are not the same meaning)\n\n"
        "Output rules:\n"
        "- Output ONLY S on one line, or exactly EMPTY.\n"
        "- No quotes, no labels, no brackets, no explanation.\n"
        "- Do NOT copy labels such as 候选A, A, [A], [1], [2], or list markers.\n"
        "- Do the K-of-M support check silently yourself."
    )


def build_majority_vote_prompt_v3_cot(
    observed_source: str,
    committed_norm: str,
    fragments: List[str],
    K: int,
    start_anchor: str = "",
    max_chars: int = 20,
) -> str:
    M = len(fragments)
    frag_str = "\n".join(f'候选{chr(65 + i)}: "{f}"' for i, f in enumerate(fragments))
    anchor_block = (
        f'Literal start anchor shared by many fragments: "{start_anchor}"\n'
        "- Your final answer must start with this anchor exactly.\n"
        "- Do NOT skip this anchor and jump to a later content word.\n\n"
        if start_anchor
        else ""
    )
    return (
        "You are finding the Semantic Common Prefix (SCP) of Chinese translation fragments.\n\n"
        "Literal LCP means the exact same starting characters.\n"
        "Semantic LCP means the same starting meaning, even if wording differs.\n"
        "Synonyms count as the same meaning. Ignore leading connectives/punctuation differences "
        "such as 而且 / 并且 / 而 / ，.\n\n"
        "Common synonym families:\n"
        "- 编辑 ↔ 编者\n"
        "- 任务 ↔ 职责 ↔ 工作\n"
        "- 寻找 ↔ 搜寻 ↔ 去找\n"
        "- 道歉 ↔ 表示歉意\n"
        "- 重要 ↔ 关键 ↔ 至关重要\n\n"
        "Task:\n"
        f"- Read all {M} fragments.\n"
        f"- Find the longest short prefix S (at most {max_chars} Chinese characters) such that at least K={K} fragments begin with the SAME MEANING as S.\n"
        "- S must stay aligned to the START of the fragments.\n"
        "- Do NOT skip earlier words and jump to a later content word.\n"
        "- Stop before any divergence in key fact, noun, role, subject, object, or predicate.\n"
        "- Do NOT invent content not grounded in at least K fragments.\n"
        "- If fewer than K fragments support a safe S, output EMPTY.\n\n"
        "Reasoning procedure (do this silently, do NOT output your reasoning):\n"
        "1. Normalize leading connectives/punctuation.\n"
        "2. Compare the fragment beginnings one by one.\n"
        "3. Identify the earliest shared meaning.\n"
        "4. Extend S only while at least K fragments still support the SAME starting meaning.\n"
        "5. The moment support drops below K, stop.\n"
        "6. Before answering, check again that your final S starts at the fragment beginning and is supported by at least K fragments.\n\n"
        f'English observed so far: "{observed_source}"\n'
        f'Committed Chinese (already output, do NOT repeat it): "{committed_norm}"\n\n'
        f"{anchor_block}"
        f"Fragments (M={M}, need K={K} support):\n{frag_str}\n\n"
        "Example 1 — synonyms merge:\n"
        "K=3\n"
        "候选A: \"编辑的职责是寻找故事\"\n"
        "候选B: \"编者的任务就是搜寻故事\"\n"
        "候选C: \"编辑的工作是找故事\"\n"
        "SCP: 编辑的任务是寻找故事\n\n"
        "Example 2 — stop before divergence:\n"
        "K=3\n"
        "候选A: \"而且这些介绍既单调又无用\"\n"
        "候选B: \"而且这些介绍冗长且令人厌倦\"\n"
        "候选C: \"而且这些介绍既枯燥又无效\"\n"
        "SCP: 这些介绍\n\n"
        "Example 3 — EMPTY due to key noun conflict:\n"
        "K=3\n"
        "候选A: \"编辑只是编辑\"\n"
        "候选B: \"作者只是作者\"\n"
        "候选C: \"出版商只是出版商\"\n"
        "SCP: EMPTY\n\n"
        "Example 4 — do not skip the beginning:\n"
        "K=3\n"
        "候选A: \"迫使编辑承认\"\n"
        "候选B: \"迫使编辑声明\"\n"
        "候选C: \"迫使编辑解释\"\n"
        "SCP: 迫使编辑\n"
        "Bad: 编辑\n\n"
        "Output rules:\n"
        "- Think step by step silently, then output ONLY the final answer.\n"
        "- Output EXACTLY ONE line: either S or EMPTY.\n"
        "- No quotes, no labels, no brackets, no explanation.\n"
        "- Do NOT copy labels such as 候选A, A, [A], [1], [2], or list markers."
    )


def build_majority_vote_prompt(
    observed_source: str,
    committed_norm: str,
    fragments: List[str],
    K: int,
    max_chars: int = 20,
) -> str:
    """Compatibility wrapper. Current runtime default = V2."""
    return build_majority_vote_prompt_v2(
        observed_source=observed_source,
        committed_norm=committed_norm,
        fragments=fragments,
        K=K,
        max_chars=max_chars,
    )


#
# SIMPLE V1 PROMPT
# - Kept only for side-by-side comparison with the current V2 prompt above.
# - Simpler and shorter, but weaker on divergence handling and anti-hallucination guidance.
# - Can be selected explicitly via --majority-vote-prompt-version simple_v1.
#
def build_majority_vote_prompt_simple_v1(
    observed_source: str,
    committed_norm: str,
    fragments: List[str],
    K: int,
    max_chars: int = 20,
) -> str:
    """Minimal majority-vote baseline prompt with examples covering committed="" and committed!="" cases."""
    M = len(fragments)
    frag_str = "\n".join(f'候选{chr(65 + i)}: "{f}"' for i, f in enumerate(fragments))
    return (
        "Task: find one short semantic prefix that best matches the common beginning "
        "of the Chinese fragments below.\n\n"
        "A semantic prefix means: the fragments may use different words, but they begin "
        "with the same meaning.\n\n"
        "IMPORTANT: The fragments are already the NEW part after what is committed. "
        "Your output must be a continuation of the committed text, NOT repeat it.\n\n"
        f'English (observed so far): "{observed_source}"\n'
        f'Committed Chinese (do NOT repeat): "{committed_norm}"\n'
        f"Fragments (M={M}):\n{frag_str}\n\n"
        "Example 1 — committed empty, synonyms merge:\n"
        "Committed: \"\"\n"
        "候选A: \"编辑的职责是寻找好故事\"\n"
        "候选B: \"编者的任务就是搜寻好故事\"\n"
        "候选C: \"编辑的工作是去找好故事\"\n"
        "SCP: 编辑的任务是寻找好故事\n\n"
        "Example 2 — committed non-empty, output only the new delta:\n"
        "Committed: \"编辑的\"\n"
        "候选A: \"任务是寻找好故事\"\n"
        "候选B: \"工作是搜寻好故事\"\n"
        "候选C: \"职责是找到好故事\"\n"
        "SCP: 任务是寻找好故事\n"
        "Bad: 编辑的任务是寻找好故事  ← wrong, repeats committed\n\n"
        "Example 3 — EMPTY due to key noun conflict:\n"
        "Committed: \"\"\n"
        "候选A: \"编辑只是编辑\"\n"
        "候选B: \"作者只是作者\"\n"
        "候选C: \"发现者只是发现者\"\n"
        "SCP: EMPTY\n\n"
        "Example 4 — committed non-empty, EMPTY because delta diverges immediately:\n"
        "Committed: \"这是一个\"\n"
        "候选A: \"重要的科学突破\"\n"
        "候选B: \"关键的政治决定\"\n"
        "候选C: \"深刻的历史时刻\"\n"
        "SCP: EMPTY\n\n"
        "Rules:\n"
        f"- Output ONLY S (at most {max_chars} Chinese characters), or exactly EMPTY.\n"
        "- S must not repeat the committed text.\n"
        "- No labels, no quotes, no explanation.\n"
        "- Follow the pattern shown in the examples."
    )


def build_majority_vote_verify_prompt(
    observed_source: str,
    committed_norm: str,
    fragments: List[str],
    candidate: str,
) -> str:
    frag_str = "\n".join(f'[{i + 1}] "{f}"' for i, f in enumerate(fragments))
    return (
        "You are a strict verifier for a candidate Chinese incremental prefix.\n\n"
        f'Observed English so far:\n"{observed_source}"\n\n'
        f'Committed Chinese (already fixed):\n"{committed_norm}"\n\n'
        "Fragments:\n"
        f"{frag_str}\n\n"
        f'Candidate S:\n"{candidate}"\n\n'
        "Judge each fragment:\n"
        "- support: fragment can naturally continue with S without contradiction and same meaning\n"
        "- Treat simple synonym / paraphrase substitutions as support when meaning is unchanged "
        "(e.g., 编辑/编者, 任务/职责/工作, 寻找/搜寻/去找)\n"
        "- When judging support, ignore leading discourse markers/punctuation differences "
        "(而且/而/并且 and leading ，、 etc.)\n"
        "- contradict: fragment explicitly conflicts with S\n"
        "- unknown: neither clear support nor contradiction\n\n"
        "Output STRICT JSON only:\n"
        '{"support_ids":[1,2],"contradict_ids":[3],"unknown_ids":[4]}'
    )


_MAJORITY_VOTE_LEADING_PUNCT = re.compile(r"^[，。、；：！？,.!?;:\s]+")
_MAJORITY_VOTE_DISCOURSE_MARKERS = (
    "而且", "并且", "同时", "此外", "然后", "另外", "而", "并",
)


def _strip_majority_vote_leading_variants(text: str) -> str:
    t = normalize_zh(text or "")
    t = _MAJORITY_VOTE_LEADING_PUNCT.sub("", t)
    for m in _MAJORITY_VOTE_DISCOURSE_MARKERS:
        if t.startswith(m):
            t = t[len(m):]
            t = _MAJORITY_VOTE_LEADING_PUNCT.sub("", t)
            break
    return t


def _best_quorum_prefix_from_seed(seed: str, fragments: List[str], K: int) -> str:
    if not seed or not fragments or K <= 0:
        return ""
    for end in range(len(seed), 0, -1):
        prefix = seed[:end]
        support_count = sum(1 for f in fragments if f.startswith(prefix))
        if support_count >= K:
            return prefix
    return ""


def _majority_vote_quorum_backoff(output: str, fragments: List[str], K: int) -> str:
    """Safety backoff when verifier rejects.

    Order:
      1) Literal backoff on raw fragments using model output as seed.
      2) Literal backoff on normalized fragments (strip discourse-marker variance).
      3) Pure quorum-LCP on normalized fragments.
    """
    # 1) literal backoff on raw fragments
    raw = _best_quorum_prefix_from_seed(output, fragments, K)
    if raw:
        return raw

    # 2) literal backoff after stripping leading discourse-marker variance
    norm_fragments = [_strip_majority_vote_leading_variants(f) for f in fragments if f]
    norm_fragments = [f for f in norm_fragments if f]
    if len(norm_fragments) < K:
        return ""
    norm_output = _strip_majority_vote_leading_variants(output)
    norm = _best_quorum_prefix_from_seed(norm_output, norm_fragments, K)
    if norm:
        return norm

    # 3) fallback to quorum-LCP on normalized fragments
    return longest_prefix_with_quorum(norm_fragments, K)


def _compute_majority_vote_start_anchor(fragments: List[str], K: int) -> str:
    if not fragments or K <= 0:
        return ""
    norm_fragments = [_strip_majority_vote_leading_variants(f) for f in fragments if f]
    norm_fragments = [f for f in norm_fragments if f]
    if len(norm_fragments) < K:
        return ""
    return longest_prefix_with_quorum(norm_fragments, K)


def _is_latin_heavy_delta(text: str, ratio: float = 0.30) -> bool:
    if not text:
        return False
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return (latin / max(1, len(text))) > ratio


_MAJORITY_VOTE_LEAK_PREFIX_RE = re.compile(
    r'^(?:\s*(?:候选[A-Z]|Candidate\s*[A-Z]|[A-Z])\s*[:：]\s*|\s*\[[A-Z0-9]+\]\s*|["“”\'`])+'
)


def sanitize_majority_vote_output(text: str) -> str:
    text = normalize_zh(text or "")
    if not text:
        return ""
    text = _MAJORITY_VOTE_LEAK_PREFIX_RE.sub("", text)
    text = text.lstrip("\"'“”`[]:： \t")
    return text.strip()


def get_majority_vote_delta_via_llm(
    api_base: str,
    model: str,
    instruct_tokenizer,
    observed_source: str,
    committed: str,
    candidate_safe_prefixes: List[str],
    consensus_ratio: float = 0.7,
    disable_backoff: bool = False,
    prompt_version: str = "v2",
    use_reasoning: bool = False,
    reasoning_temperature: float = 0.6,
    reasoning_max_tokens: int = 512,
) -> str:
    """Simple semantic-LCP baseline for majority_vote.

    Single synthesis call only:
      - no verifier
      - no quorum backoff
      - prompt-based K-of-M reasoning only
    This is intentionally a simpler experimental setting to inspect what the
    LLM itself thinks the semantic common prefix is.
    """
    if not candidate_safe_prefixes:
        return ""

    committed_norm, deltas = _extract_deltas_from_safe_prefixes(
        committed, candidate_safe_prefixes # ← 输入是完整 prefix
    )
    fragments = [d for d in deltas if d] ## ← 传给 prompt 的是 delta！
    print(f"fragments={fragments}") 
    M = len(deltas) 
    K = max(1, math.ceil(consensus_ratio * M))
    if len(fragments) < K:
        return ""

    client = _make_sync_client(api_base)
    start_anchor = _compute_majority_vote_start_anchor(fragments, K)

    if prompt_version == "simple_v1":
        synth_prompt = build_majority_vote_prompt_simple_v1(
            observed_source, committed_norm, fragments, K
        )
    elif prompt_version == "v3_cot":
        synth_prompt = build_majority_vote_prompt_v3_cot(
            observed_source, committed_norm, fragments, K,
            start_anchor=start_anchor,
        )
    else:
        synth_prompt = build_majority_vote_prompt_v2(
            observed_source, committed_norm, fragments, K,
            start_anchor=start_anchor,
        )
    assistant_prefix = "" if use_reasoning else "</think>\n"
    prompt_text = _build_instruct_generate_prompt(
        instruct_tokenizer, synth_prompt, assistant_prefix=assistant_prefix
    )
    try:
        resp = client.completions.create(
            model=model,
            prompt=prompt_text,
            temperature=reasoning_temperature if use_reasoning else 0.0,
            max_tokens=reasoning_max_tokens if use_reasoning else 128,
            stop=["<|im_end|>"],
        )
        raw = (resp.choices[0].text or "").strip()
        first_line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
        output = sanitize_majority_vote_output(clean_llm_output(first_line))
    except Exception as e:
        print(f"[MajorityVote] synthesis exception: {e}", flush=True)
        return ""

    if not output or output.upper() == "EMPTY":
        print("[MajorityVoteSimple] synthesis returned EMPTY", flush=True)
        return ""
    if committed_norm and output.startswith(committed_norm):
        output = output[len(committed_norm):]
    if prompt_version != "simple_v1" and start_anchor:
        if not output.startswith(start_anchor):
            if start_anchor.endswith(output) and len(start_anchor) >= 2:
                output = start_anchor
    if not output:
        return ""
    if _is_latin_heavy_delta(output):
        print(
            f"[MajorityVoteSimple] latin-heavy synthesis output dropped: {repr(output[:60])}",
            flush=True,
        )
        return ""

    # Cap length to longest fragment (sanity guard)
    output = output[: max(len(f) for f in fragments)]
    if len(output) > 20:
        output = output[:20]
    print(
        f"[MajorityVoteSimple] prompt={prompt_version} reasoning={use_reasoning} "
        f"anchor={repr(start_anchor)} output={repr(output[:60])} "
        f"K={K} fragments={len(fragments)}",
        flush=True,
    )
    return output


# ===================================================================
# Sentence-scoped alignment: precompute translations & chunk→sentence map
# ===================================================================

def precompute_sentence_translations(
    api_base: str,
    model: str,
    instruct_tokenizer,
    sentences: List[str],
    verbose_log_file: Optional[Any] = None,
) -> List[str]:
    if not sentences:
        return []
    raw_list = asyncio.run(
        _translate_batch_with_client_async(
            api_base, model, instruct_tokenizer, sentences, committed="",
            use_sentence_prompt=True,
            force_close_think=_uses_thinking_model(model),
        )
    )
    out = []
    for i, raw in enumerate(raw_list):
        cleaned = normalize_zh(clean_translation_for_alignment(raw or ""))
        out.append(cleaned)
    if verbose_log_file is not None:
        _vlog(verbose_log_file, "[precompute_sentence_translations] per-sentence translations:")
        for i, t in enumerate(out):
            _vlog(verbose_log_file, f"  sent[{i}]: \"{t}\"")
    return out


def build_chunk_to_sentence_map(
    sentences: List[str],
    trajectory: List[str],
) -> Tuple[List[int], List[int], List[int]]:
    n_sent = len(sentences)
    sent_word_start: List[int] = [0]
    for s in sentences:
        sent_word_start.append(sent_word_start[-1] + len(s.strip().split()))
    sent_word_end = sent_word_start[1:]
    sent_word_start = sent_word_start[:-1]
    if n_sent == 0:
        return [], [], []

    chunk_sentence_ids: List[int] = []
    obs_word_count = 0
    last_nonempty_sid = 0
    for chunk in trajectory:
        words_in_chunk = len(chunk.strip().split()) if chunk.strip() else 0
        if words_in_chunk == 0:
            chunk_sentence_ids.append(last_nonempty_sid)
            continue
        obs_word_count += words_in_chunk
        sid = 0
        for i in range(n_sent):
            if obs_word_count <= sent_word_end[i]:
                sid = i
                break
        else:
            sid = n_sent - 1
        last_nonempty_sid = sid
        chunk_sentence_ids.append(sid)
    return chunk_sentence_ids, sent_word_start, sent_word_end


# ===================================================================
# Core Processing
# ===================================================================

def process_one_utterance(
    base_llm: LLM,
    api_base: str,
    instruct_model: str,
    instruct_tokenizer,
    majority_vote_api_base: str,
    majority_vote_model: str,
    majority_vote_tokenizer,
    align_model,
    align_tokenizer,
    utt_id: str,
    sentences: List[str],
    trajectory: List[str],
    row: Dict[str, str],
    args: argparse.Namespace,
    verbose_log_file: Optional[Any] = None,
    translation_cache: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    full_source_text = " ".join(sentences)
    n_chunks = len(trajectory)
    timing_totals: Dict[str, float] = {
        "chunk_total_s": 0.0,
        "step1_future_sampling_s": 0.0,
        "step2_translate_candidates_s": 0.0,
        "step3_alignment_total_s": 0.0,
        "step3_alignment_model_s": 0.0,
        "step3_truncate_s": 0.0,
        "step4_majority_vote_s": 0.0,
        "translate_final_s": 0.0,
    }

    _vlog(verbose_log_file, f"\n{'#'*60}")
    _vlog(verbose_log_file, f"# Utterance: {utt_id}")
    _vlog(verbose_log_file, f"# Full text: {full_source_text}")
    _vlog(verbose_log_file, f"# Chunks: {n_chunks}")
    _vlog(verbose_log_file, f"# M={args.num_candidates}")
    _vlog(verbose_log_file, f"{'#'*60}")

    sentence_translations: List[str] = []
    chunk_sentence_ids: List[int] = []
    sent_word_start: List[int] = []
    sent_word_end: List[int] = []
    committed_sentence_prefix_lens: List[int] = []
    if sentences:
        sentence_translations = precompute_sentence_translations(
            api_base, instruct_model, instruct_tokenizer, sentences, verbose_log_file
        )
        chunk_sentence_ids, sent_word_start, sent_word_end = build_chunk_to_sentence_map(
            sentences, trajectory
        )
        committed_sentence_prefix_lens = [0] * len(sentences)
        total_sent_words = sum(len(s.strip().split()) for s in sentences)
        total_traj_words = sum(len(c.strip().split()) for c in trajectory if c.strip())
        if abs(total_sent_words - total_traj_words) > 3:
            _vlog(verbose_log_file,
                  f"  [WARN] sentence word count ({total_sent_words}) != trajectory word count ({total_traj_words}), sentence path may misalign")

    committed_norm = ""
    accumulated_source = ""
    decisions: List[Tuple[str, str]] = []
    all_details: List[Optional[Dict]] = []

    for chunk_pos, chunk in enumerate(trajectory):
        chunk_t0 = time.perf_counter()
        stripped = chunk.strip()
        if stripped:
            accumulated_source = (accumulated_source + " " + stripped).strip()

        is_last = (chunk_pos == n_chunks - 1)

        _vlog(verbose_log_file, f"\n{'='*60}")
        _vlog(verbose_log_file,
              f"  Chunk {chunk_pos+1}/{n_chunks}: \"{stripped}\"")
        _vlog(verbose_log_file,
              f"  accumulated: \"{accumulated_source}\"")
        _vlog(verbose_log_file,
              f"  committed:   \"{committed_norm}\"")

        # --- Case 1: Empty chunk -> READ ---
        if not stripped and not is_last:
            decisions.append(("READ", ""))
            _vlog(verbose_log_file, "  -> READ (empty chunk)")
            chunk_elapsed = time.perf_counter() - chunk_t0
            timing_totals["chunk_total_s"] += chunk_elapsed
            _vlog(verbose_log_file, f"  [Timing] chunk_total={chunk_elapsed:.3f}s (early empty)")
            if args.save_details:
                all_details.append({"reason": "empty_chunk"})
            continue

        # --- Case 2: Last chunk -> force commit ---
        if is_last:
            t_final0 = time.perf_counter()
            if args.final_commit_backend == "base":
                full_translation = translate_final_base(
                    base_llm,
                    instruct_tokenizer,
                    accumulated_source, committed_norm,
                    force_close_think=_uses_thinking_model(args.instruct_model_name),
                )
            else:
                full_translation = translate_final(
                    api_base,
                    instruct_model,
                    instruct_tokenizer,
                    accumulated_source, committed_norm,
                )
            t_final = time.perf_counter() - t_final0
            timing_totals["translate_final_s"] += t_final
            if len(full_translation) > len(committed_norm):
                remaining = full_translation[len(committed_norm):]
                committed_norm = full_translation
                decisions.append(("WRITE", remaining))
                _vlog(verbose_log_file,
                      f"  -> WRITE (end) \"{remaining}\"")
            else:
                decisions.append(("READ", ""))
                _vlog(verbose_log_file, "  -> READ (end, nothing new)")
            chunk_elapsed = time.perf_counter() - chunk_t0
            timing_totals["chunk_total_s"] += chunk_elapsed
            _vlog(verbose_log_file,
                  f"  [Timing] translate_final={t_final:.3f}s chunk_total={chunk_elapsed:.3f}s")
            if args.save_details:
                all_details.append({
                    "reason": "end_of_utterance",
                    "full_translation": full_translation,
                })
            continue

        # --- Case 3: Too few words -> READ ---
        observed_words = len(accumulated_source.split())
        if observed_words < args.min_observed_words:
            decisions.append(("READ", ""))
            _vlog(verbose_log_file,
                  f"  -> READ (too few words: {observed_words})")
            chunk_elapsed = time.perf_counter() - chunk_t0
            timing_totals["chunk_total_s"] += chunk_elapsed
            _vlog(verbose_log_file,
                  f"  [Timing] chunk_total={chunk_elapsed:.3f}s (early too_few_words)")
            if args.save_details:
                all_details.append({"reason": "early_skip",
                                    "observed_words": observed_words})
            continue

        # --- Case 4: Future sampling + alignment + delta selection ---

        # Step 1: Base model generates M future continuations
        t1_0 = time.perf_counter()
        futures = sample_source_futures(
            base_llm, accumulated_source,
            args.num_candidates, args.future_tokens,
            args.sample_temperature,
        )
        t1 = time.perf_counter() - t1_0
        timing_totals["step1_future_sampling_s"] += t1
        _vlog(verbose_log_file,
              f"  [Step 1] {len(futures)} future continuations ({t1:.3f}s):")
        for fi, f in enumerate(futures):
            _vlog(verbose_log_file, f"    {fi}: \"{f}\"")

        if len(futures) < 2:
            decisions.append(("READ", ""))
            _vlog(verbose_log_file, "  -> READ (too few futures)")
            chunk_elapsed = time.perf_counter() - chunk_t0
            timing_totals["chunk_total_s"] += chunk_elapsed
            _vlog(verbose_log_file,
                  f"  [Timing] chunk_total={chunk_elapsed:.3f}s (too_few_futures)")
            if args.save_details:
                all_details.append({"reason": "too_few_futures"})
            continue

        # Step 2: Instruct model translates all candidates, continuing from committed
        t2_0 = time.perf_counter()
        all_translations = translate_candidates(
            api_base, instruct_model,
            instruct_tokenizer,
            accumulated_source, futures, committed_norm,
        )
        t2 = time.perf_counter() - t2_0
        timing_totals["step2_translate_candidates_s"] += t2
        _vlog(verbose_log_file,
              f"  [Step 2] {len(all_translations)} translations ({t2:.3f}s):")
        for ti, t in enumerate(all_translations):
            _vlog(verbose_log_file, f"    {ti}: \"{t}\"")

        # Step 3: Alignment-truncate all candidates.
        t3_0 = time.perf_counter()
        t3_truncate_sum = 0.0
        t3_align_model_sum = 0.0
        candidate_infos: List[Dict[str, Any]] = []
        current_sent_idx_ctx: Optional[int] = None
        local_obs_src_ctx = ""
        sent_translation_ctx = ""
        sent_translation_sent = ""
        coverage = 0.0
        last_word = ""
        sentence_ctx_available = bool(
            sentence_translations and chunk_sentence_ids and chunk_pos < len(chunk_sentence_ids)
        )
        if sentence_ctx_available:
            current_sent_idx_ctx = chunk_sentence_ids[chunk_pos]
            all_obs_words = accumulated_source.strip().split()
            i = current_sent_idx_ctx
            start_w = sent_word_start[i] if i < len(sent_word_start) else 0
            end_w = min(len(all_obs_words), sent_word_end[i]) if i < len(sent_word_end) else len(all_obs_words)
            local_obs_words = all_obs_words[start_w:end_w]
            local_obs_src_ctx = " ".join(local_obs_words).strip()
            if not local_obs_src_ctx and all_obs_words:
                local_obs_src_ctx = all_obs_words[-1]
            sent_translation_sent = sentence_translations[current_sent_idx_ctx] if current_sent_idx_ctx < len(sentence_translations) else ""
            sent_translation_ctx = sent_translation_sent

            sent_total_words = len(sentences[current_sent_idx_ctx].strip().split()) if current_sent_idx_ctx < len(sentences) else 1
            local_obs_count_for_gate = len(local_obs_src_ctx.strip().split()) if local_obs_src_ctx.strip() else 0
            coverage = local_obs_count_for_gate / max(1, sent_total_words)
            last_word = local_obs_src_ctx.strip().split()[-1].lower() if local_obs_src_ctx.strip() else ""

        use_sentence_path = sentence_ctx_available
        if getattr(args, "disable_sentence_path", False):
            if use_sentence_path:
                _vlog(verbose_log_file, "  [Step 3 sentence-scoped] DISABLED by --disable-sentence-path")
            use_sentence_path = False
        if use_sentence_path:
            if coverage < _SENTENCE_COVERAGE_GATE:
                _vlog(verbose_log_file, f"  [Step 3 sentence-scoped] SKIP: coverage={coverage:.2f} < {_SENTENCE_COVERAGE_GATE} (sent_idx={current_sent_idx_ctx})")
                use_sentence_path = False
            elif last_word in _BAD_LAST_WORDS:
                _vlog(verbose_log_file, f"  [Step 3 sentence-scoped] SKIP: last_word=\"{last_word}\" in be-verb stoplist (sent_idx={current_sent_idx_ctx})")
                use_sentence_path = False

        if use_sentence_path:
            sent_translation_log = (sent_translation_sent[:80] + "…") if len(sent_translation_sent) > 80 else sent_translation_sent
            _vlog(verbose_log_file, f"  [Step 3 sentence-scoped] current_sent_idx={current_sent_idx_ctx} local_obs_src=\"{local_obs_src_ctx}\" sent_translation=\"{sent_translation_log}\" coverage={coverage:.2f}")

            t3a_0 = time.perf_counter()
            alignments_sent = get_word_alignments(
                local_obs_src_ctx, sent_translation_sent, align_model, align_tokenizer
            )
            t3_align_model_sum = time.perf_counter() - t3a_0
            if _VERBOSE_ALIGNMENT_DEBUG and getattr(args, "align_method", "awesome_align") == "simalign":
                _emit_simalign_alignment_debug(
                    verbose_log_file,
                    "  [Step 3 sentence alignments]",
                    local_obs_src_ctx,
                    local_obs_src_ctx,
                    sent_translation_sent,
                    alignments_sent,
                )
            full_src_sent = sentences[current_sent_idx_ctx] if current_sent_idx_ctx < len(sentences) else local_obs_src_ctx
            t3b_0 = time.perf_counter()
            local_safe_prefix_sent = truncate_by_alignment(
                full_src_sent, local_obs_src_ctx, sent_translation_sent, alignments_sent
            )
            t3_truncate_sum = time.perf_counter() - t3b_0
            already = committed_sentence_prefix_lens[current_sent_idx_ctx] if current_sent_idx_ctx < len(committed_sentence_prefix_lens) else 0
            new_in_sent = local_safe_prefix_sent[already:] if len(local_safe_prefix_sent) > already else ""
            _vlog(verbose_log_file,
                  f"  [Step 3 sentence] local_safe_prefix_sent=\"{local_safe_prefix_sent[:60]}{'…' if len(local_safe_prefix_sent) > 60 else ''}\" already={already} new_in_sent_len={len(new_in_sent)}")

            local_obs_count_sent = len(local_obs_src_ctx.strip().split()) if local_obs_src_ctx.strip() else 0
            alignment_ok = True
            if local_obs_count_sent > 0:
                need_src_idx = local_obs_count_sent - 1
                has_last = any(s_idx == need_src_idx for s_idx, _ in alignments_sent)
                if not has_last:
                    alignment_ok = False
                else:
                    last_t = [t for s, t in alignments_sent if s == need_src_idx]
                    if last_t:
                        spread = max(last_t) - min(last_t)
                        t_idx_used = sorted(last_t)[len(last_t) // 2]
                        if spread > _ALIGNMENT_SPREAD_THRESHOLD or t_idx_used > len(sent_translation_sent) - _ALIGNMENT_VERY_END_MARGIN:
                            alignment_ok = False

            length_ok_sent = len(new_in_sent) >= args.min_commit_chars
            if alignment_ok and length_ok_sent:
                new_chars = strip_committed_suffix_from_delta(committed_norm, new_in_sent)
                risky = _ends_on_word_head(new_chars)
                t3 = time.perf_counter() - t3_0
                timing_totals["step3_alignment_total_s"] += t3
                timing_totals["step3_alignment_model_s"] += t3_align_model_sum
                timing_totals["step3_truncate_s"] += t3_truncate_sum
                if len(new_chars) >= args.min_commit_chars and not risky:
                    committed_norm = committed_norm + new_chars
                    decisions.append(("WRITE", new_chars))
                    if sentence_translations and chunk_sentence_ids and chunk_pos < len(chunk_sentence_ids):
                        sid = chunk_sentence_ids[chunk_pos]
                        if sid < len(committed_sentence_prefix_lens):
                            prefix_before = "".join(sentence_translations[:sid])
                            chars_in_sent = len(committed_norm) - len(prefix_before)
                            committed_sentence_prefix_lens[sid] = max(
                                committed_sentence_prefix_lens[sid],
                                max(0, chars_in_sent),
                            )
                    _vlog(verbose_log_file,
                          f"  -> WRITE (sentence-path) \"{new_chars}\"  committed=\"{committed_norm}\"")
                else:
                    decisions.append(("READ", ""))
                    if risky:
                        _vlog(verbose_log_file,
                              f"  -> READ (sentence-path word_head_guard: ends on '{new_chars[-1] if new_chars else ''}')")
                    else:
                        _vlog(verbose_log_file,
                              f"  -> READ (sentence-path new_chars len={len(new_chars)} < min={args.min_commit_chars})")
                chunk_elapsed = time.perf_counter() - chunk_t0
                timing_totals["chunk_total_s"] += chunk_elapsed
                _vlog(verbose_log_file,
                      f"  [Timing] step1={t1:.3f}s step2={t2:.3f}s step3={t3:.3f}s chunk_total={chunk_elapsed:.3f}s (sentence-path)")
                if args.save_details:
                    all_details.append({
                        "observed": accumulated_source,
                        "futures": futures,
                        "translations": all_translations,
                        "new_chars": new_chars if decisions[-1][0] == "WRITE" else "",
                        "committed_after": committed_norm,
                        "action": decisions[-1][0],
                        "current_sent_idx": current_sent_idx_ctx,
                        "local_obs_src": local_obs_src_ctx,
                        "sent_translation": sent_translation_ctx,
                        "reason": "sentence_path",
                    })
                continue
            else:
                _vlog(verbose_log_file, f"  [Step 3 sentence] alignment_ok={alignment_ok} length_ok={length_ok_sent} -> fallback to windowed path")
                use_sentence_path = False

        if not use_sentence_path:
            use_sentence_local_alignment = (
                getattr(args, "align_method", "awesome_align") == "simalign"
                and current_sent_idx_ctx is not None
                and current_sent_idx_ctx < len(sentences)
                and bool(local_obs_src_ctx.strip())
                and bool(sentence_translations)
            )
            if use_sentence_local_alignment:
                _vlog(
                    verbose_log_file,
                    f"  [Step 3 sentence-local] using strict per-sentence alignment (sent_idx={current_sent_idx_ctx})",
                )
            else:
                _vlog(verbose_log_file, "  [Step 3] falling back to windowed alignment")

            # Fallback: strict per-sentence alignment for simalign; otherwise windowed alignment
            full_srcs = [
                (accumulated_source + " " + future).strip() if future else accumulated_source
                for future in futures
            ]
            local_views: List[Tuple[str, str, str, int]] = []
            alignment_pairs: List[Tuple[str, str]] = []
            alignment_srcs: List[str] = []
            for full_src_for_candidate, translation in zip(full_srcs, all_translations):
                if use_sentence_local_alignment:
                    prefix_before = "".join(sentence_translations[:current_sent_idx_ctx])
                    sentence_start_word = sent_word_start[current_sent_idx_ctx] if current_sent_idx_ctx < len(sent_word_start) else 0
                    local_full_src, local_observed_src = _build_sentence_local_source_view(
                        full_src_for_candidate,
                        accumulated_source,
                        sentence_start_word,
                    )
                    if not local_observed_src:
                        local_observed_src = local_obs_src_ctx
                    if not local_full_src:
                        local_full_src = local_observed_src
                    tgt_offset = _estimate_sentence_local_target_offset(
                        translation,
                        prefix_before,
                        committed_norm,
                    )
                    local_translation = translation[tgt_offset:]
                    if not local_translation:
                        local_translation = translation[tgt_offset:]
                else:
                    local_full_src, local_observed_src, local_translation, tgt_offset = build_local_alignment_windows(
                        full_src_for_candidate,
                        accumulated_source,
                        translation,
                        committed_norm,
                    )
                local_views.append((local_full_src, local_observed_src, local_translation, tgt_offset))
                alignment_src = local_full_src if use_sentence_local_alignment else local_observed_src
                alignment_srcs.append(alignment_src)
                alignment_pairs.append((alignment_src, local_translation))

            t3a_0 = time.perf_counter()
            batch_alignments = get_word_alignments_batch(
                alignment_pairs,
                align_model, align_tokenizer,
            )
            t3_align_model_sum += time.perf_counter() - t3a_0

            for ci, (future, translation, full_src_for_candidate, alignments, local_view) in enumerate(
                zip(futures, all_translations, full_srcs, batch_alignments, local_views)
            ):
                local_full_src, local_observed_src, local_translation, tgt_offset = local_view
                t3b_0 = time.perf_counter()
                local_safe_prefix = truncate_by_alignment(
                    local_full_src, local_observed_src,
                    local_translation, alignments,
                )
                safe_end = max(0, min(len(translation), tgt_offset + len(local_safe_prefix)))
                safe_prefix = translation[:safe_end]
                if committed_norm and translation.startswith(committed_norm) and len(safe_prefix) < len(committed_norm):
                    safe_prefix = committed_norm
                if sentence_translations and current_sent_idx_ctx is not None and current_sent_idx_ctx < len(sentence_translations):
                    prefix_before = "".join(sentence_translations[:current_sent_idx_ctx])
                    max_total_len = len(prefix_before) + len(sentence_translations[current_sent_idx_ctx])
                    if max_total_len >= len(committed_norm) and len(safe_prefix) > max_total_len:
                        safe_prefix = safe_prefix[:max_total_len]
                t3_truncate_sum += (time.perf_counter() - t3b_0)

                monotonic_ok = (not committed_norm) or safe_prefix.startswith(committed_norm)
                delta = ""
                if monotonic_ok and safe_prefix and len(safe_prefix) > len(committed_norm):
                    delta = safe_prefix[len(committed_norm):]
                length_ok = len(delta) >= args.min_commit_chars

                # Alignment quality gate: use last CONTENT word of observed src,
                # skipping function words that BERT aligns unreliably.
                # If the content word is not aligned: fall through to truncate_by_alignment's
                # own fallback (safe_tgt_idx / ratio) rather than zeroing delta.
                local_obs_words = local_observed_src.strip().split() if local_observed_src.strip() else []
                local_obs_count = len(local_obs_words)
                if local_obs_count > 0:
                    # Find last content word index (skip trailing function words)
                    content_src_idx = None
                    for _wi in range(local_obs_count - 1, -1, -1):
                        if local_obs_words[_wi].lower().rstrip(".,;:!?") not in _BAD_LAST_WORDS:
                            content_src_idx = _wi
                            break
                    # If no content word found, use the last word (best effort)
                    if content_src_idx is None:
                        content_src_idx = local_obs_count - 1

                    content_t_list = [t for s, t in alignments if s == content_src_idx]
                    if content_t_list:
                        # Content word is aligned: apply spread + very-end quality checks.
                        # Failure → do NOT zero delta, let truncate_by_alignment fallback stand.
                        spread = max(content_t_list) - min(content_t_list)
                        t_idx_used = sorted(content_t_list)[len(content_t_list) // 2]
                        if spread > _ALIGNMENT_SPREAD_THRESHOLD:
                            # Spread too large: alignment noisy, trust fallback already in safe_prefix
                            pass  # keep delta from truncate_by_alignment
                        elif t_idx_used > len(local_translation) - _ALIGNMENT_VERY_END_MARGIN:
                            # Content word mapped to very end: sentence-end attraction,
                            # safe_prefix is likely over-truncated; keep whatever truncate gave us
                            pass  # keep delta from truncate_by_alignment
                    # If content word not aligned at all: also keep delta from truncate_by_alignment

                candidate_infos.append({
                    "idx": ci,
                    "future": future,
                    "translation": translation,
                    "full_src_for_candidate": full_src_for_candidate,
                    "alignment_src": alignment_srcs[ci],
                    "local_full_src": local_full_src,
                    "local_observed_src": local_observed_src,
                    "local_translation": local_translation,
                    "tgt_offset": tgt_offset,
                    "alignments": [(s, t) for s, t in alignments],
                    "safe_prefix": safe_prefix,
                    "delta": delta,
                    "monotonic_ok": monotonic_ok,
                    "length_ok": length_ok,
                })
        t3 = time.perf_counter() - t3_0
        timing_totals["step3_alignment_total_s"] += t3
        timing_totals["step3_alignment_model_s"] += t3_align_model_sum
        timing_totals["step3_truncate_s"] += t3_truncate_sum

        _vlog(verbose_log_file,
              f"  [Step 3] alignment-truncated candidates ({t3:.3f}s, "
              f"align_model={t3_align_model_sum:.3f}s, truncate={t3_truncate_sum:.3f}s):")
        for c in candidate_infos:
            _vlog(
                verbose_log_file,
                f'    {c["idx"]}: monotonic={c["monotonic_ok"]} '
                f'len_ok={c["length_ok"]} delta_len={len(c["delta"])} '
                f'safe_prefix="{c["safe_prefix"]}"'
            )
            if _VERBOSE_ALIGNMENT_DEBUG and getattr(args, "align_method", "awesome_align") == "simalign":
                _emit_simalign_alignment_debug(
                    verbose_log_file,
                    f'      [candidate {c["idx"]}]',
                    c.get("alignment_src", ""),
                    c.get("local_observed_src", ""),
                    c.get("local_translation", ""),
                    c["alignments"],
                    tgt_offset=c.get("tgt_offset"),
                )

        valid_candidates = [c for c in candidate_infos if c["length_ok"]]
        if not valid_candidates:
            decisions.append(("READ", ""))
            _vlog(verbose_log_file,
                  "  -> READ (no alignment-safe candidate exceeds min_commit_chars)")
            chunk_elapsed = time.perf_counter() - chunk_t0
            timing_totals["chunk_total_s"] += chunk_elapsed
            _vlog(verbose_log_file,
                  f"  [Timing] chunk_total={chunk_elapsed:.3f}s (no_valid_truncated_candidate)")
            if args.save_details:
                d = {
                    "observed": accumulated_source,
                    "futures": futures,
                    "translations": all_translations,
                    "candidates": candidate_infos,
                    "reason": "no_valid_truncated_candidate",
                    "action": "READ",
                }
                if current_sent_idx_ctx is not None:
                    d["current_sent_idx"] = current_sent_idx_ctx
                    d["local_obs_src"] = local_obs_src_ctx
                    d["sent_translation"] = sent_translation_ctx
                all_details.append(d)
            continue

        new_chars = ""
        t4 = 0.0
        candidate_safe_prefixes = [c["safe_prefix"] for c in valid_candidates]
        selection_mode = getattr(args, "selection_mode", "majority_vote")
        global_quorum_k = max(1, math.ceil(args.consensus_ratio * len(candidate_infos)))
        t4_0 = time.perf_counter()
        if selection_mode == "lcp_code":
            if len(valid_candidates) < 2:
                new_chars = ""
                _vlog(verbose_log_file, f"  [Step 4] fewer than 2 valid candidates, READ.")
            else:
                deltas_for_check = [c["delta"] for c in valid_candidates if c.get("delta")]
                if len(deltas_for_check) < 2:
                    new_chars = ""
                    _vlog(verbose_log_file, f"  [Step 4] too few deltas for direction check, READ.")
                else:
                    is_consistent, dir_info = check_direction(deltas_for_check, n=3, min_ratio=0.35)
                    if not is_consistent:
                        new_chars = ""
                        _vlog(verbose_log_file, f"  [Step 4] direction inconsistent, READ. {dir_info}")
                    else:
                        new_chars = get_quorum_lcp_delta_code(
                            committed_norm, candidate_safe_prefixes, consensus_ratio=1.0
                        )
            step4_tag = "LCPCode100"
        elif selection_mode == "lcp70_code":
            if len(valid_candidates) < 2:
                new_chars = ""
                _vlog(verbose_log_file, f"  [Step 4] fewer than 2 valid candidates, READ.")
            else:
                deltas_for_check = [c["delta"] for c in valid_candidates if c.get("delta")]
                if len(deltas_for_check) < 2:
                    new_chars = ""
                    _vlog(verbose_log_file, f"  [Step 4] too few deltas for direction check, READ.")
                else:
                    is_consistent, dir_info = check_direction(deltas_for_check, n=3, min_ratio=0.35)
                    if not is_consistent:
                        new_chars = ""
                        _vlog(verbose_log_file, f"  [Step 4] direction inconsistent, READ. {dir_info}")
                    else:
                        new_chars = get_quorum_lcp_delta_code(
                            committed_norm, candidate_safe_prefixes, consensus_ratio=args.consensus_ratio
                        )
            step4_tag = f"LCPCode{int(round(args.consensus_ratio * 100))}"
        elif selection_mode == "lcp70_llm":
            new_chars = get_lcp70_llm_delta_via_llm(
                api_base, instruct_model, instruct_tokenizer,
                accumulated_source, committed_norm, candidate_safe_prefixes,
                consensus_ratio=args.consensus_ratio,
                with_boundary_rule=True,
            )
            step4_tag = f"LLMQuorumBoundary{int(round(args.consensus_ratio * 100))}"
        elif selection_mode == "majority_vote":
            if len(valid_candidates) < global_quorum_k:
                new_chars = ""
                _vlog(
                    verbose_log_file,
                    f"  [Step 4] global quorum failed for majority_vote: "
                    f"valid={len(valid_candidates)} < required={global_quorum_k} "
                    f"from total_candidates={len(candidate_infos)}",
                )
            else:
                new_chars = get_majority_vote_delta_via_llm(
                    majority_vote_api_base, majority_vote_model, majority_vote_tokenizer,
                    accumulated_source, committed_norm, candidate_safe_prefixes,
                    consensus_ratio=args.consensus_ratio,
                    disable_backoff=getattr(args, "majority_vote_disable_backoff", False),
                    prompt_version=getattr(args, "majority_vote_prompt_version", "v2"),
                    use_reasoning=bool(getattr(args, "majority_vote_use_reasoning", False)),
                    reasoning_temperature=float(getattr(args, "majority_vote_reasoning_temperature", 0.6)),
                    reasoning_max_tokens=int(getattr(args, "majority_vote_reasoning_max_tokens", 512)),
                )
            step4_tag = (
                f"MajorityVote-{getattr(args, 'majority_vote_prompt_version', 'v2')}-"
                f"{int(round(args.consensus_ratio * 100))}"
            )
        else:
            raise ValueError(f"Unsupported selection_mode after canonicalization: {selection_mode}")
        t4 = time.perf_counter() - t4_0
        timing_totals["step4_majority_vote_s"] += t4
        _vlog(verbose_log_file,
              f"  [Step 4 {step4_tag}] delta ({t4:.3f}s): \"{new_chars}\"")

        new_chars = strip_committed_suffix_from_delta(committed_norm, new_chars)
        repeat_guard = False
        sentence_start_guard = False
        if new_chars and selection_mode == "majority_vote":
            candidate_commit = committed_norm + new_chars
            repeat_guard = (
                _has_repeated_substring_pattern(new_chars)
                or _has_repeated_substring_pattern(candidate_commit)
            )
            local_obs_count = len(local_obs_src_ctx.strip().split()) if local_obs_src_ctx.strip() else 0
            if current_sent_idx_ctx is not None and local_obs_count <= 2:
                raw_support = _count_prefix_support(new_chars, candidate_safe_prefixes, normalize_leading=False)
                norm_support = _count_prefix_support(new_chars, candidate_safe_prefixes, normalize_leading=True)
                if max(raw_support, norm_support) < global_quorum_k:
                    sentence_start_guard = True
                    _vlog(
                        verbose_log_file,
                        f"  [Step 4 guard] sentence-start guard: support={max(raw_support, norm_support)} "
                        f"< required={global_quorum_k} for new_chars=\"{new_chars}\"",
                    )
            if repeat_guard:
                _vlog(
                    verbose_log_file,
                    f"  [Step 4 guard] repeated-pattern guard fired for new_chars=\"{new_chars}\"",
                )
        risky = _ends_on_word_head(new_chars)
        if len(new_chars) >= args.min_commit_chars and not risky and not repeat_guard and not sentence_start_guard:
            committed_norm = committed_norm + new_chars
            decisions.append(("WRITE", new_chars))
            if sentence_translations and chunk_sentence_ids and chunk_pos < len(chunk_sentence_ids):
                sid = chunk_sentence_ids[chunk_pos]
                if sid < len(committed_sentence_prefix_lens):
                    prefix_before = "".join(sentence_translations[:sid])
                    chars_in_sent = len(committed_norm) - len(prefix_before)
                    committed_sentence_prefix_lens[sid] = max(
                        committed_sentence_prefix_lens[sid],
                        max(0, chars_in_sent),
                    )
            _vlog(verbose_log_file,
                  f"  -> WRITE \"{new_chars}\"  committed=\"{committed_norm}\"")
        else:
            decisions.append(("READ", ""))
            if repeat_guard:
                _vlog(verbose_log_file, "  -> READ (repeat_pattern_guard)")
            elif sentence_start_guard:
                _vlog(verbose_log_file, "  -> READ (sentence_start_guard)")
            elif risky:
                _vlog(verbose_log_file,
                      f"  -> READ (word_head_guard: ends on '{new_chars[-1] if new_chars else ''}')")
            else:
                _vlog(verbose_log_file,
                      f"  -> READ (new_chars len={len(new_chars)} < min={args.min_commit_chars})")
        chunk_elapsed = time.perf_counter() - chunk_t0
        timing_totals["chunk_total_s"] += chunk_elapsed
        _vlog(verbose_log_file,
              f"  [Timing] step1={t1:.3f}s step2={t2:.3f}s step3={t3:.3f}s "
              f"step4={t4:.3f}s chunk_total={chunk_elapsed:.3f}s")

        if args.save_details:
            detail: Dict[str, Any] = {
                "observed": accumulated_source,
                "futures": futures,
                "translations": all_translations,
                "candidates": candidate_infos,
                "valid_candidate_indices": [c["idx"] for c in valid_candidates],
                "new_chars": new_chars,
                "committed_after": committed_norm,
                "action": decisions[-1][0],
            }
            detail["selection_mode"] = selection_mode
            if current_sent_idx_ctx is not None:
                detail["current_sent_idx"] = current_sent_idx_ctx
                detail["local_obs_src"] = local_obs_src_ctx
                detail["sent_translation"] = sent_translation_ctx
            all_details.append(detail)

    # ---- Assemble output ----
    target_deltas = [d[1] for d in decisions]
    action_list = [d[0] for d in decisions]
    system_output_text = "".join(d for d in target_deltas if d)

    laal_reference_text = ""
    laal_value = float("nan")
    laal_error: Optional[str] = None
    bleu_char_value = float("nan")
    bleu_char_error: Optional[str] = None
    laal_reference_mode = "llm_full_translation"
    try:
        cached_translation = (translation_cache or {}).get(utt_id)
        if cached_translation:
            laal_reference_text = cached_translation
            laal_reference_mode = "cache"
        else:
            laal_reference_text = translate_final(
                api_base, instruct_model, instruct_tokenizer,
                full_source_text, "",
            )
        laal_value = compute_laal(
            trajectory,
            target_deltas,
            action_list,
            laal_reference_text,
        )
        bleu_char_value = compute_bleu_char(
            system_output_text,
            laal_reference_text,
        )
    except Exception as e:
        laal_error = str(e)
        bleu_char_error = str(e)

    result: Dict[str, Any] = {
        "utt_id": utt_id,
        "original_text": full_source_text,
        "input_sentences": sentences,
        "source_future_sampling": trajectory,
        "target_future_sampling": target_deltas,
        "actions": action_list,
        "laal_reference_text": laal_reference_text,
        "metrics": {
            "laal_text": laal_value,
            "laal_reference_mode": laal_reference_mode,
            "bleu_char": bleu_char_value,
            "bleu_reference_mode": laal_reference_mode,
            "effective_source_chunks": sum(1 for c in trajectory if str(c).strip()),
            "system_output_chars": len(system_output_text),
            "reference_chars": len(laal_reference_text.replace(" ", "")) if laal_reference_text else 0,
            "laal_error": laal_error,
            "bleu_char_error": bleu_char_error,
        },
        "config": {
            "version": f"final_dual_model_{getattr(args, 'selection_mode', 'majority_vote')}",
            "selection_mode": getattr(args, "selection_mode", "majority_vote"),
            "patches": [
                "local_alignment_window",
                "word_head_guard",
                "prefix_selection_hard_validation",
                "monotonic_alignment",  # NEW: _make_monotonic applied to awesome-align output
            ],
            "num_candidates": args.num_candidates,
            "future_tokens": args.future_tokens,
            "min_commit_chars": args.min_commit_chars,
            "min_observed_words": args.min_observed_words,
            "consensus_ratio": args.consensus_ratio,
            "disable_sentence_path": bool(getattr(args, "disable_sentence_path", False)),
            "base_model": args.base_model_path,
            "instruct_model": args.instruct_model_name,
            "majority_vote_use_reasoning": bool(getattr(args, "majority_vote_use_reasoning", False)),
            "majority_vote_reasoning_model": (
                args.majority_vote_reasoning_model_name
                if getattr(args, "majority_vote_use_reasoning", False)
                else None
            ),
            "majority_vote_reasoning_temperature": (
                args.majority_vote_reasoning_temperature
                if getattr(args, "majority_vote_use_reasoning", False)
                else None
            ),
            "majority_vote_reasoning_max_tokens": (
                args.majority_vote_reasoning_max_tokens
                if getattr(args, "majority_vote_use_reasoning", False)
                else None
            ),
        },
    }
    result["timing"] = {
        **timing_totals,
        "n_chunks": n_chunks,
        "n_nonempty_chunks": sum(1 for c in trajectory if str(c).strip()),
    }
    _vlog(
        verbose_log_file,
        "[Timing Summary] " + " ".join(
            f"{k}={v:.3f}s" for k, v in result["timing"].items() if isinstance(v, float)
        ),
    )
    for k in ["audio", "n_frames", "speaker", "src_lang", "tgt_lang"]:
        if k in row:
            result[k] = row[k]
    if args.save_details:
        result["details"] = all_details
    return result


# ===================================================================
# Data-Parallel I/O
# ===================================================================

def iter_assigned_rows(input_tsv: str, task_id: int, num_tasks: int):
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if row_idx % num_tasks != task_id:
                continue
            yield row_idx, row


def count_assigned_rows(input_tsv: str, task_id: int, num_tasks: int) -> int:
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        total = sum(1 for _ in f) - 1
    if total <= task_id:
        return 0
    return int(math.ceil((total - task_id) / num_tasks))


def get_one_row_by_id(
    input_tsv: str, utt_id: str, id_column: str = "id",
) -> Optional[Tuple[int, Dict[str, str]]]:
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if str(row.get(id_column, "")).strip() == str(utt_id).strip():
                return row_idx, row
    return None


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    args = parse_args()
    setup_env()

    # simalign override (kept for backward-compat with --align-method simalign)
    if getattr(args, "align_method", "awesome_align") == "simalign":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        core_v2_path = os.path.join(script_dir, "llm_future_sampling_core_v2.py")
        import importlib.util
        spec = importlib.util.spec_from_file_location("llm_future_sampling_core_v2", core_v2_path)
        core_v2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_v2)
        global load_align_model, get_word_alignments, truncate_by_alignment
        global get_word_alignments_batch, build_local_alignment_windows
        load_align_model = core_v2.load_align_model
        truncate_by_alignment = core_v2.truncate_by_alignment
        build_local_alignment_windows = core_v2.build_local_alignment_windows
        # Wrap alignment calls so they use _align_model_lock when parallel_utterances > 1.
        _orig_get_word_alignments_v2 = core_v2.get_word_alignments
        _orig_get_word_alignments_batch_v2 = core_v2.get_word_alignments_batch

        def _wrapped_get_word_alignments(src_text, tgt_text, align_model, align_tokenizer):
            if _align_model_lock is not None:
                with _align_model_lock:
                    return _orig_get_word_alignments_v2(src_text, tgt_text, align_model, align_tokenizer)
            return _orig_get_word_alignments_v2(src_text, tgt_text, align_model, align_tokenizer)

        def _wrapped_get_word_alignments_batch(pairs, align_model, align_tokenizer):
            if _align_model_lock is not None:
                with _align_model_lock:
                    return _orig_get_word_alignments_batch_v2(pairs, align_model, align_tokenizer)
            return _orig_get_word_alignments_batch_v2(pairs, align_model, align_tokenizer)

        get_word_alignments = _wrapped_get_word_alignments
        get_word_alignments_batch = _wrapped_get_word_alignments_batch
        print("[Align] Using simalign (core_v2).")

    os.makedirs(args.output_root, exist_ok=True)

    api_base = args.instruct_api_base
    instruct_model = args.instruct_model_name

    # Verify instruct server is reachable
    try:
        client = _make_sync_client(api_base)
        models = client.models.list()
        print(f"[Instruct] Server OK: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"ERROR: Cannot connect to instruct server at {api_base}: {e}")
        print("Start it first:  bash test_instruct_serve.sh")
        sys.exit(1)

    # Load word-alignment model first (so vLLM can account for its memory)
    align_dev = getattr(args, "align_device", "cuda:0")
    print(f"[Align] Loading align model on {align_dev} ...")
    align_model, align_tokenizer = load_align_model(
        cache_dir=os.environ.get("HF_HOME"),
        device=align_dev,
    )
    print("[Align] Model loaded.")

    # Load instruct tokenizer locally
    print(f"[Instruct] Loading tokenizer from {args.instruct_tokenizer_path} ...")
    instruct_tokenizer = load_instruct_tokenizer(
        args.instruct_tokenizer_path,
        cache_dir=os.environ.get("HF_HOME"),
    )
    print("[Instruct] Tokenizer loaded.")

    majority_vote_api_base = api_base
    majority_vote_model = instruct_model
    majority_vote_tokenizer = instruct_tokenizer
    if getattr(args, "majority_vote_use_reasoning", False):
        majority_vote_api_base = args.majority_vote_reasoning_api_base or api_base
        majority_vote_model = args.majority_vote_reasoning_model_name
        try:
            mv_client = _make_sync_client(majority_vote_api_base)
            mv_models = mv_client.models.list()
            print(f"[MajorityVote Reasoning] Server OK: {[m.id for m in mv_models.data]}")
        except Exception as e:
            print(f"ERROR: Cannot connect to majority-vote reasoning server at {majority_vote_api_base}: {e}")
            sys.exit(1)
        print(
            f"[MajorityVote Reasoning] Loading tokenizer from "
            f"{args.majority_vote_reasoning_tokenizer_path} ..."
        )
        majority_vote_tokenizer = load_instruct_tokenizer(
            args.majority_vote_reasoning_tokenizer_path,
            cache_dir=os.environ.get("HF_HOME"),
        )
        print("[MajorityVote Reasoning] Tokenizer loaded.")

    # Load base model
    print(f"[Base] Loading {args.base_model_path} (TP={args.tp}) ...")
    base_llm = LLM(
        model=args.base_model_path,
        dtype="auto",
        tensor_parallel_size=args.tp,
        max_model_len=4096,
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.85),
        enforce_eager=True,
    )
    print("[Base] Model loaded.")

    # Load pre-computed translation cache
    translation_cache: Dict[str, str] = {}
    if args.translation_cache_dir:
        import glob as _glob
        jsonl_files = sorted(_glob.glob(os.path.join(args.translation_cache_dir, "task_*.jsonl")))
        print(f"[Cache] Loading translation cache from {args.translation_cache_dir} ({len(jsonl_files)} files) ...")
        for jf in jsonl_files:
            with open(jf, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        uid = str(entry.get("utt_id", "")).strip()
                        tl = entry.get("llm_full_translation", "")
                        if uid and tl:
                            translation_cache[uid] = tl
                    except Exception:
                        pass
        print(f"[Cache] Loaded {len(translation_cache)} entries.")

    # Resolve rows
    if getattr(args, "test_one", False):
        args.verbose = True
        if args.utt_id:
            one = get_one_row_by_id(args.input_tsv, args.utt_id, args.id_column)
            if one is None:
                print(f"utt-id '{args.utt_id}' not found.")
                return
            row_iter: Any = [one]
        else:
            row_iter = list(itertools.islice(
                iter_assigned_rows(args.input_tsv, args.task_id, args.num_tasks), 1
            ))
        total = len(row_iter)
        if total == 0:
            print("No rows to process.")
            return
    else:
        row_iter = iter_assigned_rows(args.input_tsv, args.task_id, args.num_tasks)
        total = count_assigned_rows(args.input_tsv, args.task_id, args.num_tasks)
        if args.max_rows is not None:
            total = min(total, args.max_rows)

    print(
        f"[Task {args.task_id}] Processing {total} rows\n"
        f"  M={args.num_candidates}, future_tokens={args.future_tokens}\n"
        f"  base={args.base_model_path}\n"
        f"  instruct={instruct_model} @ {api_base}\n"
        f"  majority_vote_reasoning={'ON' if getattr(args, 'majority_vote_use_reasoning', False) else 'OFF'}"
    )

    use_tee = getattr(args, "test_one", False) and not getattr(args, "no_tee", False)

    global _base_llm_lock, _align_model_lock, _future_sampling_request_queue, _future_sampling_worker_thread
    if args.parallel_utterances > 1:
        _align_model_lock = threading.Lock()
    batch_size = getattr(args, "future_sampling_batch_size", 4)
    batch_wait = getattr(args, "future_sampling_batch_wait", 0.05)
    if args.parallel_utterances > 1 and batch_size >= 2:
        _future_sampling_request_queue = queue_module.Queue()
        _future_sampling_worker_thread = threading.Thread(
            target=_run_batch_future_sampling_worker,
            args=(
                base_llm,
                args.num_candidates,
                args.future_tokens,
                args.sample_temperature,
                batch_size,
                batch_wait,
                _future_sampling_request_queue,
            ),
            daemon=False,
        )
        _future_sampling_worker_thread.start()
        print(f"[Parallel] {args.parallel_utterances} concurrent utterances; "
              f"future sampling batched (size={batch_size}) for GPU0.")
    elif args.parallel_utterances > 1:
        _base_llm_lock = threading.Lock()
        print(f"[Parallel] {args.parallel_utterances} concurrent utterances; "
              f"base_llm.generate() serialised via lock.")

    written = skipped = failed = 0
    _counter_lock = threading.Lock()
    _timing_list: List[Dict[str, float]] = []
    _timing_lock = threading.Lock()
    pbar = tqdm(total=total, desc=f"task_{args.task_id}")

    def _do_one_row(row_idx_row):
        row_idx, row = row_idx_row
        utt_id = str(row.get(args.id_column, "")).strip()
        if not utt_id:
            utt_id = f"row_{row_idx:09d}"
        out_path = os.path.join(
            args.output_root, f"{sanitize_filename(utt_id)}.json"
        )
        if os.path.exists(out_path) and not args.overwrite and not args.verbose:
            return "skipped", utt_id, None, out_path
        try:
            sentences = parse_list_column(row.get("src_text_full"))
            trajectory = parse_list_column(row.get("src_trajectory"))
            if not sentences:
                raise ValueError("Empty src_text_full")
            if not trajectory:
                raise ValueError("Empty src_trajectory")
            verbose_log_file = None
            if args.verbose:
                verbose_log_path = os.path.join(
                    args.output_root,
                    f"verbose_{sanitize_filename(utt_id)}.log",
                )
                raw_file = open(verbose_log_path, "w", encoding="utf-8")
                verbose_log_file = _TeeWriter(raw_file) if use_tee else raw_file
            try:
                result = process_one_utterance(
                    base_llm, api_base, instruct_model,
                    instruct_tokenizer,
                    majority_vote_api_base, majority_vote_model, majority_vote_tokenizer,
                    align_model, align_tokenizer,
                    utt_id, sentences, trajectory, row, args,
                    verbose_log_file=verbose_log_file,
                    translation_cache=translation_cache,
                )
            finally:
                if verbose_log_file is not None:
                    verbose_log_file.close()
            return "ok", utt_id, result, out_path
        except Exception as e:
            return "error", utt_id, e, out_path

    def _handle_result(action, utt_id, payload, out_path):
        nonlocal written, skipped, failed
        if action == "skipped":
            with _counter_lock:
                skipped += 1
        elif action == "ok":
            result = payload
            if result and "timing" in result:
                with _timing_lock:
                    _timing_list.append(result["timing"])
            if not (args.verbose and not args.overwrite and os.path.exists(out_path)):
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                with _counter_lock:
                    written += 1
            else:
                with _counter_lock:
                    skipped += 1
        else:
            e = payload
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"utt_id": utt_id, "error": str(e)}, f,
                          ensure_ascii=False, indent=2)
            with _counter_lock:
                failed += 1
        pbar.update(1)

    _wall_start = time.perf_counter()
    if args.parallel_utterances <= 1:
        submitted = 0
        for row_idx, row in row_iter:
            if args.max_rows is not None and submitted >= args.max_rows:
                break
            submitted += 1
            _handle_result(*_do_one_row((row_idx, row)))
    else:
        N = args.parallel_utterances
        submitted = 0
        row_source = iter(row_iter)
        in_flight: set = set()

        with ThreadPoolExecutor(max_workers=N) as pool:
            def _fill():
                nonlocal submitted
                while len(in_flight) < N * 3:
                    if args.max_rows is not None and submitted >= args.max_rows:
                        break
                    try:
                        ri, r = next(row_source)
                        submitted += 1
                        in_flight.add(pool.submit(_do_one_row, (ri, r)))
                    except StopIteration:
                        break

            _fill()
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    in_flight.discard(fut)
                    _handle_result(*fut.result())
                _fill()

    pbar.close()
    if _future_sampling_worker_thread is not None:
        _future_sampling_request_queue.put((None, None))
        _future_sampling_worker_thread.join(timeout=10.0)
        _future_sampling_request_queue = None
        _future_sampling_worker_thread = None
    _wall_elapsed = time.perf_counter() - _wall_start
    print(
        f"\n[Task {args.task_id}] Done. "
        f"written={written}, skipped={skipped}, failed={failed}"
    )
    if _timing_list and written + failed > 0:
        agg = {}
        for k in ["step1_future_sampling_s", "step2_translate_candidates_s",
                  "step3_alignment_total_s", "step3_alignment_model_s",
                  "step4_majority_vote_s", "translate_final_s", "chunk_total_s"]:
            agg[k] = sum(t.get(k, 0.0) for t in _timing_list)
        gpu0_approx = agg["step1_future_sampling_s"] + agg["step3_alignment_total_s"]
        gpu1_approx = agg["step2_translate_candidates_s"] + agg["step4_majority_vote_s"] + agg["translate_final_s"]
        n_workers = max(1, args.parallel_utterances)
        print(
            f"[Timing] wall={_wall_elapsed:.1f}s | "
            f"step1(base)={agg['step1_future_sampling_s']:.1f}s step2(instruct)={agg['step2_translate_candidates_s']:.1f}s "
            f"step3(align)={agg['step3_alignment_total_s']:.1f}s step4(vote)={agg['step4_majority_vote_s']:.1f}s "
            f"final={agg['translate_final_s']:.1f}s"
        )
        print(
            f"[GPU hint] GPU0(base+align)≈{gpu0_approx:.1f}s | GPU1(instruct)≈{gpu1_approx:.1f}s | "
            f"parallel_utterances={n_workers} → 若 GPU1 占比高可考虑 2 卡做 instruct serve"
        )


if __name__ == "__main__":
    main()
