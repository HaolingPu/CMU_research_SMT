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
  CUDA_VISIBLE_DEVICES=0 python llm_future_sampling_majority_vote.py \\
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


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Future source sampling (baseline2 direct-LLM-LCP) -- dual model."
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
                   help="Device for awesome-align (e.g. cuda:0 to share with base for speed; use cpu if OOM).")

    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)

    p.add_argument("--num-candidates", type=int, default=10,
                   help="M: number of future source samples.")
    p.add_argument("--future-tokens", type=int, default=12,
                   help="Max tokens per continuation.")
    p.add_argument("--sample-temperature", type=float, default=0.8)
    p.add_argument("--min-commit-chars", type=int, default=1)
    p.add_argument("--min-observed-words", type=int, default=2)
    p.add_argument("--score-threshold", type=int, default=80,
                   help="Deprecated for current pipeline (lcp_llm only).")
    p.add_argument("--consensus-ratio", type=float, default=0.6,
                   help="Deprecated for current pipeline (lcp_llm only).")

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
                        "base_llm.generate() calls are serialised via a lock; "
                        "HTTP (translate/judge) and alignment calls overlap freely.  "
                        "Recommended: 2-4 on a single-node setup.")
    p.add_argument("--judge-prompt-version", choices=["full", "short"], default="short",
                   help="Deprecated for current pipeline (lcp_llm only).")
    p.add_argument("--selection-mode", choices=["lcp_llm"], default="lcp_llm",
                   help="Only mode: lcp_llm (no score-consensus judge).")
    p.add_argument("--no-tee", action="store_true",
                   help="With --test-one/--verbose: write verbose log only to file, not to stdout (avoids duplicate output).")

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


def normalize_zh(text: str) -> str:
    text = unicodedata.normalize("NFC", text.strip())
    text = re.sub(r"\s+", "", text)
    return text


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
    """若译文末尾出现同一短语多次重复（模型陷入重复），截断到只保留第一段，避免整段重复污染 committed。

    例如 "……他就是编辑他就是编辑，而非作者他就是编辑他就是编辑，而非作者……" 截成 "……他就是编辑他就是编辑，而非作者"。
    """
    if not text or len(text) < 2 * min_period:
        return text
    out = text
    for p in range(min_period, min(max_period, len(out) // 2) + 1):
        # 从末尾不断去掉“重复的一整段”，直到末尾不再等于前一段
        while len(out) > 2 * p and out[-p:] == out[-2 * p : -p]:
            out = out[:-p]
    return out


# 如果模型输出把 observed 又复述了一遍，就把前缀裁掉；遇换行截断，保持单行结构
def clean_continuation(observed: str, raw_output: str, max_words: int = 15) -> str:
    text = raw_output.strip()
    # 换行会破坏 trajectory 结构，只保留第一行
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
    """If delta starts with a suffix of committed (phrase repeat), strip it so we don't double-commit.

    E.g. committed='...他就是编辑，' delta='他就是编辑，而不' -> return '而不'
    """
    if not committed or not delta:
        return delta
    for k in range(min(len(committed), len(delta)), 0, -1):
        suffix = committed[-k:]
        if delta.startswith(suffix):
            return normalize_zh(delta[len(suffix):].strip())
    return delta


_LEADING_PUNCT = re.compile(r"^[，。、；：！？\s]+")
_NOISE_PREFIXES = ("一们", "一为", "一名", "一位", "们", "为", "名", "位")


def _semantic_normalize(delta: str, key_len: int = 5) -> str:
    """Strip structural noise prefixes and return first key_len chars as direction key.

    Examples:
      "为著名的科学家"  -> "著名的科学"
      "们杰出的科学家"  -> "杰出的科学"
      "著名科学家"      -> "著名科学家"
      "名称誉卓著"      -> "称誉卓著"
      "位杰出的科学家"  -> "杰出的科学"
    """
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
    """Check whether candidate deltas point in a consistent direction.

    Uses the first n chars (or shorter if delta is shorter) as a coarse prefix key.
    Returns (is_consistent, debug_info).
对每个 delta 先 _semantic_normalize()：
去掉开头标点
去掉一些结构噪音前缀（"一们","一为","一名","一位","们","为","名","位"）
取前 n=3 个字符作为 key
看 top_key 的占比是否 >= 0.5

取每个 delta 的“语义化前 n 个字”，做 majority vote，
如果最多的那个前缀 ≥ 50%，认为方向一致。
    """
    keys = []
    for d in deltas:
        d = (d or "").strip()
        if not d:
            continue
        sem_key = _semantic_normalize(d, key_len=n) # 去掉开头标点 and  去掉结构噪音前缀
        #eg： 因为 LLM continuation 常出现：位杰出的科学家； 为著名的科学家； 一位杰出的科学家
        key = sem_key if sem_key else d[: min(n, len(d))]  # 取前 n 个字符作为 key
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
    timeline: List[int] = [] # timeline[i] = 第 i+1 个输出字符产生时，系统已读了多少源词。
    source_read = 0

    for chunk, delta, action in zip(source_chunks, target_deltas, actions):
        words_in_chunk = len(str(chunk).strip().split()) if str(chunk).strip() else 0
        source_read += words_in_chunk
        if action == "WRITE" and str(delta).strip():
            for _ in str(delta).strip():
                timeline.append(source_read) # timeline[i] = 第 i+1 个输出字符产生时，系统已读了多少源词。

    y = "".join(d for d in target_deltas if d)
    y_len = len(y) # 系统输出总长度（把所有 target_deltas 拼接后长度）
    yref_len = len(str(reference).replace(" ", ""))
    x_len = sum(  # 源文本总词数
        len(str(c).strip().split())
        for c in source_chunks
        if str(c).strip()
    )

    if y_len == 0 or x_len == 0 or yref_len == 0:
        return float("nan")

    denom = max(y_len, yref_len)  # laal 关键改动
    if denom <= 0 or len(timeline) == 0:
        return float("nan")
    # Per Papi et al. 2022 (LAAL): sum from i=1 to max(|Y|,|Y*|).
    # For positions i > |Y| (system output too short), d(i) = x_len (fully delayed).
    # Divide by denom = max(|Y|, |Y*|).
    total_lagging = 0.0
    for i in range(1, denom + 1):
        # 如果系统输出太短（i 超出 timeline），用 x_len（表示“拖到最后才出”）
        d_i = timeline[i - 1] if i <= len(timeline) else x_len 
        d_star_i = (i - 1) * x_len / denom
        total_lagging += (d_i - d_star_i)

    return total_lagging / denom  # 最后除以 denom 得到平均延迟。


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
# Word Alignment  (awesome-align, CPU-only)
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


def get_word_alignments(
    src_text: str,
    tgt_text: str,
    align_model,
    align_tokenizer,
) -> List[Tuple[int, int]]:
    """Extract word-level alignment pairs.

    src_text is split on whitespace (English words).
    tgt_text is split per character (Chinese characters).
    Returns list of (src_word_idx, tgt_char_idx) pairs using
    bidirectional argmax intersection, falling back to forward-only.
    """
    src_words = src_text.strip().split()  # 按空格切成词
    tgt_chars = list(tgt_text.strip().replace(" ", ""))  # 按“单个汉字”切字符

    if not src_words or not tgt_chars:
        return []

    src_subwords = [align_tokenizer.tokenize(w) for w in src_words]
    tgt_subwords = [align_tokenizer.tokenize(c) for c in tgt_chars]
    src_subwords = [sw if sw else [align_tokenizer.unk_token] for sw in src_subwords]
    tgt_subwords = [sw if sw else [align_tokenizer.unk_token] for sw in tgt_subwords]

    # awesome-align (BERT) 单次前向最多支持 512 positions:
    # [CLS] + src + [SEP] + tgt + [SEP] -> src+tgt budget = max_pos-3
    max_pos = int(getattr(getattr(align_model, "config", None), "max_position_embeddings", 512) or 512)
    if max_pos <= 3:
        max_pos = 512
    per_pass_budget = max_pos - 3
    max_joint_subwords = 1024  # user requested upper limit

    def _align_single(
        src_sw: List[List[str]],
        tgt_sw: List[List[str]],
    ) -> List[Tuple[int, int]]:
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
            (s, fwd[s].item())
            for s in range(len(src_sw))
            if bwd[fwd[s].item()].item() == s
        ]
        if inter:
            return inter
        return [(s, fwd[s].item()) for s in range(len(src_sw))]

    src_token_lens = [len(sw) for sw in src_subwords]
    tgt_token_lens = [len(sw) for sw in tgt_subwords]
    src_total = sum(src_token_lens)
    tgt_total = sum(tgt_token_lens)
    joint_total = src_total + tgt_total

    # Fast path: fits in one pass.
    if joint_total <= per_pass_budget:
        return _align_single(src_subwords, tgt_subwords)

    # Hard guard requested: do not attempt alignment beyond 1024 subwords.
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

        # Need: left_tgt <= per_pass_budget-left_src and right_tgt <= per_pass_budget-right_src
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
    left = _align_single(src_subwords[:s_split], tgt_subwords[:t_split])
    right = _align_single(src_subwords[s_split:], tgt_subwords[t_split:])

    out: List[Tuple[int, int]] = []
    out.extend(left)
    out.extend((s + s_split, t + t_split) for s, t in right)
    return out


def truncate_by_alignment(
    full_src: str,
    observed_src: str,
    translation: str,
    alignments: List[Tuple[int, int]],
) -> str:
    """Keep only the part of *translation* aligned with *observed_src*.

    Falls back to a conservative length-ratio estimate when alignment
    returns nothing useful.
    """
# idea: 只看对齐到 observed 的英文词，这些词最靠后的那个对齐到中文的哪个字，就说明你“最多可以安全输出到那里”
    observed_count = len(observed_src.strip().split()) # 先算 observed 覆盖多少 src words
    src_word_count = len(full_src.strip().split())

    safe_tgt_idx = -1
    for s_idx, t_idx in alignments:
        if s_idx < observed_count:
            safe_tgt_idx = max(safe_tgt_idx, t_idx)

    if safe_tgt_idx >= 0:
        return translation[: safe_tgt_idx + 1]

    if src_word_count == 0:
        return ""
    # 如果 alignment 没有任何 useful 对齐，fallback（很保守） 按英文长度比例估
    ratio = observed_count / src_word_count
    safe_chars = int(len(translation) * ratio * 0.8)  #乘 0.8 是为了“宁可少写一点”
    return translation[:safe_chars] if safe_chars >= 2 else ""


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

def sample_source_futures(
    base_llm: LLM,
    observed_source: str,
    num_candidates: int,
    future_tokens: int,
    temperature: float,
) -> List[str]:
    """Pure text continuation using base model. Stops at newline to keep single-line structure."""
    params = SamplingParams(
        temperature=temperature,
        max_tokens=future_tokens,
        n=num_candidates,
        top_p=0.95,
        top_k=50,
        presence_penalty=0.6,
        stop=["\n"],  # 遇换行即停，避免破坏 trajectory 结构
    )
    with (_base_llm_lock if _base_llm_lock is not None else contextlib.nullcontext()):
        outputs = base_llm.generate([observed_source], params)

    futures: List[str] = []
    for out in outputs[0].outputs:
        cleaned = clean_continuation(observed_source, out.text)  #如果模型输出把 observed 又复述了一遍，就把前缀裁掉
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


def _build_translation_prompt_text(
    tokenizer: Any,
    observed_source: str,
    committed: str,
) -> str:
    """Build prompt for completion API: committed translation is FIXED (in prompt), model generates ONLY the continuation.

    Used with completions.create(): the server returns only the new tokens after the prompt,
    so we get committed (unchanged) + continuation without re-generating committed.
    """
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
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    # Assistant turn: prefill with committed so completion = only new translation.
    text += "<|im_start|>assistant\n"
    if committed:
        text += normalize_zh(committed)
    return text


def _build_continue_prompt(full_source: str, committed: str) -> str:
    """Prompt that asks the model to continue translating from committed."""
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
    """Return True if text is predominantly Chinese characters.

    Filters out English-explanation outputs that Qwen3 sometimes generates
    when the source looks incomplete (e.g. 'Explanation: the text is cut off...').
    """
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
) -> List[str]:
    """Translate a batch of sources concurrently, continuing from committed."""
    async def _one(src: str) -> str:
        prompt_text = _build_translation_prompt_text(
            instruct_tokenizer,
            observed_source=src,
            committed=committed,
        )
        resp = await client.completions.create(
            model=model,
            prompt=prompt_text,
            temperature=0.0,
            max_tokens=512,
        )
        raw = clean_llm_output((resp.choices[0].text or "").strip())
        raw = truncate_translation_repetition(raw)  # 防止模型陷入“他就是编辑……”等短语无限重复
        return raw if _is_chinese_output(raw) else ""

    results = await asyncio.gather(*[_one(s) for s in sources])
    return list(results)


async def _translate_batch_with_client_async(
    api_base: str,
    model: str,
    instruct_tokenizer,
    sources: List[str],
    committed: str = "",
) -> List[str]:
    """Create/use/close AsyncOpenAI client inside one event loop."""
    client = _make_async_client(api_base)
    try:
        return await _translate_batch_async(
            client, model, instruct_tokenizer, sources, committed
        )
    finally:
        # Ensure httpx AsyncClient closes before asyncio.run tears down the loop.
        await client.close()


def translate_candidates(
    api_base: str,
    model: str,
    instruct_tokenizer,
    observed_source: str,
    futures: List[str],
    committed: str = "",
) -> List[str]:
    """Translate observed+future for each candidate via completion: committed is FIXED, only continuation is generated.

    Uses completions.create() with prompt ending at committed; server returns new tokens only.
    Returns M full translations (committed + extension).
    """
    sources = []
    for future in futures:
        full = (observed_source + " " + future).strip() if future else observed_source
        sources.append(full)

    # Prompt ends with committed → API returns only continuation. If server ever returns full sequence, we dedupe below.
    extensions = asyncio.run(
        _translate_batch_with_client_async(
            api_base, model, instruct_tokenizer, sources, committed
        )
    )
    committed_norm = normalize_zh(committed) if committed else ""
    out = []
    for ext in extensions:
        ext_norm = normalize_zh(ext)
        # Strip any leading repetition of committed (model may echo committed despite prompt).
        while committed_norm and len(ext_norm) > len(committed_norm) and ext_norm.startswith(committed_norm):
            ext_norm = normalize_zh(ext_norm[len(committed_norm):].strip())
        if committed_norm and ext_norm.startswith(committed_norm):
            out.append(ext_norm)  # model returned committed + continuation (no extra repeat)
        else:
            out.append(committed_norm + ext_norm)
    return out


def translate_final(
    api_base: str, model: str,
    instruct_tokenizer,
    full_source: str, committed: str,
) -> str:
    """Final translation at utterance end."""
    client = _make_sync_client(api_base)
    prompt_text = _build_translation_prompt_text(
        instruct_tokenizer,
        observed_source=full_source,
        committed=committed,
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
) -> str:
    """Final tail commit using base model generate (no instruct API call).

    Builds the chat-template prefill with the committed translation already in
    the assistant turn, then lets the base model continue to the end.  This
    avoids calling the instruct server and any fragile character-splitting.
    """
    prompt_text = _build_translation_prompt_text(
        instruct_tokenizer,
        observed_source=full_source,
        committed=committed,
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
    # The model continues from the prefill, so its output is only the new part.
    return committed_norm + continuation


def select_best_candidate(
    api_base: str, model: str,
    observed_source: str,
    committed: str,
    candidate_translations: List[str],
) -> int:
    """Ask instruct model to pick the best candidate. Returns 0-based index."""
    client = _make_sync_client(api_base)
    prompt = build_select_prompt(
        observed_source, committed, candidate_translations,
    )
    # Kept for backward compatibility / debugging; not used in current pipeline.
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
    """LLM scores alignment-truncated candidate prefixes. Returns aligned list.

    prompt_version: 'full' = build_score_prompt + max_tokens=256; 'short' = build_score_prompt_short + max_tokens=64.
    """
    if not candidate_items:
        return []

    use_short = prompt_version == "short"
    user_prompt = (
        build_score_prompt_short(observed_source, committed, candidate_items)
        if use_short
        else build_score_prompt(observed_source, committed, candidate_items)
    )
    max_tokens = 64 if use_short else 256

    # LLM 打分器 + 容错解析器
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

    #把解析结果标准化成 score_map
    score_map: Dict[int, Dict[str, Any]] = {}
    for item in parsed_items:
        # 这里做了很多容错
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

    # Fallback parser for non-JSON outputs: lines like "1: 85"
    if not score_map:
        for line in text.splitlines():
            m = re.search(r"(\d+)\D+?(\d{1,3})", line)
            if not m:
                continue
            cid_int = int(m.group(1))
            score = max(0, min(100, int(m.group(2))))
            score_map[cid_int] = {"score": score, "tags": ["fallback_parse"]}

    results: List[Dict[str, Any]] = []
    # 上面都是容错
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
# Direct LCP via LLM output (delta after committed)
# ===================================================================

def build_direct_lcp_delta_prompt(committed: str, candidate_safe_prefixes: List[str]) -> str:
    """Ask LLM to output delta directly for LCP(committed + candidates)."""
    candidates_json = json.dumps(candidate_safe_prefixes, ensure_ascii=False)
    return (
        "You are a deterministic string-processing function.\n"
        "DO NOT translate. DO NOT paraphrase. DO NOT infer meaning.\n"
        "Only compare raw Chinese strings.\n\n"
        f'Committed prefix (must stay unchanged): "{committed}"\n'
        "The assistant reply is prefilled with the committed prefix.\n"
        "You must only continue after it.\n\n"
        f"Candidates JSON: {candidates_json}\n\n"
        "Task:\n"
        "1) Verify every candidate starts with committed exactly.\n"
        "2) Compute LCP of all candidates.\n"
        "3) Output only DELTA = LCP minus committed.\n"
        "4) If no common extension exists, empty delta is valid.\n\n"
        "Output rules:\n"
        "- Output only the delta text.\n"
        "- Single line only.\n"
        "- No labels, no JSON, no explanation.\n"
    )


def get_common_prefix_delta_via_llm(
    api_base: str,
    model: str,
    instruct_tokenizer,
    committed: str,
    candidate_safe_prefixes: List[str],
) -> Tuple[str, bool]:
    """Compute LCP delta via direct LLM output. Returns (delta, used_llm).

    Programmatic LCP delta is kept as deterministic fallback.
    """
    if not candidate_safe_prefixes:
        return "", False
    committed_norm = normalize_zh(committed) if committed else ""

    # Deterministic fallback from programmatic LCP.
    lcp = longest_common_prefix(candidate_safe_prefixes)
    fallback_delta = ""
    if committed_norm:
        if lcp.startswith(committed_norm) and len(lcp) > len(committed_norm):
            fallback_delta = lcp[len(committed_norm):]
    else:
        fallback_delta = lcp

    user_prompt = build_direct_lcp_delta_prompt(committed_norm, candidate_safe_prefixes)
    client = _make_sync_client(api_base)
    prompt_text = _build_instruct_generate_prompt(
        instruct_tokenizer, user_prompt, assistant_prefix=committed_norm
    )
    try:
        resp = client.completions.create(
            model=model,
            prompt=prompt_text,
            temperature=0.0,
            max_tokens=max(64, len(fallback_delta) + 16),
            stop=["\n"],
        )
        raw = (resp.choices[0].text or "").strip()
        delta = normalize_zh(clean_llm_output(raw))
        # Rare echo: model may still return committed+delta.
        if committed_norm and delta.startswith(committed_norm):
            delta = normalize_zh(delta[len(committed_norm):].strip())
        # Hard validation: committed+delta must be a prefix of every candidate.
        full_prefix = committed_norm + delta
        if all(c.startswith(full_prefix) for c in candidate_safe_prefixes):
            return delta, True
    except Exception:
        pass
    return fallback_delta, False


# ===================================================================
# Core Processing
# ===================================================================

def process_one_utterance(
    base_llm: LLM,
    api_base: str,
    instruct_model: str,
    instruct_tokenizer,
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
        "step4_lcp_llm_s": 0.0,
        "translate_final_s": 0.0,
    }

    _vlog(verbose_log_file, f"\n{'#'*60}")
    _vlog(verbose_log_file, f"# Utterance: {utt_id}")
    _vlog(verbose_log_file, f"# Full text: {full_source_text}")
    _vlog(verbose_log_file, f"# Chunks: {n_chunks}")
    _vlog(verbose_log_file, f"# M={args.num_candidates}")
    _vlog(verbose_log_file, f"{'#'*60}")

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

        # --- Case 4: Future sampling + LCP delta + Word alignment ---

        # Step 1: Base model generates M future continuations
        t1_0 = time.perf_counter()
        futures = sample_source_futures( # GPU0 上的base model
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

        # Step 3: Alignment-truncate all candidates first (judge evaluates real outputs)
        t3_0 = time.perf_counter()
        t3_align_model_sum = 0.0
        t3_truncate_sum = 0.0
        candidate_infos: List[Dict[str, Any]] = []
        for ci, (future, translation) in enumerate(zip(futures, all_translations)):
            full_src_for_candidate = (
                (accumulated_source + " " + future).strip()
                if future else accumulated_source
            )
            t3a_0 = time.perf_counter()
            alignments = get_word_alignments(
                full_src_for_candidate, translation,
                align_model, align_tokenizer,
            )
            t3_align_model_sum += (time.perf_counter() - t3a_0)
            t3b_0 = time.perf_counter()
            safe_prefix = truncate_by_alignment(
                full_src_for_candidate, accumulated_source,
                translation, alignments,
            )
            t3_truncate_sum += (time.perf_counter() - t3b_0)

            # 是否以已经提交的译文 committed_norm 开头
            monotonic_ok = (not committed_norm) or safe_prefix.startswith(committed_norm)
            delta = ""
            if monotonic_ok and safe_prefix and len(safe_prefix) > len(committed_norm):
                delta = safe_prefix[len(committed_norm):]  # 比已提交内容更长时，才取新增部分
            length_ok = len(delta) >= args.min_commit_chars

            candidate_infos.append({
                "idx": ci,
                "future": future,
                "translation": translation,
                "full_src_for_candidate": full_src_for_candidate,
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
                all_details.append({
                    "observed": accumulated_source,
                    "futures": futures,
                    "translations": all_translations,
                    "candidates": candidate_infos,
                    "reason": "no_valid_truncated_candidate",
                    "action": "READ",
                })
            continue

        new_chars = ""
        t4 = 0.0
        # Step 4 (LCP): LLM outputs common-prefix delta only; committed is never changed.
        candidate_safe_prefixes = [c["safe_prefix"] for c in valid_candidates]
        t4_0 = time.perf_counter()
        new_chars, lcp_used_llm = get_common_prefix_delta_via_llm(
            api_base, instruct_model, instruct_tokenizer,
            committed_norm, candidate_safe_prefixes,
        )
        t4 = time.perf_counter() - t4_0
        timing_totals["step4_lcp_llm_s"] += t4
        _vlog(verbose_log_file,
              f"  [Step 4 LCP] delta (used_llm={lcp_used_llm}) ({t4:.3f}s): \"{new_chars}\"")

        # Strip delta prefix that repeats a suffix of committed (e.g. "他就是编辑，" then "他就是编辑，而不" -> "而不")
        new_chars = strip_committed_suffix_from_delta(committed_norm, new_chars)
        # Guards: min_commit_chars; word_head guard (for lcp_llm always apply when ending on word head)
        risky = _ends_on_word_head(new_chars)
        if len(new_chars) >= args.min_commit_chars and not risky:
            committed_norm = committed_norm + new_chars
            decisions.append(("WRITE", new_chars))
            _vlog(verbose_log_file,
                  f"  -> WRITE \"{new_chars}\"  committed=\"{committed_norm}\"")
        else:
            decisions.append(("READ", ""))
            if risky:
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
            detail["selection_mode"] = "lcp_llm"
            detail["lcp_used_llm"] = lcp_used_llm  # False = fallback to programmatic LCP
            all_details.append(detail)

    # ---- Assemble output ----
    target_deltas = [d[1] for d in decisions]
    action_list = [d[0] for d in decisions]
    system_output_text = "".join(d for d in target_deltas if d)

    # LAAL reference: use pre-computed cache if available, otherwise call translate_final().
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
            "version": "baseline2_direct_llm_lcp_delta",
            "selection_mode": getattr(args, "selection_mode", "lcp_llm"),
            "patches": [
                "word_head_guard",
                "direct_llm_lcp_delta_with_hard_validation",
            ],
            "num_candidates": args.num_candidates,
            "future_tokens": args.future_tokens,
            "min_commit_chars": args.min_commit_chars,
            "min_observed_words": args.min_observed_words,
            "base_model": args.base_model_path,
            "instruct_model": args.instruct_model_name,
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

    # Load word-alignment model first (so vLLM can account for its memory when base shares GPU)
    print(f"[Align] Loading awesome-align model on {getattr(args, 'align_device', 'cuda:0')} ...")
    align_model, align_tokenizer = load_align_model(
        cache_dir=os.environ.get("HF_HOME"),
        device=getattr(args, "align_device", "cuda:0"),
    )
    print("[Align] Model loaded.")

    # Load instruct tokenizer locally (for manual chat-template -> generate prompts)
    print(f"[Instruct] Loading tokenizer from {args.instruct_tokenizer_path} ...")
    instruct_tokenizer = load_instruct_tokenizer(
        args.instruct_tokenizer_path,
        cache_dir=os.environ.get("HF_HOME"),
    )
    print("[Instruct] Tokenizer loaded.")

    # Load base model (4B fits on one card; leave headroom if align is on same GPU)
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

    # Load pre-computed translation cache (utt_id -> llm_full_translation)
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
        f"  instruct={instruct_model} @ {api_base}"
    )

    use_tee = getattr(args, "test_one", False) and not getattr(args, "no_tee", False)

    # Initialise base_llm lock for parallel utterance processing.
    global _base_llm_lock
    if args.parallel_utterances > 1:
        _base_llm_lock = threading.Lock()
        print(f"[Parallel] {args.parallel_utterances} concurrent utterances; "
              f"base_llm.generate() serialised via lock.")

    written = skipped = failed = 0
    _counter_lock = threading.Lock()
    pbar = tqdm(total=total, desc=f"task_{args.task_id}")

    # ----------------------------------------------------------------
    # Per-row worker: runs in thread pool or inline.
    # Returns ("skipped"|"ok"|"error", utt_id, payload, out_path)
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # Result handler (called from main thread after future completes).
    # ----------------------------------------------------------------
    def _handle_result(action, utt_id, payload, out_path):
        nonlocal written, skipped, failed
        if action == "skipped":
            with _counter_lock:
                skipped += 1
        elif action == "ok":
            result = payload
            if not (args.verbose and not args.overwrite and os.path.exists(out_path)):
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                with _counter_lock:
                    written += 1
            else:
                with _counter_lock:
                    skipped += 1
        else:  # "error"
            e = payload
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"utt_id": utt_id, "error": str(e)}, f,
                          ensure_ascii=False, indent=2)
            with _counter_lock:
                failed += 1
        pbar.update(1)

    # ----------------------------------------------------------------
    # Main dispatch: sequential or thread pool.
    # ----------------------------------------------------------------
    if args.parallel_utterances <= 1:
        # Sequential (original behaviour, zero overhead).
        submitted = 0
        for row_idx, row in row_iter:
            if args.max_rows is not None and submitted >= args.max_rows:
                break
            submitted += 1
            _handle_result(*_do_one_row((row_idx, row)))
    else:
        # Parallel: bounded sliding window over the row generator so we never
        # load the whole dataset into memory.
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

            _fill()  # seed the pool
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    in_flight.discard(fut)
                    _handle_result(*fut.result())
                _fill()  # refill to keep pool busy

    pbar.close()
    print(
        f"\n[Task {args.task_id}] Done. "
        f"written={written}, skipped={skipped}, failed={failed}"
    )


if __name__ == "__main__":
    main()


# Baseline2 (direct LLM LCP delta) - run 10 cases
# INPUT_TSV="$(find /data/user_data/haolingp/data_synthesis -type f -name '*.tsv' | head -n 1)" && \
# echo "Using INPUT_TSV=$INPUT_TSV" && \
# CUDA_VISIBLE_DEVICES=0 python /data/user_data/haolingp/data_synthesis/codes/gigaspeech/future_sampling/llm_future_sampling_b2.py \
#   --input-tsv "$INPUT_TSV" \
#   --output-root /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final/baseline2_direct_llm_lcp_10 \
#   --task-id 0 \
#   --num-tasks 1 \
#   --max-rows 10 \
#   --selection-mode lcp_llm \
#   --instruct-api-base http://localhost:8100/v1 \
#   --overwrite
