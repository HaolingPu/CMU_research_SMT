#!/usr/bin/env python3
"""Token-ID-level consensus decoding with future sampling via vLLM serve API.

Two base models each generate future continuations of the observed source
prefix.  An instruct model provides next-token distributions for each
hypothesised full source, and the consensus intersection selects the
committed token.  Candidate sets are built via either top-k or min-p.

Pipeline per chunk:
  1. Sample source futures from two base LMs (via vLLM completions API).
  2. For each future, query the instruct model for next-token distribution
     over the target language, conditioned on the committed target prefix
     (passed as token IDs to avoid re-encoding drift).
  3. Intersect candidate sets across futures; pick the token with the
     highest average probability.
  4. Repeat until consensus fails or max steps reached, then trim the
     pending token buffer to a clean Unicode boundary and commit.
  5. At the last chunk, force-complete the remaining translation.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TSV_PATH = (
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/"
    "eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"
)
DEFAULT_INSTRUCT_API_BASE = os.environ.get("INSTRUCT_API_BASE", "")
DEFAULT_INSTRUCT_API_MODEL = os.environ.get("INSTRUCT_API_MODEL", "qwen3-instruct")
TOP_K = 6
MIN_P = 0.0


def setup_env() -> None:
    os.environ.setdefault("HF_HOME", "/data/user_data/haolingp/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/data/user_data/haolingp/hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data/user_data/haolingp/hf_cache/transformers")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consensus decoding with future sampling via vLLM serve.")
    p.add_argument("--input-tsv", default=DEFAULT_TSV_PATH)
    p.add_argument("--id-column", default="id")
    # primary base model
    p.add_argument("--base-model-path", default="")
    p.add_argument("--base-api-base", required=True)
    p.add_argument("--base-api-model", required=True)
    p.add_argument("--base-api-timeout", type=float, default=120.0)
    # secondary base model
    p.add_argument("--secondary-base-model-path", default="")
    p.add_argument("--secondary-base-api-base", default="")
    p.add_argument("--secondary-base-api-model", default="")
    p.add_argument("--secondary-base-api-timeout", type=float, default=120.0)
    # instruct model
    p.add_argument("--instruct-tokenizer-path", required=True)
    p.add_argument("--instruct-api-base", required=True)
    p.add_argument("--instruct-api-model", default=DEFAULT_INSTRUCT_API_MODEL)
    p.add_argument("--instruct-api-timeout", type=float, default=120.0)
    # sampling / decoding
    p.add_argument("--num-futures", type=int, default=20)
    p.add_argument("--secondary-num-futures", type=int, default=10)
    p.add_argument("--future-tokens", type=int, default=20)
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--max-consensus-steps", type=int, default=12)
    p.add_argument("--min-consensus-horizon", type=int, default=1,
                   help="Minimum number of consensus-confirmed tokens required before committing.")
    p.add_argument("--final-max-tokens", type=int, default=128,
                   help="Maximum tokens for the final tail-completion step.")
    p.add_argument("--candidate-top-k", type=int, default=TOP_K)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.0,
                   help="Nucleus (top-p) candidate selection: keep smallest set with cumulative prob >= top-p.")
    # target language
    p.add_argument("--target-lang", default="Chinese",
                   help="Target language name for prompts (e.g. Chinese, Japanese, German)")
    p.add_argument("--future-source-window-chunks", type=int, default=0,
                   help="For future sampling, keep only the most recent N observed source chunks. "
                        "Use 0 to keep the full observed prefix.")
    # output
    p.add_argument("--output-jsonl", default=None)
    p.add_argument("--row-idx", type=int, default=0)
    p.add_argument("--utt-id", default=None)
    p.add_argument("--max-rows", type=int, default=1)
    p.add_argument("--test-one", action="store_true")
    p.add_argument("--num-concurrent-cases", type=int, default=1)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

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


def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_model_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.split("<|im_end|>")[0]
    text = text.split("<|endoftext|>")[0]
    return text.strip()


def clean_future_text(observed_source: str, raw_text: str) -> str:
    text = clean_model_text(raw_text)
    if text.startswith(observed_source):
        text = text[len(observed_source):].lstrip()
    text = text.splitlines()[0].strip() if text else ""
    text = re.sub(r"^[\.\s\u2026\-]+", "", text)
    obs_trailing = observed_source.strip()
    if obs_trailing and text:
        for k in range(min(len(obs_trailing), len(text)), 2, -1):
            tail = obs_trailing[-k:]
            if text.startswith(tail):
                text = text[k:].lstrip()
                break
            tail_space = " " + tail
            if text.startswith(tail_space):
                text = text[len(tail_space):].lstrip()
                break
    return text


def is_valid_future_text(text: str) -> bool:
    if not text:
        return False
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text):
        return False
    lowered = text.lower()
    banned = [
        "translate", "translation", "grammar analysis", "analyze", "analysis",
        "这句话", "翻译", "语法", "句子结构",
    ]
    return not any(b in lowered for b in banned)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_future_sampling_prompt(observed_source: str) -> str:
    return observed_source


def build_future_sampling_chat_messages(observed_source: str, num_futures: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": (
            "You are an expert linguist predicting how an incomplete spoken English sentence "
            "might continue. Stay on the same domain and style as the input. Generate diverse "
            "continuations that vary in topic emphasis and sentence structure but remain plausible."
        )},
        {"role": "user", "content": (
            f"Generate exactly {num_futures} different continuations of this English text. "
            f"Each should be 15-30 words. Output English only (no Chinese, no analysis).\n\n"
            f"IMPORTANT: each numbered item must contain ONLY the new words that come AFTER the "
            f"input - do NOT repeat any words from the input text, do NOT prepend '...' or any "
            f"placeholder. Start each item with the very next word.\n\n"
            f"Text: {observed_source}\n\n"
            f"Format strictly as:\n1. <new words after input>\n2. <new words after input>\n"
            f"...\n{num_futures}. <new words after input>"
        )},
    ]


def build_future_sampling_chat_messages_single(observed_source: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": (
            "You are an expert linguist predicting how an incomplete spoken English sentence "
            "might continue. Stay on the same domain and style as the input."
        )},
        {"role": "user", "content": (
            f"Continue this English text with one plausible continuation of 15-30 words. "
            f"Output English only (no Chinese, no analysis).\n\n"
            f"IMPORTANT: output ONLY the new words that come AFTER the input - do NOT repeat any "
            f"words from the input text, do NOT prepend '...' or any placeholder. Start with the "
            f"very next word.\n\n"
            f"Text: {observed_source}\n\n"
            f"Continuation:"
        )},
    ]


_NUMBERED_LIST_RE = re.compile(r"^\s*(\d+)[\.\)]\s*(.+?)\s*$", re.MULTILINE)


def parse_method_a_output(raw_text: str, num_expected: int) -> List[str]:
    if not raw_text:
        return []
    items: List[Tuple[int, str]] = []
    for match in _NUMBERED_LIST_RE.finditer(raw_text):
        idx = int(match.group(1))
        text = re.sub(r"^[\.\s\-\u2026]+", "", match.group(2).strip())
        if 1 <= idx <= num_expected and text:
            items.append((idx, text))
    items.sort(key=lambda x: x[0])
    seen_idx: set = set()
    out: List[str] = []
    for idx, text in items:
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        out.append(text)
    return out


def _is_chat_endpoint_model(api_model: str) -> bool:
    name = str(api_model or "").lower()
    return any(tag in name for tag in ("-it", "instruct", "chat"))


def _build_translation_messages(
    full_source: str, has_target_prefix: bool, target_lang: str = "Chinese",
) -> List[Dict[str, str]]:
    """Build the chat messages for a translation probe (shared by text and token-id prompts)."""
    if not has_target_prefix:
        return [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nStart the {target_lang} translation from the beginning "
            "and output only the next continuation token(s)."
        )}]
    return [{"role": "user", "content": (
        f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
        f"[INPUT]\n{full_source}\n\n"
        f"[IMPORTANT]\nA partial {target_lang} translation is already committed "
        "at the start of the assistant reply. You must continue from that "
        "exact prefix and produce only the continuation."
    )}]


def build_translation_probe_prefix_token_ids(
    tokenizer: Any,
    full_source: str,
    has_target_prefix: bool,
    target_lang: str = "Chinese",
) -> List[int]:
    """Return token IDs for [translation instruction + assistant prefix].

    The caller appends the committed target prefix token IDs to form the
    full prompt, avoiding text re-encoding drift.
    """
    messages = _build_translation_messages(full_source, has_target_prefix, target_lang)
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids.get("input_ids", [])
    elif hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    assistant_prefix_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    return list(prompt_ids) + list(assistant_prefix_ids)


def build_final_completion_prompt(
    tokenizer: Any, full_source: str, committed_text: str, target_lang: str = "Chinese",
) -> str:
    """Build a text prompt for force-completing the translation at the last chunk."""
    if not str(committed_text or "").strip():
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nOutput the complete {target_lang} translation only.\n"
            "Do not add explanations, summaries, examples, lists, numbering, markdown, or background knowledge."
        )}]
    else:
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nA partial {target_lang} translation is already committed "
            "at the start of the assistant reply.\n"
            f"Continue ONLY with the source content that is not yet covered by the committed {target_lang} prefix.\n"
            "Do not repeat the committed prefix.\n"
            "Do not add explanations, summaries, examples, lists, numbering, markdown, or background knowledge.\n"
            "Do not translate beyond the source.\n"
            "If there is no remaining source content to translate, output nothing.\n"
            f"Output only the remaining {target_lang} continuation."
        )}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    prompt += "<|im_start|>assistant\n"
    if str(committed_text or "").strip():
        prompt += committed_text
    return prompt


# ---------------------------------------------------------------------------
# Model / API helpers
# ---------------------------------------------------------------------------

def load_tokenizer(path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def normalize_api_base(api_base: str) -> str:
    base = str(api_base or "").strip().rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _http_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode('utf-8', errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e}") from e


def _http_get_json(url: str, timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer dummy"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode('utf-8', errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e}") from e


def verify_api(api_base: str, timeout: float) -> List[str]:
    data = _http_get_json(f"{normalize_api_base(api_base)}/models", timeout=timeout)
    return [str(item.get("id", "")) for item in data.get("data", []) if item.get("id")]


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _single_token_text(tokenizer: Any, tok_id: int) -> str:
    return tokenizer.decode([tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _disallowed_token_reason(tokenizer: Any, tok_id: int) -> Optional[str]:
    """Return a reason string if this token should never be emitted, else None."""
    if tok_id is None:
        return "missing"
    if tok_id in set(getattr(tokenizer, "all_special_ids", []) or []):
        return "special"
    token_text = _single_token_text(tokenizer, tok_id)
    for frag in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        if frag in token_text:
            return "special"
    if re.search(r"[A-Za-z]", token_text):
        return "ascii"
    # Do NOT filter U+FFFD here — byte-level BPE tokens for rare CJK characters
    # decode to U+FFFD individually but combine correctly in the pending buffer.
    if any(ch in {"\u200d", "\ufe0f"} for ch in token_text):
        return "zero_width"
    if any(unicodedata.category(ch) in {"Cc", "Cs"} for ch in token_text):
        return "control"
    return None


def filter_distribution(tokenizer: Any, dist: Dict[int, float]) -> Dict[int, float]:
    """Remove disallowed tokens from a next-token probability distribution."""
    return {
        tok_id: prob
        for tok_id, prob in dist.items()
        if _disallowed_token_reason(tokenizer, tok_id) is None
    }


# ---------------------------------------------------------------------------
# Future sampling (API)
# ---------------------------------------------------------------------------

def sample_source_futures_api(
    observed_source: str,
    num_futures: int,
    future_tokens: int,
    sample_temperature: float,
    api_base: str,
    api_model: str,
    api_timeout: float,
) -> List[str]:
    if not observed_source.strip():
        return []

    base = normalize_api_base(api_base)
    if _is_chat_endpoint_model(api_model):
        method = os.environ.get("FUTURE_SAMPLING_METHOD", "A").strip().upper()
        if method == "B":
            payload = {
                "model": api_model,
                "messages": build_future_sampling_chat_messages_single(observed_source),
                "max_tokens": future_tokens,
                "temperature": sample_temperature,
                "top_p": 0.95,
                "n": num_futures,
            }
            data = _http_json(f"{base}/chat/completions", payload=payload, timeout=api_timeout)
            futures: List[str] = []
            for choice in data.get("choices", []):
                msg = choice.get("message", {}) if isinstance(choice, dict) else {}
                raw = str(msg.get("content", "")) if isinstance(msg, dict) else ""
                cleaned = clean_future_text(observed_source, raw)
                if cleaned and is_valid_future_text(cleaned):
                    futures.append(cleaned)
            return futures

        max_tokens = max(future_tokens * num_futures, 40 * num_futures + 50)
        payload = {
            "model": api_model,
            "messages": build_future_sampling_chat_messages(observed_source, num_futures),
            "max_tokens": max_tokens,
            "temperature": sample_temperature,
            "top_p": 0.95,
            "n": 1,
        }
        data = _http_json(f"{base}/chat/completions", payload=payload, timeout=api_timeout)
        choices = data.get("choices", [])
        if not choices:
            return []
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        raw_text = str(msg.get("content", "")) if isinstance(msg, dict) else ""
        futures = []
        for raw in parse_method_a_output(raw_text, num_expected=num_futures):
            cleaned = clean_future_text(observed_source, raw)
            if cleaned and is_valid_future_text(cleaned):
                futures.append(cleaned)
        return futures

    payload = {
        "model": api_model,
        "prompt": observed_source,
        "max_tokens": future_tokens,
        "temperature": sample_temperature,
        "top_p": 0.95,
        "n": num_futures,
        "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    }
    data = _http_json(f"{base}/completions", payload=payload, timeout=api_timeout)
    futures: List[str] = []
    for choice in data.get("choices", []):
        raw = str(choice.get("text", "")) if isinstance(choice, dict) else ""
        cleaned = clean_future_text(observed_source, raw)
        if cleaned and is_valid_future_text(cleaned):
            futures.append(cleaned)
    return futures


def sample_source_futures_multi(
    base_specs: List[Dict[str, Any]],
    observed_source: str,
    future_tokens: int,
    sample_temperature: float,
) -> List[str]:
    """Sample futures from multiple base models, deduplicate."""
    merged: List[str] = []
    seen: set = set()
    for spec in base_specs:
        requested = int(spec.get("num_futures", 0) or 0)
        if requested <= 0:
            continue
        futures = sample_source_futures_api(
            observed_source=observed_source,
            num_futures=requested,
            future_tokens=future_tokens,
            sample_temperature=sample_temperature,
            api_base=spec["api_base"],
            api_model=spec["api_model"],
            api_timeout=spec["api_timeout"],
        )
        for future in futures:
            key = future.lower()
            if key not in seen:
                seen.add(key)
                merged.append(future)
    return merged


# ---------------------------------------------------------------------------
# Next-token distribution (batch API)
# ---------------------------------------------------------------------------

def _parse_token_id_string(raw: str) -> Optional[int]:
    match = re.fullmatch(r"token_id:(\d+)", str(raw or "").strip())
    return int(match.group(1)) if match else None


def _parse_completion_top_logprobs(
    top_logprobs: Optional[List[Optional[Dict[str, float]]]],
) -> Dict[int, float]:
    """Parse the first step of vLLM top_logprobs into {token_id: probability}."""
    if not top_logprobs or not top_logprobs[0]:
        return {}
    dist: Dict[int, float] = {}
    for raw_token, logprob in top_logprobs[0].items():
        tok_id = _parse_token_id_string(raw_token)
        if tok_id is not None:
            dist[tok_id] = math.exp(float(logprob))
    return dist


def batch_get_next_token_distributions(
    tokenizer: Any,
    full_sources: List[str],
    target_prefix_token_ids: List[int],
    top_k: int = TOP_K,
    min_p: float = 0.0,
    top_p: float = 0.0,
    api_base: str = "",
    api_model: str = "",
    api_timeout: float = 120.0,
    target_lang: str = "Chinese",
) -> List[Dict[int, float]]:
    """Query instruct model for next-token distributions across all futures.

    Each prompt is: [chat template token IDs] + [committed target prefix token IDs].
    Returns one filtered distribution per future.
    """
    prompts = [
        build_translation_probe_prefix_token_ids(
            tokenizer, src,
            has_target_prefix=bool(target_prefix_token_ids),
            target_lang=target_lang,
        ) + list(target_prefix_token_ids)
        for src in full_sources
    ]
    if not prompts:
        return []
    # When using min-p or top-p, request more logprobs to have enough candidates after filtering.
    logprobs_n = max(top_k, 100) if (min_p > 0 or top_p > 0) else top_k
    payload = {
        "model": api_model,
        "prompt": prompts,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": logprobs_n,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
    }
    data = _http_json(f"{normalize_api_base(api_base)}/completions", payload=payload, timeout=api_timeout)
    choices = data.get("choices", [])
    results: List[Dict[int, float]] = []
    for i in range(len(prompts)):
        if i >= len(choices):
            results.append({})
            continue
        choice = choices[i]
        logprobs = choice.get("logprobs", {}) if isinstance(choice, dict) else {}
        dist = _parse_completion_top_logprobs(logprobs.get("top_logprobs"))
        results.append(filter_distribution(tokenizer, dist))
    return results


# ---------------------------------------------------------------------------
# Consensus logic
# ---------------------------------------------------------------------------

def topk_token_ids(dist: Dict[int, float], k: int = TOP_K) -> List[int]:
    return [tok_id for tok_id, _ in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def minp_token_ids(dist: Dict[int, float], min_p: float) -> List[int]:
    """Return token IDs whose probability >= min_p (absolute threshold)."""
    if not dist:
        return []
    return [tok_id for tok_id, prob in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
            if prob >= min_p]


def topp_token_ids(dist: Dict[int, float], top_p: float) -> List[int]:
    """Return the smallest set of token IDs whose cumulative probability >= top_p."""
    if not dist:
        return []
    sorted_items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    result: List[int] = []
    cumsum = 0.0
    for tok_id, prob in sorted_items:
        result.append(tok_id)
        cumsum += prob
        if cumsum >= top_p:
            break
    return result


def _select_candidates(dist: Dict[int, float], top_p: float = 0.0,
                       min_p: float = 0.0, top_k: int = TOP_K) -> List[int]:
    """Select candidate token IDs using the active policy (top-p > min-p > top-k)."""
    if top_p > 0:
        return topp_token_ids(dist, top_p)
    if min_p > 0:
        return minp_token_ids(dist, min_p)
    return topk_token_ids(dist, top_k)


def choose_consensus_token(
    distributions: List[Dict[int, float]],
    min_p: float = MIN_P,
    candidate_top_k: int = TOP_K,
    top_p: float = 0.0,
) -> Optional[int]:
    """Intersect candidate sets from all futures; return the best token or None."""
    if not distributions:
        return None
    candidate_lists = [_select_candidates(dist, top_p=top_p, min_p=min_p, top_k=candidate_top_k)
                       for dist in distributions]

    intersection = set(candidate_lists[0])
    for clist in candidate_lists[1:]:
        intersection &= set(clist)
    if not intersection:
        return None

    return max(intersection, key=lambda tok: sum(d.get(tok, 0.0) for d in distributions) / len(distributions))


# ---------------------------------------------------------------------------
# Token buffer management
# ---------------------------------------------------------------------------

def decode_token_ids(tokenizer: Any, token_ids: List[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def has_suspicious_content(text: str) -> bool:
    """Check if decoded text contains broken Unicode or dangerous trailing characters."""
    if not text:
        return False
    if "\ufffd" in text or "�" in text:
        return True
    last_char = text[-1]
    if last_char in {"\u200d", "\ufe0f"}:
        return True
    if unicodedata.category(last_char) in {"Mn", "Mc", "Me", "Cc", "Cs"}:
        return True
    return False


def sanitize_token_ids(tokenizer: Any, token_ids: List[int]) -> List[int]:
    """Remove disallowed tokens from a token ID sequence."""
    return [t for t in token_ids if _disallowed_token_reason(tokenizer, t) is None]


def trim_to_clean_boundary(
    tokenizer: Any, pending_token_ids: List[int],
) -> Tuple[List[int], str]:
    """Trim pending tokens to the longest prefix that decodes to clean Unicode.

    Returns (clean_token_ids, decoded_text).
    """
    work = sanitize_token_ids(tokenizer, pending_token_ids)
    if not work:
        return [], ""
    # Walk forward to find the last position that decodes without broken chars.
    last_clean = -1
    for i in range(len(work)):
        partial = decode_token_ids(tokenizer, work[: i + 1])
        if not has_suspicious_content(partial):
            last_clean = i
    if last_clean >= 0:
        clean_ids = work[: last_clean + 1]
        return clean_ids, decode_token_ids(tokenizer, clean_ids)
    return [], ""


def commit_pending_tokens(
    tokenizer: Any, committed_text: str, pending_token_ids: List[int],
) -> Tuple[str, str, List[int]]:
    """Trim pending tokens and commit to the translation.

    Returns (new_committed_text, delta_text, committed_token_ids).
    """
    trimmed_ids, delta = trim_to_clean_boundary(tokenizer, pending_token_ids)
    return committed_text + delta, delta, trimmed_ids


# ---------------------------------------------------------------------------
# Core consensus loop
# ---------------------------------------------------------------------------

def extend_pending_tokens(
    instruct_tokenizer: Any,
    source_observed: str,
    futures: List[str],
    committed_token_ids: List[int],
    max_consensus_steps: int,
    candidate_top_k: int = TOP_K,
    instruct_api_base: str = "",
    instruct_api_model: str = "",
    instruct_api_timeout: float = 120.0,
    min_p: float = 0.0,
    top_p: float = 0.0,
    target_lang: str = "Chinese",
) -> List[int]:
    """Grow the pending token buffer via iterative consensus.

    At each step, queries the instruct model for next-token distributions
    conditioned on each future, intersects the candidate sets, and appends
    the consensus token.  Stops when consensus fails or max steps reached.
    """
    pending: List[int] = []
    full_sources = [append_text_continuation(source_observed, f) for f in futures]

    for _ in range(max_consensus_steps):
        prefix_ids = list(committed_token_ids) + list(pending)

        distributions = batch_get_next_token_distributions(
            tokenizer=instruct_tokenizer,
            full_sources=full_sources,
            target_prefix_token_ids=prefix_ids,
            top_k=candidate_top_k,
            min_p=min_p,
            top_p=top_p,
            api_base=instruct_api_base,
            api_model=instruct_api_model,
            api_timeout=instruct_api_timeout,
            target_lang=target_lang,
        )
        if any(not d for d in distributions):
            break

        token_id = choose_consensus_token(distributions, min_p=min_p, candidate_top_k=candidate_top_k, top_p=top_p)
        if token_id is None:
            break
        pending.append(token_id)

    return pending


# ---------------------------------------------------------------------------
# Force-complete last chunk
# ---------------------------------------------------------------------------

def force_complete_translation(
    tokenizer: Any,
    full_source: str,
    committed_text: str,
    api_base: str,
    api_model: str,
    api_timeout: float = 120.0,
    target_lang: str = "Chinese",
    max_tokens: int = 128,
) -> str:
    prompt = build_final_completion_prompt(tokenizer, full_source, committed_text, target_lang=target_lang)
    payload = {
        "model": api_model,
        "prompt": prompt,
        "max_tokens": max(1, int(max_tokens)),
        "temperature": 0.0,
        "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    }
    data = _http_json(f"{normalize_api_base(api_base)}/completions", payload=payload, timeout=api_timeout)
    choices = data.get("choices", [])
    if not choices:
        return ""
    return clean_model_text(str(choices[0].get("text", "")))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _extract_reference_text_from_row(row: Dict[str, Any], target_lang: str = "Chinese") -> Optional[str]:
    lang_suffix_map = {"Japanese": "ja", "German": "de", "French": "fr", "Spanish": "es"}
    lang_suffix = lang_suffix_map.get(target_lang, "")
    keys: List[str] = []
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


# ---------------------------------------------------------------------------
# Run one utterance
# ---------------------------------------------------------------------------

def run_one_utterance(
    row: Dict[str, Any],
    args: argparse.Namespace,
    base_specs: List[Dict[str, Any]],
    instruct_tokenizer: Any,
) -> Dict[str, Any]:
    utt_id = str(row.get(args.id_column, row.get("id", f"row_{args.row_idx}")))
    chunks = parse_trajectory(row["src_trajectory"])
    source_units = parse_source_units(row.get("src_text_full"))
    full_source_text = get_full_source_text(row)

    committed_text = ""
    committed_token_ids: List[int] = []
    target_deltas: List[str] = []
    actions: List[str] = []

    if args.top_p > 0:
        candidate_policy = f"top_p({args.top_p})"
    elif args.min_p > 0:
        candidate_policy = f"min_p({args.min_p})"
    else:
        candidate_policy = f"top_k({args.candidate_top_k})"

    for t in range(len(chunks)):
        source_observed_full = build_source_observed(chunks, t)
        source_observed = build_source_observed_recent_units(
            source_units=source_units,
            observed_full=source_observed_full,
            num_units=args.future_source_window_chunks,
        )

        # Last chunk: force-complete the remaining translation.
        if t == len(chunks) - 1:
            final_delta = force_complete_translation(
                tokenizer=instruct_tokenizer,
                full_source=source_observed_full,
                committed_text=committed_text,
                api_base=args.instruct_api_base,
                api_model=args.instruct_api_model,
                api_timeout=args.instruct_api_timeout,
                target_lang=args.target_lang,
                max_tokens=args.final_max_tokens,
            )
            if final_delta:
                committed_text += final_delta
                target_deltas.append(final_delta)
                actions.append("WRITE")
            else:
                target_deltas.append("")
                actions.append("READ")
            continue

        # Sample source futures from base models.
        futures = sample_source_futures_multi(
            base_specs=base_specs,
            observed_source=source_observed,
            future_tokens=args.future_tokens,
            sample_temperature=args.sample_temperature,
        )
        if len(futures) <= 3:
            target_deltas.append("")
            actions.append("READ")
            continue

        # Grow pending tokens via iterative consensus.
        pending_token_ids = extend_pending_tokens(
            instruct_tokenizer=instruct_tokenizer,
            source_observed=source_observed_full,
            futures=futures,
            committed_token_ids=committed_token_ids,
            max_consensus_steps=args.max_consensus_steps,
            candidate_top_k=args.candidate_top_k,
            instruct_api_base=args.instruct_api_base,
            instruct_api_model=args.instruct_api_model,
            instruct_api_timeout=args.instruct_api_timeout,
            min_p=args.min_p,
            top_p=args.top_p,
            target_lang=args.target_lang,
        )

        if args.min_consensus_horizon > 1 and 0 < len(pending_token_ids) < args.min_consensus_horizon:
            pending_token_ids = []

        # Trim to clean boundary and commit.
        new_committed, delta, delta_token_ids = commit_pending_tokens(
            tokenizer=instruct_tokenizer,
            committed_text=committed_text,
            pending_token_ids=pending_token_ids,
        )
        target_deltas.append(delta)
        actions.append("WRITE" if delta else "READ")
        committed_text = new_committed
        committed_token_ids.extend(delta_token_ids)

    result: Dict[str, Any] = {
        "utt_id": utt_id,
        "source_full_text": full_source_text,
        "src_text_full": source_units,
        "src_trajectory": chunks,
        "target_trajectory": target_deltas,
        "actions": actions,
        "prediction": committed_text,
        "decoder_impl": {"candidate_policy": candidate_policy, "backend": "vllm_completion"},
    }

    reference_text = _extract_reference_text_from_row(row, target_lang=args.target_lang)
    laal_value = float("nan")
    bleu_char_value = float("nan")
    length_ratio_ref = float("nan")
    try:
        if not reference_text:
            raise ValueError("reference_text_unavailable")
        laal_value = compute_laal(chunks, target_deltas, actions, reference_text)
        bleu_char_value = compute_bleu_char(committed_text, reference_text)
        length_ratio_ref = compute_length_ratio_ref(committed_text, reference_text)
    except Exception:
        pass
    length_ratio_src = compute_length_ratio_src(committed_text, full_source_text)

    result["reference_text"] = reference_text or ""
    result["metrics"] = {
        "laal_text": laal_value,
        "bleu_char": bleu_char_value,
        "length_ratio_ref": length_ratio_ref,
        "length_ratio_src": length_ratio_src,
        "pred_chars": _nonspace_char_count(committed_text),
        "ref_chars": _nonspace_char_count(reference_text or ""),
        "src_words": len(str(full_source_text or "").split()),
    }
    return result


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------

def select_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.utt_id is not None:
        selected = df[df[args.id_column].astype(str) == str(args.utt_id)]
        if selected.empty:
            raise ValueError(f"utt_id not found: {args.utt_id}")
        return selected.iloc[:1] if args.test_one else selected
    if args.test_one:
        return df.iloc[[args.row_idx]]
    start = max(0, int(args.row_idx))
    end = min(len(df), start + max(1, int(args.max_rows)))
    return df.iloc[start:end]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def main() -> None:
    setup_env()
    args = parse_args()

    primary_num_futures = args.num_futures - args.secondary_num_futures
    if primary_num_futures <= 0:
        raise ValueError("primary base model must keep at least 1 future")

    df = pd.read_csv(args.input_tsv, sep="\t")
    rows = select_rows(df, args)

    # Build base model specs.
    base_specs: List[Dict[str, Any]] = []
    for name, api_base, api_model, api_timeout, model_path, num_futures in [
        ("primary", args.base_api_base, args.base_api_model,
         args.base_api_timeout, args.base_model_path, primary_num_futures),
        ("secondary", args.secondary_base_api_base, args.secondary_base_api_model,
         args.secondary_base_api_timeout, args.secondary_base_model_path, args.secondary_num_futures),
    ]:
        if not api_base or num_futures <= 0:
            continue
        models = verify_api(api_base, api_timeout)
        if api_model not in models:
            raise RuntimeError(f"{name} api_model '{api_model}' not found; available={models}")
        base_specs.append({
            "name": name, "path": model_path, "num_futures": num_futures,
            "api_base": api_base, "api_model": api_model, "api_timeout": api_timeout,
        })
        print(f"[Base] {name}: model={api_model} api={normalize_api_base(api_base)} futures={num_futures}")

    # Verify instruct API.
    models = verify_api(args.instruct_api_base, args.instruct_api_timeout)
    if args.instruct_api_model not in models:
        raise RuntimeError(f"instruct_api_model '{args.instruct_api_model}' not found; available={models}")
    instruct_tokenizer = load_tokenizer(args.instruct_tokenizer_path)
    print(f"[Instruct] model={args.instruct_api_model} api={normalize_api_base(args.instruct_api_base)}")

    output_dir: Optional[str] = None
    if args.output_jsonl:
        output_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
        os.makedirs(output_dir, exist_ok=True)

    def _process_one_row(row_idx: int, row: Dict[str, Any]) -> Dict[str, Any]:
        utt_id = str(row.get(args.id_column, row.get("id", f"row_{row_idx}")))
        out_path = (
            os.path.join(output_dir, f"{sanitize_filename(utt_id)}.json")
            if output_dir is not None else None
        )
        if args.skip_existing and out_path is not None and os.path.exists(out_path):
            print(f"[SKIP] {out_path}")
            return {"utt_id": utt_id, "skipped": True}

        result = run_one_utterance(row=row, args=args, base_specs=base_specs,
                                   instruct_tokenizer=instruct_tokenizer)
        if out_path is not None:
            write_json(out_path, result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    num_concurrent = max(1, args.num_concurrent_cases)
    row_items = [(idx, s.to_dict()) for idx, (_, s) in enumerate(rows.iterrows())]
    failures: List[str] = []

    if num_concurrent <= 1:
        for row_idx, row_dict in row_items:
            _process_one_row(row_idx, row_dict)
    else:
        print(f"[Concurrent] {len(row_items)} rows, {num_concurrent} workers")
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futs = {executor.submit(_process_one_row, ri, rd): ri for ri, rd in row_items}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as exc:
                    row_id = futs[fut]
                    print(f"[ERROR] Row {row_id} raised: {exc}", file=sys.stderr)
                    failures.append(f"Row {row_id}: {exc}")

    if failures:
        print(f"[FATAL] {len(failures)} row(s) failed", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
