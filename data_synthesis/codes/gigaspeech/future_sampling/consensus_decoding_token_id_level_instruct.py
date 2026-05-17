#!/usr/bin/env python3
"""Consensus decoding with future sampling via vLLM serve API.

Two base models (e.g. gemma4-E2B + Qwen3-4B-Base) each generate future
continuations of the observed source prefix.  An instruct model
(Qwen3-30B-A3B-Instruct) provides next-token distributions for each
hypothesised full source, and the consensus intersection selects the
committed token.  Candidate sets are built via either top-k or min-p.
"""
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


DEFAULT_TSV_PATH = "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"
DEFAULT_INSTRUCT_API_BASE = os.environ.get("INSTRUCT_API_BASE", "")
DEFAULT_INSTRUCT_API_MODEL = os.environ.get("INSTRUCT_API_MODEL", "qwen3-instruct")
TOP_K = 6
MIN_P = 0.0


def setup_env() -> None:
    os.environ.setdefault("HF_HOME", "/data/user_data/haolingp/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/data/user_data/haolingp/hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data/user_data/haolingp/hf_cache/transformers")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consensus decoding with future sampling via vLLM serve.")
    p.add_argument("--input-tsv", default=DEFAULT_TSV_PATH)
    p.add_argument("--id-column", default="id")
    # primary base model (future sampling)
    p.add_argument("--base-model-path", default="")
    p.add_argument("--base-api-base", default="",
                   help="Required unless --use-targeted-instruct-sampling is set.")
    p.add_argument("--base-api-model", default="",
                   help="Required unless --use-targeted-instruct-sampling is set.")
    p.add_argument("--base-api-timeout", type=float, default=120.0)
    # secondary base model (future sampling)
    p.add_argument("--secondary-base-model-path", default="")
    p.add_argument("--secondary-base-api-base", default="")
    p.add_argument("--secondary-base-api-model", default="")
    p.add_argument("--secondary-base-api-timeout", type=float, default=120.0)
    # instruct model (next-token distribution)
    p.add_argument("--instruct-tokenizer-path", required=True)
    p.add_argument("--instruct-api-base", required=True)
    p.add_argument("--instruct-api-model", default=DEFAULT_INSTRUCT_API_MODEL)
    p.add_argument("--instruct-api-timeout", type=float, default=120.0)
    # sampling / decoding
    p.add_argument("--num-futures", type=int, default=20)
    p.add_argument("--secondary-num-futures", type=int, default=10)
    p.add_argument("--future-tokens", type=int, default=10)
    p.add_argument("--sample-temperature", type=float, default=1.0)
    # targeted instruct future sampling (Method C) — bypass base-LM sampling and use the
    # instruct model to generate ambiguity-targeted continuations via a 2-step JSON prompt.
    p.add_argument("--use-targeted-instruct-sampling", action="store_true",
                   help="Use the instruct model (via --instruct-api-base) to generate "
                        "ambiguity-targeted continuations via structured JSON, instead of "
                        "base-LM sampling. When set, base_specs are ignored for future sampling.")
    p.add_argument("--targeted-num-futures", type=int, default=20,
                   help="Number of diverse continuations sampled in one /completions call (n=K).")
    p.add_argument("--targeted-sample-temperature", type=float, default=1.0,
                   help="Sampling temperature for the prefill sampler — higher → more diversity.")
    p.add_argument("--targeted-top-p", type=float, default=0.98)
    p.add_argument("--targeted-max-tokens", type=int, default=40,
                   help="Per-completion token budget (4-15 words ≈ 8-30 tokens; bumped to 40 to fight short/empty outputs).")
    p.add_argument("--targeted-sampler-tokenizer-path", default="",
                   help="HF path/name of the tokenizer matching the sampler model. "
                        "Required because the sampler may use a different chat template "
                        "than the probe (e.g. gemma vs qwen). Defaults to --instruct-tokenizer-path.")
    # Optional split: route the sampler to a different endpoint than the probe so
    # the model that *generates* futures is not the same model being *probed* for
    # next-token distributions (mitigates shared-blindspot confirmation bias).
    # Defaults are empty → fall back to --instruct-api-* (same-model mode).
    p.add_argument("--targeted-sampler-api-base", default="",
                   help="If set, route the targeted-instruct sampler to this endpoint "
                        "(cross-family setup, e.g. gemma-4-it). Defaults to --instruct-api-base.")
    p.add_argument("--targeted-sampler-api-model", default="",
                   help="Served-model-name of the sampler endpoint. Defaults to --instruct-api-model.")
    p.add_argument("--targeted-sampler-api-timeout", type=float, default=0.0,
                   help="Timeout for sampler API. 0 or negative → fall back to --instruct-api-timeout.")
    # Optional second sampler (ensemble across families). When all three of
    # --targeted-sampler2-* are set, the targeted prefill pipeline runs 4
    # sub-batches: (sampler1, subj_cont), (sampler1, focus_shift),
    # (sampler2, subj_cont), (sampler2, focus_shift), each with n=K, then
    # merges all futures with cross-batch dedup.
    p.add_argument("--targeted-sampler2-tokenizer-path", default="",
                   help="HF path/name of the 2nd sampler's tokenizer (different family). "
                        "Enables dual-model ensemble when all 3 sampler2-* flags are set.")
    p.add_argument("--targeted-sampler2-api-base", default="",
                   help="API endpoint of the 2nd sampler (e.g. http://localhost:8100/v1 to "
                        "reuse the probe's Qwen3 server).")
    p.add_argument("--targeted-sampler2-api-model", default="",
                   help="Served-model-name of the 2nd sampler endpoint.")
    p.add_argument("--targeted-sampler2-api-timeout", type=float, default=0.0,
                   help="Timeout for 2nd sampler API. 0 or negative → fall back to --instruct-api-timeout.")
    p.add_argument("--max-consensus-steps", type=int, default=12)
    p.add_argument("--min-consensus-horizon", type=int, default=2,
                   help="Minimum number of consensus-confirmed tokens required before committing. "
                        "If the pending buffer ends up shorter than this (after consensus breaks), "
                        "discard all pending tokens and READ instead. Default 2: avoid single-token "
                        "commits that the translator may misinterpret (reduces early lock-in).")
    p.add_argument("--final-max-tokens", type=int, default=128,
                   help="Maximum tokens for the final tail-completion step.")
    p.add_argument("--candidate-top-k", type=int, default=TOP_K)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.0,
                   help="Nucleus (top-p) candidate selection: keep smallest set with cumulative prob >= top-p.")
    # target language
    p.add_argument("--target-lang", default="Chinese",
                   help="Target language name for prompts (e.g. Chinese, Japanese, German)")
    p.add_argument(
        "--future-source-window-chunks",
        type=int,
        default=0,
        help="For future sampling, keep only the most recent N observed source chunks. "
             "Use 0 to keep the full observed prefix.",
    )
    # output
    p.add_argument("--output-jsonl", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-dir", default=None)
    p.add_argument("--row-idx", type=int, default=0)
    p.add_argument("--utt-id", default=None)
    p.add_argument("--max-rows", type=int, default=1)
    p.add_argument("--test-one", action="store_true")
    p.add_argument("--num-concurrent-cases", type=int, default=8)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip rows whose per-utterance output JSON already exists.")
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


def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))


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


# ---------------------------------------------------------------------------
# Verbose logging (lightweight)
# ---------------------------------------------------------------------------

def _vlog(f: Optional[Any], msg: str) -> None:
    if f is None:
        return
    line = str(msg) if msg.endswith("\n") else str(msg) + "\n"
    f.write(line)
    f.flush()


def _vlog_pretty_value(f: Optional[Any], label: str, value: Any) -> None:
    if f is None:
        return
    if isinstance(value, (list, dict)):
        pretty = json.dumps(value, ensure_ascii=False, indent=2)
        lines = pretty.splitlines()
        if not lines:
            _vlog(f, f"# {label}: []")
            return
        _vlog(f, f"# {label}: {lines[0]}")
        for line in lines[1:]:
            _vlog(f, f"#   {line}")
        return
    _vlog(f, f"# {label}: {value}")


class _TeeWriter:
    """Write to both a file and stdout (used only for --test-one)."""
    def __init__(self, fobj: Any):
        self._f = fobj
    def write(self, msg: str) -> None:
        self._f.write(msg); sys.stdout.write(msg)
    def flush(self) -> None:
        self._f.flush(); sys.stdout.flush()
    def close(self) -> None:
        self._f.close()


def write_pretty_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


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


def strip_markdown_wrappers(text: str) -> str:
    text = str(text or "").strip()
    # Gemma-style completions sometimes wrap continuations in markdown bold.
    # The continuation itself can still be useful, but the markup must never
    # enter either the source future or the committed Chinese token stream.
    text = re.sub(r"^\*{1,3}\s*(.*?)\s*\*{1,3}$", r"\1", text)
    text = re.sub(r"^`+\s*(.*?)\s*`+$", r"\1", text)
    return text.strip()


def clean_future_text(observed_source: str, raw_text: str) -> str:
    # 只保留第一行 (如果模型输出了多行，我们只取第一行作为续写)，并且如果第一行以observed source开头，就去掉这个重复的prefix（有些模型喜欢在续写里重复输入的source prefix）
    text = strip_markdown_wrappers(clean_model_text(raw_text))
    if text.startswith(observed_source):
        text = text[len(observed_source):].lstrip()
    text = text.splitlines()[0].strip() if text else ""
    text = strip_markdown_wrappers(text)
    # Strip leading placeholders like "..." or "…" that instruct models sometimes prepend.
    text = re.sub(r"^[\.\s…\-]+", "", text)
    # If the cleaned continuation still starts by repeating the trailing words of the
    # observed source (instruct-model habit), strip the longest matching suffix-prefix.
    obs_trailing = observed_source.strip()
    if obs_trailing and text:
        # Try the longest possible suffix of observed_source that matches the start of text.
        for k in range(min(len(obs_trailing), len(text)), 2, -1):
            tail = obs_trailing[-k:]
            if text.startswith(tail):
                text = text[k:].lstrip()
                break
            # Also try after a leading space (text starts with a space in some models)
            tail_space = " " + tail
            if text.startswith(tail_space):
                text = text[len(tail_space):].lstrip()
                break
    return text


def is_valid_future_text(text: str) -> bool:
    if not text:
        return False
    if any(ch in text for ch in ("*", "`")):
        return False
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text):
        return False
    lowered = text.lower()
    banned = ["translate", "translation", "grammar analysis", "analyze", "analysis",
              "这句话", "翻译", "语法", "句子结构"]
    return not any(b in lowered for b in banned)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_future_sampling_prompt(observed_source: str) -> str:
    return observed_source


# ---------------------------------------------------------------------------
# Method A: instruct-model future sampling via /chat/completions
# Generates N diverse continuations in a single call as a numbered list.
# ---------------------------------------------------------------------------

def build_future_sampling_chat_messages(observed_source: str, num_futures: int) -> List[Dict[str, str]]:
    """Method A: ask the model to produce a numbered list of `num_futures` continuations in one call."""
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
            f"input — do NOT repeat any words from the input text, do NOT prepend '...' or any "
            f"placeholder. Start each item with the very next word.\n\n"
            f"Text: {observed_source}\n\n"
            f"Format strictly as:\n1. <new words after input>\n2. <new words after input>\n"
            f"...\n{num_futures}. <new words after input>"
        )},
    ]


def build_future_sampling_chat_messages_single(observed_source: str) -> List[Dict[str, str]]:
    """Method B: ask the model for ONE continuation; caller invokes with n=num_futures for independent samples."""
    return [
        {"role": "system", "content": (
            "You are an expert linguist predicting how an incomplete spoken English sentence "
            "might continue. Stay on the same domain and style as the input."
        )},
        {"role": "user", "content": (
            f"Continue this English text with one plausible continuation of 15-30 words. "
            f"Output English only (no Chinese, no analysis).\n\n"
            f"IMPORTANT: output ONLY the new words that come AFTER the input — do NOT repeat any "
            f"words from the input text, do NOT prepend '...' or any placeholder. Start with the "
            f"very next word.\n\n"
            f"Text: {observed_source}\n\n"
            f"Continuation:"
        )},
    ]


_NUMBERED_LIST_RE = re.compile(r"^\s*(\d+)[\.\)]\s*(.+?)\s*$", re.MULTILINE)


def parse_method_a_output(raw_text: str, num_expected: int) -> List[str]:
    """Parse a numbered list response (1. ... 2. ... N. ...) into a list of continuations."""
    if not raw_text:
        return []
    items: List[Tuple[int, str]] = []
    for match in _NUMBERED_LIST_RE.finditer(raw_text):
        idx = int(match.group(1))
        text = match.group(2).strip()
        # Some models output "1. ... so it was in <continuation>" — strip leading "..." or "so it was in"
        text = re.sub(r"^[\.\s\-…]+", "", text)
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


# ---------------------------------------------------------------------------
# Method C: targeted instruct future sampling via structured JSON.
# The instruct model first identifies up to K ambiguity spans in the heard
# English prefix whose Chinese translation could still change, proposes 2
# competing branches per span, and writes one short English continuation per
# branch — plus a few neutral continuations as a control group. This produces
# branch-coverage-oriented futures (vs probability-mass futures from base LM).
# ---------------------------------------------------------------------------

def build_targeted_future_messages(
    observed_source: str,
    committed_text: str,
    target_lang: str,
    max_spans: int,
    num_neutrals: int,
) -> List[Dict[str, str]]:
    """Build the 2-step JSON prompt for ambiguity-targeted continuation sampling."""
    system = (
        f"You help a simultaneous English-to-{target_lang} interpreter test whether "
        f"the current {target_lang} prefix is safe to commit. Your job is to surface "
        f"ambiguities in the heard English prefix that could still flip the {target_lang} "
        f"translation, and to write short natural continuations that probe each branch."
    )
    user = (
        f"English heard so far:\n\"{observed_source}\"\n\n"
        f"{target_lang} committed so far:\n\"{committed_text}\"\n\n"
        f"Task:\n"
        f"1. Find up to {max_spans} spans in the English prefix whose {target_lang} "
        f"translation could change after hearing more words (word sense, verb sense, "
        f"PP/clause attachment, long-distance structure, proper-name vs common-noun reading).\n"
        f"2. For each span, propose 2-3 possible branches. Use 3 only when the span "
        f"genuinely has 3 or more distinct senses (e.g., a polysemous verb like "
        f"\"run\" or \"draw\"); otherwise 2 is sufficient. Do NOT pad with near-duplicate branches.\n"
        f"3. For each branch, write 1 short natural English continuation, 4-12 words, "
        f"starting immediately AFTER the heard English prefix and grammatically connecting "
        f"to it (if the prefix ends mid-clause, the continuation completes the clause).\n"
        f"4. Additionally, write {num_neutrals} neutral continuations that simply extend "
        f"the sentence without committing to any branch (control group); use span=\"none\" "
        f"and branch=\"neutral\" for these.\n\n"
        f"Rules:\n"
        f"- Output English only (not {target_lang}).\n"
        f"- Do NOT repeat the English prefix verbatim — output only what comes AFTER it.\n"
        f"- Avoid implausible or off-topic details.\n"
        f"- If there is no meaningful ambiguity, output only the {num_neutrals} neutral items.\n"
        f"- Output strictly the JSON object described below, nothing else.\n\n"
        f"Format:\n"
        f"{{\n"
        f"  \"items\": [\n"
        f"    {{\"span\": \"<exact span text or 'none'>\", "
        f"\"branch\": \"<short label of the branch>\", "
        f"\"continuation\": \"<4-12 English words>\"}}\n"
        f"  ]\n"
        f"}}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _extract_first_json_object(raw_text: str) -> Optional[str]:
    """Extract the first balanced top-level JSON object from raw_text. Tolerates
    leading/trailing prose or markdown fences around the JSON."""
    if not raw_text:
        return None
    s = raw_text
    # Strip common code fences.
    s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def parse_targeted_future_json(raw_text: str) -> List[Dict[str, str]]:
    """Parse the targeted-instruct JSON response into a list of items.

    Returns a list of {"span": str, "branch": str, "continuation": str}. Items with
    empty continuation are dropped. Returns [] on any parse failure (caller decides
    whether to fall back or READ).
    """
    blob = _extract_first_json_object(raw_text)
    if not blob:
        return []
    try:
        data = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return []
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: List[Dict[str, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        cont = str(it.get("continuation", "")).strip()
        if not cont:
            continue
        out.append({
            "span": str(it.get("span", "")).strip(),
            "branch": str(it.get("branch", "")).strip(),
            "continuation": cont,
        })
    return out


def _is_chat_endpoint_model(api_model: str) -> bool:
    """Decide whether to route to /chat/completions (instruct-style) or /completions (raw LM).

    Heuristic: model name contains an instruct suffix ('-it', 'instruct', 'chat').
    Examples:
      gemma4-e2b-it, qwen3-instruct, llama-3-chat → /chat/completions
      gemma4-e2b, qwen3-4b-base                    → /completions (raw text continuation)
    """
    name = str(api_model or "").lower()
    return any(tag in name for tag in ("-it", "instruct", "chat"))


def build_translation_probe_prompt(tokenizer: Any, full_source: str, target_prefix: str, target_lang: str = "Chinese") -> str:
    if not str(target_prefix or "").strip():
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nStart the {target_lang} translation from the beginning "
            "and output only the next continuation token(s)."
        )}]
    else:
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nA partial {target_lang} translation is already committed "
            "at the start of the assistant reply. You must continue from that "
            "exact prefix and produce only the continuation."
        )}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    prompt += "<|im_start|>assistant\n"
    if str(target_prefix or "").strip():
        prompt += target_prefix
    return prompt


def build_translation_probe_prompt_prefix_token_ids(
    tokenizer: Any,
    full_source: str,
    has_target_prefix: bool,
    target_lang: str = "Chinese",
) -> List[int]:
    if not has_target_prefix:
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nStart the {target_lang} translation from the beginning "
            "and output only the next continuation token(s)."
        )}]
    else:
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nA partial {target_lang} translation is already committed "
            "at the start of the assistant reply. You must continue from that "
            "exact prefix and produce only the continuation."
        )}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids.get("input_ids", [])
    elif hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    assistant_prefix_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    return list(prompt_ids) + list(assistant_prefix_ids) #返回翻译指令+assistant前缀的token ID列表，后面会拼上已提交翻译的token IDs


def build_final_completion_prompt(tokenizer: Any, full_source: str, committed_text: str, target_lang: str = "Chinese") -> str:
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


def _disallowed_generation_token_reason(tokenizer: Any, tok_id: int) -> Optional[str]:
    if tok_id is None:
        return "missing_token_id"
    if tok_id in set(getattr(tokenizer, "all_special_ids", []) or []):
        return "special_token_id"
    token_text = _single_token_text(tokenizer, tok_id)
    for frag in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        if frag in token_text:
            return "special_token_text"
    if any(ch in token_text for ch in ("*", "`")):
        return "markdown_token"
    if re.search(r"[A-Za-z]", token_text):
        return "ascii_letters"
    # NOTE: Do NOT filter replacement_char (U+FFFD) here. Byte-level BPE tokens
    # for rare characters (e.g. Japanese kanji "繰") decode to U+FFFD individually
    # but combine into valid characters when decoded together in the pending buffer.
    if any(ch in {"\u200d", "\ufe0f"} for ch in token_text):
        return "zero_width_or_variation_selector"
    if any(unicodedata.category(ch) in {"Cc", "Cs"} for ch in token_text):
        return "control_or_surrogate"
    return None


def filter_distribution_token_ids(
    tokenizer: Any,
    id_distribution: Dict[int, float],
    dist_debug: Dict[str, Any],
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    filtered: Dict[int, float] = {}
    removed: List[Dict[str, Any]] = []
    for tok_id, prob in id_distribution.items():
        reason = _disallowed_generation_token_reason(tokenizer, tok_id)
        if reason is not None:
            removed.append({"token_id": int(tok_id), "token_text": _single_token_text(tokenizer, tok_id),
                            "prob": round(float(prob), 6), "reason": reason})
            continue
        filtered[int(tok_id)] = float(prob)

    out_debug = dict(dist_debug)
    out_debug["filtered_disallowed_tokens"] = removed
    if filtered:
        out_debug["topk_token_ids"] = list(filtered.keys())
        out_debug["topk_token_texts"] = [_single_token_text(tokenizer, t) for t in filtered]
        out_debug["topk_true_probs"] = [round(float(filtered[t]), 6) for t in filtered]
    elif removed and out_debug.get("reason") == "ok":
        out_debug["reason"] = "all_top_tokens_filtered_as_disallowed"
    return filtered, out_debug


# ---------------------------------------------------------------------------
# Future sampling (API only)
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
        # Choose Method A (numbered-list, one call) or Method B (n independent calls in one batched request).
        # Set FUTURE_SAMPLING_METHOD=B to switch; default is A.
        method = os.environ.get("FUTURE_SAMPLING_METHOD", "A").strip().upper()

        if method == "B":
            # Method B: one /chat/completions call with n=num_futures. vLLM samples each completion
            # independently and returns them as separate `choices` (semantically equivalent to N
            # independent calls but batched on the GPU for speed).
            messages = build_future_sampling_chat_messages_single(observed_source)
            payload = {
                "model": api_model,
                "messages": messages,
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

        # Method A (default): one /chat/completions call returning a numbered list of N continuations.
        messages = build_future_sampling_chat_messages(observed_source, num_futures)
        # Allow enough output for ~30 words per item × num_futures + numbering overhead.
        max_tokens = max(future_tokens * num_futures, 40 * num_futures + 50)
        payload = {
            "model": api_model,
            "messages": messages,
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
        items = parse_method_a_output(raw_text, num_expected=num_futures)
        futures: List[str] = []
        for raw in items:
            cleaned = clean_future_text(observed_source, raw)
            if cleaned and is_valid_future_text(cleaned):
                futures.append(cleaned)
        return futures

    # Default: base model raw text continuation via /completions.
    payload = {
        "model": api_model,
        "prompt": observed_source,
        "max_tokens": future_tokens,
        "temperature": sample_temperature,
        "top_p": 0.95,
        "n": num_futures,
        "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    }
    data = _http_json(f"{base}/completions", payload=payload, timeout=api_timeout) #call base model，用observed source作为prompt，采样n条可能的未来源语言续写
    futures: List[str] = []
    for choice in data.get("choices", []):
        raw = str(choice.get("text", "")) if isinstance(choice, dict) else ""
        cleaned = clean_future_text(observed_source, raw)
        if cleaned and is_valid_future_text(cleaned):
            futures.append(cleaned)
    return futures

#Step1: primary base model采样未来续写，Step2: secondary base model采样未来续写，合并去重后得到所有候选未来续写列
def sample_source_futures_multi(
    base_specs: List[Dict[str, Any]],
    observed_source: str,
    future_tokens: int,
    sample_temperature: float,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    merged: List[str] = []
    merged_info: List[Dict[str, Any]] = []
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
            if key in seen:
                continue
            seen.add(key)
            merged.append(future)
            merged_info.append({"source": spec["name"], "path": spec["path"], "future": future})
    return merged, merged_info


_AXIS_HINT_SUBJECT_CONTINUES = (
    "For THIS batch: each continuation must keep the CURRENT subject / topic of "
    "the partial sentence as the focus — the same entity that is acting or being "
    "described continues to act or be described. Diversity comes from picking "
    "DIFFERENT actions, predicates, attributes, or outcomes for that SAME subject. "
    "Do NOT introduce a new subject, new actor, or scene shift in this batch."
)
_AXIS_HINT_FOCUS_SHIFTS = (
    "For THIS batch: each continuation must SHIFT FOCUS away from the current "
    "subject / topic of the partial sentence. Examples of shifts: a new entity "
    "is introduced and takes over, a different referent of an ambiguous pronoun "
    "or determiner is committed, the scene/setting changes, the perspective "
    "pivots to a bystander, or an aside / parenthetical intrudes. Diversity "
    "comes from picking DIFFERENT new subjects/topics/scenes. Do NOT keep the "
    "current subject as the focus in this batch."
)
# Narrower axes used in the experimental 5-axis split. focus_shift is replaced
# by 4 specific shift-types so both models can't collapse onto the same
# "corner, a shadow..." literary template.
_AXIS_HINT_NAMED_ACTOR_ENTERS = (
    "For THIS batch: introduce a NEW NAMED entity (a specific person, role, "
    "organization, or place name) that takes over as the actor of the next "
    "clause. Do NOT use vague placeholders like 'a shadow', 'a figure', "
    "'someone'. Use concrete names or specific roles (e.g. 'Lieutenant Cross', "
    "'the foreman', 'the Bureau of Mines', 'Marseille'). The new entity must "
    "be the grammatical subject of an action verb."
)
_AXIS_HINT_TIME_PIVOT = (
    "For THIS batch: the continuation must pivot to a DIFFERENT TIME FRAME "
    "than the present moment of the partial sentence — either a flash-forward "
    "(years later, the next morning, by the time), a flashback (long before "
    "that, as a child), or a frequency/habitual (every Sunday, in those days). "
    "The new clause should clearly anchor in a different temporal frame, with "
    "explicit temporal phrasing."
)
_AXIS_HINT_DIRECT_SPEECH = (
    "For THIS batch: the continuation must contain DIRECT QUOTED SPEECH or a "
    "speech-act report. Either an opening quote with attribution (\"That's "
    "enough,\" she said) or reported speech that signals a speech act "
    "(he muttered something about the price). The continuation must include "
    "a speech verb (said/whispered/shouted/snapped/muttered/argued)."
)
_AXIS_HINT_FACTUAL_ASIDE = (
    "For THIS batch: pivot the continuation into a FACTUAL, TECHNICAL, OR "
    "NUMERICAL aside that doesn't continue the narrative. Examples: cite a "
    "date or year ('the 1923 treaty had banned this'), give a measurement "
    "or quantity ('the temperature reached 38 degrees'), reference a "
    "document/law/source ('Article 14 of the constitution requires this'), "
    "or state a domain-specific fact. The continuation should read like an "
    "encyclopedia or footnote, not narrative prose."
)
# K-RISK axes: each axis is bound to a SPECIFIC translation failure mode.
# Goal: produce a continuation that would, if true, invalidate the
# already-committed Chinese prefix in exactly this way. If the risk doesn't
# apply to the current partial, produce a plausible neutral continuation.
_AXIS_HINT_WORD_SENSE_RISK = (
    "For THIS batch — WORD SENSE risk. Examine the partial for any noun, "
    "adjective, or content word with MULTIPLE distinct Chinese senses (e.g. "
    "'bank' = 银行 / 河岸, 'fan' = 风扇 / 粉丝, 'spring' = 春天 / 弹簧 / 泉水, "
    "'plant' = 植物 / 工厂). If such a word exists in the partial AND its "
    "sense is not yet pinned down, write a continuation that resolves it to "
    "a SPECIFIC sense which forces a DIFFERENT Chinese word than the most "
    "default reading would commit. If no such ambiguous word exists, write a "
    "plausible neutral continuation that follows the partial naturally."
)
_AXIS_HINT_VERB_ROLE_RISK = (
    "For THIS batch — VERB SENSE / ROLE risk. Examine the partial for any "
    "verb whose Chinese translation depends on context (e.g. 'raise' = 提高 / "
    "筹集 / 抚养 / 提出, 'run' = 经营 / 运行 / 竞选 / 跑, 'present' = 呈现 / "
    "赠送 / 介绍, 'address' = 处理 / 演讲 / 致函, 'charge' = 收费 / 控告 / "
    "充电 / 冲锋). If the partial contains a verb whose sense is still "
    "undetermined, write a continuation that pins it to a SPECIFIC sense "
    "that would force a different Chinese verb than the default. If no such "
    "verb is in the partial, write a plausible neutral continuation."
)
_AXIS_HINT_ATTACHMENT_RISK = (
    "For THIS batch — ATTACHMENT risk. Examine the partial for any "
    "prepositional phrase, relative clause, adverbial, or modifier that could "
    "plausibly attach to MULTIPLE different heads, leading to different "
    "Chinese sentence structures (classic PP-attachment 'I saw the man with "
    "the telescope', or relative-clause attachment 'the daughter of the king "
    "who was crowned yesterday'). If such an attachment ambiguity is present "
    "in the partial, write a continuation that disambiguates the attachment "
    "toward the LESS OBVIOUS reading. If no attachment ambiguity exists, "
    "write a plausible neutral continuation."
)
_AXIS_HINT_PREDICATE_DELAY_RISK = (
    "For THIS batch — MAIN PREDICATE DELAY risk. Examine the partial: has "
    "the MAIN verb of the matrix clause been heard yet? Many sentences start "
    "with a subject + relative clause / appositive / adverbial that DELAYS "
    "the main predicate (e.g. 'The agreement that the two countries signed "
    "last year...' — the matrix verb is still pending). If the main predicate "
    "is still pending, write a continuation that supplies a SPECIFIC matrix "
    "predicate which would force a non-default Chinese verb at the matrix "
    "position. If the main predicate has already been heard, write a "
    "plausible neutral continuation."
)
_AXIS_HINT_POLARITY_REVERSAL_RISK = (
    "For THIS batch — POLARITY / REVERSAL risk. Examine the partial: is its "
    "current trajectory positive / successful / affirmative / cooperative? "
    "Could it be REVERSED by a subsequent 'but', 'although', 'however', "
    "'failed to', 'turned out not to', 'unfortunately', 'instead'? If a "
    "polarity flip is plausible given the partial, write a continuation that "
    "flips it (positive → negative, success → failure, acceptance → "
    "rejection, confirmation → denial). If no such reversal is plausible, "
    "write a plausible neutral continuation that preserves the current "
    "trajectory."
)
# L axes: softened version of J's narrative-diversity axes.
# - keep SUBJECT_CONTINUES (baseline)
# - loosen NAMED_ACTOR → NEW_ACTOR_OR_TOPIC (definite NP OK, no "specific" name required)
# - extend TIME_PIVOT → add CONDITIONAL anchors (if/when/before/after)
# - relax DIRECT_SPEECH → SPEECH_ACT_OR_ATTITUDE (reported speech + mental stance, no quote required)
# - replace FACTUAL_ASIDE → POLARITY_OR_OUTCOME_FLIPS (translation-relevant direction flip)
_AXIS_HINT_NEW_ACTOR_OR_TOPIC = (
    "For THIS batch: introduce a NEW grammatical subject / topic that takes "
    "over the next clause. The new subject must be SPECIFIC — a definite NP "
    "or proper noun. ACCEPTABLE: 'the guard', 'the queen', 'the committee', "
    "'the old woman', 'the crowd', 'Lieutenant Cross', 'the foreman', "
    "'Marseille'. NOT ACCEPTABLE: 'a shadow', 'a figure', 'someone', "
    "'something', 'a voice'. The new entity must serve as the grammatical "
    "subject of an action verb in the continuation."
)
_AXIS_HINT_TIME_OR_CONDITION_PIVOT = (
    "For THIS batch: the continuation must pivot to a DIFFERENT TIME FRAME "
    "or introduce a CONDITIONAL / TEMPORAL anchor. Acceptable forms: "
    "flash-forward ('years later, the next morning, by the time'), flashback "
    "('long before that, as a child'), frequency ('every Sunday, in those "
    "days'), or conditional ('if the rain held, when the bell rang, before "
    "they noticed, after the meeting was over, unless the king agreed'). "
    "The continuation must clearly anchor in a different temporal or "
    "conditional frame from the partial."
)
_AXIS_HINT_SPEECH_ACT_OR_ATTITUDE = (
    "For THIS batch: the continuation must signal a SPEECH ACT, REPORT, or "
    "MENTAL STANCE. This includes reported speech ('asked whether...', "
    "'warned that...', 'claimed that...', 'insisted on...', 'muttered "
    "something about...', 'argued that...') and mental states ('wondered "
    "if...', 'feared that...', 'realized...', 'understood that...', "
    "'suspected...', 'doubted whether...'). Direct quotes with attribution "
    "are allowed but NOT required. The continuation must include a verb "
    "signalling speech or mental stance."
)
_AXIS_HINT_POLARITY_OUTCOME_FLIPS = (
    "For THIS batch: the continuation must FLIP the polarity or outcome of "
    "the partial's current trajectory. If the partial is heading toward "
    "success / affirmation / cooperation / approval, flip to failure / "
    "denial / refusal / rejection. If it's heading toward failure, flip to "
    "success. Examples: 'succeeded' → 'failed at the last moment', "
    "'accepted' → 'rejected outright', 'was praised' → 'was condemned', "
    "'proved true' → 'turned out false', 'agreed' → 'refused', 'survived' → "
    "'died'. The continuation must change the direction of the outcome from "
    "what the partial implies."
)


def _sample_prefill_one_batch(
    sampler_tokenizer: Any,
    observed_source: str,
    committed_text: str,
    target_lang: str,
    num_futures: int,
    axis_hint: str,
    axis_tag: str,
    api_base: str,
    api_model: str,
    api_timeout: float,
    sample_temperature: float,
    top_p: float,
    max_tokens: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """One sub-batch of prefill sampling with a specific axis hint.

    Same prefill mechanism as before (partial only in prefilled assistant turn,
    not in user msg), same (c) hard-drop filter, but the user message carries
    an axis_hint that pins this batch to ONE diversity axis ("subject continues"
    vs "focus shifts"). Two calls of this helper with the two hints together
    span both axes and prevent gemma from collapsing all N samples into a single
    axis (e.g. 16 paraphrases of "results suggest the hypothesis was flawed").
    """
    if not observed_source.strip() or num_futures <= 0:
        return [], []

    system_msg = (
        f"You are an adversarial future-probe for a simultaneous English-to-"
        f"{target_lang} interpreter. The interpreter has heard only a partial "
        f"English sentence and has already committed part of a {target_lang} "
        f"translation. Your job is NOT to write the most natural next words. "
        f"Your job is to surface continuations that would FORCE the interpreter "
        f"to revise the current {target_lang} translation if they turned out to "
        f"be true.\n\n"
        f"Hard requirements:\n"
        f"- Produce 4-15 word English continuations, one per sample.\n"
        f"- Across your N samples, each continuation must commit to a DIFFERENT "
        f"interpretation along at least one axis: (a) what an ambiguous pronoun "
        f"or determiner refers to, (b) which verb sense is intended, (c) which "
        f"phrase a PP / relative clause attaches to, (d) who/what becomes the "
        f"new subject or topic, (e) polarity / outcome (positive vs negative).\n"
        f"- Do NOT all collapse onto the same syntactic frame (e.g. all starting "
        f"with the same subject pronoun). Span the space of plausible readings.\n"
        f"- Each continuation should be one a competent reader would consider "
        f"genuinely possible given the partial sentence, but it should pick a "
        f"specific reading rather than stay vague.\n"
        f"- Output only the continuation text. No analysis, no JSON, no "
        f"markdown, no {target_lang}.\n\n"
        f"Example A — verb polysemy. Partial: \"She decided to run\"\n"
        f"BAD (all collapse to the athletic sense):\n"
        f"  - \"the marathon next weekend\"\n"
        f"  - \"five miles before breakfast\"\n"
        f"  - \"in the park every morning\"\n"
        f"GOOD (each picks a different verb sense):\n"
        f"  - \"the company after her father retired\"      # run = manage\n"
        f"  - \"for mayor in the next election\"            # run = stand for office\n"
        f"  - \"the experiment one more time tonight\"      # run = conduct\n"
        f"  - \"the water until it got warm enough\"        # run = operate (a faucet)\n"
        f"  - \"away before they noticed her absence\"      # run = flee\n\n"
        f"Example B — noun polysemy (bank = 银行 vs 河岸). Partial: \"The bank\"\n"
        f"BAD (all collapse to financial):\n"
        f"  - \"approved the loan quickly\"\n"
        f"  - \"raised its lending rates\"\n"
        f"  - \"lent us a large sum\"\n"
        f"GOOD (forces a different {target_lang} word entirely):\n"
        f"  - \"approved the loan by Friday afternoon\"     # bank = 银行\n"
        f"  - \"collapsed after three days of heavy rain\"  # bank = 河岸 (caved in)\n"
        f"  - \"was lined with old willow trees\"            # bank = 河岸 (scenery)\n"
        f"  - \"froze all withdrawals without warning\"      # bank = 银行, negative\n\n"
        f"Example C — long-distance dependency (matrix verb still pending). "
        f"Partial: \"The agreement that the two countries signed last year\"\n"
        f"BAD (no main verb chosen, all stay on the subject):\n"
        f"  - \"is a very important document\"\n"
        f"  - \"was very long and detailed\"\n"
        f"  - \"contains many clauses\"\n"
        f"GOOD (each commits to a different matrix predicate / outcome):\n"
        f"  - \"expires at the end of this month\"           # 即将失效\n"
        f"  - \"has been violated repeatedly since then\"    # 多次被违反 (negative)\n"
        f"  - \"remains the foundation of their alliance\"   # 仍是基石 (positive)\n"
        f"  - \"will be reviewed by the new government\"     # 将被审议 (future)\n"
        f"  - \"was secretly abandoned only weeks later\"    # 被废止 (reversal)"
    )
    # NOTE: Intentionally do NOT quote the partial English here. The partial
    # appears ONLY in the prefilled assistant turn below, so the model has no
    # "second copy" to rewrite from. This is the (a) fix for prefix
    # regurgitation: the model can only continue the prefill, it cannot
    # restate the partial because it never sees it as a quotable string.
    if committed_text and committed_text.strip():
        user_msg = (
            f"The assistant turn has been started for you with a partial "
            f"English sentence (mid-sentence). Continue it from exactly where "
            f"it ends, by writing 4-15 more words. Do NOT restate, paraphrase, "
            f"or capitalize the partial; begin mid-sentence in lowercase unless "
            f"the partial ended with terminal punctuation.\n\n"
            f"{target_lang} already committed by the interpreter:\n"
            f"{committed_text}\n\n"
            f"{axis_hint}\n\n"
            f"Across your N samples, each continuation MUST pick a DIFFERENT "
            f"interpretation that would force the interpreter to revise the "
            f"committed {target_lang}. Output only the continuation text."
        )
    else:
        user_msg = (
            f"The assistant turn has been started for you with a partial "
            f"English sentence (mid-sentence). Continue it from exactly where "
            f"it ends, by writing 4-15 more words. Do NOT restate, paraphrase, "
            f"or capitalize the partial; begin mid-sentence in lowercase unless "
            f"the partial ended with terminal punctuation.\n\n"
            f"{axis_hint}\n\n"
            f"Across your N samples, each continuation MUST pick a DIFFERENT "
            f"interpretation of what is currently ambiguous. Output only the "
            f"continuation text."
        )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Apply chat template up to (and including) the assistant turn opener, then
    # prefill the observed source so the model can only generate the continuation.
    prompt = sampler_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt += observed_source

    base = normalize_api_base(api_base)
    payload: Dict[str, Any] = {
        "model": api_model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": sample_temperature,
        "top_p": top_p,
        "n": num_futures,
        # Push the N samples apart in token space to fight mode collapse.
        # Reduced from 0.5 → 0.1: with 5 hard-coded axis hints already forcing
        # divergence, a high presence_penalty stacks as double-divergence and
        # over-pushes the sampler away from natural high-prob continuations.
        "presence_penalty": 0.1,
        # Common end-of-turn / line tokens across Qwen / Gemma / Llama families.
        "stop": ["\n\n", "<|im_end|>", "<end_of_turn>", "<|endoftext|>", "<|eot_id|>"],
    }
    try:
        data = _http_json(f"{base}/completions", payload=payload, timeout=api_timeout)
    except Exception:
        return [], []
    choices = data.get("choices", []) if isinstance(data, dict) else []

    # Build a normalized form of the partial (lowercased, punctuation/space
    # stripped) so the (c) regurgitation check can catch capitalized/reformatted
    # restatements like "They held on for some time, ..." that don't startswith()
    # match the raw partial.
    def _norm_for_dup(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())
    partial_norm = _norm_for_dup(observed_source)
    # Use a min length to avoid false-positives on very short partials.
    partial_norm_min_len = max(8, len(partial_norm) // 2)

    futures: List[str] = []
    items_info: List[Dict[str, Any]] = []
    seen: set = set()
    for choice in choices:
        raw = str(choice.get("text", "")) if isinstance(choice, dict) else ""
        cleaned = strip_markdown_wrappers(raw.strip())
        # vLLM /completions returns only the new tokens (not the prompt), so we
        # should already be prefix-free. Defensive strip in case any echo sneaks in.
        if cleaned.lower().startswith(observed_source.lower()):
            cleaned = cleaned[len(observed_source):].lstrip()
        # First-line only — the model may try to extend across multiple sentences.
        cleaned = strip_markdown_wrappers(cleaned.split("\n")[0].strip())
        # Reuse the existing validators (no zh chars, no analysis filler).
        cleaned = re.sub(r"^[\.\s…\-]+", "", cleaned)
        if not cleaned or not is_valid_future_text(cleaned):
            continue
        # (c) Hard-drop continuations that contain the partial (or a long
        # substring of it) anywhere — handles restart/capitalization cases
        # where the model rewrote the partial before its actual continuation.
        if partial_norm and len(partial_norm) >= partial_norm_min_len:
            cleaned_norm = _norm_for_dup(cleaned)
            if partial_norm in cleaned_norm:
                continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        futures.append(cleaned)
        items_info.append({
            "source": f"targeted_prefill_{axis_tag}",
            "path": api_model,
            "future": cleaned,
        })
    return futures, items_info


def sample_source_futures_targeted_prefill(
    sampler_tokenizer: Any,
    observed_source: str,
    committed_text: str,
    target_lang: str,
    num_futures: int,
    api_base: str,
    api_model: str,
    api_timeout: float,
    sample_temperature: float = 1.0,
    top_p: float = 0.98,
    max_tokens: int = 40,
    sampler2_tokenizer: Any = None,
    sampler2_api_base: str = "",
    sampler2_api_model: str = "",
    sampler2_api_timeout: float = 0.0,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """C-2/C-3: dual-batch prefill sampler with optional 2nd model ensemble.

    Single-sampler mode (default): runs 2 sub-batches with the same model:
      - subject continues   (n=num_futures/2)
      - focus shifts        (n=num_futures/2)

    Dual-sampler mode (when sampler2_* all set): runs 4 sub-batches:
      - sampler1 × subj_cont    (n=num_futures)
      - sampler1 × focus_shift  (n=num_futures)
      - sampler2 × subj_cont    (n=num_futures)
      - sampler2 × focus_shift  (n=num_futures)
    Each model produces 2*num_futures raw samples, total 4*num_futures, then
    cross-batch dedup. The two models come from different families (e.g. gemma
    + Qwen3) so the prior overlap that locks gemma into mode collapse on
    "sent [NOUN]" type prefixes is broken by the orthogonal Qwen3 distribution.
    """
    if not observed_source.strip():
        return [], []
    use_ensemble = bool(sampler2_tokenizer and sampler2_api_base and sampler2_api_model)
    # PRODUCTION (J): artificial-forcing 5-axis split. Empirically beat both
    # K-risk (detect-then-expose) and L (softened forcing) on BLEU / Jaccard
    # / lenRef. The "artificial" requirement (must be named entity / must
    # use quotes / must read like encyclopedia footnote) is what forces
    # small models to break out of high-prob defaults — softening or
    # replacing with risk detection lets the sampler fall back to default
    # continuations (e.g. 33-36 "letters" futures at chunk 20 in K/L vs
    # 1-2 "science" variants in J).
    AXES = [
        ("subj_cont",    _AXIS_HINT_SUBJECT_CONTINUES),
        ("named_actor",  _AXIS_HINT_NAMED_ACTOR_ENTERS),
        ("time_pivot",   _AXIS_HINT_TIME_PIVOT),
        ("direct_speech",_AXIS_HINT_DIRECT_SPEECH),
        ("factual",      _AXIS_HINT_FACTUAL_ASIDE),
    ]
    if use_ensemble:
        # num_futures is interpreted as "per-model total". With 5 axes that
        # means num_futures/5 per (model, axis) sub-batch. With the user's
        # num_futures=20: 4 samples × 5 axes × 2 models = 40 raw samples.
        per_axis = max(1, num_futures // len(AXES))
        s2_to = sampler2_api_timeout if sampler2_api_timeout and sampler2_api_timeout > 0 else api_timeout
        batches = []
        for tag, hint in AXES:
            batches.append(_sample_prefill_one_batch(
                sampler_tokenizer=sampler_tokenizer,
                observed_source=observed_source, committed_text=committed_text,
                target_lang=target_lang, num_futures=per_axis,
                axis_hint=hint, axis_tag=f"s1_{tag}",
                api_base=api_base, api_model=api_model, api_timeout=api_timeout,
                sample_temperature=sample_temperature, top_p=top_p, max_tokens=max_tokens,
            ))
            batches.append(_sample_prefill_one_batch(
                sampler_tokenizer=sampler2_tokenizer,
                observed_source=observed_source, committed_text=committed_text,
                target_lang=target_lang, num_futures=per_axis,
                axis_hint=hint, axis_tag=f"s2_{tag}",
                api_base=sampler2_api_base, api_model=sampler2_api_model, api_timeout=s2_to,
                sample_temperature=sample_temperature, top_p=top_p, max_tokens=max_tokens,
            ))
    else:
        half_a = max(1, num_futures // 2)
        half_b = max(1, num_futures - half_a)
        s1_a = _sample_prefill_one_batch(
            sampler_tokenizer=sampler_tokenizer,
            observed_source=observed_source, committed_text=committed_text,
            target_lang=target_lang, num_futures=half_a,
            axis_hint=_AXIS_HINT_SUBJECT_CONTINUES, axis_tag="subj_cont",
            api_base=api_base, api_model=api_model, api_timeout=api_timeout,
            sample_temperature=sample_temperature, top_p=top_p, max_tokens=max_tokens,
        )
        s1_b = _sample_prefill_one_batch(
            sampler_tokenizer=sampler_tokenizer,
            observed_source=observed_source, committed_text=committed_text,
            target_lang=target_lang, num_futures=half_b,
            axis_hint=_AXIS_HINT_FOCUS_SHIFTS, axis_tag="focus_shift",
            api_base=api_base, api_model=api_model, api_timeout=api_timeout,
            sample_temperature=sample_temperature, top_p=top_p, max_tokens=max_tokens,
        )
        batches = [s1_a, s1_b]
    # Merge with cross-batch dedup (lowercased).
    seen = set()
    merged_f: List[str] = []
    merged_i: List[Dict[str, Any]] = []
    for futs, infos in batches:
        for f, info in zip(futs, infos):
            k = f.lower()
            if k in seen:
                continue
            seen.add(k)
            merged_f.append(f)
            merged_i.append(info)
    return merged_f, merged_i


def sample_source_futures_targeted_instruct(
    observed_source: str,
    committed_text: str,
    target_lang: str,
    max_spans: int,
    num_neutrals: int,
    api_base: str,
    api_model: str,
    api_timeout: float,
    sample_temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 600,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Method C: ambiguity-targeted future sampling via the instruct model.

    Returns (futures, items_info) where:
      - futures: deduped list of English continuations (drop-in for downstream consensus)
      - items_info: parallel list of dicts {"source": "targeted_instruct", "span", "branch",
                    "continuation"} aligned with `futures` (for verbose logging / future
                    per-span aggregation)

    Returns ([], []) on parse failure or empty observed_source; the caller will then READ.
    """
    if not observed_source.strip():
        return [], []
    base = normalize_api_base(api_base)
    messages = build_targeted_future_messages(
        observed_source=observed_source,
        committed_text=committed_text or "",
        target_lang=target_lang,
        max_spans=max_spans,
        num_neutrals=num_neutrals,
    )
    payload: Dict[str, Any] = {
        "model": api_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": sample_temperature,
        "top_p": top_p,
        "n": 1,
        # vLLM honors this and gates decoding to valid JSON. If the backend doesn't
        # support it, vLLM ignores the field rather than erroring.
        "response_format": {"type": "json_object"},
    }
    try:
        data = _http_json(f"{base}/chat/completions", payload=payload, timeout=api_timeout)
    except Exception:
        return [], []
    choices = data.get("choices", []) if isinstance(data, dict) else []
    if not choices:
        return [], []
    msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    raw_text = str(msg.get("content", "")) if isinstance(msg, dict) else ""
    items = parse_targeted_future_json(raw_text)

    futures: List[str] = []
    items_info: List[Dict[str, Any]] = []
    seen: set = set()
    for it in items:
        cleaned = clean_future_text(observed_source, it["continuation"])
        if not cleaned or not is_valid_future_text(cleaned):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        futures.append(cleaned)
        items_info.append({
            "source": "targeted_instruct",
            "path": api_model,
            "span": it["span"],
            "branch": it["branch"],
            "future": cleaned,
        })
    return futures, items_info


# ---------------------------------------------------------------------------
# Next-token distribution (API only, batch)
# ---------------------------------------------------------------------------

def _parse_token_id_string(raw: str) -> Optional[int]:
    match = re.fullmatch(r"token_id:(\d+)", str(raw or "").strip())
    return int(match.group(1)) if match else None


def _parse_completion_top_logprobs(
    # The raw "top_logprobs" from the API is a list of dicts, 
    # one per generated token, where each dict maps
    # token_id strings like "token_id:1234" to logprobs. 
    #We only care about the first step (next token).
    top_logprobs: Optional[List[Optional[Dict[str, float]]]],
    tokenizer: Any,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    if not top_logprobs:
        return {}, {"reason": "missing_top_logprobs"}
    step = top_logprobs[0]
    if not step:
        return {}, {"reason": "empty_top_logprobs"}
    id_distribution: Dict[int, float] = {}
    unknown_tokens: List[str] = []
    for raw_token, logprob in step.items():
        tok_id = _parse_token_id_string(raw_token)
        if tok_id is None:
            unknown_tokens.append(str(raw_token))
            continue
        id_distribution[tok_id] = float(math.exp(float(logprob))) #logprob是对数概率，exp转回真实概率
    token_ids = list(id_distribution.keys())
    # {
    #     123: 0.82,
    #     456: 0.21,
    #     789: 0.07
    # }
    return id_distribution, {
        "reason": "ok" if id_distribution else "no_token_ids_in_top_logprobs",
        "topk_token_ids": token_ids,
        "topk_token_texts": [_single_token_text(tokenizer, t) for t in token_ids],
        "topk_true_probs": [round(float(id_distribution[t]), 6) for t in token_ids],
        "unknown_top_logprob_tokens": unknown_tokens,
    }


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
) -> List[Tuple[Dict[int, float], Dict[str, Any]]]:
# Step 1: 用 chat template 把翻译指令编码成 token IDs
#         → [<|im_start|>, user, \n, [TASK], Translate..., <|im_end|>]
        
# Step 2: 拼上 assistant 前缀的 token IDs
#         → 上面 + encode("<|im_start|>assistant\n")
        
# Step 3: 拼上已提交翻译的 token IDs
#         → 上面 + [32108, 45621]
    prompts = [
        build_translation_probe_prompt_prefix_token_ids(
            tokenizer,
            src,
            has_target_prefix=bool(target_prefix_token_ids),
            target_lang=target_lang,
        ) + list(target_prefix_token_ids) # Step 3
        for src in full_sources
    ]
    # [user task tokens] + [assistant prefix tokens] + [already committed target prefix tokens]
    if not prompts:
        return []
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
    data = _http_json(f"{normalize_api_base(api_base)}/completions", payload=payload, timeout=api_timeout) #call instruct model，传token ID列表，只生成1个token，拿top-k logprobs分布
    choices = data.get("choices", []) #vllm做batch处理，每个choice对应一个future的next-token分布
    results: List[Tuple[Dict[int, float], Dict[str, Any]]] = []
    for i in range(len(prompts)):
        if i >= len(choices):
            results.append(({}, {"reason": "missing_choice", "raw_response": data}))
            continue
        choice = choices[i] 
        # choice = {
        #     "text": "而",
        #     "logprobs": {
        #         "tokens": ["token_id:123"],
        #         "top_logprobs": [
        #             {
        #                 "token_id:123": -0.1,
        #                 "token_id:456": -1.2,
        #                 "token_id:789": -2.0,
        #                 "token_id:999": -3.1,
        #                 "token_id:555": -4.0,
        #             }
        #         ]
        #     }
        # }
        logprobs = choice.get("logprobs", {}) if isinstance(choice, dict) else {} #每个choice里有一个"logprobs"字段，里面的"top_logprobs"是一个列表，每个元素是一个dict，表示对应生成token的top-k logprobs分布；我们只关心第一个生成token的分布，所以取top_logprobs[0]来解析出token ID分布

        dist, dist_debug = _parse_completion_top_logprobs(logprobs.get("top_logprobs"), tokenizer=tokenizer) #解析出token ID分布，并记录debug信息
        dist_debug["api_backend"] = "vllm_completion"
        if min_p > 0:
            dist_debug["candidate_policy"] = "min_p"
            dist_debug["min_p"] = min_p
        else:
            dist_debug["candidate_policy"] = "top_k"
            dist_debug["candidate_top_k"] = top_k
        results.append(filter_distribution_token_ids(tokenizer, dist, dist_debug)) #过滤掉不合规的token ID，并在debug信息里记录被过滤掉的token ID和原因
    return results


# ---------------------------------------------------------------------------
# Consensus logic
# ---------------------------------------------------------------------------

def topk_token_ids(dist: Dict[int, float], k: int = TOP_K) -> List[int]:
    # Return the top-k token IDs by probability, sorted in descending order of probability.
    return [tok_id for tok_id, _ in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def minp_token_ids(dist: Dict[int, float], min_p: float) -> List[int]:
    """Return token IDs whose probability >= min_p (absolute threshold)."""
    if not dist:
        return []
    return [tok_id for tok_id, prob in sorted(dist.items(), key=lambda kv: kv[1], reverse=True) if prob >= min_p]


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
) -> Tuple[Optional[int], Dict[str, Any]]:
    if not distributions:
        return None, {"reason": "no_distributions"}
    candidate_lists = [_select_candidates(dist, top_p=top_p, min_p=min_p, top_k=candidate_top_k)
                       for dist in distributions]
    # Soft voting / sum-of-probabilities ensemble (product-of-experts style).
    # Replaces the previous strict set-intersection: now the token with the
    # highest summed probability across all futures wins, even if not every
    # future ranked it in their top-k. This lets ≥majority agreement override
    # a single dissenting future.
    agg: Counter = Counter()
    for dist in distributions:
        for tok, p in dist.items():
            agg[tok] += p
    if not agg:
        return None, {"reason": "no_distributions"}
    best_token, best_score = agg.most_common(1)[0]
    return best_token, {
        "reason": "ok",
        "intersection": [tok for tok, _ in agg.most_common(candidate_top_k)],
        "avg_score": best_score / len(distributions),
        "candidate_lists": candidate_lists,
    }


# ---------------------------------------------------------------------------
# Token buffer management
# ---------------------------------------------------------------------------

def decode_token_ids_to_text(tokenizer: Any, token_ids: List[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def inspect_token_ids(tokenizer: Any, token_ids: List[int]) -> Dict[str, Any]:
    # Decode the entire token ID buffer to text for debugging and logging.  This is NOT used for any model input or state management, to avoid drift from re-encoding incomplete token sequences.
    decoded_text = decode_token_ids_to_text(tokenizer, token_ids)
    last_token_id = token_ids[-1] if token_ids else None
    last_token_text = _single_token_text(tokenizer, last_token_id) if last_token_id is not None else ""
    return {"decoded_text": decoded_text, "last_token_id": last_token_id, "last_token_text": last_token_text}


def has_suspicious_content(text: str) -> bool:
    if not text:
        return False
    # Check the ENTIRE decoded text for replacement characters, not just tail.
    # Byte-level BPE tokens may form broken sequences anywhere in the buffer
    # when consensus picks a byte token but fails to pick its pair.
    if "\ufffd" in text or "�" in text:
        return True
    last_char = text[-1]
    if last_char in {"\u200d", "\ufe0f"}:
        return True
    if unicodedata.category(last_char) in {"Mn", "Mc", "Me", "Cc", "Cs"}:
        return True
    return False


def sanitize_pending_token_ids(
    tokenizer: Any, pending_token_ids: List[int],
) -> Tuple[List[int], List[Dict[str, Any]]]:
# Filter out any disallowed tokens from the pending token ID buffer, and log the removed tokens with reasons.  This ensures that we never commit or feed back into the model any token that could cause decoding issues or model confusion.
    kept: List[int] = []
    removed: List[Dict[str, Any]] = []
    for idx, tok_id in enumerate(pending_token_ids):
        reason = _disallowed_generation_token_reason(tokenizer, tok_id)
        if reason is None:
            kept.append(tok_id)
        else:
            removed.append({"position": idx, "token_id": int(tok_id),
                            "token_text": _single_token_text(tokenizer, tok_id), "reason": reason})
    return kept, removed


def trim_pending_tokens_to_complete_boundary(
    tokenizer: Any, committed_text: str, pending_token_ids: List[int],
) -> Tuple[List[int], List[Dict[str, Any]], List[int], Dict[str, Any]]:
    work, removed_disallowed = sanitize_pending_token_ids(tokenizer, pending_token_ids)
    if not work:
        return [], removed_disallowed, [], {
            "decoded_text": "", "last_token_id": None, "last_token_text": "", "full_text": committed_text,
        }
    # Commit only an exact prefix of the original token-id buffer.
    # This preserves the true model state and avoids drift from re-encoding text
    # such as ["而", "且"] into a merged token like ["而且"].
    last_committable_idx = -1 #从前往后逐个decode，找到最后一个"解码后没有乱码"的位置
    for i in range(len(work)):
        partial = decode_token_ids_to_text(tokenizer, work[:i + 1])
        if not has_suspicious_content(partial): #没有U+FFFD等异常字符，说明到这里的token组合是完整的
            last_committable_idx = i
    if last_committable_idx >= 0:
        clean_ids = work[:last_committable_idx + 1]
        removed_tail = work[last_committable_idx + 1:]
        view = inspect_token_ids(tokenizer, clean_ids)
        view["full_text"] = committed_text + view["decoded_text"]
        return clean_ids, removed_disallowed, removed_tail, view
    return [], removed_disallowed, work, {
        "decoded_text": "", "last_token_id": None, "last_token_text": "", "full_text": committed_text,
    }


def finalize_external_commit(
    tokenizer: Any, committed_text: str, pending_token_ids: List[int],
) -> Tuple[str, str, List[int], Dict[str, Any]]:
    trimmed, removed_disallowed, removed_tail, view = trim_pending_tokens_to_complete_boundary(
        tokenizer, committed_text, pending_token_ids)
    commit_text = view["decoded_text"]
    return committed_text + commit_text, commit_text, trimmed, {
        "pending_before_trim": decode_token_ids_to_text(tokenizer, pending_token_ids),
        "pending_after_trim": commit_text,
        "removed_disallowed_tokens": removed_disallowed,
        "removed_tail_text": decode_token_ids_to_text(tokenizer, removed_tail),
    }


# ---------------------------------------------------------------------------
# Core consensus loop
# ---------------------------------------------------------------------------

def extend_pending_tokens(
    instruct_tokenizer: Any,
    source_observed: str,
    futures: List[str],
    committed_text: str,
    committed_token_ids: List[int],
    max_consensus_steps: int,
    candidate_top_k: int = TOP_K,
    instruct_api_base: str = "",
    instruct_api_model: str = "",
    instruct_api_timeout: float = 120.0,
    min_p: float = 0.0,
    top_p: float = 0.0,
    target_lang: str = "Chinese",
) -> Tuple[List[int], List[Dict[str, Any]]]:
    pending_token_ids: List[int] = []
    grow_logs: List[Dict[str, Any]] = []

    for step_idx in range(max_consensus_steps): #最多32 步
        # 2a) 每条 future 拼成完整 source，问 instruct 模型："给定这个完整英文 + 已有中文前缀，下一个中文 token 是啥？"
        target_prefix_token_ids = list(committed_token_ids) + list(pending_token_ids)
        full_sources = [append_text_continuation(source_observed, f) for f in futures]

        batch_results = batch_get_next_token_distributions(
            tokenizer=instruct_tokenizer,
            full_sources=full_sources,
            target_prefix_token_ids=target_prefix_token_ids,
            top_k=candidate_top_k,
            min_p=min_p,
            top_p=top_p,
            api_base=instruct_api_base,
            api_model=instruct_api_model,
            api_timeout=instruct_api_timeout,
            target_lang=target_lang,
        )

        distributions: List[Dict[int, float]] = []
        per_future: List[Dict[str, Any]] = []
        for i, (dist, dist_debug) in enumerate(batch_results):
            if not dist:
                grow_logs.append({"step": step_idx, "stop": "empty_distribution",
                                  "future": futures[i], "dist_debug": dist_debug})
                return pending_token_ids, grow_logs
            distributions.append(dist)
            # select 3 policies: topk, topp, minp
            candidate_ids = _select_candidates(dist, top_p=top_p, min_p=min_p, top_k=candidate_top_k)
            per_future.append({
                "future": futures[i],
                "candidate_texts": [_single_token_text(instruct_tokenizer, t) for t in candidate_ids],
                "candidate_probs": [dist.get(t, 0.0) for t in candidate_ids],
                "num_candidates": len(candidate_ids),
            })

        # 共识选 token
        consensus_token_id, meta = choose_consensus_token(distributions, min_p=min_p, candidate_top_k=candidate_top_k, top_p=top_p)
        if consensus_token_id is None: #没有共识token，停止本轮生长
            grow_logs.append({"step": step_idx, "stop": "no_consensus_token",
                              "per_future": per_future, "meta": meta})
            break

        pending_token_ids.append(consensus_token_id) #共识成功，追加到pending buffer，继续下一步
        view = inspect_token_ids(instruct_tokenizer, pending_token_ids) #解码当前pending buffer，准备日志
        log_entry = {
            "step": step_idx,
            "accepted_token_id": consensus_token_id,
            "accepted_token_text": view["last_token_text"],
            "pending_text": view["decoded_text"],
            "llm_prefix": decode_token_ids_to_text(instruct_tokenizer, target_prefix_token_ids),
            "llm_prefix_token_ids": target_prefix_token_ids,
            "per_future": per_future,
            "meta": meta,
        }
        grow_logs.append(log_entry)

    return pending_token_ids, grow_logs


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
    data = _http_json(f"{normalize_api_base(api_base)}/completions", payload=payload, timeout=api_timeout) #call instruct model，带上已提交翻译前缀，让模型把剩余翻译补完
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


# ---------------------------------------------------------------------------
# Run one utterance
# ---------------------------------------------------------------------------

def run_one_utterance(
    row: Dict[str, Any],
    args: argparse.Namespace,
    base_specs: List[Dict[str, Any]],
    instruct_tokenizer: Any,
    sampler_tokenizer: Any = None,
    sampler2_tokenizer: Any = None,
    verbose_log_file: Optional[Any] = None,
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

    # ── verbose header ──
    _vlog(verbose_log_file, "############################################################")
    _vlog(verbose_log_file, f"# utt_id: {utt_id}")
    _vlog(verbose_log_file, f"# source_full_text: {full_source_text}")
    _vlog_pretty_value(verbose_log_file, "src_text_full", source_units)
    _vlog_pretty_value(verbose_log_file, "src_trajectory", chunks)
    _vlog(verbose_log_file, f"# Chunks: {len(chunks)}")
    _vlog(verbose_log_file, f"# num_futures={args.num_futures}, top_k={args.candidate_top_k}, min_p={args.min_p}")
    for si, bs in enumerate(base_specs):
        label = bs.get("name", f"spec_{si}")
        _vlog(verbose_log_file, f"# base_model[{label}]: api model={bs.get('api_model','')} base={bs.get('api_base','')} num_futures={bs.get('num_futures','')}")
    _vlog(verbose_log_file, f"# instruct_backend: vllm_completion")
    _vlog(verbose_log_file, "############################################################")

    for t in range(len(chunks)):
        current_source_chunk = str(chunks[t] or "")
        source_observed_full = build_source_observed(chunks, t)
        source_observed = build_source_observed_recent_units(
            source_units=source_units,
            observed_full=source_observed_full,
            num_units=args.future_source_window_chunks,
        )
        _vlog(verbose_log_file, f"\n{'='*60}")
        _vlog(verbose_log_file, f"Chunk {t + 1}/{len(chunks)}")
        _vlog(verbose_log_file, f"source_observed: {current_source_chunk!r}")
        _vlog(verbose_log_file, f"future_source_prefix: {source_observed!r}")
        if source_observed != source_observed_full:
            _vlog(verbose_log_file, f"source_observed_full: {source_observed_full!r}")
        _vlog(verbose_log_file, f"committed_before: {committed_text!r}")

        if t == len(chunks) - 1: #最后一个chunk，不再做共识，直接让instruct model把翻译补完
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
            _vlog(verbose_log_file, f"  [Final] delta={final_delta!r}")
            continue

        if getattr(args, "use_targeted_instruct_sampling", False):
            # C-1: prefill-based sampler. The observed English prefix is pushed
            # into the assistant turn via apply_chat_template + manual prefill, so
            # the model can only emit what comes AFTER it. One /completions call
            # with n=K returns K diverse continuations.
            sampler_api_base = args.targeted_sampler_api_base or args.instruct_api_base
            sampler_api_model = args.targeted_sampler_api_model or args.instruct_api_model
            sampler_api_timeout = (
                args.targeted_sampler_api_timeout
                if args.targeted_sampler_api_timeout and args.targeted_sampler_api_timeout > 0
                else args.instruct_api_timeout
            )
            futures, future_infos = sample_source_futures_targeted_prefill(
                sampler_tokenizer=sampler_tokenizer,
                observed_source=source_observed,
                committed_text=committed_text,
                target_lang=args.target_lang,
                num_futures=args.targeted_num_futures,
                api_base=sampler_api_base,
                api_model=sampler_api_model,
                api_timeout=sampler_api_timeout,
                sample_temperature=args.targeted_sample_temperature,
                top_p=args.targeted_top_p,
                max_tokens=args.targeted_max_tokens,
                sampler2_tokenizer=sampler2_tokenizer,
                sampler2_api_base=args.targeted_sampler2_api_base,
                sampler2_api_model=args.targeted_sampler2_api_model,
                sampler2_api_timeout=args.targeted_sampler2_api_timeout,
            )
        else:
            futures, future_infos = sample_source_futures_multi( #call base models，用当前observed source采样多条未来续写
                base_specs=base_specs,
                observed_source=source_observed,
                future_tokens=args.future_tokens,
                sample_temperature=args.sample_temperature,
            )

        # ── verbose: list futures ──
        _vlog(verbose_log_file, f"[Step 1-2] future_sampling total={len(futures)}")
        if verbose_log_file is not None:
            for fi, ftxt in enumerate(futures):
                info = future_infos[fi] if fi < len(future_infos) else {}
                label = info.get("source", "?")
                # For targeted instruct, also surface span/branch so we can later
                # validate whether the instruct model is finding real ambiguities.
                span = info.get("span")
                branch = info.get("branch")
                if span or branch:
                    _vlog(verbose_log_file,
                          f"  future[{fi}] ({label}) span={span!r} branch={branch!r}: {ftxt!r}")
                else:
                    _vlog(verbose_log_file, f"  future[{fi}] ({label}): {ftxt!r}")

        if len(futures) <= 3: #future太少无法做共识，等待更多源语言输入
            target_deltas.append("")
            actions.append("READ")
            _vlog(verbose_log_file, "  -> READ (too few futures)")
            continue

        pending_token_ids, grow_logs = extend_pending_tokens(
            instruct_tokenizer=instruct_tokenizer,
            source_observed=source_observed_full,
            futures=futures,
            committed_text=committed_text,
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

        # ── min-consensus-horizon filter ──
        # If the consensus only carried us a few tokens before breaking, the path is fragile.
        # Discard the whole pending buffer and READ instead, to avoid early-commit lock-in.
        if (
            args.min_consensus_horizon > 1
            and 0 < len(pending_token_ids) < args.min_consensus_horizon
        ):
            pending_text_dropped = decode_token_ids_to_text(instruct_tokenizer, pending_token_ids)
            _vlog(verbose_log_file,
                  f"[Step 5.5] horizon_filter: dropped pending={pending_text_dropped!r} "
                  f"(len={len(pending_token_ids)} < min_horizon={args.min_consensus_horizon}) -> READ")
            grow_logs.append({"step": "filter", "stop": "below_min_horizon",
                              "horizon": len(pending_token_ids),
                              "min_horizon": args.min_consensus_horizon})
            pending_token_ids = []

        # ── verbose: consensus steps ──
        if verbose_log_file is not None and grow_logs:
            _vlog(verbose_log_file, f"[Step 4-5] consensus summary:")
            for gl in grow_logs:
                step = gl.get("step", "?")
                stop = gl.get("stop", "")
                meta = gl.get("meta", {})
                intersection = meta.get("intersection", [])
                if stop:
                    intersection_texts = [_single_token_text(instruct_tokenizer, tid) for tid in intersection] if intersection else []
                    _vlog(verbose_log_file, f"  step={step} stop={stop} intersection={intersection_texts} pending={decode_token_ids_to_text(instruct_tokenizer, pending_token_ids)!r}")
                else:
                    accepted = gl.get("accepted_token_text", "?")
                    pending = gl.get("pending_text", "")
                    _vlog(verbose_log_file, f"  step={step} accepted={accepted!r} pending={pending!r}")
                per_future = gl.get("per_future", [])
                for pf in per_future:
                    texts = pf.get("candidate_texts", [])
                    probs = pf.get("candidate_probs", [])
                    pairs = ", ".join(f"{t!r}:{p:.3f}" for t, p in zip(texts, probs))
                    idx = per_future.index(pf)
                    _vlog(verbose_log_file, f"    future[{idx}] candidates={len(texts)}: [{pairs}]")

        new_committed, delta, committed_delta_token_ids, finalize_meta = finalize_external_commit( #修剪pending tokens，只提交解码后没有乱码的前缀部分
            tokenizer=instruct_tokenizer,
            committed_text=committed_text,
            pending_token_ids=pending_token_ids,
        )

        # ── verbose: finalize ──
        pending_text = decode_token_ids_to_text(instruct_tokenizer, pending_token_ids)
        _vlog(verbose_log_file, f"[Step 6-7] pending_before_trim={pending_text!r}")
        _vlog(verbose_log_file, f"[Step 6-7] commit_after_trim={delta!r}")

        action = "WRITE" if delta else "READ"
        target_deltas.append(delta)
        actions.append(action)
        _vlog(verbose_log_file, f"-> {action} delta={delta!r}")
        _vlog(verbose_log_file, f"committed_after: {new_committed!r}")
        committed_text = new_committed
        committed_token_ids.extend(committed_delta_token_ids)

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
    _vlog(
        verbose_log_file,
        f"  prediction={committed_text!r} bleu={bleu_char_value:.2f} laal={laal_value:.2f} "
        f"len_ratio_ref={length_ratio_ref:.2f} len_ratio_src={length_ratio_src:.2f}",
    )
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

def main() -> None:
    setup_env()
    args = parse_args()

    if args.use_targeted_instruct_sampling:
        # Targeted instruct sampling bypasses base-LM future sampling entirely.
        primary_num_futures = 0
    else:
        if not args.base_api_base or not args.base_api_model:
            raise ValueError(
                "--base-api-base and --base-api-model are required when "
                "--use-targeted-instruct-sampling is not set."
            )
        primary_num_futures = args.num_futures - args.secondary_num_futures # 两个base 各10个 future sampling
        if primary_num_futures <= 0:
            raise ValueError("primary base model must keep at least 1 future")

    df = pd.read_csv(args.input_tsv, sep="\t")
    rows = select_rows(df, args)

    # Build base model specs (API only). Skipped entirely when targeted instruct
    # sampling is enabled — the instruct endpoint generates futures directly.
    base_specs: List[Dict[str, Any]] = []
    if not args.use_targeted_instruct_sampling:
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
    else:
        # If the sampler is routed to a separate endpoint, verify it now.
        if args.targeted_sampler_api_base:
            sampler_models = verify_api(
                args.targeted_sampler_api_base,
                args.targeted_sampler_api_timeout if args.targeted_sampler_api_timeout > 0
                else args.instruct_api_timeout,
            )
            sampler_model_name = args.targeted_sampler_api_model or args.instruct_api_model
            if sampler_model_name not in sampler_models:
                raise RuntimeError(
                    f"targeted_sampler_api_model '{sampler_model_name}' not found at "
                    f"{args.targeted_sampler_api_base}; available={sampler_models}"
                )
            print(f"[Sampler] targeted_instruct (cross-family): "
                  f"model={sampler_model_name} api={normalize_api_base(args.targeted_sampler_api_base)}")
        else:
            print(f"[Sampler] targeted_instruct (same-model as probe): "
                  f"model={args.instruct_api_model} api={normalize_api_base(args.instruct_api_base)}")
        print(f"[Sampler] mode=prefill n={args.targeted_num_futures} "
              f"temp={args.targeted_sample_temperature} top_p={args.targeted_top_p} "
              f"max_tokens={args.targeted_max_tokens}")

    # Verify instruct API
    models = verify_api(args.instruct_api_base, args.instruct_api_timeout)
    if args.instruct_api_model not in models:
        raise RuntimeError(f"instruct_api_model '{args.instruct_api_model}' not found; available={models}")
    instruct_tokenizer = load_tokenizer(args.instruct_tokenizer_path)
    print(f"[Instruct] model={args.instruct_api_model} api={normalize_api_base(args.instruct_api_base)}")

    # Sampler tokenizer for the prefill chat-template path. Defaults to the
    # probe's tokenizer (same-model setup); set --targeted-sampler-tokenizer-path
    # for cross-family (e.g. gemma sampler + qwen probe).
    sampler_tokenizer = None
    sampler2_tokenizer = None
    if args.use_targeted_instruct_sampling:
        sampler_tok_path = args.targeted_sampler_tokenizer_path or args.instruct_tokenizer_path
        sampler_tokenizer = load_tokenizer(sampler_tok_path)
        print(f"[Sampler tokenizer] {sampler_tok_path}")
        if (args.targeted_sampler2_tokenizer_path
                and args.targeted_sampler2_api_base
                and args.targeted_sampler2_api_model):
            sampler2_tokenizer = load_tokenizer(args.targeted_sampler2_tokenizer_path)
            print(f"[Sampler2 tokenizer] {args.targeted_sampler2_tokenizer_path}")
            print(f"[Sampler2] model={args.targeted_sampler2_api_model} "
                  f"api={normalize_api_base(args.targeted_sampler2_api_base)}")

    output_dir: Optional[str] = None
    if args.output_jsonl:
        output_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
        os.makedirs(output_dir, exist_ok=True)
    if args.verbose and args.verbose_dir:
        os.makedirs(args.verbose_dir, exist_ok=True)

    def _process_one_row(row_idx: int, series_dict: Dict[str, Any]) -> Dict[str, Any]:
        row = series_dict
        utt_id = str(row.get(args.id_column, row.get("id", f"row_{row_idx}")))
        out_path = (
            os.path.join(output_dir, f"{sanitize_filename(utt_id)}.json")
            if output_dir is not None else None
        )

        if args.skip_existing and out_path is not None and os.path.exists(out_path):
            print(f"[SKIP existing] {out_path}")
            return {"utt_id": utt_id, "skipped_existing": True}

        verbose_log_file: Optional[Any] = None
        if args.verbose:
            if args.verbose_dir:
                vpath = os.path.join(args.verbose_dir, f"verbose_{sanitize_filename(utt_id)}.log")
                raw_file = open(vpath, "w", encoding="utf-8")
                verbose_log_file = _TeeWriter(raw_file) if args.test_one else raw_file
            elif args.test_one:
                verbose_log_file = sys.stdout

        try:
            result = run_one_utterance(
                row=row, args=args, base_specs=base_specs,
                instruct_tokenizer=instruct_tokenizer,
                sampler_tokenizer=sampler_tokenizer,
                sampler2_tokenizer=sampler2_tokenizer,
                verbose_log_file=verbose_log_file,
            )
        finally:
            if verbose_log_file is not None and args.verbose_dir:
                verbose_log_file.close()

        if output_dir is not None:
            write_pretty_json(out_path or os.path.join(output_dir, f"{sanitize_filename(result['utt_id'])}.json"), result)
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
