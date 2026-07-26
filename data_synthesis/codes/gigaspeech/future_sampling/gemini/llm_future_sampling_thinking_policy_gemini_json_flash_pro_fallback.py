#!/usr/bin/env python3
"""
Gemini JSON-structured thinking-policy ablation without simalign post-check.

Pipeline:
  1. Use Gemini Flash first for the per-chunk safe-delta decision.
  2. Skip simalign and keep the raw normalized delta.
  3. Escalate directly to Pro when Flash looks untrustworthy, especially on
     uncertain future-sampling chunks or obvious duplication artifacts.
  4. Add BLEU and LAAL metrics into each output JSON when a reference is available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import llm_future_sampling_thinking_policy as local_policy
import llm_future_sampling_thinking_policy_json as local_json
import llm_future_sampling_thinking_policy_gemini_json_flash as flash_json


_GEMINI_API_KEY: str = ""
_GEMINI_TIMEOUT: float = 600.0
_GEMINI_PRIMARY_MODEL: str = "gemini-3-flash-preview"
_GEMINI_FALLBACK_MODEL: str = "gemini-pro-latest"
_GEMINI_FINAL_COMPLETION_MAX_TOKENS: int = 2048

_FALLBACK_STATS_LOCK = threading.Lock()
_FALLBACK_STATS: Dict[str, int] = {
    "triggered": 0,
    "attempted": 0,
    "used": 0,
    "kept_flash": 0,
    "primary_reject": 0,
    "primary_trim": 0,
    "primary_uncertainty": 0,
    "primary_duplication": 0,
    "second_flash_runs": 0,
    "second_flash_disagreement": 0,
    "second_flash_agree": 0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Thinking-policy simultaneous interpretation with local future sampling, "
            "Gemini Flash JSON output, direct Pro fallback on uncertain chunks, "
            "and no simalign post-check."
        )
    )
    p.add_argument("--input-tsv", required=True, help="Manifest TSV with src_text_full, src_trajectory.")
    p.add_argument("--output-root", required=True)

    p.add_argument("--base-model-path", default="/data/user_data/haolingp/models/Qwen3-4B-Base")
    p.add_argument(
        "--thinking-api-base",
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    p.add_argument(
        "--thinking-api-bases",
        default="",
        help=(
            "Comma-separated list of Gemini OpenAI-compatible API bases. "
            "Normally you only need the default Gemini endpoint."
        ),
    )
    p.add_argument("--thinking-model-name", default="gemini-3-flash-preview")
    p.add_argument(
        "--fallback-model-name",
        default="gemini-pro-latest",
        help=(
            "Pro model used when Flash looks untrustworthy, especially on uncertain "
            "chunks. Set empty string or use --disable-pro-fallback to turn fallback off."
        ),
    )
    p.add_argument(
        "--final-completion-model-name",
        default="",
        help="Optional override for the last-step force-complete call. Defaults to the Flash model.",
    )
    p.add_argument(
        "--thinking-prompt-version",
        default="json_v1",
        help=(
            "Accepted for CLI compatibility, but ignored by this ablation. "
            "This script always uses the JSON-structured prompt path."
        ),
    )
    p.add_argument("--gemini-api-key-env", default="GEMINI_API_KEY")
    p.add_argument("--gemini-timeout", type=float, default=600.0)
    p.add_argument(
        "--thinking-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        default="low",
        help="Reasoning effort for the primary Flash request.",
    )
    p.add_argument(
        "--fallback-reasoning-effort",
        choices=["", "none", "minimal", "low", "medium", "high"],
        default="",
        help="Reasoning effort for fallback Pro requests. Empty means inherit primary reasoning effort.",
    )
    p.add_argument(
        "--gemini-include-thoughts",
        action="store_true",
        help=(
            "Accepted for CLI compatibility, but ignored. "
            "This ablation always disables thought summaries."
        ),
    )
    p.add_argument(
        "--no-gemini-include-thoughts",
        action="store_true",
        help="Accepted for CLI compatibility; thoughts are already disabled.",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)

    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--num-futures", type=int, default=5, help="N future continuations per step.")
    p.add_argument("--future-tokens", type=int, default=10)
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--thinking-temperature", type=float, default=0.1)
    p.add_argument(
        "--fallback-temperature",
        type=float,
        default=None,
        help="Optional override for fallback Pro temperature. Defaults to --thinking-temperature.",
    )
    p.add_argument(
        "--thinking-max-tokens",
        type=int,
        default=256,
        help="Max tokens for the per-chunk JSON thinking call. Default 256 for this ablation.",
    )
    p.add_argument(
        "--fallback-max-tokens",
        type=int,
        default=0,
        help="Optional override for fallback Pro max tokens. 0 means reuse --thinking-max-tokens.",
    )
    p.add_argument(
        "--final-completion-max-tokens",
        type=int,
        default=2048,
        help="Max tokens for the final completion JSON call.",
    )
    p.add_argument(
        "--align-device",
        default="cuda:0",
        help="Device for simalign check model (e.g. cuda:0 or cpu).",
    )
    p.add_argument(
        "--parallel-utterances",
        type=int,
        default=1,
        help=(
            "Number of utterances to process concurrently. With Gemini-hosted "
            "thinking, this is the main throughput knob."
        ),
    )
    p.add_argument(
        "--future-sampling-batch-size",
        type=int,
        default=4,
        help=(
            "When parallel-utterances>1: batch this many future-sampling "
            "requests into one base_llm.generate([src1, ...]) call."
        ),
    )
    p.add_argument(
        "--future-sampling-batch-wait",
        type=float,
        default=0.05,
        help="Seconds to wait for more future-sampling requests before flushing a batch.",
    )

    p.add_argument("--disable-pro-fallback", action="store_true")
    p.add_argument(
        "--no-fallback-on-alignment-reject",
        action="store_true",
        help="Do not trigger Pro fallback when Flash is rejected by simalign.",
    )
    p.add_argument(
        "--no-fallback-on-alignment-trim",
        action="store_true",
        help="Do not trigger Pro fallback when Flash is trimmed by simalign.",
    )
    p.add_argument(
        "--fallback-min-trim-chars",
        type=int,
        default=1,
        help="Minimum number of trimmed Chinese characters to trigger Pro fallback.",
    )
    p.add_argument(
        "--fallback-min-trim-ratio",
        type=float,
        default=0.0,
        help="Minimum trimmed fraction (0-1) to trigger Pro fallback.",
    )
    p.add_argument(
        "--disable-second-flash-check",
        action="store_true",
        help=(
            "Skip the cheap second Flash validation pass that probes uncertain chunks "
            "before escalating to Pro."
        ),
    )
    p.add_argument(
        "--second-flash-order",
        choices=["reverse", "rotate"],
        default="reverse",
        help="How to perturb the future list for the second Flash validation call.",
    )
    p.add_argument(
        "--second-flash-temperature",
        type=float,
        default=None,
        help="Optional override for the second Flash call. Defaults to --thinking-temperature.",
    )
    p.add_argument(
        "--second-flash-max-tokens",
        type=int,
        default=0,
        help="Optional override for the second Flash call. 0 means reuse --thinking-max-tokens.",
    )
    p.add_argument(
        "--second-flash-reasoning-effort",
        choices=["", "none", "minimal", "low", "medium", "high"],
        default="",
        help="Optional override for the second Flash call. Empty means reuse primary Flash effort.",
    )
    p.add_argument(
        "--no-fallback-on-uncertainty",
        action="store_true",
        help=(
            "Do not probe divergent future-sampling chunks with a second Flash pass "
            "before deciding whether to escalate."
        ),
    )
    p.add_argument(
        "--uncertainty-first-token-ratio-threshold",
        type=float,
        default=0.60,
        help=(
            "If the dominant first-token ratio across sampled futures falls below this, "
            "mark the chunk as uncertain."
        ),
    )
    p.add_argument(
        "--uncertainty-max-shared-prefix-words",
        type=int,
        default=0,
        help=(
            "Only mark chunks as uncertain when futures share at most this many initial words."
        ),
    )
    p.add_argument(
        "--no-fallback-on-duplication",
        action="store_true",
        help="Do not escalate when Flash emits obvious duplication/restart artifacts.",
    )

    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--id-column", default="id")
    p.add_argument("--test-one", action="store_true")
    p.add_argument("--utt-id", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--disable-post-simalign-check",
        action="store_true",
        help="Skip the post-thinking simalign safety truncation and use the raw delta directly.",
    )

    return p.parse_args()


def resolve_thinking_api_bases(args: argparse.Namespace) -> List[str]:
    return flash_json.resolve_thinking_api_bases(args)


class GeminiChatCompletionsPool:
    """Load-balance Gemini chat-completion requests across one or more endpoints."""

    def __init__(self, api_bases: List[str]):
        bases = [b.strip() for b in api_bases if (b or "").strip()]
        if not bases:
            raise ValueError("GeminiChatCompletionsPool requires at least one API base.")
        if not _GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is empty. Set the configured API key env var first.")

        self._slots = [
            {
                "api_base": api_base,
                "client": OpenAI(api_key=_GEMINI_API_KEY, base_url=api_base, timeout=_GEMINI_TIMEOUT),
                "inflight": 0,
                "requests": 0,
            }
            for api_base in bases
        ]
        self._lock = threading.Lock()
        self._rr = 0
        self._total_requests = 0
        self._by_model: Dict[str, Dict[str, int]] = {}

    def __len__(self) -> int:
        return len(self._slots)

    def _acquire_slot(self, exclude: Optional[set] = None) -> Tuple[int, Dict[str, Any]]:
        exclude = exclude or set()
        with self._lock:
            candidates = [
                (idx, slot)
                for idx, slot in enumerate(self._slots)
                if idx not in exclude
            ]
            if not candidates:
                raise RuntimeError("No available Gemini slot.")
            min_inflight = min(slot["inflight"] for _, slot in candidates)
            tied = [(idx, slot) for idx, slot in candidates if slot["inflight"] == min_inflight]
            pick_idx = self._rr % len(tied)
            self._rr += 1
            idx, slot = tied[pick_idx]
            slot["inflight"] += 1
            slot["requests"] += 1
            return idx, slot

    def _release_slot(self, idx: int) -> None:
        with self._lock:
            self._slots[idx]["inflight"] = max(0, self._slots[idx]["inflight"] - 1)

    def list_models(self) -> List[Tuple[str, List[str]]]:
        targets = {m for m in [_GEMINI_PRIMARY_MODEL, _GEMINI_FALLBACK_MODEL] if (m or "").strip()}
        results: List[Tuple[str, List[str]]] = []
        for slot in self._slots:
            models = slot["client"].models.list()
            model_ids = [m.id for m in models.data]
            visible = [m for m in model_ids if m in targets]
            if not visible:
                visible = model_ids[:20]
            results.append((slot["api_base"], visible))
        return results

    def chat_completions_create(self, **kwargs) -> Tuple[Any, str]:
        errors: List[str] = []
        tried: set = set()
        model_name = str(kwargs.get("model", "") or "")
        for _ in range(len(self._slots)):
            idx, slot = self._acquire_slot(exclude=tried)
            tried.add(idx)
            try:
                resp = slot["client"].chat.completions.create(**kwargs)
                usage = getattr(resp, "usage", None)
                with self._lock:
                    self._total_requests += 1
                    if model_name not in self._by_model:
                        self._by_model[model_name] = {
                            "requests": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                        }
                    self._by_model[model_name]["requests"] += 1
                    if usage is not None:
                        self._by_model[model_name]["prompt_tokens"] += usage.prompt_tokens or 0
                        self._by_model[model_name]["completion_tokens"] += usage.completion_tokens or 0
                return resp, slot["api_base"]
            except Exception as e:
                errors.append(f"{slot['api_base']}: {e}")
            finally:
                self._release_slot(idx)
        raise RuntimeError("All Gemini chat endpoints failed: " + " | ".join(errors))

    def stats(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "api_base": slot["api_base"],
                    "inflight": slot["inflight"],
                    "requests": slot["requests"],
                }
                for slot in self._slots
            ]

    def token_summary(self) -> Dict[str, Any]:
        with self._lock:
            per_model = {}
            total_prompt = 0
            total_completion = 0
            total_requests = 0
            for model_name, stats in self._by_model.items():
                prompt = stats["prompt_tokens"]
                completion = stats["completion_tokens"]
                requests = stats["requests"]
                total_prompt += prompt
                total_completion += completion
                total_requests += requests
                per_model[model_name] = {
                    "requests": requests,
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                }
            return {
                "total_requests": total_requests,
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "per_model": per_model,
            }

    def close(self) -> None:
        return None


def build_thinking_prompt(observed_source: str, futures: List[str], committed_chinese: str) -> str:
    return flash_json.build_thinking_prompt(observed_source, futures, committed_chinese)


def build_final_completion_prompt(full_source: str, committed_chinese: str) -> str:
    return flash_json.build_final_completion_prompt(full_source, committed_chinese)


def _record_fallback(event: str) -> None:
    with _FALLBACK_STATS_LOCK:
        _FALLBACK_STATS[event] = _FALLBACK_STATS.get(event, 0) + 1


def _fallback_reasoning_effort(args: argparse.Namespace) -> str:
    return (args.fallback_reasoning_effort or args.thinking_reasoning_effort or "low").strip()


def _fallback_temperature(args: argparse.Namespace) -> float:
    return args.thinking_temperature if args.fallback_temperature is None else float(args.fallback_temperature)


def _fallback_max_tokens(args: argparse.Namespace) -> int:
    return int(args.fallback_max_tokens) if int(args.fallback_max_tokens) > 0 else int(args.thinking_max_tokens)


def _second_flash_temperature(args: argparse.Namespace) -> float:
    return (
        args.thinking_temperature
        if args.second_flash_temperature is None
        else float(args.second_flash_temperature)
    )


def _second_flash_max_tokens(args: argparse.Namespace) -> int:
    return (
        int(args.second_flash_max_tokens)
        if int(args.second_flash_max_tokens) > 0
        else int(args.thinking_max_tokens)
    )


def _second_flash_reasoning_effort(args: argparse.Namespace) -> str:
    return (
        args.second_flash_reasoning_effort
        or args.thinking_reasoning_effort
        or "low"
    ).strip()


def _thinking_system_prompt() -> str:
    return flash_json._thinking_system_prompt()


def _final_completion_system_prompt() -> str:
    return flash_json._final_completion_system_prompt()


def _call_json_thinking_model(
    thinking_pool: GeminiChatCompletionsPool,
    model: str,
    user_content: str,
    *,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
) -> Tuple[str, Dict[str, Any]]:
    request_kwargs = flash_json._build_chat_kwargs(
        model=model,
        system_prompt=_thinking_system_prompt(),
        user_content=user_content,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    resp, api_base = thinking_pool.chat_completions_create(**request_kwargs)
    choice = resp.choices[0]
    message = choice.message
    reasoning_text, content_text, raw_message_fields = flash_json._extract_message_debug(message)
    raw_content = raw_message_fields.get("message.content", "") or content_text

    structured_parse_error = None
    structured_response: Optional[Dict[str, Any]] = None
    json_text = local_json._extract_json_object_from_text(raw_content)
    try:
        delta, structured_response = local_json.parse_thinking_json_response(json_text)
        content_text = json_text or content_text
    except Exception as e:
        structured_parse_error = str(e)
        fallback_answer = local_policy._extract_answer_candidate(content_text or raw_content)
        delta = "" if (not fallback_answer or fallback_answer.upper() == "EMPTY") else local_policy.normalize_zh(fallback_answer)

    return delta, {
        "server_api_base": api_base,
        "raw_message_fields": raw_message_fields,
        "reasoning_text": reasoning_text,
        "content_text": content_text,
        "cleaned_content": delta,
        "finish_reason": getattr(choice, "finish_reason", None),
        "structured_response": structured_response,
        "structured_parse_error": structured_parse_error,
        "temperature_requested": temperature,
        "temperature_sent": request_kwargs.get("temperature"),
        "reasoning_effort_sent": request_kwargs.get("reasoning_effort"),
        "include_thoughts": False,
        "model_name": model,
    }


def call_thinking_model(
    thinking_pool: GeminiChatCompletionsPool,
    model: str,
    user_content: str,
    committed_chinese: str = "",
    temperature: float = 0.1,
    max_tokens: int = 256,
) -> Tuple[str, Dict[str, Any]]:
    del committed_chinese
    return _call_json_thinking_model(
        thinking_pool,
        model,
        user_content,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort="low",
    )


def force_complete_translation(
    thinking_pool: GeminiChatCompletionsPool,
    model: str,
    full_source: str,
    committed_chinese: str,
) -> Tuple[str, Dict[str, Any]]:
    prompt = build_final_completion_prompt(full_source, committed_chinese)
    request_kwargs = flash_json._build_chat_kwargs(
        model=model,
        system_prompt=_final_completion_system_prompt(),
        user_content=prompt,
        temperature=0.0,
        max_tokens=_GEMINI_FINAL_COMPLETION_MAX_TOKENS,
        reasoning_effort="none",
    )
    resp, api_base = thinking_pool.chat_completions_create(**request_kwargs)
    choice = resp.choices[0]
    message = choice.message
    reasoning_text, content_text, raw_message_fields = flash_json._extract_message_debug(message)
    raw_content = raw_message_fields.get("message.content", "") or content_text

    structured_parse_error = None
    structured_response: Optional[Dict[str, Any]] = None
    json_text = local_json._extract_json_object_from_text(raw_content)
    try:
        continuation, structured_response = local_json.parse_final_completion_json_response(json_text)
        content_text = json_text or content_text
    except Exception as e:
        structured_parse_error = str(e)
        fallback_answer = local_policy._extract_answer_candidate(content_text or raw_content)
        continuation = "" if (not fallback_answer or fallback_answer.upper() == "EMPTY") else local_policy.normalize_zh(fallback_answer)

    committed_norm = local_policy.normalize_zh(committed_chinese)
    new_part = local_policy.strip_committed_suffix_from_delta(committed_chinese, continuation)
    new_part = local_policy.normalize_zh(new_part)
    full_translation = committed_norm + new_part if committed_chinese else continuation
    return full_translation, {
        "server_api_base": api_base,
        "raw_message_fields": raw_message_fields,
        "reasoning_text": reasoning_text,
        "content_text": content_text,
        "cleaned_content": continuation,
        "finish_reason": getattr(choice, "finish_reason", None),
        "full_translation": full_translation,
        "structured_response": structured_response,
        "structured_parse_error": structured_parse_error,
        "temperature_requested": 0.0,
        "temperature_sent": request_kwargs.get("temperature"),
        "reasoning_effort_sent": request_kwargs.get("reasoning_effort"),
        "include_thoughts": False,
        "model_name": model,
    }


def _run_alignment_check(
    *,
    accumulated_source: str,
    futures: List[str],
    committed: str,
    delta: str,
    align_model: Any,
    align_tokenizer: Any,
    args: argparse.Namespace,
) -> Tuple[str, Dict[str, Any]]:
    del accumulated_source, futures, committed, align_model, align_tokenizer, args
    return local_policy.normalize_zh(delta), {
        "status": "skipped_no_simalign",
        "raw_delta_used": True,
    }


_FUTURE_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_START_DOUBLE_CHAR_RE = re.compile(r"^([㐀-䶿一-鿿豈-﫿])\1")
_PUNCT_STRIP_RE = re.compile(r"[，。！？；：、（）《》【】“”‘’…,.!?;:'\"()\-]+")


def _future_words(text: str) -> List[str]:
    return _FUTURE_WORD_RE.findall((text or "").lower())


def _shared_prefix_word_count(futures: List[str]) -> int:
    tokenized = [_future_words(f) for f in futures if (f or "").strip()]
    if len(tokenized) < 2:
        return 0
    limit = min(len(words) for words in tokenized)
    count = 0
    for idx in range(limit):
        token = tokenized[0][idx]
        if all(words[idx] == token for words in tokenized[1:]):
            count += 1
            continue
        break
    return count


def _future_uncertainty_info(
    futures: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "trigger": False,
        "reason": "",
        "dominant_first_token": "",
        "dominant_first_token_ratio": 1.0,
        "unique_first_tokens": 0,
        "shared_prefix_words": 0,
    }
    cleaned = [f for f in futures if (f or "").strip()]
    if args.no_fallback_on_uncertainty or len(cleaned) < 2:
        return info

    first_tokens = [words[0] for words in (_future_words(f) for f in cleaned) if words]
    if len(first_tokens) < 2:
        return info

    counts = Counter(first_tokens)
    dominant_first_token, dominant_count = counts.most_common(1)[0]
    dominant_ratio = dominant_count / max(1, len(first_tokens))
    shared_prefix_words = _shared_prefix_word_count(cleaned)

    info.update(
        {
            "dominant_first_token": dominant_first_token,
            "dominant_first_token_ratio": round(dominant_ratio, 4),
            "unique_first_tokens": len(counts),
            "shared_prefix_words": shared_prefix_words,
        }
    )

    if (
        dominant_ratio < float(args.uncertainty_first_token_ratio_threshold)
        and shared_prefix_words <= int(args.uncertainty_max_shared_prefix_words)
        and len(counts) >= 2
    ):
        info["trigger"] = True
        info["reason"] = (
            f"future_divergence:first_token_ratio={dominant_ratio:.3f},"
            f"shared_prefix_words={shared_prefix_words},unique_first_tokens={len(counts)}"
        )
    return info


def _normalize_compare_text(text: str) -> str:
    return _PUNCT_STRIP_RE.sub("", local_policy.normalize_zh(text))


def _tail_head_overlap(committed: str, delta: str, max_chars: int = 12) -> int:
    committed_norm = local_policy.normalize_zh(committed)
    delta_norm = local_policy.normalize_zh(delta)
    if not committed_norm or not delta_norm:
        return 0
    max_probe = min(len(committed_norm), len(delta_norm), max_chars)
    for probe in range(max_probe, 1, -1):
        if committed_norm.endswith(delta_norm[:probe]):
            return probe
    return 0


def _find_adjacent_repeated_span(text: str, min_chars: int = 2, max_chars: int = 8) -> str:
    norm = _normalize_compare_text(text)
    if len(norm) < min_chars * 2:
        return ""
    max_probe = min(max_chars, len(norm) // 2)
    for width in range(max_probe, min_chars - 1, -1):
        for start in range(0, len(norm) - width * 2 + 1):
            span = norm[start : start + width]
            if span and norm[start + width : start + width * 2] == span:
                return span
    return ""


def _duplication_info(
    committed: str,
    raw_delta: str,
    checked_delta: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "trigger": False,
        "reasons": [],
        "committed_overlap_chars": 0,
        "start_double_char": "",
        "adjacent_repeat_span": "",
    }
    if args.no_fallback_on_duplication:
        return info

    probe_text = checked_delta or raw_delta
    probe_norm = local_policy.normalize_zh(probe_text)
    if not probe_norm:
        return info

    reasons: List[str] = []
    overlap_chars = _tail_head_overlap(committed, raw_delta or checked_delta)
    if overlap_chars >= 2:
        reasons.append(f"committed_overlap:{overlap_chars}")

    start_match = _START_DOUBLE_CHAR_RE.match(_normalize_compare_text(probe_norm))
    if start_match:
        reasons.append(f"double_start_char:{start_match.group(1)}")

    repeat_span = _find_adjacent_repeated_span(probe_norm)
    if repeat_span:
        reasons.append(f"adjacent_repeat:{repeat_span}")

    info.update(
        {
            "trigger": bool(reasons),
            "reasons": reasons,
            "committed_overlap_chars": overlap_chars,
            "start_double_char": start_match.group(1) if start_match else "",
            "adjacent_repeat_span": repeat_span,
        }
    )
    return info


def _reorder_futures_for_second_flash(
    futures: List[str],
    args: argparse.Namespace,
) -> List[str]:
    cleaned = list(futures)
    if len(cleaned) < 2:
        return cleaned
    if args.second_flash_order == "rotate":
        return cleaned[1:] + cleaned[:1]
    return list(reversed(cleaned))


def _common_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    idx = 0
    while idx < limit and a[idx] == b[idx]:
        idx += 1
    return idx


def _second_flash_disagreement_info(
    primary_checked: str,
    secondary_checked: str,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "trigger": False,
        "reason": "",
        "primary": local_policy.normalize_zh(primary_checked),
        "secondary": local_policy.normalize_zh(secondary_checked),
        "common_prefix_chars": 0,
    }
    primary_cmp = _normalize_compare_text(primary_checked)
    secondary_cmp = _normalize_compare_text(secondary_checked)

    if not primary_cmp and not secondary_cmp:
        return info
    if not primary_cmp or not secondary_cmp:
        info["trigger"] = primary_cmp != secondary_cmp
        info["reason"] = "one_empty_one_nonempty"
        return info
    if primary_cmp == secondary_cmp:
        return info

    common_prefix_chars = _common_prefix_len(primary_cmp, secondary_cmp)
    info["common_prefix_chars"] = common_prefix_chars

    shorter = min(len(primary_cmp), len(secondary_cmp))
    longer = max(len(primary_cmp), len(secondary_cmp))
    if (
        (primary_cmp.startswith(secondary_cmp) or secondary_cmp.startswith(primary_cmp))
        and shorter >= 2
        and (shorter / max(1, longer)) >= 0.80
    ):
        info["reason"] = "near_prefix_match"
        return info

    info["trigger"] = True
    info["reason"] = (
        f"delta_disagreement:primary={primary_cmp!r},secondary={secondary_cmp!r},"
        f"common_prefix={common_prefix_chars}"
    )
    return info


def _fallback_trigger_info(
    raw_delta: str,
    checked_delta: str,
    alignment_debug: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[bool, Dict[str, Any]]:
    raw_norm = local_policy.normalize_zh(raw_delta)
    checked_norm = local_policy.normalize_zh(checked_delta)
    status = str(alignment_debug.get("status", "") or "")
    trimmed_chars = max(0, len(raw_norm) - len(checked_norm))
    trim_ratio = (trimmed_chars / len(raw_norm)) if raw_norm else 0.0

    reasons: List[str] = []
    if not raw_norm or args.disable_post_simalign_check or args.disable_pro_fallback:
        return False, {
            "status": status,
            "trimmed_chars": trimmed_chars,
            "trim_ratio": trim_ratio,
            "reasons": reasons,
        }

    reject_enabled = not args.no_fallback_on_alignment_reject
    trim_enabled = not args.no_fallback_on_alignment_trim

    if reject_enabled and status.startswith("reject_"):
        reasons.append(status)
    if trim_enabled and trimmed_chars >= max(0, int(args.fallback_min_trim_chars)) and trim_ratio >= max(0.0, float(args.fallback_min_trim_ratio)):
        if trimmed_chars > 0:
            reasons.append(f"trim:{trimmed_chars}:{trim_ratio:.3f}")

    if status.startswith("reject_"):
        _record_fallback("primary_reject")
    elif trimmed_chars > 0:
        _record_fallback("primary_trim")

    return bool(reasons), {
        "status": status,
        "trimmed_chars": trimmed_chars,
        "trim_ratio": trim_ratio,
        "reasons": reasons,
    }


def _should_use_fallback(
    primary_checked: str,
    primary_alignment_debug: Dict[str, Any],
    fallback_checked: str,
    fallback_alignment_debug: Dict[str, Any],
    *,
    prefer_fallback: bool = False,
) -> Tuple[bool, str]:
    primary_norm = local_policy.normalize_zh(primary_checked)
    fallback_norm = local_policy.normalize_zh(fallback_checked)
    primary_status = str(primary_alignment_debug.get("status", "") or "")
    fallback_status = str(fallback_alignment_debug.get("status", "") or "")

    if prefer_fallback and fallback_status == "applied" and fallback_norm and fallback_norm != primary_norm:
        return True, "prefer_fallback_for_untrusted_flash"
    if fallback_norm and not primary_norm:
        return True, "fallback_nonempty_vs_primary_empty"
    if len(fallback_norm) > len(primary_norm):
        return True, f"fallback_longer_{len(fallback_norm)}>{len(primary_norm)}"
    if primary_status.startswith("reject_") and fallback_status == "applied" and fallback_norm:
        return True, "fallback_applied_after_primary_reject"
    return False, "keep_primary"


def _log_thinking_attempt(
    verbose_log_file: Optional[Any],
    prefix: str,
    debug: Dict[str, Any],
    raw_delta: str,
    alignment_debug: Dict[str, Any],
    checked_delta: str,
) -> None:
    local_policy._vlog(verbose_log_file, local_policy._format_verbose_paragraph(f"  {prefix}_reasoning", debug.get("reasoning_text", "")))
    if debug.get("server_api_base"):
        local_policy._vlog(verbose_log_file, f"  {prefix}_server_api_base: {debug['server_api_base']!r}")
    if debug.get("raw_message_fields"):
        raw_fields = debug["raw_message_fields"]
        local_policy._vlog(verbose_log_file, f"  {prefix}_raw_message.reasoning: {raw_fields.get('message.reasoning', '')!r}")
        local_policy._vlog(verbose_log_file, f"  {prefix}_raw_message.reasoning_content: {raw_fields.get('message.reasoning_content', '')!r}")
        if raw_fields.get("message.content_raw", ""):
            local_policy._vlog(verbose_log_file, f"  {prefix}_raw_message.content_raw: {raw_fields.get('message.content_raw', '')!r}")
        local_policy._vlog(verbose_log_file, f"  {prefix}_raw_message.content: {raw_fields.get('message.content', '')!r}")
    if "temperature_ignored" in debug:
        local_policy._vlog(verbose_log_file, f"  {prefix}_temperature_requested: {debug.get('temperature_requested')!r}")
        local_policy._vlog(verbose_log_file, f"  {prefix}_temperature_sent: {debug.get('temperature_sent')!r}")
        local_policy._vlog(verbose_log_file, f"  {prefix}_temperature_ignored: {debug.get('temperature_ignored')!r}")
    if debug.get("ran_out_of_tokens"):
        local_policy._vlog(verbose_log_file, f"  {prefix}_ran_out_of_tokens: True")
        local_policy._vlog(verbose_log_file, f"  {prefix}_incomplete_details: {debug.get('incomplete_details')!r}")
        if debug.get("partial_output"):
            local_policy._vlog(verbose_log_file, f"  {prefix}_partial_output: {debug['partial_output']!r}")
        if debug.get("ran_out_during_reasoning"):
            local_policy._vlog(verbose_log_file, f"  {prefix}_ran_out_during_reasoning: True")
    local_policy._vlog(verbose_log_file, f"  {prefix}_content_text: {debug.get('content_text', '')!r}")
    local_policy._vlog(verbose_log_file, f"  {prefix}_cleaned_content: {debug.get('cleaned_content', '')!r}")
    local_policy._vlog(verbose_log_file, f"  {prefix}_delta_raw: {raw_delta!r}")
    local_policy._vlog(verbose_log_file, f"  {prefix}_alignment_check_status: {alignment_debug.get('status', '')!r}")
    if alignment_debug.get("accepted_case"):
        accepted_case = alignment_debug["accepted_case"]
        local_policy._vlog(verbose_log_file, f"  {prefix}_alignment_selected_future: {accepted_case.get('future', '')!r}")
        local_policy._vlog(verbose_log_file, f"  {prefix}_alignment_selected_safe_prefix: {accepted_case.get('safe_prefix', '')!r}")
    if alignment_debug.get("valid_cases"):
        compact_valid = [
            {
                "future": c.get("future", ""),
                "alignment_count": c.get("alignment_count", 0),
                "safe_prefix": c.get("safe_prefix", ""),
                "safe_prefix_chars": c.get("safe_prefix_chars", 0),
            }
            for c in alignment_debug["valid_cases"]
        ]
        local_policy._vlog(verbose_log_file, f"  {prefix}_alignment_valid_cases: {json.dumps(compact_valid, ensure_ascii=False, indent=2)}")
    if alignment_debug.get("skipped_cases"):
        local_policy._vlog(verbose_log_file, f"  {prefix}_alignment_skipped_cases: {json.dumps(alignment_debug['skipped_cases'], ensure_ascii=False, indent=2)}")
    local_policy._vlog(verbose_log_file, f"  {prefix}_delta_checked: {checked_delta!r}")


def process_one_utterance(
    base_llm: Any,
    thinking_pool: GeminiChatCompletionsPool,
    align_model: Any,
    align_tokenizer: Any,
    thinking_model: str,
    utt_id: str,
    sentences: List[str],
    trajectory: List[str],
    row: Dict[str, str],
    args: argparse.Namespace,
    translation_cache: Optional[Dict[str, str]] = None,
    verbose_log_file: Optional[Any] = None,
) -> Dict[str, Any]:

    full_source = " ".join(sentences)
    n_chunks = len(trajectory)
    timing: Dict[str, float] = {
        "step1_future_sampling_s": 0.0,
        "step2_thinking_delta_s": 0.0,
        "step2_alignment_check_s": 0.0,
        "step2_fallback_thinking_s": 0.0,
        "step2_fallback_alignment_s": 0.0,
        "step3_final_complete_s": 0.0,
    }

    source_chunks: List[str] = []
    target_deltas: List[str] = []
    actions: List[str] = []
    committed = ""
    accumulated_source = ""

    local_policy._vlog(verbose_log_file, f"\n{'#' * 60}")
    local_policy._vlog(verbose_log_file, f"# Utterance: {utt_id}")
    local_policy._vlog(verbose_log_file, f"# Full text: {full_source}")
    local_policy._vlog(verbose_log_file, f"# Chunks: {n_chunks}")
    local_policy._vlog(verbose_log_file, f"# Primary thinking model: {thinking_model}")
    local_policy._vlog(verbose_log_file, f"# Fallback thinking model: {args.fallback_model_name!r}")
    local_policy._vlog(verbose_log_file, f"{'#' * 60}")

    for chunk_idx, chunk in enumerate(trajectory):
        chunk_str = (chunk or "").strip()
        if chunk_str:
            accumulated_source = (accumulated_source + " " + chunk_str).strip()
        source_chunks.append(chunk_str)

        is_last = chunk_idx == n_chunks - 1

        local_policy._vlog(verbose_log_file, f"\n{'=' * 60}")
        local_policy._vlog(verbose_log_file, f"Chunk {chunk_idx + 1}/{n_chunks}: {chunk_str!r}")
        local_policy._vlog(verbose_log_file, f"  accumulated_source: {accumulated_source!r}")
        local_policy._vlog(verbose_log_file, f"  committed_before:  {committed!r}")

        if is_last:
            final_model = (args.final_completion_model_name or thinking_model).strip() or thinking_model
            local_policy._vlog(verbose_log_file, f"  [last] force-complete from committed len={len(local_policy.normalize_zh(committed))}")
            local_policy._vlog(verbose_log_file, f"  [last] final_completion_model: {final_model!r}")
            t0 = time.perf_counter()
            full_translation, final_debug = force_complete_translation(
                thinking_pool,
                final_model,
                full_source,
                committed,
            )
            timing["step3_final_complete_s"] += time.perf_counter() - t0
            local_policy._vlog(verbose_log_file, local_policy._format_verbose_paragraph("  [last] reasoning", final_debug["reasoning_text"]))
            if final_debug.get("server_api_base"):
                local_policy._vlog(verbose_log_file, f"  [last] server_api_base: {final_debug['server_api_base']!r}")
            if final_debug.get("raw_message_fields"):
                raw_fields = final_debug["raw_message_fields"]
                local_policy._vlog(verbose_log_file, f"  [last][raw] message.reasoning: {raw_fields.get('message.reasoning', '')!r}")
                local_policy._vlog(verbose_log_file, f"  [last][raw] message.reasoning_content: {raw_fields.get('message.reasoning_content', '')!r}")
                if raw_fields.get("message.content_raw", ""):
                    local_policy._vlog(verbose_log_file, f"  [last][raw] message.content_raw: {raw_fields.get('message.content_raw', '')!r}")
                local_policy._vlog(verbose_log_file, f"  [last][raw] message.content: {raw_fields.get('message.content', '')!r}")
            local_policy._vlog(verbose_log_file, f"  [last] content_text: {final_debug['content_text']!r}")
            local_policy._vlog(verbose_log_file, f"  [last] cleaned_content: {final_debug['cleaned_content']!r}")
            local_policy._vlog(verbose_log_file, f"  [last] full_translation: {full_translation!r}")
            committed_norm = local_policy.normalize_zh(committed)
            full_norm = local_policy.normalize_zh(full_translation)
            if len(full_norm) > len(committed_norm):
                remaining = full_norm[len(committed_norm):]
                target_deltas.append(remaining)
                actions.append("WRITE")
                committed = full_translation
                local_policy._vlog(verbose_log_file, f"  -> WRITE (end) delta={remaining!r}")
            else:
                target_deltas.append("")
                actions.append("READ")
                local_policy._vlog(verbose_log_file, "  -> READ (end, nothing new)")
            continue

        t1_0 = time.perf_counter()
        futures, future_raw_outputs = local_policy.sample_futures(
            base_llm,
            accumulated_source,
            args.num_futures,
            args.future_tokens,
            args.sample_temperature,
        )
        timing["step1_future_sampling_s"] += time.perf_counter() - t1_0

        local_policy._vlog(verbose_log_file, f"  step1_future_raw_outputs: {json.dumps(future_raw_outputs, ensure_ascii=False, indent=2)}")
        local_policy._vlog(verbose_log_file, f"  step1_futures_cleaned: {json.dumps(futures, ensure_ascii=False, indent=2)}")

        if len(futures) < 2:
            target_deltas.append("")
            actions.append("READ")
            local_policy._vlog(verbose_log_file, "  -> READ (too few futures)")
            continue

        user_content = build_thinking_prompt(accumulated_source, futures, committed)

        t2_0 = time.perf_counter()
        flash_raw_delta, flash_debug = _call_json_thinking_model(
            thinking_pool,
            thinking_model,
            user_content,
            temperature=args.thinking_temperature,
            max_tokens=args.thinking_max_tokens,
            reasoning_effort=args.thinking_reasoning_effort,
        )
        timing["step2_thinking_delta_s"] += time.perf_counter() - t2_0

        t2a_0 = time.perf_counter()
        flash_checked_delta, flash_alignment_debug = _run_alignment_check(
            accumulated_source=accumulated_source,
            futures=futures,
            committed=committed,
            delta=flash_raw_delta,
            align_model=align_model,
            align_tokenizer=align_tokenizer,
            args=args,
        )
        timing["step2_alignment_check_s"] += time.perf_counter() - t2a_0

        _log_thinking_attempt(
            verbose_log_file,
            "step2_flash",
            flash_debug,
            flash_raw_delta,
            flash_alignment_debug,
            flash_checked_delta,
        )

        selected_model = thinking_model
        selected_raw_delta = flash_raw_delta
        selected_checked_delta = flash_checked_delta
        selected_debug = flash_debug
        selected_alignment_debug = flash_alignment_debug
        selected_reason = "flash_default"

        alignment_trigger, alignment_meta = _fallback_trigger_info(
            flash_raw_delta,
            flash_checked_delta,
            flash_alignment_debug,
            args,
        )
        uncertainty_meta = _future_uncertainty_info(futures, args)
        duplication_meta = _duplication_info(
            committed=committed,
            raw_delta=flash_raw_delta,
            checked_delta=flash_checked_delta,
            args=args,
        )
        if uncertainty_meta.get("trigger"):
            _record_fallback("primary_uncertainty")
        if duplication_meta.get("trigger"):
            _record_fallback("primary_duplication")

        local_policy._vlog(verbose_log_file, f"  step2_alignment_trigger_meta: {json.dumps(alignment_meta, ensure_ascii=False)}")
        local_policy._vlog(verbose_log_file, f"  step2_uncertainty_meta: {json.dumps(uncertainty_meta, ensure_ascii=False)}")
        local_policy._vlog(verbose_log_file, f"  step2_duplication_meta: {json.dumps(duplication_meta, ensure_ascii=False)}")

        fallback_reasons: List[str] = list(alignment_meta.get("reasons", []))
        if uncertainty_meta.get("trigger"):
            fallback_reasons.append(uncertainty_meta.get("reason") or "future_divergence")
        if duplication_meta.get("trigger"):
            fallback_reasons.extend(duplication_meta.get("reasons", []))

        fallback_model = (args.fallback_model_name or "").strip()
        can_escalate = bool(
            fallback_model
            and fallback_model != thinking_model
            and not args.disable_pro_fallback
        )

        second_flash_disagreement = {
            "trigger": False,
            "reason": "disabled_no_second_flash",
        }

        trigger_fallback = bool(fallback_reasons)
        trigger_meta = {
            "alignment": alignment_meta,
            "uncertainty": uncertainty_meta,
            "duplication": duplication_meta,
            "second_flash_disagreement": second_flash_disagreement,
            "reasons": fallback_reasons,
        }
        local_policy._vlog(verbose_log_file, f"  step2_fallback_triggered: {trigger_fallback!r}")
        local_policy._vlog(verbose_log_file, f"  step2_fallback_trigger_meta: {json.dumps(trigger_meta, ensure_ascii=False)}")

        if trigger_fallback and can_escalate:
            _record_fallback("triggered")
            _record_fallback("attempted")
            t2f_0 = time.perf_counter()
            fallback_raw_delta, fallback_debug = _call_json_thinking_model(
                thinking_pool,
                fallback_model,
                user_content,
                temperature=_fallback_temperature(args),
                max_tokens=_fallback_max_tokens(args),
                reasoning_effort=_fallback_reasoning_effort(args),
            )
            timing["step2_fallback_thinking_s"] += time.perf_counter() - t2f_0

            t2fa_0 = time.perf_counter()
            fallback_checked_delta, fallback_alignment_debug = _run_alignment_check(
                accumulated_source=accumulated_source,
                futures=futures,
                committed=committed,
                delta=fallback_raw_delta,
                align_model=align_model,
                align_tokenizer=align_tokenizer,
                args=args,
            )
            timing["step2_fallback_alignment_s"] += time.perf_counter() - t2fa_0

            _log_thinking_attempt(
                verbose_log_file,
                "step2_fallback",
                fallback_debug,
                fallback_raw_delta,
                fallback_alignment_debug,
                fallback_checked_delta,
            )

            use_fallback, selected_reason = _should_use_fallback(
                flash_checked_delta,
                flash_alignment_debug,
                fallback_checked_delta,
                fallback_alignment_debug,
                prefer_fallback=bool(
                    duplication_meta.get("trigger")
                    or uncertainty_meta.get("trigger")
                ),
            )
            local_policy._vlog(verbose_log_file, f"  step2_fallback_selected: {use_fallback!r}")
            local_policy._vlog(verbose_log_file, f"  step2_fallback_selection_reason: {selected_reason!r}")
            if use_fallback:
                _record_fallback("used")
                selected_model = fallback_model
                selected_raw_delta = fallback_raw_delta
                selected_checked_delta = fallback_checked_delta
                selected_debug = fallback_debug
                selected_alignment_debug = fallback_alignment_debug
            else:
                _record_fallback("kept_flash")
        elif trigger_fallback:
            _record_fallback("triggered")
            local_policy._vlog(verbose_log_file, f"  step2_fallback_skipped_reason: {'disabled_or_same_model_or_empty_model'!r}")

        local_policy._vlog(verbose_log_file, f"  step2_selected_model: {selected_model!r}")
        local_policy._vlog(verbose_log_file, f"  step2_selection_reason: {selected_reason!r}")
        local_policy._vlog(verbose_log_file, f"  step2_delta_raw: {selected_raw_delta!r}")
        local_policy._vlog(verbose_log_file, f"  step2_alignment_check_status: {selected_alignment_debug.get('status', '')!r}")
        local_policy._vlog(verbose_log_file, f"  step2_delta_checked: {selected_checked_delta!r}")

        if selected_checked_delta:
            target_deltas.append(selected_checked_delta)
            actions.append("WRITE")
            committed = (committed or "") + selected_checked_delta
            local_policy._vlog(verbose_log_file, f"  -> WRITE delta={selected_checked_delta!r}")
        else:
            target_deltas.append("")
            actions.append("READ")
            local_policy._vlog(verbose_log_file, "  -> READ (empty/invalid content)")

    system_output = "".join(d for d in target_deltas if d)
    result: Dict[str, Any] = {
        "utt_id": utt_id,
        "original_text": full_source,
        "source_full": sentences,
        "source_future_sampling": source_chunks,
        "target_future_sampling": target_deltas,
        "actions": actions,
        "system_output": system_output,
        "system_output_text": system_output,
        "timing": timing,
        "metadata": {
            "thinking_model_name": thinking_model,
            "fallback_model_name": (args.fallback_model_name or "").strip(),
            "thinking_prompt_format": "json_v1",
            "thinking_reasoning_effort": args.thinking_reasoning_effort,
            "fallback_reasoning_effort": _fallback_reasoning_effort(args),
            "second_flash_order": args.second_flash_order,
            "second_flash_enabled": False,
            "uncertainty_first_token_ratio_threshold": args.uncertainty_first_token_ratio_threshold,
            "uncertainty_max_shared_prefix_words": args.uncertainty_max_shared_prefix_words,
            "disable_post_simalign_check": True,
        },
    }

    laal_reference_text = ""
    laal_value = float("nan")
    laal_error: Optional[str] = None
    bleu_char_value = float("nan")
    bleu_char_error: Optional[str] = None
    laal_reference_mode = "manifest_reference"

    ref_text = (translation_cache or {}).get(utt_id)
    if ref_text:
        laal_reference_mode = "cache"
    else:
        ref_text = local_policy._get_reference_from_row(row)
    if ref_text is not None:
        laal_reference_text = ref_text
        result["reference_text"] = ref_text
        try:
            laal_value = local_policy.compute_laal(
                source_chunks,
                target_deltas,
                actions,
                ref_text,
            )
            bleu_char_value = local_policy.compute_bleu_char(
                system_output,
                ref_text,
            )
        except Exception as e:
            laal_error = str(e)
            bleu_char_error = str(e)
    else:
        laal_error = "reference_text_unavailable"
        bleu_char_error = "reference_text_unavailable"

    result["laal_reference_text"] = laal_reference_text
    result["laal"] = laal_value
    result["bleu"] = bleu_char_value
    result["metrics"] = {
        "laal": laal_value,
        "laal_text": laal_value,
        "laal_reference_mode": laal_reference_mode,
        "bleu": bleu_char_value,
        "bleu_char": bleu_char_value,
        "bleu_reference_mode": laal_reference_mode,
        "effective_source_chunks": sum(1 for c in source_chunks if str(c).strip()),
        "system_output_chars": len(system_output),
        "reference_chars": len(laal_reference_text.replace(" ", "")) if laal_reference_text else 0,
        "laal_error": laal_error,
        "bleu_char_error": bleu_char_error,
    }
    return result


def main() -> None:
    args = parse_args()

    api_key = os.environ.get(args.gemini_api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"ERROR: env var {args.gemini_api_key_env} is not set. "
            "Set your Gemini API key before running this script."
        )

    if args.gemini_include_thoughts:
        print("[Ablation] Ignoring --gemini-include-thoughts; this script always disables thoughts.")
    if args.no_gemini_include_thoughts:
        print("[Ablation] Thoughts are already disabled.")
    if (args.thinking_prompt_version or "").strip() not in {"", "json_v1"}:
        print(
            f"[Ablation] Ignoring --thinking-prompt-version={args.thinking_prompt_version!r}; "
            "this script always uses the JSON-structured prompt path."
        )

    args.disable_post_simalign_check = True
    args.disable_second_flash_check = True
    print("[Ablation] Simalign post-check is disabled for this script.")
    print("[Ablation] Second Flash validation is disabled; uncertain chunks go directly to Pro.")

    global _GEMINI_API_KEY
    global _GEMINI_TIMEOUT
    global _GEMINI_PRIMARY_MODEL
    global _GEMINI_FALLBACK_MODEL
    global _GEMINI_FINAL_COMPLETION_MAX_TOKENS

    _GEMINI_API_KEY = api_key
    _GEMINI_TIMEOUT = float(args.gemini_timeout)
    _GEMINI_PRIMARY_MODEL = args.thinking_model_name
    _GEMINI_FALLBACK_MODEL = args.fallback_model_name
    _GEMINI_FINAL_COMPLETION_MAX_TOKENS = int(args.final_completion_max_tokens)

    local_policy.parse_args = lambda: args
    local_policy.resolve_thinking_api_bases = resolve_thinking_api_bases
    local_policy.process_one_utterance = process_one_utterance
    local_policy.ThinkingServerPool = GeminiChatCompletionsPool
    local_policy.build_thinking_prompt = build_thinking_prompt
    local_policy.build_final_completion_prompt = build_final_completion_prompt
    local_policy.call_thinking_model = call_thinking_model
    local_policy.force_complete_translation = force_complete_translation

    local_policy.main()

    summary = None
    try:
        thinking_api_bases = resolve_thinking_api_bases(args)
        pool = GeminiChatCompletionsPool(thinking_api_bases)
        summary = pool.token_summary()
        pool.close()
    except Exception:
        summary = None

    if summary is not None:
        print(
            f"\n[Token Usage] requests={summary['total_requests']} | "
            f"prompt={summary['total_prompt_tokens']:,} | "
            f"completion={summary['total_completion_tokens']:,} | "
            f"total={summary['total_tokens']:,}"
        )
        for model_name, stats in summary.get("per_model", {}).items():
            print(
                f"[Token Usage] model={model_name} requests={stats['requests']} "
                f"prompt={stats['prompt_tokens']:,} completion={stats['completion_tokens']:,} "
                f"total={stats['total_tokens']:,}"
            )

    with _FALLBACK_STATS_LOCK:
        stats = dict(_FALLBACK_STATS)
    print(
        "[Fallback Stats] "
        f"triggered={stats.get('triggered', 0)} | "
        f"attempted={stats.get('attempted', 0)} | "
        f"used={stats.get('used', 0)} | "
        f"kept_flash={stats.get('kept_flash', 0)} | "
        f"primary_reject={stats.get('primary_reject', 0)} | "
        f"primary_trim={stats.get('primary_trim', 0)} | "
        f"primary_uncertainty={stats.get('primary_uncertainty', 0)} | "
        f"primary_duplication={stats.get('primary_duplication', 0)} | "
        f"second_flash_runs={stats.get('second_flash_runs', 0)} | "
        f"second_flash_disagreement={stats.get('second_flash_disagreement', 0)} | "
        f"second_flash_agree={stats.get('second_flash_agree', 0)}"
    )


if __name__ == "__main__":
    main()
