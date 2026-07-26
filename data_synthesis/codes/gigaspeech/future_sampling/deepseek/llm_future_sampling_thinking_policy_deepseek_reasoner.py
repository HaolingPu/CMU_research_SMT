#!/usr/bin/env python3
"""
DeepSeek-hosted thinking-policy simultaneous interpretation pipeline.

This wrapper keeps the local future-sampling + simalign logic from
llm_future_sampling_thinking_policy.py, but replaces the local vLLM thinking
backend with DeepSeek's OpenAI-compatible chat completions API using the
thinking model (`deepseek-reasoner`) directly, without any gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import llm_future_sampling_thinking_policy as local_policy


_DEEPSEEK_API_KEY: str = ""
_DEEPSEEK_TIMEOUT: float = 600.0
_DEEPSEEK_TARGET_MODEL: str = "deepseek-reasoner"
_DEEPSEEK_API_BASE_DEFAULT: str = "https://api.deepseek.com"

_DEEPSEEK_PRICING_DOC_URL: str = "https://api-docs.deepseek.com/quick_start/pricing"
_DEEPSEEK_INPUT_CACHE_HIT_USD_PER_1M: float = 0.14
_DEEPSEEK_INPUT_CACHE_MISS_USD_PER_1M: float = 0.55
_DEEPSEEK_OUTPUT_USD_PER_1M: float = 2.19


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Thinking-policy simultaneous interpretation with local base future "
            "sampling + simalign, and hosted DeepSeek reasoning model via "
            "OpenAI-compatible chat completions."
        )
    )
    p.add_argument("--input-tsv", required=True, help="Manifest TSV with src_text_full, src_trajectory.")
    p.add_argument("--output-root", required=True)

    p.add_argument("--base-model-path", default="/data/user_data/haolingp/models/Qwen3-4B-Base")
    p.add_argument("--thinking-api-base", default=_DEEPSEEK_API_BASE_DEFAULT)
    p.add_argument(
        "--thinking-api-bases",
        default="",
        help=(
            "Comma-separated list of OpenAI-compatible API bases. Normally you "
            "only need the default DeepSeek endpoint."
        ),
    )
    p.add_argument("--thinking-model-name", default="deepseek-reasoner")
    p.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--deepseek-timeout", type=float, default=600.0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)

    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--num-futures", type=int, default=5, help="N future continuations per step.")
    p.add_argument("--future-tokens", type=int, default=10)
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument(
        "--thinking-temperature",
        type=float,
        default=0.0,
        help="DeepSeek reasoner ignores temperature; kept only for CLI compatibility and debug logging.",
    )
    p.add_argument("--thinking-max-tokens", type=int, default=4096)
    p.add_argument(
        "--align-device",
        default="cuda:0",
        help="Device for simalign check model (e.g. cuda:0 or cpu).",
    )
    p.add_argument(
        "--parallel-utterances",
        type=int,
        default=1,
        help="Number of utterances to process concurrently.",
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
    raw = args.thinking_api_bases.strip()
    if raw:
        bases = [item.strip() for item in raw.split(",") if item.strip()]
        if bases:
            return bases
    return [args.thinking_api_base]


def _coerce_usage_subfields(obj: Any) -> Dict[str, int]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        items = obj.items()
    elif hasattr(obj, "model_dump"):
        try:
            items = obj.model_dump().items()
        except Exception:
            items = {}
    elif hasattr(obj, "__dict__"):
        items = vars(obj).items()
    else:
        return {}

    result: Dict[str, int] = {}
    for key, value in items:
        if value is None or isinstance(value, (str, bytes, bool)):
            continue
        if isinstance(value, (int, float)):
            result[str(key)] = int(value)
    return result


def _extract_usage_breakdown(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens_reported_by_api": None,
            "cached_tokens_reported_by_api": None,
            "input_tokens_cache_hit": 0,
            "input_tokens_cache_miss": 0,
        }

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    prompt_details = _coerce_usage_subfields(getattr(usage, "prompt_tokens_details", None))
    completion_details = _coerce_usage_subfields(getattr(usage, "completion_tokens_details", None))

    reasoning_tokens = completion_details.get("reasoning_tokens", None)
    if reasoning_tokens is not None:
        reasoning_tokens = int(reasoning_tokens)

    cached_tokens = (
        prompt_details.get("cached_tokens", None)
        or prompt_details.get("cache_hit_tokens", None)
        or prompt_details.get("prompt_cache_hit_tokens", None)
    )
    if cached_tokens is not None:
        cached_tokens = int(cached_tokens)

    input_tokens_cache_hit = max(0, int(cached_tokens or 0))
    input_tokens_cache_miss = max(0, prompt_tokens - input_tokens_cache_hit)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens_reported_by_api": reasoning_tokens,
        "cached_tokens_reported_by_api": cached_tokens,
        "input_tokens_cache_hit": input_tokens_cache_hit,
        "input_tokens_cache_miss": input_tokens_cache_miss,
    }


def _estimate_cost_usd(usage_stats: Dict[str, Any]) -> float:
    input_cache_hit = int(usage_stats.get("input_tokens_cache_hit", 0) or 0)
    input_cache_miss = int(usage_stats.get("input_tokens_cache_miss", 0) or 0)
    completion_tokens = int(usage_stats.get("completion_tokens", 0) or 0)
    return (
        input_cache_hit * _DEEPSEEK_INPUT_CACHE_HIT_USD_PER_1M
        + input_cache_miss * _DEEPSEEK_INPUT_CACHE_MISS_USD_PER_1M
        + completion_tokens * _DEEPSEEK_OUTPUT_USD_PER_1M
    ) / 1_000_000


def _build_usage_entry(model_name: str, usage_stats: Dict[str, Any]) -> Dict[str, Any]:
    prompt_tokens = int(usage_stats.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage_stats.get("completion_tokens", 0) or 0)
    reasoning_tokens = usage_stats.get("reasoning_tokens_reported_by_api", None)
    if reasoning_tokens is not None:
        reasoning_tokens = int(reasoning_tokens)
    cached_tokens = usage_stats.get("cached_tokens_reported_by_api", None)
    if cached_tokens is not None:
        cached_tokens = int(cached_tokens)
    input_cache_hit = int(usage_stats.get("input_tokens_cache_hit", 0) or 0)
    input_cache_miss = int(usage_stats.get("input_tokens_cache_miss", 0) or 0)
    estimated_cost_usd = _estimate_cost_usd(usage_stats)
    return {
        "model_name": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens_reported_by_api": reasoning_tokens,
        "cached_tokens_reported_by_api": cached_tokens,
        "input_tokens_cache_hit": input_cache_hit,
        "input_tokens_cache_miss": input_cache_miss,
        "total_tokens": prompt_tokens + completion_tokens,
        "input_cache_hit_price_per_1m_usd": _DEEPSEEK_INPUT_CACHE_HIT_USD_PER_1M,
        "input_cache_miss_price_per_1m_usd": _DEEPSEEK_INPUT_CACHE_MISS_USD_PER_1M,
        "output_price_per_1m_usd": _DEEPSEEK_OUTPUT_USD_PER_1M,
        "estimated_cost_usd": round(estimated_cost_usd, 6),
    }


def _append_usage_entry(bucket: Dict[str, Dict[str, Any]], entry: Dict[str, Any]) -> None:
    model_name = str(entry.get("model_name", "") or "").strip()
    if not model_name:
        return
    if model_name not in bucket:
        bucket[model_name] = {
            "model_name": model_name,
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens_reported_by_api": None,
            "cached_tokens_reported_by_api": None,
            "input_tokens_cache_hit": 0,
            "input_tokens_cache_miss": 0,
        }
    bucket[model_name]["requests"] += 1
    for key in ["prompt_tokens", "completion_tokens", "input_tokens_cache_hit", "input_tokens_cache_miss"]:
        bucket[model_name][key] += int(entry.get(key, 0) or 0)
    for maybe_key in ["reasoning_tokens_reported_by_api", "cached_tokens_reported_by_api"]:
        value = entry.get(maybe_key, None)
        if value is not None:
            current = bucket[model_name].get(maybe_key, None)
            if current is None:
                bucket[model_name][maybe_key] = int(value)
            else:
                bucket[model_name][maybe_key] = int(current) + int(value)


def _finalize_usage_bucket(bucket: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    per_model: Dict[str, Any] = {}
    total_prompt = 0
    total_completion = 0
    total_requests = 0
    total_cache_hit = 0
    total_cache_miss = 0
    total_cost = 0.0
    total_reasoning: Optional[int] = 0
    total_cached_reported: Optional[int] = 0

    for model_name, stats in bucket.items():
        entry = _build_usage_entry(model_name, stats)
        entry["requests"] = int(stats.get("requests", 0) or 0)
        per_model[model_name] = entry

        total_prompt += entry["prompt_tokens"]
        total_completion += entry["completion_tokens"]
        total_requests += entry["requests"]
        total_cache_hit += entry["input_tokens_cache_hit"]
        total_cache_miss += entry["input_tokens_cache_miss"]
        total_cost += float(entry["estimated_cost_usd"] or 0.0)

        entry_reasoning = entry.get("reasoning_tokens_reported_by_api", None)
        if total_reasoning is not None:
            if entry_reasoning is None:
                total_reasoning = None
            else:
                total_reasoning += int(entry_reasoning)

        entry_cached = entry.get("cached_tokens_reported_by_api", None)
        if total_cached_reported is not None:
            if entry_cached is None:
                total_cached_reported = None
            else:
                total_cached_reported += int(entry_cached)

    return {
        "total_requests": total_requests,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_reasoning_tokens_reported_by_api": total_reasoning,
        "total_cached_tokens_reported_by_api": total_cached_reported,
        "total_input_tokens_cache_hit": total_cache_hit,
        "total_input_tokens_cache_miss": total_cache_miss,
        "total_tokens": total_prompt + total_completion,
        "total_cost_usd": round(total_cost, 6),
        "per_model": per_model,
        "pricing_doc_url": _DEEPSEEK_PRICING_DOC_URL,
    }


def _to_jsonish(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return obj.model_dump()
    if isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    return repr(obj)


def _extract_message_debug(message: Any) -> Tuple[str, str, Dict[str, Any]]:
    raw_reasoning = local_policy._message_text_to_str(getattr(message, "reasoning", None))
    raw_reasoning_content = local_policy._message_text_to_str(getattr(message, "reasoning_content", None))
    raw_content = local_policy._message_text_to_str(getattr(message, "content", None))
    reasoning_text, content_text = local_policy._split_reasoning_and_content(
        raw_reasoning_content or raw_reasoning,
        raw_content,
    )
    raw_fields = {
        "message.raw": _to_jsonish(message),
        "message.reasoning": raw_reasoning,
        "message.reasoning_content": raw_reasoning_content,
        "message.content": raw_content,
    }
    return reasoning_text, content_text, raw_fields


class DeepSeekThinkingPool:
    def __init__(self, api_bases: List[str]):
        bases = [b.strip() for b in api_bases if (b or "").strip()]
        if not bases:
            raise ValueError("DeepSeekThinkingPool requires at least one API base.")
        if not _DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY is empty. Set the configured API key env var first.")
        self._slots = [
            {
                "api_base": api_base,
                "client": OpenAI(api_key=_DEEPSEEK_API_KEY, base_url=api_base, timeout=_DEEPSEEK_TIMEOUT),
                "inflight": 0,
                "requests": 0,
            }
            for api_base in bases
        ]
        self._lock = threading.Lock()
        self._rr = 0
        self._by_model: Dict[str, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._slots)

    def _acquire_slot(self, exclude: Optional[set] = None) -> Tuple[int, Dict[str, Any]]:
        exclude = exclude or set()
        with self._lock:
            candidates = [(idx, slot) for idx, slot in enumerate(self._slots) if idx not in exclude]
            if not candidates:
                raise RuntimeError("No available DeepSeek slot.")
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
        results: List[Tuple[str, List[str]]] = []
        for slot in self._slots:
            try:
                models = slot["client"].models.list()
                model_ids = [m.id for m in models.data]
                visible = model_ids[:20] if model_ids else [_DEEPSEEK_TARGET_MODEL]
            except Exception:
                visible = [_DEEPSEEK_TARGET_MODEL]
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
                usage_stats = _extract_usage_breakdown(getattr(resp, "usage", None))
                with self._lock:
                    if model_name not in self._by_model:
                        self._by_model[model_name] = {
                            "requests": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "reasoning_tokens_reported_by_api": None,
                            "cached_tokens_reported_by_api": None,
                            "input_tokens_cache_hit": 0,
                            "input_tokens_cache_miss": 0,
                        }
                    self._by_model[model_name]["requests"] += 1
                    self._by_model[model_name]["prompt_tokens"] += usage_stats["prompt_tokens"]
                    self._by_model[model_name]["completion_tokens"] += usage_stats["completion_tokens"]
                    self._by_model[model_name]["input_tokens_cache_hit"] += usage_stats["input_tokens_cache_hit"]
                    self._by_model[model_name]["input_tokens_cache_miss"] += usage_stats["input_tokens_cache_miss"]
                    for maybe_key in ["reasoning_tokens_reported_by_api", "cached_tokens_reported_by_api"]:
                        value = usage_stats.get(maybe_key, None)
                        if value is not None:
                            current = self._by_model[model_name].get(maybe_key, None)
                            if current is None:
                                self._by_model[model_name][maybe_key] = int(value)
                            else:
                                self._by_model[model_name][maybe_key] = int(current) + int(value)
                return resp, slot["api_base"]
            except Exception as e:
                errors.append(f"{slot['api_base']}: {e}")
            finally:
                self._release_slot(idx)
        raise RuntimeError("All DeepSeek chat endpoints failed: " + " | ".join(errors))

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
            return _finalize_usage_bucket(self._by_model)

    def close(self) -> None:
        return None


def call_thinking_model(
    thinking_pool: DeepSeekThinkingPool,
    model: str,
    user_content: str,
    committed_chinese: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Tuple[str, Dict[str, Any]]:
    del committed_chinese
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp, api_base = thinking_pool.chat_completions_create(**request_kwargs)
    choice = resp.choices[0]
    message = choice.message
    reasoning_text, content_text, raw_message_fields = _extract_message_debug(message)
    delta = "" if (not content_text or content_text.upper() == "EMPTY") else local_policy.normalize_zh(content_text)
    usage_entry = _build_usage_entry(model, _extract_usage_breakdown(getattr(resp, "usage", None)))
    return delta, {
        "server_api_base": api_base,
        "usage": usage_entry,
        "raw_message_fields": raw_message_fields,
        "reasoning_text": reasoning_text,
        "content_text": content_text,
        "cleaned_content": delta,
        "finish_reason": getattr(choice, "finish_reason", None),
        "temperature_requested": temperature,
        "temperature_sent": None,
        "temperature_ignored": True,
        "model_name": model,
    }


def force_complete_translation(
    thinking_pool: DeepSeekThinkingPool,
    model: str,
    full_source: str,
    committed_chinese: str,
) -> Tuple[str, Dict[str, Any]]:
    prompt = local_policy.build_final_completion_prompt(full_source, committed_chinese)
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "stream": False,
    }
    resp, api_base = thinking_pool.chat_completions_create(**request_kwargs)
    choice = resp.choices[0]
    message = choice.message
    reasoning_text, content_text, raw_message_fields = _extract_message_debug(message)
    continuation = "" if (not content_text or content_text.upper() == "EMPTY") else local_policy.normalize_zh(content_text)

    committed_norm = local_policy.normalize_zh(committed_chinese)
    new_part = local_policy.strip_committed_suffix_from_delta(committed_chinese, continuation)
    new_part = local_policy.normalize_zh(new_part)
    full_translation = committed_norm + new_part if committed_chinese else continuation
    usage_entry = _build_usage_entry(model, _extract_usage_breakdown(getattr(resp, "usage", None)))
    return full_translation, {
        "server_api_base": api_base,
        "usage": usage_entry,
        "raw_message_fields": raw_message_fields,
        "reasoning_text": reasoning_text,
        "content_text": content_text,
        "cleaned_content": continuation,
        "finish_reason": getattr(choice, "finish_reason", None),
        "full_translation": full_translation,
        "temperature_requested": 0.0,
        "temperature_sent": None,
        "temperature_ignored": True,
        "model_name": model,
    }


def process_one_utterance(
    base_llm: Any,
    thinking_pool: DeepSeekThinkingPool,
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
        "step3_final_complete_s": 0.0,
    }
    usage_bucket: Dict[str, Dict[str, Any]] = {}

    source_chunks: List[str] = []
    target_deltas: List[str] = []
    actions: List[str] = []
    committed = ""
    accumulated_source = ""

    local_policy._vlog(verbose_log_file, f"\n{'#' * 60}")
    local_policy._vlog(verbose_log_file, f"# Utterance: {utt_id}")
    local_policy._vlog(verbose_log_file, f"# Full text: {full_source}")
    local_policy._vlog(verbose_log_file, f"# Chunks: {n_chunks}")
    local_policy._vlog(verbose_log_file, f"# Thinking model: {thinking_model}")
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
            local_policy._vlog(verbose_log_file, f"  [last] force-complete from committed len={len(local_policy.normalize_zh(committed))}")
            t0 = time.perf_counter()
            full_translation, final_debug = force_complete_translation(
                thinking_pool, thinking_model, full_source, committed
            )
            timing["step3_final_complete_s"] += time.perf_counter() - t0
            _append_usage_entry(usage_bucket, final_debug.get("usage", {}))
            local_policy._vlog(verbose_log_file, local_policy._format_verbose_paragraph("  [last] reasoning", final_debug["reasoning_text"]))
            if final_debug.get("server_api_base"):
                local_policy._vlog(verbose_log_file, f"  [last] server_api_base: {final_debug['server_api_base']!r}")
            if final_debug.get("raw_message_fields"):
                raw_fields = final_debug["raw_message_fields"]
                local_policy._vlog(verbose_log_file, f"  [last][raw] message.reasoning: {raw_fields.get('message.reasoning', '')!r}")
                local_policy._vlog(verbose_log_file, f"  [last][raw] message.reasoning_content: {raw_fields.get('message.reasoning_content', '')!r}")
                local_policy._vlog(verbose_log_file, f"  [last][raw] message.content: {raw_fields.get('message.content', '')!r}")
            local_policy._vlog(verbose_log_file, f"  [last] temperature_requested: {final_debug.get('temperature_requested')!r}")
            local_policy._vlog(verbose_log_file, f"  [last] temperature_sent: {final_debug.get('temperature_sent')!r}")
            local_policy._vlog(verbose_log_file, f"  [last] temperature_ignored: {final_debug.get('temperature_ignored')!r}")
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

        user_content = local_policy.build_thinking_prompt(accumulated_source, futures, committed)
        t2_0 = time.perf_counter()
        delta, thinking_debug = call_thinking_model(
            thinking_pool,
            thinking_model,
            user_content,
            committed_chinese=committed,
            temperature=args.thinking_temperature,
            max_tokens=args.thinking_max_tokens,
        )
        timing["step2_thinking_delta_s"] += time.perf_counter() - t2_0
        _append_usage_entry(usage_bucket, thinking_debug.get("usage", {}))

        local_policy._vlog(verbose_log_file, local_policy._format_verbose_paragraph("  step2_reasoning", thinking_debug["reasoning_text"]))
        if thinking_debug.get("server_api_base"):
            local_policy._vlog(verbose_log_file, f"  step2_server_api_base: {thinking_debug['server_api_base']!r}")
        if thinking_debug.get("raw_message_fields"):
            raw_fields = thinking_debug["raw_message_fields"]
            local_policy._vlog(verbose_log_file, f"  step2_raw_message.reasoning: {raw_fields.get('message.reasoning', '')!r}")
            local_policy._vlog(verbose_log_file, f"  step2_raw_message.reasoning_content: {raw_fields.get('message.reasoning_content', '')!r}")
            local_policy._vlog(verbose_log_file, f"  step2_raw_message.content: {raw_fields.get('message.content', '')!r}")
        local_policy._vlog(verbose_log_file, f"  step2_temperature_requested: {thinking_debug.get('temperature_requested')!r}")
        local_policy._vlog(verbose_log_file, f"  step2_temperature_sent: {thinking_debug.get('temperature_sent')!r}")
        local_policy._vlog(verbose_log_file, f"  step2_temperature_ignored: {thinking_debug.get('temperature_ignored')!r}")
        local_policy._vlog(verbose_log_file, f"  step2_content_text: {thinking_debug['content_text']!r}")
        local_policy._vlog(verbose_log_file, f"  step2_cleaned_content: {thinking_debug['cleaned_content']!r}")

        raw_delta = delta
        if args.disable_post_simalign_check:
            alignment_debug = {"status": "skipped_disabled", "raw_delta_used": True}
        else:
            t2a_0 = time.perf_counter()
            delta, alignment_debug = local_policy.apply_simalign_delta_check(
                accumulated_source=accumulated_source,
                futures=futures,
                committed=committed,
                delta=delta,
                align_model=align_model,
                align_tokenizer=align_tokenizer,
            )
            timing["step2_alignment_check_s"] += time.perf_counter() - t2a_0

        local_policy._vlog(verbose_log_file, f"  step2_delta_raw: {raw_delta!r}")
        local_policy._vlog(verbose_log_file, f"  step2_alignment_check_status: {alignment_debug.get('status', '')!r}")
        if alignment_debug.get("accepted_case"):
            accepted_case = alignment_debug["accepted_case"]
            local_policy._vlog(verbose_log_file, f"  step2_alignment_selected_future: {accepted_case.get('future', '')!r}")
            local_policy._vlog(verbose_log_file, f"  step2_alignment_selected_safe_prefix: {accepted_case.get('safe_prefix', '')!r}")
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
            local_policy._vlog(verbose_log_file, f"  step2_alignment_valid_cases: {json.dumps(compact_valid, ensure_ascii=False, indent=2)}")
        if alignment_debug.get("skipped_cases"):
            local_policy._vlog(verbose_log_file, f"  step2_alignment_skipped_cases: {json.dumps(alignment_debug['skipped_cases'], ensure_ascii=False, indent=2)}")
        local_policy._vlog(verbose_log_file, f"  step2_delta_checked: {delta!r}")

        if delta:
            target_deltas.append(delta)
            actions.append("WRITE")
            committed = (committed or "") + delta
            local_policy._vlog(verbose_log_file, f"  -> WRITE delta={delta!r}")
        else:
            target_deltas.append("")
            actions.append("READ")
            local_policy._vlog(verbose_log_file, "  -> READ (empty/invalid content)")

    system_output = "".join(d for d in target_deltas if d)
    usage_summary = _finalize_usage_bucket(usage_bucket)
    result: Dict[str, Any] = {
        "utt_id": utt_id,
        "original_text": full_source,
        "input_sentences": sentences,
        "source_future_sampling": source_chunks,
        "target_future_sampling": target_deltas,
        "actions": actions,
        "system_output_text": system_output,
        "config": {
            "version": "thinking_policy_deepseek_reasoner",
            "num_futures": args.num_futures,
            "future_tokens": args.future_tokens,
            "thinking_model": thinking_model,
            "thinking_api_pool_size": len(thinking_pool),
            "disable_post_simalign_check": bool(args.disable_post_simalign_check),
        },
        "timing": timing,
        "usage": usage_summary,
        "cost": {
            "total_cost_usd": usage_summary["total_cost_usd"],
            "total_input_tokens_cache_hit": usage_summary["total_input_tokens_cache_hit"],
            "total_input_tokens_cache_miss": usage_summary["total_input_tokens_cache_miss"],
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
    result["metrics"] = {
        "laal_text": laal_value,
        "laal_reference_mode": laal_reference_mode,
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

    api_key = os.environ.get(args.deepseek_api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"ERROR: env var {args.deepseek_api_key_env} is not set. "
            "Export your DeepSeek API key before running."
        )

    global _DEEPSEEK_API_KEY, _DEEPSEEK_TIMEOUT, _DEEPSEEK_TARGET_MODEL
    _DEEPSEEK_API_KEY = api_key
    _DEEPSEEK_TIMEOUT = float(args.deepseek_timeout)
    _DEEPSEEK_TARGET_MODEL = args.thinking_model_name

    local_policy.parse_args = lambda: args
    local_policy.resolve_thinking_api_bases = resolve_thinking_api_bases
    local_policy.ThinkingServerPool = DeepSeekThinkingPool
    local_policy.call_thinking_model = call_thinking_model
    local_policy.force_complete_translation = force_complete_translation
    local_policy.process_one_utterance = process_one_utterance

    _thinking_pool_ref: List[DeepSeekThinkingPool] = []

    class _TrackingPool(DeepSeekThinkingPool):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            _thinking_pool_ref.append(self)

    local_policy.ThinkingServerPool = _TrackingPool

    try:
        local_policy.main()
    finally:
        summary = None
        try:
            if _thinking_pool_ref:
                summary = _thinking_pool_ref[0].token_summary()
        except Exception:
            summary = None

        if summary is not None:
            print(
                f"\n[DeepSeek Usage] requests={summary['total_requests']} | "
                f"prompt={summary['total_prompt_tokens']:,} | "
                f"completion={summary['total_completion_tokens']:,} | "
                f"reasoning_reported_by_api={summary.get('total_reasoning_tokens_reported_by_api')} | "
                f"cache_hit={summary.get('total_input_tokens_cache_hit', 0):,} | "
                f"cache_miss={summary.get('total_input_tokens_cache_miss', 0):,} | "
                f"total={summary['total_tokens']:,} | "
                f"estimated_cost=${summary.get('total_cost_usd', 0.0):.6f} USD"
            )
            for model_name, stats in summary.get("per_model", {}).items():
                print(
                    f"[DeepSeek Usage] model={model_name} requests={stats['requests']} "
                    f"prompt={stats['prompt_tokens']:,} completion={stats['completion_tokens']:,} "
                    f"reasoning_reported_by_api={stats.get('reasoning_tokens_reported_by_api')} "
                    f"cache_hit={stats.get('input_tokens_cache_hit', 0):,} "
                    f"cache_miss={stats.get('input_tokens_cache_miss', 0):,} "
                    f"total={stats['total_tokens']:,} "
                    f"estimated_cost=${(stats.get('estimated_cost_usd') or 0.0):.6f} USD"
                )
            try:
                usage_path = Path(args.output_root) / "deepseek_usage_summary.json"
                usage_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"[DeepSeek Usage] wrote summary to {usage_path}")
            except Exception as e:
                print(f"[DeepSeek Usage] failed to write summary: {e}")


if __name__ == "__main__":
    main()
