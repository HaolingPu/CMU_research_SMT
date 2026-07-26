#!/usr/bin/env python3
"""
Gemini future-distribution routing with a fixed API-served 30B gate.

Current pipeline:
  1. Use the local 4B base model for future sampling.
  2. Use a served 30B instruct model to read next-token logprobs for a small number of futures.
  3. Filter formatting-only gate tokens, then compute entropy and JS divergence.
  4. Route each non-final chunk directly to Flash or 3.1-Pro.
  5. Skip simalign and keep the normalized raw delta.
  6. Use Pro for the final completion step.
  7. Add BLEU and LAAL to each output JSON when a reference is available.
"""

from __future__ import annotations

import argparse
import json
import math
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
_GEMINI_FALLBACK_MODEL: str = "gemini-3.1-pro-preview"
_GEMINI_FINAL_COMPLETION_MAX_TOKENS: int = 2048
_GEMINI_FINAL_COMPLETION_REASONING_EFFORT: str = "low"
_GEMINI_INCLUDE_THOUGHTS: bool = False

_GEMINI_PRICING_DOC_URL: str = "https://ai.google.dev/gemini-api/docs/pricing"
_GEMINI_MODEL_PRICING_USD_PER_1M: Dict[str, Dict[str, float]] = {
    "gemini-3-flash-preview": {
        "input": 0.50,
        "output": 3.00,
    },
    "gemini-3.1-pro-preview<=200k": {
        "input": 2.00,
        "output": 12.00,
    },
    "gemini-3.1-pro-preview>200k": {
        "input": 4.00,
        "output": 18.00,
    },
}

_FIXED_BASE_GPU_MEMORY_UTILIZATION: float = 0.80
_FIXED_PARALLEL_UTTERANCES: int = 1
_FIXED_FUTURE_SAMPLING_BATCH_SIZE: int = 1
_FIXED_FUTURE_SAMPLING_BATCH_WAIT: float = 0.05

_ORIGINAL_LOAD_BASE_MODEL = local_policy.load_base_model
_LOCAL_GATE_API_BASE: str = ""
_LOCAL_GATE_API_KEY: str = "EMPTY"
_LOCAL_GATE_API_MODEL_NAME: str = "qwen3-instruct"
_LOCAL_GATE_API_TIMEOUT: float = 120.0
_LOCAL_GATE_CLIENT: Optional[OpenAI] = None

_FALLBACK_STATS_LOCK = threading.Lock()
_FALLBACK_STATS: Dict[str, int] = {
    "probe_runs": 0,
    "probe_triggered": 0,
    "probe_entropy_trigger": 0,
    "probe_js_trigger": 0,
    "probe_semantic_mass_trigger": 0,
    "routed_pro": 0,
    "routed_flash": 0,
    "probe_skipped": 0,
}


def _coerce_usage_subfields(obj: Any) -> Dict[str, int]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        items = obj.items()
    elif hasattr(obj, "model_dump"):
        items = obj.model_dump().items()
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
            "cached_tokens": 0,
        }

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    prompt_details = _coerce_usage_subfields(getattr(usage, "prompt_tokens_details", None))
    completion_details = _coerce_usage_subfields(getattr(usage, "completion_tokens_details", None))

    reasoning_tokens = completion_details.get("reasoning_tokens", None)
    if reasoning_tokens is not None:
        reasoning_tokens = int(reasoning_tokens)
    cached_tokens = int(prompt_details.get("cached_tokens", 0) or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens_reported_by_api": reasoning_tokens,
        "cached_tokens": cached_tokens,
    }


def _gemini_price_rates_for_model(model_name: str, prompt_tokens: int) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    model_name = (model_name or "").strip()
    if model_name == "gemini-3-flash-preview":
        rates = _GEMINI_MODEL_PRICING_USD_PER_1M["gemini-3-flash-preview"]
        return "gemini-3-flash-preview", rates["input"], rates["output"]
    if model_name == "gemini-3.1-pro-preview":
        tier = "<=200k" if int(prompt_tokens) <= 200_000 else ">200k"
        key = f"gemini-3.1-pro-preview{tier}"
        rates = _GEMINI_MODEL_PRICING_USD_PER_1M[key]
        return key, rates["input"], rates["output"]
    return None, None, None


def _estimate_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    _, input_rate, output_rate = _gemini_price_rates_for_model(model_name, prompt_tokens)
    if input_rate is None or output_rate is None:
        return None
    return ((int(prompt_tokens) * input_rate) + (int(completion_tokens) * output_rate)) / 1_000_000


def _build_usage_entry(model_name: str, usage_stats: Dict[str, Any]) -> Dict[str, Any]:
    prompt_tokens = int(usage_stats.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage_stats.get("completion_tokens", 0) or 0)
    reasoning_tokens = usage_stats.get("reasoning_tokens_reported_by_api", None)
    if reasoning_tokens is not None:
        reasoning_tokens = int(reasoning_tokens)
    cached_tokens = int(usage_stats.get("cached_tokens", 0) or 0)
    price_key, input_rate, output_rate = _gemini_price_rates_for_model(model_name, prompt_tokens)
    estimated_cost_usd = _estimate_cost_usd(model_name, prompt_tokens, completion_tokens)
    return {
        "model_name": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens_reported_by_api": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "pricing_tier": price_key,
        "input_price_per_1m_usd": input_rate,
        "output_price_per_1m_usd": output_rate,
        "estimated_cost_usd": None if estimated_cost_usd is None else round(estimated_cost_usd, 6),
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
            "cached_tokens": 0,
        }
    bucket[model_name]["requests"] += 1
    for key in ["prompt_tokens", "completion_tokens", "cached_tokens"]:
        bucket[model_name][key] += int(entry.get(key, 0) or 0)
    entry_reasoning = entry.get("reasoning_tokens_reported_by_api", None)
    if entry_reasoning is not None:
        current_reasoning = bucket[model_name].get("reasoning_tokens_reported_by_api", None)
        if current_reasoning is None:
            bucket[model_name]["reasoning_tokens_reported_by_api"] = int(entry_reasoning)
        else:
            bucket[model_name]["reasoning_tokens_reported_by_api"] = int(current_reasoning) + int(entry_reasoning)


def _finalize_usage_bucket(bucket: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    per_model: Dict[str, Any] = {}
    total_prompt = 0
    total_completion = 0
    total_reasoning: Optional[int] = 0
    total_cached = 0
    total_requests = 0
    total_cost = 0.0
    flash_cost = 0.0
    pro_cost = 0.0

    for model_name, stats in bucket.items():
        entry = _build_usage_entry(model_name, stats)
        entry["requests"] = int(stats.get("requests", 0) or 0)
        per_model[model_name] = entry

        total_prompt += entry["prompt_tokens"]
        total_completion += entry["completion_tokens"]
        entry_reasoning = entry.get("reasoning_tokens_reported_by_api", None)
        if total_reasoning is not None:
            if entry_reasoning is None:
                total_reasoning = None
            else:
                total_reasoning += int(entry_reasoning)
        total_cached += entry["cached_tokens"]
        total_requests += entry["requests"]

        cost_value = float(entry["estimated_cost_usd"] or 0.0)
        total_cost += cost_value
        if "flash" in model_name.lower():
            flash_cost += cost_value
        if "pro" in model_name.lower():
            pro_cost += cost_value

    return {
        "total_requests": total_requests,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_reasoning_tokens_reported_by_api": total_reasoning,
        "total_cached_tokens": total_cached,
        "total_tokens": total_prompt + total_completion,
        "flash_cost_usd": round(flash_cost, 6),
        "pro_cost_usd": round(pro_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "per_model": per_model,
        "pricing_doc_url": _GEMINI_PRICING_DOC_URL,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Thinking-policy simultaneous interpretation with local 4B future sampling, "
            "a served 30B next-token-logprobs gate, direct Flash/3.1-Pro routing, and "
            "no simalign post-check."
        )
    )
    p.add_argument("--input-tsv", required=True, help="Manifest TSV with src_text_full, src_trajectory.")
    p.add_argument("--output-root", required=True)
    p.add_argument("--base-model-path", default="/data/user_data/haolingp/models/Qwen3-4B-Base")

    p.add_argument(
        "--gate-api-base",
        required=True,
        help="OpenAI-compatible API base for the served 30B gate, e.g. http://127.0.0.1:8100/v1",
    )
    p.add_argument(
        "--gate-api-key",
        default="EMPTY",
        help="API key for the local gate server. Local vLLM serve usually accepts EMPTY.",
    )
    p.add_argument(
        "--gate-api-model-name",
        default="qwen3-instruct",
        help="Served model name exposed by the gate server.",
    )
    p.add_argument(
        "--gate-api-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for gate API calls.",
    )

    p.add_argument(
        "--thinking-api-base",
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    p.add_argument(
        "--thinking-api-bases",
        default="",
        help="Comma-separated Gemini OpenAI-compatible API bases. Normally one endpoint is enough.",
    )
    p.add_argument("--thinking-model-name", default="gemini-3-flash-preview")
    p.add_argument("--fallback-model-name", default="gemini-3.1-pro-preview")
    p.add_argument(
        "--final-completion-model-name",
        default="gemini-3.1-pro-preview",
        help="Model used for the last-step completion. Defaults to 3.1-Pro.",
    )
    p.add_argument("--gemini-api-key-env", default="GEMINI_API_KEY")
    p.add_argument("--gemini-timeout", type=float, default=600.0)
    p.add_argument(
        "--gemini-include-thoughts",
        action="store_true",
        help="Request Gemini thought summaries and expose them in verbose logs.",
    )
    p.add_argument(
        "--no-gemini-include-thoughts",
        action="store_true",
        help="Disable Gemini thought summaries even if the default changes later.",
    )
    p.add_argument(
        "--thinking-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        default="high",
    )
    p.add_argument(
        "--fallback-reasoning-effort",
        choices=["", "none", "minimal", "low", "medium", "high"],
        default="low",
        help="Reasoning effort for the Pro fallback path. Defaults to low.",
    )

    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--num-futures", type=int, default=5, help="Number of future continuations per step.")
    p.add_argument("--future-tokens", type=int, default=10)
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--thinking-temperature", type=float, default=0.1)
    p.add_argument(
        "--fallback-temperature",
        type=float,
        default=None,
        help="Optional override for Pro temperature. Defaults to thinking temperature.",
    )
    p.add_argument("--thinking-max-tokens", type=int, default=8192)
    p.add_argument(
        "--fallback-max-tokens",
        type=int,
        default=8192,
        help="0 means reuse thinking-max-tokens.",
    )
    p.add_argument("--final-completion-max-tokens", type=int, default=8192)

    p.add_argument("--probe-max-futures", type=int, default=2)
    p.add_argument("--probe-top-k-logprobs", type=int, default=10)
    p.add_argument("--probe-rollout-tokens", type=int, default=3)
    p.add_argument("--probe-rollout-max-chars", type=int, default=4)
    p.add_argument("--probe-distribution-chars", type=int, default=2)
    p.add_argument("--probe-avg-entropy-threshold", type=float, default=0.65)
    p.add_argument("--probe-js-threshold", type=float, default=0.10)
    p.add_argument(
        "--probe-min-semantic-mass",
        type=float,
        default=0.10,
        help=(
            "Minimum average retained probability mass on semantic gate tokens. "
            "If most probability stays on whitespace/newlines, treat the probe as noisy."
        ),
    )

    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--id-column", default="id")
    p.add_argument("--test-one", action="store_true")
    p.add_argument("--utt-id", default=None)
    p.add_argument("--verbose", action="store_true")
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
                            "reasoning_tokens_reported_by_api": None,
                            "cached_tokens": 0,
                        }
                    self._by_model[model_name]["requests"] += 1
                    usage_stats = _extract_usage_breakdown(usage)
                    self._by_model[model_name]["prompt_tokens"] += usage_stats["prompt_tokens"]
                    self._by_model[model_name]["completion_tokens"] += usage_stats["completion_tokens"]
                    self._by_model[model_name]["cached_tokens"] += usage_stats["cached_tokens"]
                    usage_reasoning = usage_stats.get("reasoning_tokens_reported_by_api", None)
                    if usage_reasoning is not None:
                        current_reasoning = self._by_model[model_name].get("reasoning_tokens_reported_by_api", None)
                        if current_reasoning is None:
                            self._by_model[model_name]["reasoning_tokens_reported_by_api"] = int(usage_reasoning)
                        else:
                            self._by_model[model_name]["reasoning_tokens_reported_by_api"] = int(current_reasoning) + int(usage_reasoning)
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
            total_reasoning: Optional[int] = 0
            total_cached = 0
            total_requests = 0
            total_cost = 0.0
            for model_name, stats in self._by_model.items():
                prompt = stats["prompt_tokens"]
                completion = stats["completion_tokens"]
                reasoning = stats.get("reasoning_tokens_reported_by_api", None)
                cached = stats.get("cached_tokens", 0)
                requests = stats["requests"]
                total_prompt += prompt
                total_completion += completion
                if total_reasoning is not None:
                    if reasoning is None:
                        total_reasoning = None
                    else:
                        total_reasoning += int(reasoning)
                total_cached += cached
                total_requests += requests
                price_key, input_rate, output_rate = _gemini_price_rates_for_model(model_name, prompt)
                estimated_cost_usd = _estimate_cost_usd(model_name, prompt, completion)
                if estimated_cost_usd is not None:
                    total_cost += estimated_cost_usd
                per_model[model_name] = {
                    "requests": requests,
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "reasoning_tokens_reported_by_api": reasoning,
                    "cached_tokens": cached,
                    "total_tokens": prompt + completion,
                    "pricing_tier": price_key,
                    "input_price_per_1m_usd": input_rate,
                    "output_price_per_1m_usd": output_rate,
                    "estimated_cost_usd": None if estimated_cost_usd is None else round(estimated_cost_usd, 6),
                }
            return {
                "total_requests": total_requests,
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_reasoning_tokens_reported_by_api": total_reasoning,
                "total_cached_tokens": total_cached,
                "total_tokens": total_prompt + total_completion,
                "estimated_cost_usd": round(total_cost, 6),
                "pricing_doc_url": _GEMINI_PRICING_DOC_URL,
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


def _default_reasoning_effort_for_model(model_name: str) -> str:
    model_name = (model_name or "").strip().lower()
    return "low" if "pro" in model_name else "high"


def _fallback_reasoning_effort(args: argparse.Namespace) -> str:
    return (args.fallback_reasoning_effort or args.thinking_reasoning_effort or "low").strip()


def _fallback_temperature(args: argparse.Namespace) -> float:
    return args.thinking_temperature if args.fallback_temperature is None else float(args.fallback_temperature)


def _fallback_max_tokens(args: argparse.Namespace) -> int:
    return int(args.fallback_max_tokens) if int(args.fallback_max_tokens) > 0 else int(args.thinking_max_tokens)


def _gate_client() -> OpenAI:
    global _LOCAL_GATE_CLIENT
    if _LOCAL_GATE_CLIENT is None:
        _LOCAL_GATE_CLIENT = OpenAI(
            api_key=_LOCAL_GATE_API_KEY or "EMPTY",
            base_url=_LOCAL_GATE_API_BASE,
            timeout=float(_LOCAL_GATE_API_TIMEOUT),
            max_retries=2,
        )
    return _LOCAL_GATE_CLIENT


def _load_base_and_gate_model(model_path: str, gpu_memory_utilization: float) -> Any:
    return _ORIGINAL_LOAD_BASE_MODEL(model_path, gpu_memory_utilization)


def _default_final_completion_model(args: argparse.Namespace, thinking_model: str) -> str:
    explicit = (args.final_completion_model_name or "").strip()
    if explicit:
        return explicit
    fallback_model = (args.fallback_model_name or "").strip()
    if fallback_model:
        return fallback_model
    return thinking_model


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
        include_thoughts=_GEMINI_INCLUDE_THOUGHTS,
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
        "usage": _build_usage_entry(model, _extract_usage_breakdown(getattr(resp, "usage", None))),
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
        "include_thoughts": _GEMINI_INCLUDE_THOUGHTS,
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
        reasoning_effort=_default_reasoning_effort_for_model(model),
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
        reasoning_effort=_GEMINI_FINAL_COMPLETION_REASONING_EFFORT,
        include_thoughts=_GEMINI_INCLUDE_THOUGHTS,
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
        "usage": _build_usage_entry(model, _extract_usage_breakdown(getattr(resp, "usage", None))),
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
        "include_thoughts": _GEMINI_INCLUDE_THOUGHTS,
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


def _build_probe_prompt(
    observed_source: str,
    future: str,
    committed_chinese: str,
) -> str:
    return (
        "You are a professional English-to-Chinese simultaneous interpreter.\n"
        "Translate incrementally and continue from the committed Chinese prefix.\n"
        "Do not explain anything. Just continue the translation itself.\n\n"
        f"Observed English source so far:\n{observed_source}\n\n"
        f"Assume this one future continuation is true:\n{future}\n\n"
        f"Committed Chinese so far:\n{committed_chinese or '<EMPTY>'}\n\n"
        "Continue the Chinese translation now:"
    )


def _normalize_compare_text(text: str) -> str:
    return re.sub(r"[，。！？；：、（）《》【】“”‘’…,.!?;:'\"()\-]+", "", local_policy.normalize_zh(text))


def _strip_probe_token_for_gate(token: str) -> str:
    token = local_policy.normalize_zh(str(token or ""))
    token = token.replace("�", "")
    token = re.sub(r"\s+", "", token)
    token = re.sub(r"[，。！？；：、（）《》【】“”‘’…,.!?;:'\"()\[\]{}<>\-_/|`~]+", "", token)
    return token


def _filter_probe_distribution(probs: Dict[str, float]) -> Dict[str, float]:
    kept: Dict[str, float] = {}
    for token, value in probs.items():
        if value <= 0.0:
            continue
        if not _strip_probe_token_for_gate(token):
            continue
        kept[token] = value
    total = sum(kept.values())
    if total <= 0.0:
        return {}
    return {token: value / total for token, value in kept.items()}


def _extract_probe_prefix(raw_content: str, committed_chinese: str, max_chars: int) -> str:
    candidate = (raw_content or "").splitlines()[0] if raw_content else ""
    candidate = local_policy.strip_committed_suffix_from_delta(committed_chinese, candidate)
    candidate = _normalize_compare_text(candidate)
    if not candidate:
        return ""
    cjk_chars = re.findall(r"[㐀-䶿一-鿿豈-﫿]", candidate)
    normalized = "".join(cjk_chars) if cjk_chars else local_policy.normalize_zh(candidate)
    return normalized[: max(0, int(max_chars))]

#  gate 怎么真正拿 distribution
def _call_probe_prefix_model(
    thinking_pool: GeminiChatCompletionsPool,
    model: str,
    observed_source: str,
    future: str,
    committed_chinese: str,
    *,
    top_k_logprobs: int,
    rollout_tokens: int,
    rollout_max_chars: int,
) -> Dict[str, Any]:
    del thinking_pool, model

    prompt = _build_probe_prompt(observed_source, future, committed_chinese)
    client = _gate_client()
    #调用本地 gate server，要求模型“继续翻译，但不要解释，只输出 continuation”
    response = client.completions.create(
        model=_LOCAL_GATE_API_MODEL_NAME,
        prompt=prompt,
        max_tokens=max(1, int(rollout_tokens)),
        temperature=0.0,
        logprobs=max(1, int(top_k_logprobs)),  
        stop=["\n"],
    ) # ---> 接下来的几个 target token 是什么？它们的概率分布是什么
    choice = response.choices[0]
    raw_content = getattr(choice, "text", "") or ""
    cleaned_prefix = _extract_probe_prefix(raw_content, committed_chinese, rollout_max_chars)
    choice_logprobs = getattr(choice, "logprobs", None)
    top_logprobs = list(getattr(choice_logprobs, "top_logprobs", []) or []) if choice_logprobs is not None else []
    tokens = list(getattr(choice_logprobs, "tokens", []) or []) if choice_logprobs is not None else []
    # 第一个生成位置 的 top-logprobs
    first_entry = top_logprobs[0] if top_logprobs else None
    first_distribution: Dict[str, float] = {}
    raw_first_distribution: Dict[str, float] = {}
    semantic_mass = 0.0
    if first_entry:
        scored = [(str(token), float(logprob)) for token, logprob in first_entry.items() if logprob is not None]
        if scored:
            max_lp = max(lp for _, lp in scored)
            weights = [(token, math.exp(lp - max_lp)) for token, lp in scored]
            total = sum(weight for _, weight in weights)
            if total > 0.0:
                raw_first_distribution = {token: weight / total for token, weight in weights}
                # 过滤空格、换行、纯标点等格式 token，再重新归一化
                first_distribution = _filter_probe_distribution(raw_first_distribution)
                semantic_mass = sum(
                    value for token, value in raw_first_distribution.items() if _strip_probe_token_for_gate(token)
                )
    first_top1_token_id = None
    if first_distribution:
        first_top1_token_id = max(first_distribution.items(), key=lambda kv: kv[1])[0]
    first_top_candidates = []
    if first_entry:
        first_top_candidates = [
            {"token_id": str(token), "decoded_token": str(token), "logprob": round(float(logprob), 4)}
            for token, logprob in sorted(first_entry.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_k_logprobs))]
            if logprob is not None
        ]
    return {
        "content_text": raw_content,
        "cleaned_prefix": cleaned_prefix,
        "first_distribution": first_distribution,
        "raw_first_distribution": raw_first_distribution,
        "semantic_mass": semantic_mass,
        "first_top_candidates": first_top_candidates,
        "first_top1_token_id": first_top1_token_id,
        "generated_token_ids": [str(tok) for tok in tokens],
        "finish_reason": getattr(choice, "finish_reason", "api_generate"),
        "model_name": _LOCAL_GATE_API_MODEL_NAME,
    }

# 对单个 future 下的 first-token distribution 算 entropy
def _normalized_entropy_from_probs(probs: Dict[str, float]) -> float:
    # entropy 低：模型在这个 future 下对下一 token 很确定
    # entropy 高：模型在这个 future 下自己都犹豫
    probs = {key: value for key, value in probs.items() if value > 0.0}
    support = len(probs)
    if support <= 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in probs.values())
    return entropy / math.log(support) # 再除以 log(support) 做归一化, ---> [0, 1]


def _normalized_js_divergence(probs_list: List[Dict[str, float]]) -> float:
    probs_list = [{k: v for k, v in probs.items() if v > 0.0} for probs in probs_list if probs]
    if len(probs_list) < 2:
        return 0.0
    support = sorted({key for probs in probs_list for key in probs.keys()})
    if len(support) <= 1:
        return 0.0

    def _entropy(prob_values: List[float]) -> float:
        return -sum(p * math.log(p) for p in prob_values if p > 0.0)

    distributions: List[List[float]] = []
    for probs in probs_list:
        distributions.append([probs.get(key, 0.0) for key in support])
    mean_distribution = [
        sum(dist[idx] for dist in distributions) / len(distributions)
        for idx in range(len(support))
    ]
    js = _entropy(mean_distribution) - sum(_entropy(dist) for dist in distributions) / len(distributions)
    return max(0.0, js / math.log(len(support)))




def _future_distribution_probe_info(
    thinking_pool: GeminiChatCompletionsPool,
    thinking_model: str,
    observed_source: str,
    futures: List[str],
    committed_chinese: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "enabled": False,
        "trigger": False,
        "reason": "",
        "avg_entropy": 0.0,
        "js_divergence": 0.0,
        "avg_semantic_mass": 0.0,
        "entropy_trigger": False,
        "js_trigger": False,
        "semantic_mass_trigger": False,
        "futures_evaluated": 0,
        "top_k_logprobs": int(args.probe_top_k_logprobs),
        "rollout_tokens": int(args.probe_rollout_tokens),
        "rollout_max_chars": int(args.probe_rollout_max_chars),
        "per_future": [],
    }
    if args.no_fallback_on_uncertainty:
        info["reason"] = "disabled_by_no_fallback_on_uncertainty"
        return info

    cleaned = [f for f in futures if (f or "").strip()]
    if len(cleaned) < 2:
        info["reason"] = "too_few_futures"
        return info

    max_futures = int(args.probe_max_futures)
    if max_futures > 0:
        cleaned = cleaned[:max_futures]
    rollout_tokens = max(1, int(args.probe_rollout_tokens))
    rollout_max_chars = max(1, int(args.probe_rollout_max_chars))
    top_k_logprobs = max(1, int(args.probe_top_k_logprobs))

    info["enabled"] = True
    distributions: List[Dict[str, float]] = []
    per_future_meta: List[Dict[str, Any]] = []
    semantic_masses: List[float] = []

    # 对每个 future 单独 probe 一次
    for future in cleaned:
        # 真正拿概率分布
        probe = _call_probe_prefix_model( # 要求模型“继续翻译，但不要解释，只输出 continuation”
            thinking_pool,
            thinking_model,
            observed_source,
            future,
            committed_chinese,
            top_k_logprobs=top_k_logprobs,
            rollout_tokens=rollout_tokens,
            rollout_max_chars=rollout_max_chars,
        )
        first_distribution = probe.get("first_distribution", {})
        # 对单个 future 下的 first-token distribution 算 entropy
        entropy_value = _normalized_entropy_from_probs(first_distribution)
        distributions.append(first_distribution)
        semantic_mass = float(probe.get("semantic_mass", 0.0) or 0.0)
        semantic_masses.append(semantic_mass)
        per_future_meta.append(
            {
                "future": future,
                "greedy_prefix": probe.get("cleaned_prefix", "") or "",
                "semantic_mass": round(semantic_mass, 4),
                "first_distribution": {k: round(v, 4) for k, v in sorted(first_distribution.items(), key=lambda kv: kv[1], reverse=True)[:top_k_logprobs]},
                "raw_first_distribution": {
                    k: round(v, 4)
                    for k, v in sorted(probe.get("raw_first_distribution", {}).items(), key=lambda kv: kv[1], reverse=True)[:top_k_logprobs]
                },
                "entropy": round(entropy_value, 4),
                "first_top1_token_id": probe.get("first_top1_token_id"),
                "first_top_candidates": probe.get("first_top_candidates", []),
                "generated_token_ids": probe.get("generated_token_ids", []),
                "content_text": probe.get("content_text", ""),
            }
        )
    avg_entropy = sum(item["entropy"] for item in per_future_meta) / len(per_future_meta)
    js_divergence = _normalized_js_divergence(distributions)
    avg_semantic_mass = sum(semantic_masses) / len(semantic_masses) if semantic_masses else 0.0

    entropy_trigger = avg_entropy >= float(args.probe_avg_entropy_threshold)
    js_trigger = js_divergence >= float(args.probe_js_threshold)
    semantic_mass_trigger = (
        avg_semantic_mass < float(args.probe_min_semantic_mass)
        and js_divergence > 0.0
    )
    reason_parts = []
    if entropy_trigger:
        reason_parts.append(f"avg_entropy={avg_entropy:.3f}>={float(args.probe_avg_entropy_threshold):.3f}")
    if js_trigger:
        reason_parts.append(f"js_divergence={js_divergence:.3f}>={float(args.probe_js_threshold):.3f}")
    if semantic_mass_trigger:
        reason_parts.append(
            f"avg_semantic_mass={avg_semantic_mass:.3f}<{float(args.probe_min_semantic_mass):.3f}"
        )

    info.update(
        {
            "trigger": bool(reason_parts),
            "reason": ";".join(reason_parts),
            "avg_entropy": round(avg_entropy, 4),
            "js_divergence": round(js_divergence, 4),
            "avg_semantic_mass": round(avg_semantic_mass, 4),
            "entropy_trigger": entropy_trigger,
            "js_trigger": js_trigger,
            "semantic_mass_trigger": semantic_mass_trigger,
            "futures_evaluated": len(cleaned),
            "top_k_logprobs": top_k_logprobs,
            "rollout_tokens": rollout_tokens,
            "rollout_max_chars": rollout_max_chars,
            "per_future": per_future_meta,
        }
    )
    return info




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
        "step2_gate_probe_s": 0.0,
        "step2_thinking_delta_s": 0.0,
        "step2_alignment_check_s": 0.0,
        "step2_fallback_thinking_s": 0.0,
        "step2_fallback_alignment_s": 0.0,
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
            final_model = _default_final_completion_model(args, thinking_model)
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
            _append_usage_entry(usage_bucket, final_debug.get("usage", {}))
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
        # step1: future sampling
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

        t_gate_0 = time.perf_counter()
        # step3: gate --- future probe
        probe_meta = _future_distribution_probe_info(
            thinking_pool,
            thinking_model,
            accumulated_source,
            futures,
            committed,
            args,
        )
        timing["step2_gate_probe_s"] += time.perf_counter() - t_gate_0
        if probe_meta.get("enabled"):
            _record_fallback("probe_runs")
        else:
            _record_fallback("probe_skipped")
        if probe_meta.get("entropy_trigger"):
            _record_fallback("probe_entropy_trigger")
        if probe_meta.get("js_trigger"):
            _record_fallback("probe_js_trigger")
        if probe_meta.get("semantic_mass_trigger"):
            _record_fallback("probe_semantic_mass_trigger")
        if probe_meta.get("trigger"):
            _record_fallback("probe_triggered")

        local_policy._vlog(verbose_log_file, f"  step2_future_distribution_probe: {json.dumps(probe_meta, ensure_ascii=False)}")

        fallback_model = (args.fallback_model_name or "").strip()
        can_escalate = bool(
            fallback_model
            and fallback_model != thinking_model
            and not args.disable_pro_fallback
        )
        route_to_pro = bool(probe_meta.get("trigger")) and can_escalate
        selected_model = fallback_model if route_to_pro else thinking_model
        if route_to_pro:
            _record_fallback("routed_pro")
            selected_reason = probe_meta.get("reason") or "future_distribution_probe"
        else:
            _record_fallback("routed_flash")
            if probe_meta.get("trigger") and not can_escalate:
                selected_reason = "probe_triggered_but_no_pro_available"
            elif probe_meta.get("enabled"):
                selected_reason = "future_distribution_probe_flash"
            else:
                selected_reason = probe_meta.get("reason") or "probe_disabled_flash"

        local_policy._vlog(verbose_log_file, f"  step2_selected_model: {selected_model!r}")
        local_policy._vlog(verbose_log_file, f"  step2_selection_reason: {selected_reason!r}")

        t2_0 = time.perf_counter()
        selected_raw_delta, selected_debug = _call_json_thinking_model(
            thinking_pool,
            selected_model,
            user_content,
            temperature=_fallback_temperature(args) if route_to_pro else args.thinking_temperature,
            max_tokens=_fallback_max_tokens(args) if route_to_pro else args.thinking_max_tokens,
            reasoning_effort=_fallback_reasoning_effort(args) if route_to_pro else args.thinking_reasoning_effort,
        )
        timing["step2_fallback_thinking_s" if route_to_pro else "step2_thinking_delta_s"] += time.perf_counter() - t2_0
        _append_usage_entry(usage_bucket, selected_debug.get("usage", {}))

        t2a_0 = time.perf_counter()
        selected_checked_delta, selected_alignment_debug = _run_alignment_check(
            accumulated_source=accumulated_source,
            futures=futures,
            committed=committed,
            delta=selected_raw_delta,
            align_model=align_model,
            align_tokenizer=align_tokenizer,
            args=args,
        )
        timing["step2_fallback_alignment_s" if route_to_pro else "step2_alignment_check_s"] += time.perf_counter() - t2a_0

        _log_thinking_attempt(
            verbose_log_file,
            "step2_selected",
            selected_debug,
            selected_raw_delta,
            selected_alignment_debug,
            selected_checked_delta,
        )
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
        "usage": _finalize_usage_bucket(usage_bucket),
        "cost": {},
        "metadata": {
            "thinking_model_name": thinking_model,
            "fallback_model_name": (args.fallback_model_name or "").strip(),
            "thinking_prompt_format": "json_v1",
            "thinking_reasoning_effort": args.thinking_reasoning_effort,
            "fallback_reasoning_effort": _fallback_reasoning_effort(args),
            "probe_max_futures": args.probe_max_futures,
            "probe_top_k_logprobs": args.probe_top_k_logprobs,
            "probe_rollout_tokens": args.probe_rollout_tokens,
            "probe_rollout_max_chars": args.probe_rollout_max_chars,
            "probe_avg_entropy_threshold": args.probe_avg_entropy_threshold,
            "probe_js_threshold": args.probe_js_threshold,
            "probe_min_semantic_mass": args.probe_min_semantic_mass,
            "gate_api_base": _LOCAL_GATE_API_BASE,
            "gate_api_model_name": _LOCAL_GATE_API_MODEL_NAME,
            "disable_post_simalign_check": True,
        },
    }
    result["cost"] = {
        "flash_cost_usd": result["usage"]["flash_cost_usd"],
        "pro_cost_usd": result["usage"]["pro_cost_usd"],
        "total_cost_usd": result["usage"]["total_cost_usd"],
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

    args.gpu_memory_utilization = _FIXED_BASE_GPU_MEMORY_UTILIZATION
    args.parallel_utterances = _FIXED_PARALLEL_UTTERANCES
    args.future_sampling_batch_size = _FIXED_FUTURE_SAMPLING_BATCH_SIZE
    args.future_sampling_batch_wait = _FIXED_FUTURE_SAMPLING_BATCH_WAIT
    args.disable_post_simalign_check = True
    args.disable_second_flash_check = True
    args.disable_pro_fallback = False
    args.no_fallback_on_uncertainty = False
    args.align_device = "cuda:0"

    print("[Ablation] Simalign post-check is disabled for this script.")
    print("[Ablation] Second Flash validation is disabled for this script.")
    print("[Ablation] Future-distribution routing is enabled: next-token logprobs + greedy prefix agreement decide whether to use Pro.")
    print(f"[Ablation] Gate server: {args.gate_api_base!r} model={args.gate_api_model_name!r}")
    print(f"[Ablation] Fixed base GPU memory utilization: {_FIXED_BASE_GPU_MEMORY_UTILIZATION:.2f}")

    global _GEMINI_API_KEY
    global _GEMINI_TIMEOUT
    global _GEMINI_PRIMARY_MODEL
    global _GEMINI_FALLBACK_MODEL
    global _GEMINI_FINAL_COMPLETION_MAX_TOKENS
    global _GEMINI_FINAL_COMPLETION_REASONING_EFFORT
    global _GEMINI_INCLUDE_THOUGHTS
    global _LOCAL_GATE_API_BASE
    global _LOCAL_GATE_API_KEY
    global _LOCAL_GATE_API_MODEL_NAME
    global _LOCAL_GATE_API_TIMEOUT
    global _LOCAL_GATE_CLIENT

    _GEMINI_API_KEY = api_key
    _GEMINI_TIMEOUT = float(args.gemini_timeout)
    _GEMINI_PRIMARY_MODEL = args.thinking_model_name
    _GEMINI_FALLBACK_MODEL = args.fallback_model_name
    _GEMINI_FINAL_COMPLETION_MAX_TOKENS = int(args.final_completion_max_tokens)
    _GEMINI_FINAL_COMPLETION_REASONING_EFFORT = (_fallback_reasoning_effort(args) or "minimal")
    _GEMINI_INCLUDE_THOUGHTS = bool(args.gemini_include_thoughts and not args.no_gemini_include_thoughts)
    print(f"[Gemini] include_thoughts={_GEMINI_INCLUDE_THOUGHTS}")
    _LOCAL_GATE_API_BASE = (args.gate_api_base or "").strip()
    _LOCAL_GATE_API_KEY = args.gate_api_key
    _LOCAL_GATE_API_MODEL_NAME = (args.gate_api_model_name or "qwen3-instruct").strip()
    _LOCAL_GATE_API_TIMEOUT = float(args.gate_api_timeout)
    _LOCAL_GATE_CLIENT = None

    local_policy.parse_args = lambda: args
    local_policy.resolve_thinking_api_bases = resolve_thinking_api_bases
    local_policy.process_one_utterance = process_one_utterance
    local_policy.load_base_model = _load_base_and_gate_model
    local_policy.build_thinking_prompt = build_thinking_prompt
    local_policy.build_final_completion_prompt = build_final_completion_prompt
    local_policy.call_thinking_model = call_thinking_model
    local_policy.force_complete_translation = force_complete_translation

    _thinking_pool_ref: List[GeminiChatCompletionsPool] = []

    class _TrackingPool(GeminiChatCompletionsPool):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            _thinking_pool_ref.append(self)

    local_policy.ThinkingServerPool = _TrackingPool

    try:
        local_policy.main()
    finally:
        local_policy.load_base_model = _ORIGINAL_LOAD_BASE_MODEL

    summary = None
    try:
        if _thinking_pool_ref:
            summary = _thinking_pool_ref[0].token_summary()
    except Exception:
        summary = None

    if summary is not None:
        print(
            f"\n[Token Usage] requests={summary['total_requests']} | "
            f"prompt={summary['total_prompt_tokens']:,} | "
            f"completion={summary['total_completion_tokens']:,} | "
            f"reasoning_reported_by_api={summary.get('total_reasoning_tokens_reported_by_api')} | "
            f"total={summary['total_tokens']:,} | "
            f"estimated_cost=${summary.get('estimated_cost_usd', 0.0):.6f} USD"
        )
        for model_name, stats in summary.get("per_model", {}).items():
            print(
                f"[Token Usage] model={model_name} requests={stats['requests']} "
                f"prompt={stats['prompt_tokens']:,} completion={stats['completion_tokens']:,} "
                f"reasoning_reported_by_api={stats.get('reasoning_tokens_reported_by_api')} "
                f"cached={stats.get('cached_tokens', 0):,} "
                f"total={stats['total_tokens']:,} "
                f"pricing_tier={stats.get('pricing_tier')} "
                f"estimated_cost=${(stats.get('estimated_cost_usd') or 0.0):.6f} USD"
            )
        try:
            usage_path = Path(args.output_root) / "gemini_usage_summary.json"
            usage_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[Token Usage] wrote summary to {usage_path}")
        except Exception as e:
            print(f"[Token Usage] failed to write usage summary: {e}")

    with _FALLBACK_STATS_LOCK:
        stats = dict(_FALLBACK_STATS)
    print(
        "[Probe Routing Stats] "
        f"probe_runs={stats.get('probe_runs', 0)} | "
        f"probe_triggered={stats.get('probe_triggered', 0)} | "
        f"probe_entropy_trigger={stats.get('probe_entropy_trigger', 0)} | "
        f"probe_js_trigger={stats.get('probe_js_trigger', 0)} | "
        f"probe_semantic_mass_trigger={stats.get('probe_semantic_mass_trigger', 0)} | "
        f"routed_pro={stats.get('routed_pro', 0)} | "
        f"routed_flash={stats.get('routed_flash', 0)} | "
        f"probe_skipped={stats.get('probe_skipped', 0)}"
    )


if __name__ == "__main__":
    main()
