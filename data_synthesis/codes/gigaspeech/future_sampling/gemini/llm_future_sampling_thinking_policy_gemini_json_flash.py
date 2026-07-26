#!/usr/bin/env python3
"""
Gemini Flash JSON-structured thinking-policy ablation.

This wrapper keeps the local future-sampling + simalign logic from
llm_future_sampling_thinking_policy.py, but replaces the thinking/final
Gemini calls with a JSON-structured prompt/response style.

Default ablation settings:
  - Gemini Flash model
  - no thought summaries
  - JSON-only outputs for step delta and final completion
  - thinking max tokens = 256

This file is intended to be easy to A/B against
llm_future_sampling_thinking_policy_gemini.py without modifying the original.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import llm_future_sampling_thinking_policy as local_policy
import llm_future_sampling_thinking_policy_json as local_json


_GEMINI_API_KEY: str = ""
_GEMINI_TIMEOUT: float = 600.0
_GEMINI_REASONING_EFFORT: str = "low"
_GEMINI_TARGET_MODEL: str = "gemini-3-flash-preview"
_GEMINI_THINKING_MAX_TOKENS: int = 256
_GEMINI_FINAL_COMPLETION_MAX_TOKENS: int = 2048
_GEMINI_INCLUDE_THOUGHTS: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Thinking-policy simultaneous interpretation with local base future "
            "sampling and Gemini Flash JSON-structured outputs."
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
        help="Gemini reasoning_effort passed through the OpenAI-compatible API.",
    )
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
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)

    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--num-futures", type=int, default=5, help="N future continuations per step.")
    p.add_argument("--future-tokens", type=int, default=10)
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--thinking-temperature", type=float, default=0.1)
    p.add_argument(
        "--thinking-max-tokens",
        type=int,
        default=256,
        help="Max tokens for the per-chunk JSON thinking call. Default 256 for this ablation.",
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
    raw_message = _to_jsonish(message)
    reasoning_parts: List[str] = []
    answer_parts: List[str] = []

    raw_reasoning = local_policy._message_text_to_str(getattr(message, "reasoning", None))
    raw_reasoning_content = local_policy._message_text_to_str(getattr(message, "reasoning_content", None))
    if raw_reasoning:
        reasoning_parts.append(raw_reasoning)
    if raw_reasoning_content and raw_reasoning_content != raw_reasoning:
        reasoning_parts.append(raw_reasoning_content)

    content = getattr(message, "content", None)
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                text = local_policy._message_text_to_str(item)
                if text:
                    answer_parts.append(text)
                continue
            item_type = str(item.get("type", "") or "").strip().lower()
            text = local_policy._message_text_to_str(item.get("text"))
            if not text:
                continue
            if item_type in {"output_text", "text"}:
                answer_parts.append(text)
            elif "thought" in item_type or "reason" in item_type or "summary" in item_type:
                reasoning_parts.append(text)
            else:
                answer_parts.append(text)
    else:
        raw_content = local_policy._message_text_to_str(content)
        if raw_content:
            answer_parts.append(raw_content)

    import re as _re

    merged = "".join(part for part in answer_parts if part)
    thought_matches = _re.findall(r"<thought>(.*?)</thought>", merged, flags=_re.DOTALL)
    if thought_matches:
        for t in thought_matches:
            reasoning_parts.append(t.strip())
        merged = _re.sub(r"<thought>.*?</thought>", "", merged, flags=_re.DOTALL).strip()
        answer_parts = [merged]

    raw_content_text = "".join(part for part in answer_parts if part).strip()
    content_text = raw_content_text.strip()

    reasoning_text = "\n".join(part for part in reasoning_parts if part).strip()
    raw_fields = {
        "message.raw": raw_message,
        "message.reasoning": raw_reasoning,
        "message.reasoning_content": raw_reasoning_content,
        "message.content": raw_content_text,
    }
    return reasoning_text, content_text, raw_fields


def _build_chat_kwargs(
    *,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: Optional[float],
    reasoning_effort: Optional[str],
    include_thoughts: bool = False,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    # Gemini OpenAI-compat rejects combining top-level reasoning_effort with
    # custom google.thinking_config. When we request thought summaries, move the
    # reasoning control into thinking_config instead.
    if include_thoughts:
        thinking_config: Dict[str, Any] = {"include_thoughts": True}
        if reasoning_effort and reasoning_effort != "none":
            thinking_config["thinking_level"] = reasoning_effort
        kwargs["extra_body"] = {
            "extra_body": {
                "google": {
                    "thinking_config": thinking_config,
                }
            }
        }
    elif reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return kwargs


def build_thinking_prompt(
    observed_source: str,
    futures: List[str],
    committed_chinese: str,
) -> str:
    return local_json.build_thinking_json_prompt(observed_source, futures, committed_chinese)


def build_final_completion_prompt(full_source: str, committed_chinese: str) -> str:
    return local_json.build_final_completion_json_prompt(full_source, committed_chinese)


def _thinking_system_prompt() -> str:
    return (
        "You are a professional English-to-Chinese simultaneous interpreter.\n"
        "Return only one JSON object with keys action and delta.\n"
        "No prose, no markdown, no code fences, no explanations."
    )


def _final_completion_system_prompt() -> str:
    return (
        "You are a professional English-to-Chinese translator.\n"
        "Return only one JSON object with the required continuation field.\n"
        "No prose, no markdown, no code fences, no explanations."
    )


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
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_requests = 0

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
        results: List[Tuple[str, List[str]]] = []
        for slot in self._slots:
            models = slot["client"].models.list()
            model_ids = [m.id for m in models.data]
            if _GEMINI_TARGET_MODEL and _GEMINI_TARGET_MODEL in model_ids:
                visible = [_GEMINI_TARGET_MODEL]
            else:
                visible = model_ids[:20]
            results.append((slot["api_base"], visible))
        return results

    def chat_completions_create(self, **kwargs) -> Tuple[Any, str]:
        errors: List[str] = []
        tried: set = set()
        for _ in range(len(self._slots)):
            idx, slot = self._acquire_slot(exclude=tried)
            tried.add(idx)
            try:
                resp = slot["client"].chat.completions.create(**kwargs)
                if getattr(resp, "usage", None):
                    with self._lock:
                        self._total_prompt_tokens += resp.usage.prompt_tokens or 0
                        self._total_completion_tokens += resp.usage.completion_tokens or 0
                        self._total_requests += 1
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
                    "total_prompt_tokens": self._total_prompt_tokens,
                    "total_completion_tokens": self._total_completion_tokens,
                    "total_requests": self._total_requests,
                }
                for slot in self._slots
            ]

    def token_summary(self) -> Dict[str, Any]:
        with self._lock:
            prompt = self._total_prompt_tokens
            completion = self._total_completion_tokens
            cost_usd = (prompt * 0.50 + completion * 3.00) / 1_000_000
            return {
                "total_requests": self._total_requests,
                "total_prompt_tokens": prompt,
                "total_completion_tokens": completion,
                "total_tokens": prompt + completion,
                "estimated_cost_usd": round(cost_usd, 6),
            }

    def close(self) -> None:
        return None


def call_thinking_model(
    thinking_pool: GeminiChatCompletionsPool,
    model: str,
    user_content: str,
    committed_chinese: str = "",
    temperature: float = 0.1,
    max_tokens: int = 256,
) -> Tuple[str, Dict[str, Any]]:
    del committed_chinese

    request_kwargs = _build_chat_kwargs(
        model=model,
        system_prompt=_thinking_system_prompt(),
        user_content=user_content,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=_GEMINI_REASONING_EFFORT,
        include_thoughts=_GEMINI_INCLUDE_THOUGHTS,
    )
    resp, api_base = thinking_pool.chat_completions_create(**request_kwargs)
    choice = resp.choices[0]
    message = choice.message
    reasoning_text, content_text, raw_message_fields = _extract_message_debug(message)
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
        "include_thoughts": _GEMINI_INCLUDE_THOUGHTS,
    }


def force_complete_translation(
    thinking_pool: GeminiChatCompletionsPool,
    model: str,
    full_source: str,
    committed_chinese: str,
) -> Tuple[str, Dict[str, Any]]:
    prompt = build_final_completion_prompt(full_source, committed_chinese)
    request_kwargs = _build_chat_kwargs(
        model=model,
        system_prompt=_final_completion_system_prompt(),
        user_content=prompt,
        temperature=0.0,
        max_tokens=_GEMINI_FINAL_COMPLETION_MAX_TOKENS,
        reasoning_effort="none",
        include_thoughts=_GEMINI_INCLUDE_THOUGHTS,
    )
    resp, api_base = thinking_pool.chat_completions_create(**request_kwargs)
    choice = resp.choices[0]
    message = choice.message
    reasoning_text, content_text, raw_message_fields = _extract_message_debug(message)
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
        "include_thoughts": _GEMINI_INCLUDE_THOUGHTS,
    }


def main() -> None:
    args = parse_args()

    api_key = os.environ.get(args.gemini_api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"ERROR: env var {args.gemini_api_key_env} is not set. "
            "Set your Gemini API key before running this script."
        )

    if args.gemini_include_thoughts and args.no_gemini_include_thoughts:
        print("[Ablation] Both --gemini-include-thoughts and --no-gemini-include-thoughts were set; disabling thoughts.")
    if (args.thinking_prompt_version or "").strip() not in {"", "json_v1"}:
        print(
            f"[Ablation] Ignoring --thinking-prompt-version={args.thinking_prompt_version!r}; "
            "this script always uses the JSON-structured prompt path."
        )

    global _GEMINI_API_KEY
    global _GEMINI_TIMEOUT
    global _GEMINI_REASONING_EFFORT
    global _GEMINI_TARGET_MODEL
    global _GEMINI_THINKING_MAX_TOKENS
    global _GEMINI_FINAL_COMPLETION_MAX_TOKENS
    global _GEMINI_INCLUDE_THOUGHTS

    _GEMINI_API_KEY = api_key
    _GEMINI_TIMEOUT = float(args.gemini_timeout)
    _GEMINI_REASONING_EFFORT = args.thinking_reasoning_effort
    _GEMINI_TARGET_MODEL = args.thinking_model_name
    _GEMINI_THINKING_MAX_TOKENS = int(args.thinking_max_tokens)
    _GEMINI_FINAL_COMPLETION_MAX_TOKENS = int(args.final_completion_max_tokens)
    _GEMINI_INCLUDE_THOUGHTS = bool(args.gemini_include_thoughts and not args.no_gemini_include_thoughts)
    print(f"[Gemini] include_thoughts={_GEMINI_INCLUDE_THOUGHTS}")

    local_policy.parse_args = parse_args
    local_policy.resolve_thinking_api_bases = resolve_thinking_api_bases
    local_policy.call_thinking_model = call_thinking_model
    local_policy.force_complete_translation = force_complete_translation
    local_policy.build_thinking_prompt = build_thinking_prompt
    local_policy.build_final_completion_prompt = build_final_completion_prompt

    _thinking_pool_ref: List[Any] = []

    class _TrackingPool(GeminiChatCompletionsPool):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            _thinking_pool_ref.append(self)

    local_policy.ThinkingServerPool = _TrackingPool

    local_policy.main()

    if _thinking_pool_ref:
        pool = _thinking_pool_ref[0]
        summary = pool.token_summary()
        print(
            f"\n[Token Usage] requests={summary['total_requests']} | "
            f"prompt={summary['total_prompt_tokens']:,} | "
            f"completion={summary['total_completion_tokens']:,} | "
            f"total={summary['total_tokens']:,} | "
            f"estimated_cost=${summary['estimated_cost_usd']:.4f} USD"
        )


if __name__ == "__main__":
    main()
