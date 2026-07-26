#!/usr/bin/env python3
"""
OpenAI-hosted thinking-policy simultaneous interpretation pipeline.

This wrapper keeps the local future-sampling + simalign logic from
llm_future_sampling_thinking_policy.py, but replaces the local vLLM thinking
backend with the OpenAI Responses API (for hosted models such as gpt-5.4).
"""

from __future__ import annotations

import argparse
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

import llm_future_sampling_thinking_policy as local_policy


_OPENAI_API_KEY: str = ""
_OPENAI_TIMEOUT: float = 600.0
_OPENAI_REASONING_EFFORT: str = "medium"
_OPENAI_REASONING_SUMMARY: str = "auto"
_OPENAI_TEXT_VERBOSITY: str = "low"
_OPENAI_TARGET_MODEL: str = "gpt-5.4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Thinking-policy simultaneous interpretation with local base future "
            "sampling and hosted OpenAI thinking model via Responses API."
        )
    )
    p.add_argument("--input-tsv", required=True, help="Manifest TSV with src_text_full, src_trajectory.")
    p.add_argument("--output-root", required=True)

    p.add_argument("--base-model-path", default="/data/user_data/haolingp/models/Qwen3-4B-Base")
    p.add_argument("--thinking-api-base", default="https://api.openai.com/v1")
    p.add_argument(
        "--thinking-api-bases",
        default="",
        help=(
            "Comma-separated list of OpenAI-compatible API bases. Normally you "
            "only need the default OpenAI endpoint."
        ),
    )
    p.add_argument("--thinking-model-name", default="gpt-5.4")
    p.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--openai-timeout", type=float, default=600.0)
    p.add_argument(
        "--thinking-reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default="medium",
        help="OpenAI reasoning effort passed to the Responses API.",
    )
    p.add_argument(
        "--thinking-reasoning-summary",
        choices=["auto", "concise", "detailed"],
        default="auto",
        help="Reasoning summary style requested from the Responses API.",
    )
    p.add_argument(
        "--thinking-verbosity",
        choices=["low", "medium", "high"],
        default="low",
        help="Verbosity control for OpenAI text output.",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)

    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--num-futures", type=int, default=5, help="N future continuations per step.")
    p.add_argument("--future-tokens", type=int, default=10)
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--thinking-temperature", type=float, default=0.1)
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
        help=(
            "Number of utterances to process concurrently. With OpenAI-hosted "
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


def _extract_message_text(output_items: List[Any]) -> str:
    texts: List[str] = []
    for item in output_items:
        item_type = getattr(item, "type", None)
        if item_type != "message":
            continue
        for part in getattr(item, "content", None) or []:
            part_type = getattr(part, "type", None)
            if part_type == "output_text":
                text = getattr(part, "text", None)
                if text:
                    texts.append(str(text))
    return "".join(texts).strip()


def _extract_response_debug(resp: Any) -> Tuple[str, str, Dict[str, Any]]:
    raw_output_items = [_to_jsonish(item) for item in (getattr(resp, "output", None) or [])]

    reasoning_parts: List[str] = []
    for item in getattr(resp, "output", None) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        for part in getattr(item, "summary", None) or []:
            text = getattr(part, "text", None)
            if text:
                reasoning_parts.append(str(text).strip())

    raw_content_text = (getattr(resp, "output_text", None) or "").strip()
    if not raw_content_text:
        raw_content_text = _extract_message_text(getattr(resp, "output", None) or [])

    content_text = local_policy._extract_answer_candidate(raw_content_text)
    if not content_text:
        content_text = local_policy.clean_llm_output(raw_content_text)
    content_text = (content_text or "").strip()

    reasoning_text = "\n".join(part for part in reasoning_parts if part).strip()
    raw_fields = {
        "response.id": getattr(resp, "id", None),
        "response.status": getattr(resp, "status", None),
        "response.output_text": raw_content_text,
        "response.usage": _to_jsonish(getattr(resp, "usage", None)),
        "response.output": raw_output_items,
        "message.reasoning": reasoning_text,
        "message.reasoning_content": reasoning_text,
        "message.content": raw_content_text,
    }
    return reasoning_text, content_text, raw_fields


def _max_output_tokens_debug(resp: Any, raw_fields: Dict[str, Any], stage: str) -> Dict[str, Any]:
    incomplete_details = _to_jsonish(getattr(resp, "incomplete_details", None))
    incomplete_reason = None
    if isinstance(incomplete_details, dict):
        incomplete_reason = incomplete_details.get("reason")

    ran_out_of_tokens = (
        getattr(resp, "status", None) == "incomplete"
        and incomplete_reason == "max_output_tokens"
    )
    partial_output = str(raw_fields.get("response.output_text", "") or "").strip()
    ran_out_during_reasoning = ran_out_of_tokens and not partial_output

    if ran_out_of_tokens:
        print(f"[OpenAI:{stage}] Ran out of tokens")
        if partial_output:
            print(f"[OpenAI:{stage}] Partial output: {partial_output}")
        else:
            print(f"[OpenAI:{stage}] Ran out of tokens during reasoning")

    return {
        "response_status": getattr(resp, "status", None),
        "incomplete_details": incomplete_details,
        "ran_out_of_tokens": ran_out_of_tokens,
        "partial_output": partial_output,
        "ran_out_during_reasoning": ran_out_during_reasoning,
    }


def _supports_temperature(model: str, reasoning_effort: str) -> bool:
    model_name = (model or "").strip().lower()
    effort = (reasoning_effort or "").strip().lower()

    if model_name.startswith("gpt-5.4") or model_name.startswith("gpt-5.2"):
        return effort == "none"
    if model_name.startswith("gpt-5"):
        return False
    return True


def _build_responses_kwargs(
    *,
    model: str,
    instructions: str,
    input_text: str,
    max_output_tokens: int,
    temperature: Optional[float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "max_output_tokens": max_output_tokens,
        "store": False,
        "reasoning": {
            "effort": _OPENAI_REASONING_EFFORT,
            "summary": _OPENAI_REASONING_SUMMARY,
        },
        "text": {"verbosity": _OPENAI_TEXT_VERBOSITY},
    }

    temperature_ignored = False
    if temperature is not None:
        if _supports_temperature(model, _OPENAI_REASONING_EFFORT):
            kwargs["temperature"] = temperature
        else:
            temperature_ignored = True
            print(
                f"[OpenAI] Ignoring temperature={temperature} for model={model} "
                f"with reasoning.effort={_OPENAI_REASONING_EFFORT}"
            )

    return kwargs, {
        "temperature_requested": temperature,
        "temperature_sent": kwargs.get("temperature"),
        "temperature_ignored": temperature_ignored,
    }


class OpenAIResponsesPool:
    """Load-balance Responses API calls across one or more compatible endpoints."""

    def __init__(self, api_bases: List[str]):
        bases = [b.strip() for b in api_bases if (b or "").strip()]
        if not bases:
            raise ValueError("OpenAIResponsesPool requires at least one API base.")
        if not _OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is empty. Set the configured API key env var first.")

        self._slots = [
            {
                "api_base": api_base,
                "client": OpenAI(api_key=_OPENAI_API_KEY, base_url=api_base, timeout=_OPENAI_TIMEOUT),
                "inflight": 0,
                "requests": 0,
            }
            for api_base in bases
        ]
        self._lock = threading.Lock()
        self._rr = 0

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
                raise RuntimeError("No available OpenAI Responses slot.")
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
            if _OPENAI_TARGET_MODEL and _OPENAI_TARGET_MODEL in model_ids:
                visible = [_OPENAI_TARGET_MODEL]
            else:
                visible = model_ids[:20]
            results.append((slot["api_base"], visible))
        return results

    def responses_create(self, **kwargs) -> Tuple[Any, str]:
        errors: List[str] = []
        tried: set = set()
        for _ in range(len(self._slots)):
            idx, slot = self._acquire_slot(exclude=tried)
            tried.add(idx)
            try:
                resp = slot["client"].responses.create(**kwargs)
                return resp, slot["api_base"]
            except Exception as e:
                errors.append(f"{slot['api_base']}: {e}")
            finally:
                self._release_slot(idx)
        raise RuntimeError("All OpenAI Responses endpoints failed: " + " | ".join(errors))

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

    def close(self) -> None:
        return None


def call_thinking_model(
    thinking_pool: OpenAIResponsesPool,
    model: str,
    user_content: str,
    committed_chinese: str = "",
    temperature: float = 0.3,
    max_tokens: int = 256,
) -> Tuple[str, Dict[str, Any]]:
    del committed_chinese

    request_kwargs, request_debug = _build_responses_kwargs(
        model=model,
        instructions=(
            "You are a simultaneous interpretation policy model. Think carefully, "
            "but output ONLY the next safe Chinese segment to emit, or EMPTY if "
            "no new Chinese is safe yet. Never explain your answer."
        ),
        temperature=temperature,
        input_text=user_content,
        max_output_tokens=max_tokens,
    )
    resp, api_base = thinking_pool.responses_create(**request_kwargs)
    reasoning_text, content_text, raw_message_fields = _extract_response_debug(resp)
    incomplete_debug = _max_output_tokens_debug(resp, raw_message_fields, stage="thinking")
    delta = ""
    if content_text and content_text.upper() != "EMPTY":
        delta = local_policy.normalize_zh(content_text)
    return delta, {
        "server_api_base": api_base,
        "raw_message_fields": raw_message_fields,
        "reasoning_text": reasoning_text,
        "content_text": content_text,
        "cleaned_content": delta,
        "finish_reason": getattr(resp, "status", None),
        **request_debug,
        **incomplete_debug,
    }


def force_complete_translation(
    thinking_pool: OpenAIResponsesPool,
    model: str,
    full_source: str,
    committed_chinese: str,
) -> Tuple[str, Dict[str, Any]]:
    prompt = local_policy.build_final_completion_prompt(full_source, committed_chinese)
    request_kwargs, request_debug = _build_responses_kwargs(
        model=model,
        instructions=(
            "You are a professional translator. Return ONLY the remaining Chinese "
            "continuation after the committed prefix. No explanation."
        ),
        temperature=0.0,
        input_text=prompt,
        max_output_tokens=2048,
    )
    resp, api_base = thinking_pool.responses_create(**request_kwargs)
    reasoning_text, content_text, raw_message_fields = _extract_response_debug(resp)
    incomplete_debug = _max_output_tokens_debug(resp, raw_message_fields, stage="final_complete")
    continuation = ""
    if content_text and content_text.upper() != "EMPTY":
        continuation = local_policy.normalize_zh(content_text)

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
        "finish_reason": getattr(resp, "status", None),
        "full_translation": full_translation,
        **request_debug,
        **incomplete_debug,
    }


def main() -> None:
    args = parse_args()

    api_key = os.environ.get(args.openai_api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"ERROR: env var {args.openai_api_key_env} is not set. "
            "Set your OpenAI API key before running this script."
        )

    global _OPENAI_API_KEY
    global _OPENAI_TIMEOUT
    global _OPENAI_REASONING_EFFORT
    global _OPENAI_REASONING_SUMMARY
    global _OPENAI_TEXT_VERBOSITY
    global _OPENAI_TARGET_MODEL

    _OPENAI_API_KEY = api_key
    _OPENAI_TIMEOUT = float(args.openai_timeout)
    _OPENAI_REASONING_EFFORT = args.thinking_reasoning_effort
    _OPENAI_REASONING_SUMMARY = args.thinking_reasoning_summary
    _OPENAI_TEXT_VERBOSITY = args.thinking_verbosity
    _OPENAI_TARGET_MODEL = args.thinking_model_name

    local_policy.parse_args = parse_args
    local_policy.resolve_thinking_api_bases = resolve_thinking_api_bases
    local_policy.ThinkingServerPool = OpenAIResponsesPool
    local_policy.call_thinking_model = call_thinking_model
    local_policy.force_complete_translation = force_complete_translation
    local_policy.main()


if __name__ == "__main__":
    main()
