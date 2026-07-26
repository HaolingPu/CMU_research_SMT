#!/usr/bin/env python3
"""
Local self-hosted DeepSeek-R1-Distill-Qwen-32B thinking-policy pipeline.

This wrapper keeps the local future-sampling + simalign logic from
llm_future_sampling_thinking_policy.py, but points the thinking backend to a
locally served OpenAI-compatible endpoint hosting DeepSeek-R1-Distill-Qwen-32B.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import llm_future_sampling_thinking_policy as local_policy


_TARGET_MODEL = "deepseek-r1-distill-qwen-32b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Thinking-policy simultaneous interpretation with local base future "
            "sampling + simalign, and a self-hosted DeepSeek-R1-Distill-Qwen-32B "
            "thinking model served through an OpenAI-compatible endpoint."
        )
    )
    p.add_argument("--input-tsv", required=True, help="Manifest TSV with src_text_full, src_trajectory.")
    p.add_argument("--output-root", required=True)

    p.add_argument("--base-model-path", default="/data/user_data/haolingp/models/Qwen3-4B-Base")
    p.add_argument("--thinking-api-base", default="http://127.0.0.1:8200/v1")
    p.add_argument(
        "--thinking-api-bases",
        default="",
        help=(
            "Comma-separated list of OpenAI-compatible API bases. If set, "
            "requests are load-balanced across these servers."
        ),
    )
    p.add_argument("--thinking-model-name", default="deepseek-r1-distill-qwen-32b")
    p.add_argument("--thinking-api-key", default="EMPTY")
    p.add_argument("--thinking-timeout", type=float, default=600.0)
    p.add_argument("--thinking-tokenizer-path", default="/data/user_data/haolingp/models/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)

    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--num-futures", type=int, default=5, help="N future continuations per step.")
    p.add_argument("--future-tokens", type=int, default=10)
    p.add_argument("--sample-temperature", type=float, default=1.0)
    p.add_argument("--thinking-temperature", type=float, default=0.3)
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


class LocalThinkingServerPool:
    def __init__(self, api_bases: List[str], api_key: str):
        bases = [b.strip() for b in api_bases if (b or "").strip()]
        if not bases:
            raise ValueError("LocalThinkingServerPool requires at least one API base.")
        self._slots = [
            {
                "api_base": api_base,
                "client": OpenAI(base_url=api_base, api_key=api_key),
                "inflight": 0,
                "requests": 0,
            }
            for api_base in bases
        ]
        self._lock = local_policy.threading.Lock()
        self._rr = 0

    def __len__(self) -> int:
        return len(self._slots)

    def _acquire_slot(self, exclude: Optional[set] = None) -> Tuple[int, Dict[str, Any]]:
        exclude = exclude or set()
        with self._lock:
            candidates = [(idx, slot) for idx, slot in enumerate(self._slots) if idx not in exclude]
            if not candidates:
                raise RuntimeError("No available local thinking server slot.")
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
            visible = model_ids[:20] if model_ids else [_TARGET_MODEL]
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
                return resp, slot["api_base"]
            except Exception as e:
                errors.append(f"{slot['api_base']}: {type(e).__name__}: {e}")
            finally:
                self._release_slot(idx)
        raise RuntimeError("All local thinking servers failed: " + " | ".join(errors))

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
        for slot in self._slots:
            close_fn = getattr(slot["client"], "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass


def resolve_thinking_api_bases(args: argparse.Namespace) -> List[str]:
    raw = args.thinking_api_bases.strip()
    if raw:
        bases = [item.strip() for item in raw.split(",") if item.strip()]
        if bases:
            return bases
    return [args.thinking_api_base]


def call_thinking_model(
    thinking_pool: LocalThinkingServerPool,
    model: str,
    user_content: str,
    committed_chinese: str = "",
    temperature: float = 0.3,
    max_tokens: int = 256,
) -> Tuple[str, Dict[str, Any]]:
    del committed_chinese
    messages = [{"role": "user", "content": user_content}]
    resp, api_base = thinking_pool.chat_completions_create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    choice = resp.choices[0]
    message = choice.message
    raw_message_fields = local_policy._raw_message_debug_fields(message)
    reasoning_text, content_text = local_policy._split_reasoning_and_content(
        raw_message_fields["message.reasoning_content"] or raw_message_fields["message.reasoning"],
        raw_message_fields["message.content"],
    )
    delta = "" if (not content_text or content_text.upper() == "EMPTY") else local_policy.normalize_zh(content_text)
    return delta, {
        "server_api_base": api_base,
        "raw_message_fields": raw_message_fields,
        "reasoning_text": reasoning_text,
        "content_text": content_text,
        "cleaned_content": delta,
        "finish_reason": getattr(choice, "finish_reason", None),
    }


def force_complete_translation(
    thinking_pool: LocalThinkingServerPool,
    model: str,
    full_source: str,
    committed_chinese: str,
) -> Tuple[str, Dict[str, Any]]:
    prompt = local_policy.build_final_completion_prompt(full_source, committed_chinese)
    messages = [{"role": "user", "content": prompt}]
    resp, api_base = thinking_pool.chat_completions_create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = resp.choices[0]
    message = choice.message
    raw_message_fields = local_policy._raw_message_debug_fields(message)
    _, content_text = local_policy._split_reasoning_and_content(
        raw_message_fields["message.reasoning_content"] or raw_message_fields["message.reasoning"],
        raw_message_fields["message.content"],
    )
    continuation = "" if (not content_text or content_text.upper() == "EMPTY") else local_policy.normalize_zh(content_text)

    committed_norm = local_policy.normalize_zh(committed_chinese)
    new_part = local_policy.strip_committed_suffix_from_delta(committed_chinese, continuation)
    new_part = local_policy.normalize_zh(new_part)
    full_translation = committed_norm + new_part if committed_chinese else continuation
    return full_translation, {
        "server_api_base": api_base,
        "raw_message_fields": raw_message_fields,
        "reasoning_text": "",
        "content_text": content_text,
        "cleaned_content": continuation,
        "finish_reason": getattr(choice, "finish_reason", None),
        "full_translation": full_translation,
    }


def main() -> None:
    args = parse_args()
    local_policy.setup_env()

    local_policy.parse_args = lambda: args
    local_policy.resolve_thinking_api_bases = resolve_thinking_api_bases
    local_policy.call_thinking_model = call_thinking_model
    local_policy.force_complete_translation = force_complete_translation

    api_key = args.thinking_api_key
    bases = resolve_thinking_api_bases(args)

    class _PoolFactory(LocalThinkingServerPool):
        def __init__(self, api_bases: List[str]):
            super().__init__(api_bases, api_key=api_key)

    local_policy.ThinkingServerPool = _PoolFactory

    local_policy.main()


if __name__ == "__main__":
    main()
