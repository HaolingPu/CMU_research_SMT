"""Future sampling: ask base model(s) for N continuations of the observed
source prefix, then merge + dedup across models.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from .cleaning import is_valid_future_text
from .http_client import _http_json, normalize_api_base
from .prompts import (
    _is_chat_endpoint_model,
    build_future_sampling_chat_messages,
    build_future_sampling_chat_messages_single,
    parse_method_a_output,
)


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
                if raw and is_valid_future_text(raw):
                    futures.append(raw)
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
            cleaned = raw
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
        "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "\n"],
    }
    data = _http_json(f"{base}/completions", payload=payload, timeout=api_timeout) #call base model，用observed source作为prompt，采样n条可能的未来源语言续写
    futures: List[str] = []
    for choice in data.get("choices", []):
        raw = str(choice.get("text", "")) if isinstance(choice, dict) else ""
        cleaned = raw
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
