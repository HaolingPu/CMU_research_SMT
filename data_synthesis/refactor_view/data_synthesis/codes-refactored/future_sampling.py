#!/usr/bin/env python3
"""Future source continuation sampling via a vLLM-served model (OpenAI API).

Given a partial English sentence, sample N possible future continuations
using a causal language model served by ``vllm serve``. These futures are
used downstream to decide what Chinese translation is safe to emit now.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from openai import OpenAI


_FIRST_SENTENCE_END_RE = re.compile(r'[.!?](?:["\')\]]+)?(?=\s|$)')


def create_base_client(base_url: str, model_name: str) -> Tuple[OpenAI, str]:
    """Create an OpenAI client pointing at a vLLM server.

    Returns (client, model_name) tuple so callers don't need to track the
    model name separately.
    """
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    return client, model_name


def sample_futures(
    base_llm: Tuple[OpenAI, str],
    observed_source: str,
    num_futures: int = 5,
    future_tokens: int = 10,
    temperature: float = 1.0,
) -> List[str]:
    """Sample N future English continuations from the vLLM server.

    Returns a deduplicated list of cleaned continuation strings.
    """
    if not (observed_source or "").strip():
        return []

    client, model_name = base_llm

    response = client.completions.create(
        model=model_name,
        prompt=observed_source.strip(),
        max_tokens=future_tokens,
        n=num_futures,
        temperature=temperature,
        top_p=0.90,
        stop=["\n"],
    )

    futures: List[str] = []
    seen: set = set()
    for choice in response.choices:
        raw = (choice.text or "").strip()
        cleaned = _clean_continuation(observed_source, raw)
        if cleaned:
            cleaned = _truncate_to_first_sentence(cleaned)
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                futures.append(cleaned)
    return futures


def _clean_continuation(observed: str, raw_output: str, max_words: int = 15) -> str:
    """Extract continuation text after observed prefix, cap length."""
    text = (raw_output or "").strip()
    if "\n" in text:
        text = text.split("\n")[0].strip()
    obs_lower = (observed or "").lower().strip()
    text_lower = (text or "").lower()
    if obs_lower and text_lower.startswith(obs_lower):
        text = text[len(obs_lower):].strip()
    words = (text or "").split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return (text or "").strip()


def _truncate_to_first_sentence(text: str) -> str:
    """Keep at most the first English sentence."""
    text = (text or "").strip()
    if not text:
        return ""
    m = _FIRST_SENTENCE_END_RE.search(text)
    if not m:
        return text
    return text[: m.end()].strip()
