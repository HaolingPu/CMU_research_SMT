#!/usr/bin/env python3
"""Gemini-based simultaneous translation via OpenAI-compatible API.

Provides a GeminiTranslator that, given observed English + future
continuations + committed Chinese, returns the next safe Chinese delta.
Also handles force-completing remaining translation at utterance end.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


@dataclasses.dataclass(frozen=True)
class GeminiConfig:
    """Immutable configuration for the Gemini API."""
    api_key: str
    model: str = "gemini-3-flash-preview"
    reasoning_effort: str = "low"
    timeout: float = 600.0


class GeminiTranslator:
    """Simultaneous EN->ZH translator backed by Gemini's chat completions API."""

    _GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self, config: GeminiConfig):
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=self._GEMINI_API_BASE,
            timeout=config.timeout,
        )

    def get_safe_delta(
        self,
        observed_source: str,
        futures: List[str],
        committed_chinese: str,
    ) -> str:
        """Return the next safe Chinese segment to emit, or empty string."""
        user_content = _build_thinking_prompt(observed_source, futures, committed_chinese)
        system_prompt = (
            "You are a professional English-to-Chinese simultaneous interpreter.\n"
            "Follow the user's task instructions carefully.\n"
            "Return only the requested Chinese output, or EMPTY when instructed.\n"
            "Do not include explanations, reasoning, bullets, or quotation marks."
        )
        raw = self._call(system_prompt, user_content)
        if not raw or raw.upper() == "EMPTY":
            return ""
        delta = _normalize_zh(raw)
        delta = _strip_committed_suffix(committed_chinese, delta)
        return delta

    def complete_translation(
        self,
        full_source: str,
        committed_chinese: str,
    ) -> str:
        """Force-complete the remaining translation. Returns FULL translation."""
        committed_block = committed_chinese if committed_chinese else "(none yet)"
        user_content = (
            "Complete the translation from the committed Chinese prefix.\n\n"
            "Rules:\n"
            "- You are given the FULL English source and the Chinese text already committed.\n"
            "- Return ONLY the REMAINING Chinese continuation after the committed prefix.\n"
            "- Do not repeat or rewrite already committed Chinese.\n"
            "- If nothing remains to be translated, output EMPTY.\n"
            "- Do not output explanation, notes, bullets, or quotes.\n\n"
            f"Full English source:\n{full_source}\n\n"
            f"Committed Chinese prefix:\n{committed_block}\n\n"
            "Return ONLY the remaining Chinese continuation."
        )
        system_prompt = (
            "You are a professional English-to-Chinese translator.\n"
            "Follow the user's task instructions carefully.\n"
            "Return only the requested Chinese continuation.\n"
            "Do not include explanations, reasoning, bullets, or quotation marks."
        )
        raw = self._call(system_prompt, user_content)
        continuation = ""
        if raw and raw.upper() != "EMPTY":
            continuation = _normalize_zh(raw)

        committed_norm = _normalize_zh(committed_chinese)
        new_part = _strip_committed_suffix(committed_chinese, continuation)
        new_part = _normalize_zh(new_part)
        return committed_norm + new_part if committed_chinese else continuation

    def _call(self, system_prompt: str, user_content: str) -> str:
        """Send a chat completion request to Gemini and extract the answer text."""
        kwargs: Dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 1.0,
        }
        effort = self._config.reasoning_effort
        if effort:
            kwargs["reasoning_effort"] = effort

        resp = self._client.chat.completions.create(**kwargs)
        message = resp.choices[0].message
        return _extract_chinese(message)


# ---------------------------------------------------------------------------
# Text helpers (private to this module)
# ---------------------------------------------------------------------------

def _normalize_zh(text: str) -> str:
    """NFC-normalize and strip all whitespace from Chinese text."""
    text = unicodedata.normalize("NFC", (text or "").strip())
    text = re.sub(r"\s+", "", text)
    return text


def _clean_llm_output(text: str) -> str:
    """Strip thinking tags and stray quotes from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("</think>", "")
    text = (text or "").strip().strip('"').strip("'")
    text = text.strip("\u201c\u201d\u2018\u2019")
    return text


def _strip_committed_suffix(committed: str, delta: str) -> str:
    """If delta starts with a suffix of committed (overlap), strip it."""
    delta = (delta or "").strip()
    if not delta:
        return ""
    if not committed:
        return _normalize_zh(delta)
    committed_norm = _normalize_zh(committed)
    delta_norm = _normalize_zh(delta)
    max_k = min(len(committed_norm), len(delta_norm))
    for k in range(max_k, 0, -1):
        suffix = committed_norm[-k:]
        if delta_norm.startswith(suffix):
            return _normalize_zh(delta_norm[k:].strip())
    return delta_norm


def _extract_chinese(message: Any) -> str:
    """Extract usable Chinese text from a Gemini chat completion message."""
    content = getattr(message, "content", None)
    if content is None:
        return ""

    # Handle structured content (list of parts)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                item_type = str(item.get("type", "")).strip().lower()
                if "thought" in item_type or "reason" in item_type:
                    continue  # skip thinking parts
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        text = "".join(parts).strip()
    else:
        text = str(content).strip()

    # Strip inline <thought> tags that Gemini sometimes returns
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()

    # Clean model artifacts
    text = _clean_llm_output(text)
    return text


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_thinking_prompt(
    observed_source: str,
    futures: List[str],
    committed_chinese: str,
) -> str:
    """Build the user prompt for the thinking model (base version)."""
    lines = [f"  {i+1}. {f}" for i, f in enumerate(futures) if (f or "").strip()]
    futures_block = "\n".join(lines) if lines else "  1. (none)"
    committed_block = committed_chinese if committed_chinese else "(none yet)"

    return (
        "[Task]\n"
        "Given a partial English sentence, several possible future continuations, "
        "and the Chinese translation emitted so far, output the next Chinese "
        "increment that is safe to emit.\n\n"
        "[Rule]\n"
        "The increment is safe only if appending it to the committed Chinese "
        "remains correct under all possible continuations.\n"
        "Do not repeat or modify previously emitted Chinese.\n"
        "If no safe increment exists, output exactly EMPTY.\n\n"
        "[Input]\n"
        f"Partial English sentence: {observed_source}\n"
        f"Possible future continuations:\n{futures_block}\n"
        f"Committed Chinese so far: {committed_block}\n\n"
        "[Output]\n"
    )
