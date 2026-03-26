#!/usr/bin/env python3
"""vLLM-based simultaneous translation via OpenAI-compatible API.

Drop-in replacement for GeminiTranslator. Uses a locally served vLLM model
for the thinking/translation step instead of the Gemini API.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from transformers import AutoTokenizer

"""
CUDA_VISIBLE_DEVICES=1 vllm serve \
    /data/user_data/siqiouya/ckpts/pretrained/llm/Qwen3-30B-A3B-Instruct-2507-FP8/ \
    --max-model-len 4096
"""

@dataclasses.dataclass(frozen=True)
class VllmTranslatorConfig:
    """Immutable configuration for the vLLM translator server."""
    base_url: str = "http://localhost:8001/v1"
    model: str = "Qwen3-4B"
    reasoning_effort: str = ""
    timeout: float = 600.0


class VllmTranslator:
    """Simultaneous EN->ZH translator backed by a local vLLM server."""

    _SYSTEM_PROMPT = (
        "You are a professional English-to-Chinese simultaneous interpreter.\n"
        "Translate the given English source into Chinese.\n"
        "Output only the Chinese translation, nothing else."
    )

    def __init__(self, config: VllmTranslatorConfig, tokenizer_path: Optional[str] = None):
        self._config = config
        self._client = OpenAI(
            api_key="EMPTY",
            base_url=config.base_url,
            timeout=config.timeout,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or config.model, trust_remote_code=True
        )

    # -- Prompt construction ----------------------------------------------

    def _build_prompt(self, source: str, committed_chinese: str) -> str:
        """Apply the chat template and strip the trailing EOT/EOS.

        This produces a raw text prompt that the completions API can
        continue from, while preserving the chat format the model was
        trained on.
        """
        user_content = f"Translate the following English into Chinese:\n{source}"
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        if committed_chinese:
            messages.append({"role": "assistant", "content": committed_chinese})

        # add_generation_prompt=True appends the assistant header when
        # there is no assistant message yet; when there IS one the
        # template already ends after the assistant content + EOT.
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not committed_chinese,
        )
        if committed_chinese:
            # TODO

        return prompt

    # -- Core: next-token distributions -----------------------------------

    def get_next_token_distributions(
        self,
        observed_source: str,
        futures: List[str],
        committed_chinese: str,
        top_logprobs: int = 20,
    ) -> List[Tuple[str, Dict[str, float]]]:
        """For each future, decode 1 token and return its probability distribution.

        Uses the completions API so the model continues directly from
        committed_chinese rather than starting a new assistant turn.

        Returns a list (one per future) of (greedy_token, {token: prob}) tuples.
        """
        import math

        results: List[Tuple[str, Dict[str, float]]] = []
        for future in futures:
            full_source = f"{observed_source} {future}".strip()
            prompt = self._build_prompt(full_source, committed_chinese)

            resp = self._client.completions.create(
                model=self._config.model,
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
                logprobs=top_logprobs,
            )

            choice = resp.choices[0]
            greedy_token = (choice.text or "").strip()

            prob_dist: Dict[str, float] = {}
            if choice.logprobs and choice.logprobs.top_logprobs:
                for token, logprob in choice.logprobs.top_logprobs[0].items():
                    prob_dist[token] = math.exp(logprob)

            results.append((greedy_token, prob_dist))

        return results

    # -- Interface methods (stubs) ----------------------------------------

    def get_safe_delta(
        self,
        observed_source: str,
        futures: List[str],
        committed_chinese: str,
    ) -> str:
        """Return the next safe Chinese segment to emit, or empty string."""
        raise NotImplementedError("TODO: implement get_safe_delta decision logic")

    def complete_translation(
        self,
        full_source: str,
        committed_chinese: str,
    ) -> str:
        """Force-complete the remaining translation. Returns FULL translation."""
        raise NotImplementedError("TODO: implement vLLM complete_translation")
