"""Chat-template / completion prompt builders for futures, probes, and tail."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


_NUMBERED_LIST_RE = re.compile(r"^\s*(\d+)[\.\)]\s*(.+?)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Method A: instruct-model future sampling via /chat/completions
# Generates N diverse continuations in a single call as a numbered list.
# ---------------------------------------------------------------------------

def build_future_sampling_chat_messages(observed_source: str, num_futures: int) -> List[Dict[str, str]]:
    """Method A: ask the model to produce a numbered list of `num_futures` continuations in one call."""
    return [
        {"role": "system", "content": (
            "You are an expert linguist predicting how an incomplete spoken English sentence "
            "might continue. Stay on the same domain and style as the input. Generate diverse "
            "continuations that vary in topic emphasis and sentence structure but remain plausible."
        )},
        {"role": "user", "content": (
            f"Generate exactly {num_futures} different continuations of this English text. "
            f"Each should be 15-30 words. Output English only (no Chinese, no analysis).\n\n"
            f"IMPORTANT: each numbered item must contain ONLY the new words that come AFTER the "
            f"input — do NOT repeat any words from the input text, do NOT prepend '...' or any "
            f"placeholder. Start each item with the very next word.\n\n"
            f"Text: {observed_source}\n\n"
            f"Format strictly as:\n1. <new words after input>\n2. <new words after input>\n"
            f"...\n{num_futures}. <new words after input>"
        )},
    ]


def build_future_sampling_chat_messages_single(observed_source: str) -> List[Dict[str, str]]:
    """Method B: ask the model for ONE continuation; caller invokes with n=num_futures for independent samples."""
    return [
        {"role": "system", "content": (
            "You are an expert linguist predicting how an incomplete spoken English sentence "
            "might continue. Stay on the same domain and style as the input."
        )},
        {"role": "user", "content": (
            f"Continue this English text with one plausible continuation of 15-30 words. "
            f"Output English only (no Chinese, no analysis).\n\n"
            f"IMPORTANT: output ONLY the new words that come AFTER the input — do NOT repeat any "
            f"words from the input text, do NOT prepend '...' or any placeholder. Start with the "
            f"very next word.\n\n"
            f"Text: {observed_source}\n\n"
            f"Continuation:"
        )},
    ]


def parse_method_a_output(raw_text: str, num_expected: int) -> List[str]:
    """Parse a numbered list response (1. ... 2. ... N. ...) into a list of continuations."""
    if not raw_text:
        return []
    items: List[Tuple[int, str]] = []
    for match in _NUMBERED_LIST_RE.finditer(raw_text):
        idx = int(match.group(1))
        text = match.group(2).strip()
        # Some models output "1. ... so it was in <continuation>" — strip leading "..." or "so it was in"
        text = re.sub(r"^[\.\s\-…]+", "", text)
        if 1 <= idx <= num_expected and text:
            items.append((idx, text))
    items.sort(key=lambda x: x[0])
    seen_idx: set = set()
    out: List[str] = []
    for idx, text in items:
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        out.append(text)
    return out


def _is_chat_endpoint_model(api_model: str) -> bool:
    """Decide whether to route to /chat/completions (instruct-style) or /completions (raw LM).

    Heuristic: model name contains an instruct suffix ('-it', 'instruct', 'chat').
    Examples:
      gemma4-e2b-it, qwen3-instruct, llama-3-chat → /chat/completions
      gemma4-e2b, qwen3-4b-base                    → /completions (raw text continuation)
    """
    name = str(api_model or "").lower()
    return any(tag in name for tag in ("-it", "instruct", "chat"))


# ---------------------------------------------------------------------------
# Translation probe + final completion prompts
# ---------------------------------------------------------------------------

def build_translation_probe_prompt_prefix_token_ids(
    tokenizer: Any,
    full_source: str,
    has_target_prefix: bool,
    target_lang: str = "Chinese",
) -> List[int]:
    messages = [{"role": "user", "content": (
        f"[TASK]\nTranslate the [INPUT] text into {target_lang}. Do not add explanations, summaries, examples, lists, numbering, markdown, or background knowledge.\n\n"
        f"[INPUT]\n{full_source}"
    )}]
    # add_generation_prompt=True lets the tokenizer's own chat template emit the
    # right assistant-turn prefix for whatever model family it is (Qwen → "<|im_start|>assistant\n",
    # Gemma → "<|turn>model\n", etc.). Caller appends the committed target prefix tokens after this.
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids.get("input_ids", [])
    elif hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    return list(prompt_ids)


def build_final_completion_prompt(tokenizer: Any, full_source: str, committed_text: str, target_lang: str = "Chinese") -> str:
    if not str(committed_text or "").strip():
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nOutput the complete {target_lang} translation only.\n"
            "Do not add explanations, summaries, examples, lists, numbering, markdown, or background knowledge."
        )}]
    else:
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nA partial {target_lang} translation is already committed "
            "at the start of the assistant reply.\n"
            f"Continue ONLY with the source content that is not yet covered by the committed {target_lang} prefix.\n"
            "Do not repeat the committed prefix.\n"
            "Do not add explanations, summaries, examples, lists, numbering, markdown, or background knowledge.\n"
            "Do not translate beyond the source.\n"
            "If there is no remaining source content to translate, output nothing.\n"
            f"Output only the remaining {target_lang} continuation."
        )}]
    # add_generation_prompt=True so the tokenizer's chat template emits the
    # right assistant-turn prefix string for any model family.
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if str(committed_text or "").strip():
        prompt += committed_text
    return prompt
