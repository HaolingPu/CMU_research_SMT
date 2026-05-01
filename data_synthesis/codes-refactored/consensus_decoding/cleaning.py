"""Output text cleaning for model continuations and futures."""
from __future__ import annotations

import re


def clean_model_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.split("<|im_end|>")[0]
    text = text.split("<|endoftext|>")[0]
    return text.strip()


def clean_future_text(observed_source: str, raw_text: str) -> str:
    # 只保留第一行 (如果模型输出了多行，我们只取第一行作为续写)，并且如果第一行以observed source开头，就去掉这个重复的prefix（有些模型喜欢在续写里重复输入的source prefix）
    text = clean_model_text(raw_text)
    if text.startswith(observed_source):
        text = text[len(observed_source):].lstrip()
    text = text.splitlines()[0].strip() if text else ""
    # Strip leading placeholders like "..." or "…" that instruct models sometimes prepend.
    text = re.sub(r"^[\.\s…\-]+", "", text)
    # If the cleaned continuation still starts by repeating the trailing words of the
    # observed source (instruct-model habit), strip the longest matching suffix-prefix.
    obs_trailing = observed_source.strip()
    if obs_trailing and text:
        # Try the longest possible suffix of observed_source that matches the start of text.
        for k in range(min(len(obs_trailing), len(text)), 2, -1):
            tail = obs_trailing[-k:]
            if text.startswith(tail):
                text = text[k:].lstrip()
                break
            # Also try after a leading space (text starts with a space in some models)
            tail_space = " " + tail
            if text.startswith(tail_space):
                text = text[len(tail_space):].lstrip()
                break
    return text


def is_valid_future_text(text: str) -> bool:
    if not text:
        return False
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text):
        return False
    lowered = text.lower()
    banned = ["translate", "translation", "grammar analysis", "analyze", "analysis",
              "这句话", "翻译", "语法", "句子结构"]
    return not any(b in lowered for b in banned)
