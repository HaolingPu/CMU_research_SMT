"""Prompt for one coordinated set of natural future continuations."""

from __future__ import annotations


PROMPT_VERSION = "future_set_v1"


def build_coordinated_future_messages(
    *,
    observed_source: str,
    target_lang: str,
    committed_text: str,
    num_candidates: int,
) -> list[dict[str, str]]:
    """Ask an instruction-tuned sampler to plan a diverse set jointly."""
    if not observed_source.strip():
        raise ValueError("observed_source must not be empty")
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")

    committed = committed_text.strip()
    commitment = (
        f"\nThe interpreter has already committed this {target_lang} text:\n"
        f"{committed}\n"
        "Include natural futures that test whether this commitment remains safe."
        if committed
        else ""
    )
    system = f"""You predict possible future English speech for a simultaneous English-to-{target_lang} interpreter.

Generate one coordinated set of natural continuations. Every item must be grammatically valid immediately after the observed prefix, remain grounded in its topic and register, and represent a genuinely plausible way the speaker could continue.

Plan the complete set before answering:
- Make the items mutually distinct in wording and semantic outcome.
- Do not provide paraphrases that differ only in one final noun or adjective.
- Avoid reusing the same first content word across items.
- Do not force an ambiguity that the prefix does not support.
- Do not invent a technical, business, scientific, or data-analysis setting unless the prefix supports it.
- Each item must contain only 4-15 new English words after the prefix.
- Do not repeat the prefix and do not output explanations, labels, Chinese, JSON, or markdown.
"""
    user = f"""Observed English prefix:
{observed_source}
{commitment}

Generate exactly {num_candidates} continuations in one response. Before writing, compare the candidates internally and remove duplicates.

Use exactly this numbered format:
1. <continuation only>
2. <continuation only>
...
{num_candidates}. <continuation only>"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
