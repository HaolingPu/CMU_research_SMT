"""Prompt for coordinated plausible and contrastive future continuations."""

from __future__ import annotations


PROMPT_VERSION = "future_set_v2_two_groups"


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
    if num_candidates <= 0 or num_candidates % 2:
        raise ValueError("num_candidates must be a positive even number")

    candidates_per_mode = num_candidates // 2

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

The set has two groups:
- Plausible: likely, ordinary continuations that follow the strongest local interpretation.
- Contrastive: less obvious but still realistic continuations that resolve a lexical, syntactic, referential, or discourse uncertainty differently enough that a careful translator might change wording or wait before committing. Contrastive does not mean bizarre, adversarial, or unrelated.

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

Generate exactly {num_candidates} continuations in one response: first {candidates_per_mode} plausible candidates, then {candidates_per_mode} contrastive candidates. Plan and compare all {num_candidates} candidates together before answering, and remove repetition both within and across the two groups.

Use exactly this numbered format:
Plausible
1. <continuation only>
...
{candidates_per_mode}. <continuation only>
Contrastive
{candidates_per_mode + 1}. <continuation only>
...
{num_candidates}. <continuation only>"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
