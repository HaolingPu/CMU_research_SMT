"""Reviewable prompt for translation-relevant future sampling."""

from __future__ import annotations

from typing import Dict


PROMPT_VERSION = "ambiguity_icl_v1"

SAMPLING_DIRECTIVES: Dict[str, str] = {
    "plausible": (
        "Generate a natural, high-probability continuation. Commit to one concrete "
        "interpretation rather than staying vague."
    ),
    "contrastive": (
        "Generate a less obvious but still genuinely plausible continuation that "
        "resolves an uncertainty differently enough to change how some part of the "
        "observed English should be translated. Do not create arbitrary topic drift."
    ),
}


def build_ambiguity_sampler_messages(
    *,
    target_lang: str,
    committed_text: str,
    sampling_mode: str,
) -> list[dict[str, str]]:
    """Build the sampler messages without repeating the observed source prefix.

    The caller appends the observed prefix to the assistant turn. Keeping it out of
    the user message prevents the model from copying or paraphrasing the prefix.
    """
    try:
        directive = SAMPLING_DIRECTIVES[sampling_mode]
    except KeyError as exc:
        raise ValueError(f"unknown sampling mode: {sampling_mode}") from exc

    system = f"""You generate possible future English continuations for a simultaneous English-to-{target_lang} interpreter.

Goal: expose uncertainty that matters to translation. A useful pair of futures is both plausible after the observed prefix but would make a careful translator choose different wording, grammar, reference, or commitment timing. Differences that only add unrelated facts or decorative details are not useful.

Rules:
- Continue the unfinished English directly with 4-15 words.
- Produce only continuation text: no analysis, labels, JSON, markdown, or {target_lang}.
- Do not repeat or paraphrase the observed prefix.
- Keep the continuation grammatical and locally coherent.
- Resolve an uncertainty concretely instead of merely remaining ambiguous.

In-context examples:

Partial: "The bank"
Possible futures:
- "approved the loan before the Friday deadline" (financial institution)
- "collapsed after three days of heavy rain" (edge of a river)

Partial: "She decided to run"
Possible futures:
- "the company after her father retired" (manage)
- "for mayor in the next local election" (seek office)
- "the experiment again with a larger sample" (conduct)
- "away before anyone noticed the missing key" (flee)

Partial: "The agreement that the two countries signed last year"
Possible futures:
- "expires at the end of this month"
- "has been violated repeatedly by both sides"
- "remains the foundation of their defense partnership"

Partial: "I saw the scientist with the telescope"
Possible futures:
- "that she had designed for the observatory" (the scientist had it)
- "and used it to identify the distant object" (the observer used it)

Partial: "The proposal was not"
Possible futures:
- "only feasible but considerably cheaper than expected" (not only)
- "acceptable to the committee in its current form" (negative evaluation)
"""

    committed = committed_text.strip()
    committed_block = (
        f"\nThe interpreter has already committed this {target_lang} prefix:\n"
        f"{committed}\n"
        "Prefer a future that tests whether this commitment remains safe."
        if committed
        else ""
    )
    user = f"""The assistant turn is prefilled with an unfinished English prefix. Continue exactly where it ends.

Sampling mode: {sampling_mode}
{directive}{committed_block}

Return only the English continuation."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
