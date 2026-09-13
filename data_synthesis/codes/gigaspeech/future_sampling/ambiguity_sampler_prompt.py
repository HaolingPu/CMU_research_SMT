"""The v3 suffix-ICL prompt that asks one sampler for plausible and contrastive futures, and its reply format."""

import json


PROMPT_VERSION = "future_set_v3_suffix_icl"
# Character budget of one suffix; the decoder cuts the oracle future to the same budget.
MAX_SUFFIX_CHARS = 200

AMBIGUITY_ICL_EXAMPLES = (
    {
        "kind": "Word sense",
        "prefix": "The visitors stopped by the bank",
        "plausible": "to withdraw some cash before continuing their walk.",
        "contrastive": "of the river to watch the ducks swimming.",
        "explanation": "The first suffix makes bank a financial institution; the second makes it the edge of a river. The already observed word bank would need different translations.",
    },
    {
        "kind": "Grammatical role",
        "prefix": "The nurse watched her",
        "plausible": "daughter cross the room without any assistance.",
        "contrastive": "cross the room without any assistance.",
        "explanation": "In the first reading, her is possessive and modifies daughter. In the second, her is the person being watched. The continuation changes the grammatical role of an already observed word.",
    },
    {
        "kind": "Phrase attachment",
        "prefix": "I saw the man with",
        "plausible": "the binoculars, which helped me see that far.",
        "contrastive": "a broken arm waiting outside the clinic.",
        "explanation": "The first suffix supports with introducing the instrument used to see. The second attaches with to the man and describes his condition. This changes how the already observed relation should be translated.",
    },
    {
        "kind": "Object versus embedded subject across a relative clause",
        "prefix": "The editor knew the author whom the reviewers, despite their reservations, praised",
        "plausible": "from a conference they had attended together.",
        "contrastive": "would refuse to revise the final chapter.",
        "explanation": "The first reading makes the author the object of knew, meaning personal acquaintance. The second makes the author the subject of would refuse inside the content clause that knew introduces. In both readings, whom is the object of praised in a relative clause modifying author; the reviewers are its subject. The relative clause and the parenthetical despite their reservations delay the decision, but do not resolve it. Wait for the suffix before choosing the meaning of knew or the outer role of the author.",
    },
    {
        "kind": "Main verb versus reduced passive relative clause",
        "prefix": "The soldiers warned about the ambush",
        "plausible": "and advised the convoy to take another route.",
        "contrastive": "were ordered to stay inside the camp overnight.",
        "explanation": "In the first reading, warned is a main-clause predicate and the soldiers give the warning. In the second, warned about the ambush is a reduced passive relative clause, equivalent to who were warned about the ambush; the soldiers receive the warning, and were ordered is the main-clause predicate. The suffix determines both the role of warned and who gives or receives the warning. Do not commit to the active reading before this is resolved.",
    },
)


def build_future_set_messages(observed_source: str, num_candidates: int) -> list[dict[str, str]]:
    """Chat messages asking for at most `num_candidates` suffixes, up to half plausible and half contrastive."""
    per_group = num_candidates // 2
    ambiguity_examples = "\n\n".join(
        f"Ambiguity example {i} ({example['kind']}):\n"
        f"Observed prefix: {example['prefix']}\n"
        f"Example response: {json.dumps({'plausible': [example['plausible']], 'contrastive': [example['contrastive']]})}\n"
        f"Teaching note (not part of the response): {example['explanation']}"
        for i, example in enumerate(AMBIGUITY_ICL_EXAMPLES, 1)
    )
    system = f"""You predict short English speech continuations for a simultaneous English-to-Chinese interpreter.

The observed prefix is the exact speech heard so far, not a topic for a new sentence. Return only the new words that can come immediately after it.

Rules, in priority order:
1. Preserve every observed word and punctuation mark. Do not repeat, rewrite, correct, or replace any part of the prefix.
2. Continue the unfinished sentence before starting another sentence. Complete its open grammatical structure: for example, an article needs a noun phrase and an unfinished verb phrase needs a compatible continuation. A topical but independent sentence is not a valid suffix.
3. Silently check PREFIX + one space + SUFFIX. This exact concatenation must be grammatical and coherent, without editing the prefix or relying on an unseen comma or period. Do not insert punctuation merely to abandon an unfinished construction. A comma in the prefix is not a sentence boundary.
4. Stay grounded in the observed topic and register. Do not invent unrelated settings, implausible events, or exaggerated reactions just to make candidates different.
5. Prefer a few good candidates to a full list of weak ones. Return at most {num_candidates} candidates total: up to {per_group} Plausible and up to {per_group} Contrastive. Either group may be shorter or empty. Never pad a list to meet a quota.
6. Plausible candidates are likely natural continuations. Contrastive candidates are less obvious but still realistic continuations that resolve a genuine uncertainty differently. Prefer contrasts that change how already observed words or relations could be translated. Merely changing the later action, adjective, or intensity under the same reading is not enough. If there is no good contrastive continuation, leave that group empty; do not force an ambiguity.
7. Seek meaningful diversity only after grammatical fit and plausibility. It is okay to share necessary opening words. Omit redundant candidates instead of distorting the sentence to make them different.
8. Each suffix should contain 4-15 new English words. Output English suffixes only, as a JSON object with the two group lists. No reasoning, translations, full rewritten sentences, or copies of the examples.

Examples of checking suffixes (illustrative, not answers to the current input):

Observed prefix: At the picnic, the spoon fell into a
VALID suffix: bowl of soup beside the bread.
Joined check: At the picnic, the spoon fell into a bowl of soup beside the bread.
INVALID suffix: The spoon fell onto the table.
Why invalid: It rewrites the event and starts a separate sentence after an unfinished article.
INVALID suffix: a bowl of soup beside the bread.
Why invalid: It repeats the already observed article, producing "a a bowl".

Observed prefix: The second runner was faster than the first
VALID suffix: runner, despite having started several seconds later.
VALID suffix: one we watched in yesterday's race.
INVALID suffix: she waved to the crowd.
Why invalid: "than the first she waved" is not grammatical. Being about the runner is not enough.

Observed prefix: We opened the box, and the tools were
VALID suffix: covered in dust from years of storage.
VALID suffix: missing, although the packing list included them.
INVALID suffix: and we began sorting them immediately.
Why invalid: It leaves "the tools were" unfinished. Continue that clause, not a different one.

Examples of genuinely different readings of the SAME prefix:
These are short valid responses, not lists that must be padded. Both readings must be realistic; the group labels illustrate a common reading and an alternative, not universal probability rankings. Use only readings supported by the actual input. Do not copy these examples or their teaching notes into your answer.

{ambiguity_examples}

For the actual input, apply the same concatenation check to every candidate. Output fewer candidates whenever this check or plausibility fails."""
    user = f"""Observed English prefix:
{observed_source}


Return only valid suffixes for this exact prefix: at most {num_candidates} total, up to {per_group} per group. Fewer is acceptable, including zero. Do not output the joined check or repeat the examples.

Respond with one JSON object of the form {{"plausible": ["<suffix>", ...], "contrastive": ["<suffix>", ...]}}. Each suffix string is one suffix only. Use an empty list for a group with no valid candidate. No placeholders, explanations, or text outside the JSON."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def future_set_response_format(num_candidates: int) -> dict:
    """vLLM structured-output fields: a JSON object with two suffix lists of at most num_candidates // 2 each."""
    suffix = {"type": "string", "minLength": 1, "maxLength": MAX_SUFFIX_CHARS}
    group = {"type": "array", "maxItems": num_candidates // 2, "items": suffix}
    return {"structured_outputs": {"json": {
        "type": "object",
        "properties": {"plausible": group, "contrastive": group},
        "required": ["plausible", "contrastive"],
        "additionalProperties": False,
    }}}


def parse_future_set(raw_text: str, finish_reason: str | None, num_candidates: int) -> list[tuple[str, str]]:
    """(group, suffix) pairs, plausible first.

    Raises ValueError for a reply cut off by the token budget, one that is not the expected
    JSON object, or one over the per-group budget, so a malformed reply never passes as
    fewer good candidates.
    """
    if finish_reason == "length":
        raise ValueError("Truncated response: finish_reason=length")
    try:
        reply = json.loads(raw_text)
    except ValueError as exc:
        raise ValueError(f"Response is not valid JSON: {exc}") from exc
    if not isinstance(reply, dict) or set(reply) != {"plausible", "contrastive"}:
        raise ValueError("Response must be an object with exactly the keys ('plausible', 'contrastive')")
    pairs = []
    for group in ("plausible", "contrastive"):
        suffixes = reply[group]
        if not isinstance(suffixes, list) or len(suffixes) > num_candidates // 2:
            raise ValueError(f"Group {group!r} must be a list of at most {num_candidates // 2} items")
        elif not all(isinstance(suffix, str) and suffix.strip() for suffix in suffixes):
            raise ValueError(f"Group {group!r} contains a non-string or blank item")
        else:
            pairs.extend((group, suffix.strip()) for suffix in suffixes)
    return pairs
