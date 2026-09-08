"""Prompt for coordinated plausible and contrastive future continuations."""

from __future__ import annotations

import json
import re


# Keep the historical default for reproducible production runs.
PROMPT_VERSION = "future_set_v2_two_groups"
SUFFIX_ICL_PROMPT_VERSION = "future_set_v3_suffix_icl"
PROMPT_VERSIONS = (PROMPT_VERSION, SUFFIX_ICL_PROMPT_VERSION)

AMBIGUITY_ICL_EXAMPLES = (
    {
        "kind": "Word sense",
        "prefix": "The visitors stopped by the bank",
        "plausible": "to withdraw some cash before continuing their walk.",
        "contrastive": "of the river to watch the ducks swimming.",
        "resolves": "bank = river edge, not the financial institution",
        "explanation": "The first suffix makes bank a financial institution; the second makes it the edge of a river. The already observed word bank would need different translations.",
    },
    {
        "kind": "Grammatical role",
        "prefix": "The nurse watched her",
        "plausible": "daughter cross the room without any assistance.",
        "contrastive": "cross the room without any assistance.",
        "resolves": "her = object of watched, not a possessive",
        "explanation": "In the first reading, her is possessive and modifies daughter. In the second, her is the person being watched. The continuation changes the grammatical role of an already observed word.",
    },
    {
        "kind": "Phrase attachment",
        "prefix": "I saw the man with",
        "plausible": "the binoculars, which helped me see that far.",
        "contrastive": "a broken arm waiting outside the clinic.",
        "resolves": "with = attribute of the man, not the instrument of seeing",
        "explanation": "The first suffix supports with introducing the instrument used to see. The second attaches with to the man and describes his condition. This changes how the already observed relation should be translated.",
    },
    {
        "kind": "Object versus embedded subject across a relative clause",
        "prefix": "The editor knew the author whom the reviewers, despite their reservations, praised",
        "plausible": "from a conference they had attended together.",
        "contrastive": "would refuse to revise the final chapter.",
        "resolves": "knew introduces a content clause; the author is its subject",
        "explanation": "The first reading makes the author the object of knew, meaning personal acquaintance. The second makes the author the subject of would refuse inside the content clause that knew introduces. In both readings, whom is the object of praised in a relative clause modifying author; the reviewers are its subject. The relative clause and the parenthetical despite their reservations delay the decision, but do not resolve it. Wait for the suffix before choosing the meaning of knew or the outer role of the author.",
    },
    {
        "kind": "Main verb versus reduced passive relative clause",
        "prefix": "The soldiers warned about the ambush",
        "plausible": "and advised the convoy to take another route.",
        "contrastive": "were ordered to stay inside the camp overnight.",
        "resolves": "warned = reduced passive relative clause; the soldiers receive the warning",
        "explanation": "In the first reading, warned is a main-clause predicate and the soldiers give the warning. In the second, warned about the ambush is a reduced passive relative clause, equivalent to who were warned about the ambush; the soldiers receive the warning, and were ordered is the main-clause predicate. The suffix determines both the role of warned and who gives or receives the warning. Do not commit to the active reading before this is resolved.",
    },
)


def build_coordinated_future_messages(
    *,
    observed_source: str,
    target_lang: str,
    committed_text: str,
    num_candidates: int,
    prompt_version: str = PROMPT_VERSION,
    contrastive_notes: bool = False,
) -> list[dict[str, str]]:
    """Ask an instruction-tuned sampler to plan a diverse set jointly."""
    if not observed_source.strip():
        raise ValueError("observed_source must not be empty")
    if num_candidates <= 0 or num_candidates % 2:
        raise ValueError("num_candidates must be a positive even number")
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(f"Unsupported future prompt version: {prompt_version}")

    candidates_per_mode = num_candidates // 2

    committed = committed_text.strip()
    commitment = (
        f"\nThe interpreter has already committed this {target_lang} text:\n"
        f"{committed}\n"
        "Include natural futures that test whether this commitment remains safe."
        if committed
        else ""
    )
    if prompt_version == SUFFIX_ICL_PROMPT_VERSION:
        return _build_suffix_icl_messages(
            observed_source, target_lang, commitment, num_candidates, contrastive_notes,
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


def _example_response(example: dict, contrastive_notes: bool) -> str:
    contrastive = ({"suffix": example["contrastive"], "resolves": example["resolves"]}
                   if contrastive_notes else example["contrastive"])
    return json.dumps({"plausible": [example["plausible"]], "contrastive": [contrastive]})


def _build_suffix_icl_messages(
    observed_source: str, target_lang: str, commitment: str, num_candidates: int,
    contrastive_notes: bool = False,
) -> list[dict[str, str]]:
    per_group = num_candidates // 2
    ambiguity_examples = "\n\n".join(
        f"Ambiguity example {i} ({example['kind']}):\n"
        f"Observed prefix: {example['prefix']}\n"
        f"Example response: {_example_response(example, contrastive_notes)}\n"
        f"Teaching note (not part of the response): {example['explanation']}"
        for i, example in enumerate(AMBIGUITY_ICL_EXAMPLES, 1)
    )
    contrastive_shape = ('{"suffix": "<suffix>", "resolves": "<reading of an already observed word or relation that this suffix settles, a few words>"}'
                         if contrastive_notes else '"<suffix>"')
    notes_rule = ("\nEach contrastive item must name a different reading in its \"resolves\" field; do not list two suffixes that settle the same reading."
                  if contrastive_notes else "")
    system = f"""You predict short English speech continuations for a simultaneous English-to-{target_lang} interpreter.

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
{commitment}

Return only valid suffixes for this exact prefix: at most {num_candidates} total, up to {per_group} per group. Fewer is acceptable, including zero. Do not output the joined check or repeat the examples.

Respond with one JSON object of the form {{"plausible": ["<suffix>", ...], "contrastive": [{contrastive_shape}, ...]}}. Each suffix string is one suffix only. Use an empty list for a group with no valid candidate. No placeholders, explanations, or text outside the JSON.{notes_rule}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


GROUP_KEYS = ("plausible", "contrastive")
MAX_SUFFIX_CHARS = 200
MAX_NOTE_CHARS = 120


def grouped_future_schema(num_candidates: int, contrastive_notes: bool = False) -> dict:
    """JSON schema the sampler server enforces: two capped lists of suffixes.

    With ``contrastive_notes`` each contrastive item is an object that must also
    name the reading it resolves, so the model commits to a distinct reading per item.
    """
    if num_candidates <= 0 or num_candidates % 2:
        raise ValueError("num_candidates must be a positive even number")
    suffix = {"type": "string", "minLength": 1, "maxLength": MAX_SUFFIX_CHARS}
    contrastive_item = ({"type": "object", "properties": {"suffix": suffix, "resolves": {"type": "string", "minLength": 1, "maxLength": MAX_NOTE_CHARS}},
                         "required": ["suffix", "resolves"], "additionalProperties": False}
                        if contrastive_notes else suffix)
    cap = num_candidates // 2
    return {"type": "object",
            "properties": {"plausible": {"type": "array", "maxItems": cap, "items": suffix},
                           "contrastive": {"type": "array", "maxItems": cap, "items": contrastive_item}},
            "required": list(GROUP_KEYS), "additionalProperties": False}


def structured_output_extras(num_candidates: int, contrastive_notes: bool = False) -> dict:
    """Extra request fields that make vLLM constrain the reply to ``grouped_future_schema``."""
    return {"structured_outputs": {"json": grouped_future_schema(num_candidates, contrastive_notes)}}


def parse_grouped_future_output(
    raw_text: str, num_candidates: int, finish_reason: str | None = None,
) -> list[tuple[str, str, str]]:
    """Parse the JSON reply into ``[(group, suffix, note), ...]`` in plausible-then-contrastive order.

    A contrastive item may be a plain string or an object with ``suffix`` and
    ``resolves``; the note is empty for plain strings.

    Raises ValueError for a reply cut off by the token budget (``finish_reason ==
    "length"``), invalid JSON, missing or extra keys, non-string or blank items, or
    more items than the per-group budget. Structured output makes these rare; the
    checks keep a malformed reply from ever passing as fewer good candidates.
    """
    if num_candidates <= 0 or num_candidates % 2:
        raise ValueError("num_candidates must be a positive even number")
    if finish_reason == "length":
        raise ValueError("Truncated response: finish_reason=length")
    try:
        data = json.loads(raw_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or set(data) != set(GROUP_KEYS):
        raise ValueError(f"Response must be an object with exactly the keys {GROUP_KEYS}")
    items: list[tuple[str, str, str]] = []
    for key in GROUP_KEYS:
        group = data[key]
        if not isinstance(group, list) or len(group) > num_candidates // 2:
            raise ValueError(f"Group {key!r} must be a list of at most {num_candidates // 2} items")
        for value in group:
            note = ""
            if key == "contrastive" and isinstance(value, dict):
                if set(value) != {"suffix", "resolves"} or not isinstance(value.get("resolves"), str):
                    raise ValueError("Contrastive object must have exactly suffix and resolves strings")
                note, value = value["resolves"].strip(), value["suffix"]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Group {key!r} contains a non-string or blank item")
            items.append((key, value.strip(), note))
    return items


def sample_grouped_futures(request, num_candidates: int, retries: int, record):
    """Request a grouped (v3) response until one parses, at most ``1 + retries`` times.

    ``request(attempt)`` returns ``(raw_text, finish_reason)`` and may raise; a
    raised request propagates unchanged. ``record(event, attempt, raw_text,
    finish_reason, error)`` is called synchronously the moment a response is
    found malformed (event ``"malformed"``), so nothing is lost if a later request
    raises; it is also called once with ``"recovered"`` when a retry succeeds or
    ``"exhausted"`` before the final ValueError. Returns ``(items, raw_text)``.
    """
    error = None
    raw_text, finish_reason = "", None
    for attempt in range(1 + max(0, int(retries))):
        raw_text, finish_reason = request(attempt)
        try:
            items = parse_grouped_future_output(raw_text, num_candidates, finish_reason)
        except ValueError as exc:
            error = exc
            record("malformed", attempt, raw_text, finish_reason, str(exc))
            continue
        if attempt:
            record("recovered", attempt, raw_text, finish_reason, None)
        return items, raw_text
    record("exhausted", attempt, raw_text, finish_reason, str(error))
    raise ValueError(f"malformed grouped response after {attempt + 1} attempt(s): {error}")
