"""Prompt for coordinated plausible and contrastive future continuations."""

from __future__ import annotations

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


def build_coordinated_future_messages(
    *,
    observed_source: str,
    target_lang: str,
    committed_text: str,
    num_candidates: int,
    prompt_version: str = PROMPT_VERSION,
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
            observed_source, target_lang, commitment, num_candidates,
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


def _build_suffix_icl_messages(
    observed_source: str, target_lang: str, commitment: str, num_candidates: int,
) -> list[dict[str, str]]:
    per_group = num_candidates // 2
    ambiguity_examples = "\n\n".join(
        f"Ambiguity example {i} ({example['kind']}):\n"
        f"Observed prefix: {example['prefix']}\n"
        "Example response:\n"
        f"Plausible\n1. {example['plausible']}\n"
        f"Contrastive\n1. {example['contrastive']}\n"
        f"Teaching note (not part of the response): {example['explanation']}"
        for i, example in enumerate(AMBIGUITY_ICL_EXAMPLES, 1)
    )
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
8. Each suffix should contain 4-15 new English words. Output English suffixes only, with the two group headings and numbered items. No reasoning, translations, full rewritten sentences, or copies of the examples.

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

Use both headings exactly as shown below. Under each heading, number only the candidates you actually provide, starting at 1. For an empty group write None on its own line. Do not include placeholders or explanations.

Plausible
<numbered suffixes, or None>
Contrastive
<numbered suffixes, or None>"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_GROUP_HEADING_RE = re.compile(
    r"^[\W_]*(plausible|contrastive|contrast|contrasting)\b[\W_]*"
    r"(?:(?:candidates?|continuations?|suffixes|suffix|readings?|group|list)\b[\W_]*)?"
    r"(?:\([^)]*\)[\W_]*)?$",
    re.IGNORECASE,
)
_NUMBERED_LINE_RE = re.compile(r"^\(?([1-9][0-9]*)[.):\-]?\)?\s*(.*)$")
_BULLET_LINE_RE = re.compile(r"^[-*\u2022\u00b7]\s+(.+)$")
_EMPTY_GROUP_RE = re.compile(
    r"^(?:none|n/?a|nothing|no (?:valid |good |suitable |additional )?"
    r"(?:candidates?|suffixes?|continuations?|contrastive[\w ]*|plausible[\w ]*|alternatives?))[\W_]*$",
    re.IGNORECASE,
)


def _clean_candidate_text(text: str) -> str:
    text = text.strip()
    for _ in range(2):
        text = text.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
            text = text[1:-1]
        elif text.startswith("**") and text.endswith("**") and len(text) > 4:
            text = text[2:-2]
        elif text.startswith("\u201c") and text.endswith("\u201d"):
            text = text[1:-1]
    return text.strip()


def parse_grouped_future_output(
    raw_text: str, num_candidates: int, finish_reason: str | None = None,
) -> list[tuple[str, str]]:
    """Parse variable-size v3 groups by heading, never by flattened position.

    Tolerated (benign) variations: preamble text before the first heading, a
    stripped ``<think>`` block, decorated headings (``**Plausible:**``,
    ``### Contrastive``, ``Plausible candidates``, ``Contrast``), ``1)``/``(1)``/``1-``/bullet
    items, bold or quoted items, and an empty group written as ``None``,
    ``None.``, ``(none)``, ``N/A`` or a numbered ``1. None``. An empty group that
    is followed by the other heading is accepted; an empty trailing group is
    accepted only when ``finish_reason == "stop"`` (the sampler ended on its own),
    so a truncated response is still reported as malformed.

    Still rejected (raised as ValueError, never silent abstention): missing or
    repeated headings, candidates before any heading, prose lines inside a group,
    mixing ``None`` with candidates, duplicate numbers, more items than the group
    budget, and a trailing empty group without ``None`` unless finish_reason is
    ``stop``.
    """
    if num_candidates <= 0 or num_candidates % 2:
        raise ValueError("num_candidates must be a positive even number")
    text = _THINK_BLOCK_RE.sub("", raw_text or "")
    groups: dict[str, list[str]] = {}
    empty_groups: set[str] = set()
    indices: dict[str, set[int]] = {}
    order: list[str] = []
    mode: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = _GROUP_HEADING_RE.match(line)
        if heading:
            name = heading.group(1).lower()
            if name in ("contrast", "contrasting"):
                name = "contrastive"  # Qwen3.8 occasionally shortens the heading
            if name in groups:
                raise ValueError(f"Repeated future group: {name}")
            mode = name
            groups[mode] = []
            indices[mode] = set()
            order.append(mode)
            continue
        numbered = _NUMBERED_LINE_RE.match(line)
        bullet = _BULLET_LINE_RE.match(line)
        if mode is None:
            if numbered or bullet:
                raise ValueError("Future candidates must follow a group heading")
            continue  # preamble such as "Here are the suffixes:" is ignored
        body = numbered.group(2) if numbered else (bullet.group(1) if bullet else line)
        body = _clean_candidate_text(body)
        bare = re.sub(r"^[\W_]+|[\W_]+$", "", body)
        if _EMPTY_GROUP_RE.match(bare) or (not body and numbered is None and bullet is None):
            if groups[mode]:
                raise ValueError(f"Conflicting empty future group: {mode}")
            empty_groups.add(mode)
            continue
        if not numbered and not bullet:
            raise ValueError(f"Invalid candidate line in future group: {mode}")
        if mode in empty_groups:
            raise ValueError(f"Conflicting empty future group: {mode}")
        if not body:
            raise ValueError(f"Empty candidate line in future group: {mode}")
        if numbered:
            index = int(numbered.group(1))
            if index in indices[mode]:
                raise ValueError(f"Repeated candidate number in future group: {mode}")
            indices[mode].add(index)
        groups[mode].append(body)
        if len(groups[mode]) > num_candidates // 2:
            raise ValueError(f"Too many candidates in future group: {mode}")
    if set(groups) != {"plausible", "contrastive"}:
        raise ValueError("Both future group headings are required")
    for position, name in enumerate(order):
        if groups[name] or name in empty_groups:
            continue
        followed_by_heading = position < len(order) - 1
        if followed_by_heading or finish_reason == "stop":
            empty_groups.add(name)
            continue
        raise ValueError(f"Empty future group must explicitly say None: {name}")
    return [(name, item) for name in ("plausible", "contrastive") for item in groups[name]]
