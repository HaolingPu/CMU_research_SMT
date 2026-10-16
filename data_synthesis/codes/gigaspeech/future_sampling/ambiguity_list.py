"""Ambiguity-list future sampling (design v3, Haoling 2026-10-09).

A separate step before decoding lists, per utterance, at most two words that are ambiguous at the chunk where they are
heard: Qwen3.8 sees the sentence as its numbered ~1 s chunks, judges each word on the chunks so far, and gives its readings
(English gloss + one Chinese word). Thinking runs with a token budget and the answer is then forced. Three samples are
merged. During decoding, every listed word already heard goes to the targeted sampler, which writes continuations of the
whole prefix that make each reading clear; a natural sampler adds likely continuations so the vote has enough voters.
Prompts are few-shot; no example word is an inventory lemma.
"""

from __future__ import annotations

import http.client
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter

from sentence_boundary_helpers import sentence_boundary_ends
from transformers import PreTrainedTokenizerBase

MAX_WORDS = 2
MAX_READINGS = 4
PER_READING = 2   # the per-reading count of the evaluated prompt; sample_messages() formats its own count in

DETECT_SYSTEM = f"""You help a simultaneous English-to-Chinese interpreter. The interpreter hears a sentence chunk by chunk (about one second of speech each) and must translate before the sentence ends, so when a chunk arrives it knows only that chunk and the chunks before it.

You get the whole sentence as its numbered chunks. Find the one or two words that are hardest to translate at the moment they are heard: words that, given only the chunks up to the one where the word is heard, still have two or more different meanings that need different Chinese words. Look at nouns (including names of people, places and companies) and verbs; adjectives and connectives (since: 自从 / 因为) count too. Skip the other words, and do not go through the sentence word by word.

Judge each word as the interpreter would at its chunk: do NOT use the later chunks to decide whether it is ambiguous. Use the later chunks only to say where its meaning becomes clear.

Rules:
1. Different meanings, not different wordings: the Chinese words of two readings must not be interchangeable. "often" (经常 / 往往) or "huge" (巨大 / 庞大) has one meaning and is not ambiguous.
2. A meaning counts if a natural continuation could make the speaker mean it, even if it is less common. A meaning the chunks heard so far already rule out does not count.
3. A name that could be a person has the reading "a person named X".
4. List at most {MAX_WORDS} words. Most sentences have one or none; return an empty list when no word qualifies.

For each word give: "word" as written; "chunk", the number of the chunk where it is heard; "readings", 2-{MAX_READINGS} meanings, each a short English gloss and one Chinese word, the meaning this sentence ends up using first; "settled_chunk", the first chunk after which only one reading is left, or null if the sentence never settles it."""

DETECT_EXAMPLES = (
    (["We walked", "down to the bank", "of the river", "and watched the boats."],
     {"ambiguous": [{"word": "bank", "chunk": 2, "readings": [{"gloss": "side of a river", "zh": "河岸"},
                                                                {"gloss": "financial institution", "zh": "银行"}],
                     "settled_chunk": 3}]},
     'At chunk 2 the interpreter has heard only "We walked down to the bank": 河岸 and 银行 are both possible. "of the river" '
     'in chunk 3 settles it. "walked" (走 / 散步) is one meaning and is not listed.'),
    (["I think Jordan", "has always been", "a safe place", "for travellers."],
     {"ambiguous": [{"word": "Jordan", "chunk": 1, "readings": [{"gloss": "the country", "zh": "约旦"},
                                                                  {"gloss": "a person named Jordan", "zh": "乔丹"}],
                     "settled_chunk": 3}]},
     'Jordan stays open through chunk 2, because "has always been" fits a person and a country; "a safe place" settles it.'),
    (["After the storm the pitch", "was too wet to play on,", "so the match", "was moved to Sunday."],
     {"ambiguous": [{"word": "pitch", "chunk": 1, "readings": [{"gloss": "sports field", "zh": "球场"},
                                                                 {"gloss": "sales proposal", "zh": "推销"}],
                     "settled_chunk": 2}]},
     '"match" is not listed: when it is heard in chunk 3, "too wet to play on" already rules out 火柴, so only 比赛 is left.'),
    (["She often", "visits her grandmother", "on weekends."],
     {"ambiguous": []},
     'No word qualifies: "often" (经常 / 往往) and "visits" (看望 / 拜访) each have one meaning.'),
)

CONTINUATION_RULES = """Rules:
1. A continuation is only the new words that come immediately after the LAST word of the prefix. Do not repeat or change the prefix. PREFIX + one space + SUFFIX must be grammatical and coherent: continue from the end of the prefix.
2. Continue the unfinished sentence before starting another one.
3. Stay realistic and grounded in the prefix's topic and register. Each continuation has 4-15 new English words, English only."""

CONCATENATION_EXAMPLES = """Examples of checking a continuation (illustrative, not answers to the actual input):
Observed prefix: At the picnic, the spoon fell into a
VALID: bowl of soup beside the bread.
INVALID: The spoon fell onto the table. (rewrites the event and starts a new sentence after an unfinished article)
INVALID: a bowl of soup beside the bread. (repeats the article: "a a bowl")
Observed prefix: The second runner was faster than the first
VALID: runner, despite having started several seconds later.
INVALID: she waved to the crowd. ("than the first she waved" is not grammatical)
Observed prefix: We opened the box, and the tools were
VALID: missing, although the packing list included them.
INVALID: and we began sorting them immediately. (leaves "the tools were" unfinished)"""

NATURAL_SYSTEM = f"""You predict short English speech continuations for a simultaneous English-to-Chinese interpreter. Give the most likely natural ways the speaker could go on, mutually distinct in wording and in what they say.

{CONTINUATION_RULES}

{CONCATENATION_EXAMPLES}"""

SAMPLE_SYSTEM = f"""You predict short English speech continuations for a simultaneous English-to-Chinese interpreter. The speaker has said the observed prefix so far. Some words already heard are still ambiguous; you get each one with its possible readings. Write continuations that make each reading clear, so that a careful translator would translate the word that way.

Rules:
1. A continuation is only the new words that come immediately after the LAST word of the prefix. Do not repeat or change the prefix. PREFIX + one space + SUFFIX must be grammatical and coherent, also when the ambiguous word was heard several words before the end of the prefix: continue from the end, not from the ambiguous word.
2. Continue the unfinished sentence before starting another one.
3. After reading PREFIX + SUFFIX, a translator must choose that reading's Chinese word for the ambiguous word. Do not skip a reading because it is less likely; make the continuation natural for it.
4. Stay realistic and grounded in the prefix's topic and register. Each continuation has 4-15 new English words, English only.
5. For every ambiguous word and every one of its readings, write exactly {{per_reading}} different continuations."""

SAMPLE_EXAMPLES = (
    ("We walked down to the bank", [("bank", ["financial institution (银行)", "side of a river (河岸)"])],
     {"futures": [{"word": "bank", "reading": 1, "suffix": "to deposit the checks before it closed for the day."},
                  {"word": "bank", "reading": 1, "suffix": "and waited in line until a teller called our number."},
                  {"word": "bank", "reading": 2, "suffix": "of the river and watched the boats drift past."},
                  {"word": "bank", "reading": 2, "suffix": "where the water was shallow enough to wade across."}]},
     'Joined check: "We walked down to the bank to deposit the checks before it closed for the day." is one grammatical '
     'sentence and makes bank 银行.'),
    ("We walked down to the bank and sat", [("bank", ["financial institution (银行)", "side of a river (河岸)"])],
     {"futures": [{"word": "bank", "reading": 1, "suffix": "in the lobby until a teller called our number."},
                  {"word": "bank", "reading": 1, "suffix": "near the cash machines, waiting for our loan officer."},
                  {"word": "bank", "reading": 2, "suffix": "by the water, watching the current carry leaves downstream."},
                  {"word": "bank", "reading": 2, "suffix": "on the muddy slope with our feet in the stream."}]},
     'The ambiguous word was heard earlier, so each suffix continues "sat". INVALID: "of the river and watched the boats." '
     'gives "and sat of the river", which continues "bank" instead of the end of the prefix.'),
)


def format_chunks(trajectory: list[str]) -> str:
    return "\n".join(f"{index}. {piece.strip() or '(no new words)'}" for index, piece in enumerate(trajectory, 1))


def detect_messages(trajectory: list[str]) -> list[dict[str, str]]:
    examples = "\n\n".join(f"Example {index}\nChunks:\n{format_chunks(chunks)}\nAnswer: {json.dumps(answer, ensure_ascii=False)}\n"
                           f"Note (not part of the answer): {note}"
                           for index, (chunks, answer, note) in enumerate(DETECT_EXAMPLES, 1))
    user = (f"Chunks:\n{format_chunks(trajectory)}\n\nRespond with one JSON object "
            '{"ambiguous": [{"word": "<word as written>", "chunk": <number>, "readings": [{"gloss": "<English gloss>", '
            '"zh": "<Chinese>"}, ...], "settled_chunk": <number or null>}]} and nothing else.')
    return [{"role": "system", "content": f"{DETECT_SYSTEM}\n\nExamples (illustrative, not answers to the actual input):\n\n{examples}"},
            {"role": "user", "content": user}]


def natural_messages(prefix: str, count: int) -> list[dict[str, str]]:
    user = (f"Observed prefix: {prefix}\n\nRespond with one JSON object {{\"plausible\": [\"<suffix>\", ...]}} with exactly "
            f"{count} continuations and nothing else.")
    return [{"role": "system", "content": NATURAL_SYSTEM}, {"role": "user", "content": user}]


CHECK_SYSTEM = """You help a simultaneous English-to-Chinese interpreter, who has heard only the words below and nothing after them.

For the given word, which was heard earlier, decide which of the listed meanings the speaker could still mean, given everything heard so far. A meaning counts if a natural continuation could still make the speaker mean it, even if it is less common. A meaning that the words heard so far rule out does not count.

Examples (illustrative, not answers to the actual input):
Heard so far: We walked down to the bank and
Word: bank — 1. financial institution (银行); 2. side of a river (河岸)
Answer: {"possible": [1, 2]}   (nothing heard yet decides it)
Heard so far: We walked down to the bank of the river and
Word: bank — 1. financial institution (银行); 2. side of a river (河岸)
Answer: {"possible": [2]}   ("of the river" rules out the financial institution)"""


def check_messages(prefix: str, word: str, readings: list[str]) -> list[dict[str, str]]:
    meanings = "; ".join(f"{index}. {reading}" for index, reading in enumerate(readings, 1))
    user = (f"Heard so far: {prefix}\nWord: {word} — {meanings}\n\nRespond with one JSON object "
            '{"possible": [<meaning numbers>]} and nothing else.')
    return [{"role": "system", "content": CHECK_SYSTEM}, {"role": "user", "content": user}]


def check_schema(count: int) -> dict:   # xgrammar: no uniqueItems
    return {"type": "object", "properties": {"possible": {"type": "array", "maxItems": count,
            "items": {"type": "integer", "minimum": 1, "maximum": count}}}, "required": ["possible"], "additionalProperties": False}


def readings_block(words: list[tuple[str, list[str]]]) -> str:
    return "\n".join(f'- "{word}": ' + "; ".join(f"{index}. {reading}" for index, reading in enumerate(readings, 1))
                     for word, readings in words)


def sample_messages(prefix: str, words: list[tuple[str, list[str]]], per_reading: int,
                    skip_ruled_out: bool = False) -> list[dict[str, str]]:
    examples = "\n\n".join(f"Example {index}\nObserved prefix: {example_prefix}\nAmbiguous words and readings:\n{readings_block(example_words)}\n"
                           f"Answer: {json.dumps(answer)}\nNote (not part of the answer): {note}"
                           for index, (example_prefix, example_words, answer, note) in enumerate(SAMPLE_EXAMPLES, 1))
    user = (f"Observed prefix: {prefix}\nAmbiguous words and readings:\n{readings_block(words)}\n\nRespond with one JSON object "
            '{"futures": [{"word": "<word>", "reading": <reading number>, "suffix": "<suffix>"}, ...]} and nothing else.')
    system = SAMPLE_SYSTEM.format(per_reading=per_reading)
    if skip_ruled_out:   # --list-skip-ruled-out (2026-10-10)
        system += ("\n6. If the prefix already rules out a reading (for example a later word made the meaning clear), write no "
                   "continuation for that reading.")
    else:
        pass
    return [{"role": "system", "content": f"{system}\n\nExamples (illustrative, not answers to the actual input):\n\n{examples}"},
            {"role": "user", "content": user}]


READING_SCHEMA = {"type": "object", "properties": {"gloss": {"type": "string", "minLength": 1, "maxLength": 80},
                                                   "zh": {"type": "string", "minLength": 1, "maxLength": 20}},
                  "required": ["gloss", "zh"], "additionalProperties": False}
DETECT_SCHEMA = {"type": "object", "properties": {"ambiguous": {"type": "array", "maxItems": MAX_WORDS, "items": {
    "type": "object", "properties": {
        "word": {"type": "string", "minLength": 1, "maxLength": 60},
        "chunk": {"type": "integer", "minimum": 1},
        "readings": {"type": "array", "minItems": 2, "maxItems": MAX_READINGS, "items": READING_SCHEMA},
        "settled_chunk": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]}},
    "required": ["word", "chunk", "readings", "settled_chunk"], "additionalProperties": False}}},
    "required": ["ambiguous"], "additionalProperties": False}


def natural_schema(count: int) -> dict:
    suffix = {"type": "string", "minLength": 1, "maxLength": 200}
    return {"type": "object", "properties": {"plausible": {"type": "array", "maxItems": count, "items": suffix}},
            "required": ["plausible"], "additionalProperties": False}


def sample_schema(max_items: int, max_reading: int) -> dict:
    item = {"type": "object", "properties": {
        "word": {"type": "string", "minLength": 1, "maxLength": 60},
        "reading": {"type": "integer", "minimum": 1, "maximum": max_reading},
        "suffix": {"type": "string", "minLength": 1, "maxLength": 200}},
        "required": ["word", "reading", "suffix"], "additionalProperties": False}
    return {"type": "object", "properties": {"futures": {"type": "array", "maxItems": max_items, "items": item}},
            "required": ["futures"], "additionalProperties": False}


def post(api_base: str, payload: dict, timeout: float) -> dict | None:
    base = api_base.rstrip("/")
    url = f"{base}/completions" if base.endswith("/v1") else f"{base}/v1/completions"
    http_request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read())
    except (OSError, http.client.HTTPException) as error:
        print(f"[ambiguity_list] request failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return None


def request(api_base: str, model: str, tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]], thinking: bool, samples: int,
            schema: dict, max_tokens: int, timeout: float = 3600.0, temperature: float = 1.0) -> tuple[list[dict], int]:
    """No thinking: structured output at `temperature` (1.0 by default; 0 for the reading check) / top-p 0.98. Thinking (0.6 / 0.95 / top-k 20): at most
    max_tokens of reasoning; a sample whose reasoning is not closed, or whose answer does not parse, gets the reasoning
    closed and one structured-output continuation (budget forcing). Per sample {"answer", "status" (ok / forced /
    bad_json / request_error), "reasoning_tail"}, and the completion tokens."""
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
    base = {"model": model, "stop": ["<|im_end|>", "<|endoftext|>", "<end_of_turn>"]}
    structured = {"structured_outputs": {"json": schema}}
    failed = [{"answer": None, "status": "request_error", "reasoning_tail": ""} for _ in range(samples)]
    if not thinking:
        reply = post(api_base, {**base, **structured, "prompt": prompt, "n": samples, "max_tokens": max_tokens,
                                "temperature": temperature, "top_p": 0.98}, timeout)
        if reply is None:
            return failed, 0
        else:
            results = []
        for choice in reply["choices"]:
            try:
                results.append({"answer": json.loads(choice["text"]), "status": "ok", "reasoning_tail": ""})
            except json.JSONDecodeError:
                results.append({"answer": None, "status": "bad_json", "reasoning_tail": ""})
        return results, reply["usage"]["completion_tokens"]
    else:
        sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
        reply = post(api_base, {**base, **sampling, "prompt": prompt, "n": samples, "max_tokens": max_tokens}, timeout)
    if reply is None:
        return failed, 0
    else:
        tokens, results = reply["usage"]["completion_tokens"], []
    for choice in reply["choices"]:
        reasoning, closed, after = choice["text"].partition("</think>")
        start = after.find("{")
        try:
            answer = json.JSONDecoder().raw_decode(after[start:])[0] if closed and start >= 0 else None
        except json.JSONDecodeError:
            answer = None
        if answer is not None:
            results.append({"answer": answer, "status": "ok", "reasoning_tail": reasoning[-1500:]})
            continue
        else:
            forced = post(api_base, {**base, **sampling, **structured, "n": 1, "max_tokens": 1500,
                                     "prompt": f"{prompt}{reasoning.rstrip()}\n\nI have to give the answer now.\n</think>\n\n"}, timeout)
        if forced is None:
            results.append({"answer": None, "status": "request_error", "reasoning_tail": reasoning[-1500:]})
            continue
        else:
            tokens += forced["usage"]["completion_tokens"]
        try:
            results.append({"answer": json.loads(forced["choices"][0]["text"]), "status": "forced", "reasoning_tail": reasoning[-1500:]})
        except json.JSONDecodeError:
            results.append({"answer": None, "status": "bad_json", "reasoning_tail": reasoning[-1500:]})
    return results, tokens


def locate(sentence: str, word: str) -> tuple[str, int] | None:
    """The word's first occurrence as written in the sentence and the offset right after it. An exact token match
    anywhere wins; otherwise a sentence token may extend the word (brace -> braces, empire -> empire's)."""
    token_pattern = r"[a-z0-9]+(?:'[a-z]+)?"
    word_tokens = re.findall(token_pattern, word.lower().replace("’", "'"))
    sentence_tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(token_pattern, sentence.lower().replace("’", "'"))]
    if not word_tokens:
        return None
    else:
        span = len(word_tokens)
    for loose in (False, True):
        for first in range(len(sentence_tokens) - span + 1):
            window = sentence_tokens[first:first + span]
            if all(token == target or (loose and len(target) >= 3 and token.startswith(target))
                   for (token, _, _), target in zip(window, word_tokens)):
                return sentence[window[0][1]:window[-1][2]], window[-1][2]
            else:
                continue
    return None


def zh_key(zh: str) -> str:
    """Readings with the same key are one meaning: punctuation and spaces removed, a trailing 的 / 地 / 性 dropped."""
    return re.sub(r"[的地性]$", "", re.sub(r"[\s，,。.;；/、()（）]", "", zh))


def merge_detections(answers: list[dict | None], heard_chunk) -> list[dict]:
    """The words of all well-formed answers (per answer its first MAX_WORDS entries with two or more meanings, as in the
    evaluated v3b union), keyed by (heard chunk, lower-cased word); readings merged by zh_key, the most frequent first, at
    most MAX_READINGS; words not in the trajectory or with fewer than two meanings dropped.
    `heard_chunk(word)` returns the 1-based chunk by which the word has been heard, or None."""
    merged: dict[tuple[int, str], dict] = {}
    for answer in answers:
        entries = answer.get("ambiguous") if isinstance(answer, dict) else None
        kept = [entry for entry in (entries if isinstance(entries, list) else [])
                if isinstance(entry, dict) and isinstance(entry.get("readings"), list)
                and len({zh_key(r["zh"]) for r in entry["readings"] if isinstance(r, dict) and isinstance(r.get("zh"), str)} - {""}) >= 2]
        for entry in kept[:MAX_WORDS]:
            valid = (isinstance(entry, dict) and isinstance(entry.get("word"), str) and isinstance(entry.get("readings"), list)
                     and all(isinstance(r, dict) and isinstance(r.get("gloss"), str) and isinstance(r.get("zh"), str)
                             for r in entry["readings"]))
            heard = heard_chunk(entry["word"]) if valid else None
            if heard is None:
                continue
            else:
                slot = merged.setdefault((heard, entry["word"].lower()), {"word": entry["word"], "chunk": heard,
                                                                          "readings": {}, "count": Counter()})
            for reading in entry["readings"]:
                key = zh_key(reading["zh"])
                if key:
                    slot["readings"].setdefault(key, reading)
                    slot["count"][key] += 1
                else:
                    continue
    return [{"word": slot["word"], "chunk": slot["chunk"],
             "readings": [slot["readings"][key] for key in sorted(slot["readings"], key=lambda key: -slot["count"][key])[:MAX_READINGS]]}
            for slot in sorted(merged.values(), key=lambda slot: slot["chunk"]) if len(slot["readings"]) >= 2]


def open_entries(observed: str, entries: list[dict], heard_by_chunk: int) -> list[dict]:
    """The listed words heard by this chunk whose sentence is still unfinished: no sentence end after the word in the
    observed text. With sentence-end completion the word's translation is fixed once its sentence ends."""
    text = observed.strip()
    ends = sentence_boundary_ends(text)
    result = []
    for entry in entries:
        located = locate(text, entry["word"]) if entry["chunk"] <= heard_by_chunk else None
        if located is not None and not any(end >= located[1] for end in ends):
            result.append(entry)
        else:
            continue
    return result


def extend_to_sentence(observed: str, anchored: str, earliest_offset: int) -> str:
    """The sentence-anchored prefix, extended back to the start of the sentence holding `earliest_offset` (where the
    earliest open word's chunk begins) when that lies before the anchored prefix."""
    text = observed.strip()
    if earliest_offset >= len(text) - len(anchored):
        return anchored
    else:
        starts = [end for end in sentence_boundary_ends(text) if end <= earliest_offset]
        return text[starts[-1] if starts else 0:].lstrip()
