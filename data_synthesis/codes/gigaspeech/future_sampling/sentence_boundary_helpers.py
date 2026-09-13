"""Sentence boundaries decided from the observed English prefix only."""

import re


SENTENCE_BOUNDARY_RE = re.compile(r"([.!?]+)[\"”’')\]]*(?=\s|$)")
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "a.m", "p.m", "no", "fig", "inc", "ltd",
}


def sentence_boundary_ends(observed: str) -> list[int]:
    """Offsets just past each sentence end that is unambiguous without seeing more speech.

    Abbreviations, initials, ellipses and a period right after a digit might not end the
    sentence, so they are skipped rather than guessed. This misses some real ends, such as
    a sentence ending in 2026.
    """
    ends = []
    for match in SENTENCE_BOUNDARY_RE.finditer(observed):
        punctuation = match.group(1)
        before = observed[:match.start()]
        last_word = re.search(r"[A-Za-z][A-Za-z.]*$", before)
        if set(punctuation) != {"."}:
            ends.append(match.end())
        elif len(punctuation) > 1 or before[-1:].isdigit():
            continue
        elif last_word and (last_word.group().lower() in ABBREVIATIONS
                            or re.fullmatch(r"[A-Za-z](?:\.[A-Za-z])*", last_word.group())):
            continue
        else:
            ends.append(match.end())
    return ends


def observed_sentence_is_complete(observed: str) -> bool:
    text = observed.rstrip()
    ends = sentence_boundary_ends(text)
    return bool(ends) and ends[-1] == len(text)


def sentence_anchored_prefix(observed: str, max_words: int) -> str:
    """The sentence the observed source is currently in, cut to its last `max_words` words.

    Comma-delimited source units do not start a new sentence. At a sentence end the
    finished sentence itself is returned rather than an empty prefix.
    """
    text = observed.strip()
    prior_ends = [end for end in sentence_boundary_ends(text) if end < len(text)]
    if prior_ends:
        sentence = text[prior_ends[-1]:].lstrip()
    else:
        sentence = text
    words = list(re.finditer(r"\S+", sentence))
    if len(words) > max_words:
        return sentence[words[-max_words].start():]
    else:
        return sentence


def source_terminal_mark(observed: str) -> str:
    """Chinese sentence-final mark matching the observed English sentence end, "" when not at one."""
    text = observed.rstrip()
    if not observed_sentence_is_complete(text):
        return ""
    else:
        punctuation = list(SENTENCE_BOUNDARY_RE.finditer(text))[-1].group(1)
        return "？" if "?" in punctuation else "！" if "!" in punctuation else "。"


def close_translation_delta(delta: str, committed: str, terminal: str) -> tuple[str, str]:
    """End the newly emitted Chinese with `terminal`; the committed prefix is never rewritten.

    A delta without text becomes just the mark. When the committed text already ends in
    other punctuation, the conflict is reported instead of emitting double punctuation.
    """
    text = delta.rstrip()
    closers = re.search(r"[\"”’'」』)\]}]*$", text).group()
    body = re.sub(r"[\s,.!?;:。！？，；：…]+$", "", text[:len(text) - len(closers)])
    previous = re.sub(r"[\s\"”’'」』)\]}]+$", "", committed)
    if body:
        closed = body + terminal + closers
        return closed, "unchanged" if closed == delta else "normalized_new_delta"
    elif not previous:
        return delta, "empty_translation"
    elif previous.endswith(terminal):
        return closers, "already_closed"
    elif previous[-1] in ",.!?;:。！？，；：…":
        return delta, "immutable_prefix_punctuation_conflict"
    else:
        return terminal + closers, "appended_terminal_only"
