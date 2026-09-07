"""Conservative, observed-prefix-only boundaries for opt-in decoder pilots."""

import re


_BOUNDARY = re.compile(r"([.!?]+)[\"\u201d\u2019')\]]*(?=\s|$)")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "a.m", "p.m", "no", "fig", "inc", "ltd",
}


def sentence_boundary_ends(observed: str) -> list[int]:
    """Return conservative boundary offsets without consulting unobserved text.

    Abbreviations, initials, ellipses and periods immediately following digits
    are ambiguous online, so defer completion rather than inventing a boundary.
    This intentionally misses some real ends, such as a sentence ending in 2026.
    """
    ends = []
    for match in _BOUNDARY.finditer(observed):
        punctuation = match.group(1)
        if set(punctuation) == {"."}:
            if len(punctuation) > 1:
                continue
            before = observed[:match.start()]
            if before and before[-1].isdigit():
                continue
            word = re.search(r"([A-Za-z][A-Za-z.]*)$", before)
            if word:
                token = word.group(1)
                if token.lower() in _ABBREVIATIONS:
                    continue
                if re.fullmatch(r"[A-Za-z](?:\.[A-Za-z])*", token):
                    continue
        ends.append(match.end())
    return ends


def observed_sentence_is_complete(observed: str) -> bool:
    text = observed.rstrip()
    ends = sentence_boundary_ends(text)
    return bool(ends) and ends[-1] == len(text)


def sentence_anchored_prefix(observed: str, max_words: int = 128) -> str:
    """Keep the current observed sentence, ignoring comma-delimited source units."""
    text = observed.strip()
    # At a sentence end, keep that sentence rather than returning an empty prefix.
    prior_ends = [end for end in sentence_boundary_ends(text) if end < len(text)]
    prefix = text[prior_ends[-1]:].lstrip() if prior_ends else text
    words = list(re.finditer(r"\S+", prefix))
    if max_words > 0 and len(words) > max_words:
        prefix = prefix[words[-max_words].start():]
    return prefix


def source_terminal_mark(observed: str) -> str:
    """Map a recognized observed English sentence end to Chinese punctuation."""
    text = observed.rstrip()
    if not observed_sentence_is_complete(text):
        return ""
    match = list(_BOUNDARY.finditer(text))[-1]
    punctuation = match.group(1)
    return "\uFF1F" if "?" in punctuation else "\uFF01" if "!" in punctuation else "\u3002"


def close_translation_delta(delta: str, committed: str, terminal: str) -> tuple[str, str]:
    """Close only newly emitted text; never rewrite an immutable target prefix.

    If all text/punctuation was already committed, report an unfixable boundary
    rather than emitting double punctuation or pretending to repair history.
    """
    if not terminal:
        return delta, "not_sentence_end"
    text = delta.rstrip()
    closers = re.search(r"[\"\u201d\u2019'\u300d\u300f)\]}]*$", text).group()
    core = text[:-len(closers)] if closers else text
    body = re.sub(r"[\s,.!?;:\u3002\uFF01\uFF1F\uFF0C\uFF1B\uFF1A\u2026]+$", "", core)
    if body:
        closed = body + terminal + closers
        return closed, "unchanged" if closed == delta else "normalized_new_delta"
    previous = re.sub(r"[\s\"\u201d\u2019'\u300d\u300f)\]}]+$", "", committed)
    if not previous:
        return delta, "empty_translation"
    if previous.endswith(terminal):
        return closers, "already_closed"
    if previous[-1] in ",.!?;:\u3002\uFF01\uFF1F\uFF0C\uFF1B\uFF1A\u2026":
        return delta, "immutable_prefix_punctuation_conflict"
    return terminal + closers, "appended_terminal_only"
