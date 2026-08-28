#!/usr/bin/env python3
"""Re-segment a blob `src_text_full` manifest into SENTENCE units by punctuation.

Replacement for `split_src_text_full_spacy.py`. The spaCy version cut BEFORE every
coordinating conjunction (comma+and), which sliced *mid-clause* and left dangling
boundary fragments ("...but.", "and she.", "Editor,") that leaked into adjacent
segments and were duplicated across them — the root cause of the qwenasr regression
(see wiki experiments/2026-06-qwenasr-asr-regression-periodfix.md).

This splitter cuts ONLY on sentence-final punctuation [.!?;。！？；], so every unit is
a complete sentence/clause — no mid-clause cuts, no boundary leakage. It is a guarded
regex (NO spaCy dependency): a "." does NOT trigger a split when it is
  (a) a decimal point  (digit . digit),
  (b) after a known abbreviation (Mr. U.S. e.g. ...) or a single capital initial (J.),
  (c) followed by a lowercase word — qwenasr capitalises real sentence starts, so a
      lowercase continuation after a "." is almost always an abbreviation/mid-sentence.
! ? ; and the full-width forms always split (never used in abbreviations/decimals).

Output schema mirrors the old manifest / the spaCy script: `src_text_full` becomes the
Python-list-repr of the units, plus joined `src_text` and the raw blob `asr`. No
multiprocessing (pure regex, fast — also avoids the spaCy n_process exit-hang).
"""
import argparse, ast, csv, re, sys
csv.field_size_limit(sys.maxsize)

# Sentence-final punctuation, half- and full-width. ';' is a clean clause break.
_END = ".!?;。！？；"
_CLOSERS = "\"')]}”’》」』』"   # closing quotes/brackets that ride with the boundary
_CONTENT_RE = re.compile(r"[A-Za-z0-9]")

# Abbreviations whose trailing "." must NOT split. lower-cased, dots stripped.
_ABBR = {
    "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "vs", "etc", "inc", "ltd",
    "co", "corp", "no", "vol", "fig", "eq", "pp", "al", "gen", "gov", "sen", "rep",
    "rev", "hon", "capt", "col", "lt", "sgt", "messrs", "mt", "ave", "blvd", "dept",
    "univ", "est", "approx", "min", "max", "ca", "cf", "ed", "esp", "dec", "jan",
    "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "phd",
    "md", "ba", "ma", "bsc", "msc", "ft", "lb", "oz", "kg", "km", "cm", "mm",
    # dotted multi-letter abbreviations (matched on the token incl. internal dots)
    "e.g", "i.e", "u.s", "u.k", "a.m", "p.m", "d.c", "u.n", "u.s.a",
}


def parse_blob(raw):
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return " ".join(str(x).strip() for x in v if str(x).strip()).strip()
        return str(v).strip()
    except Exception:
        return s


def _prev_token(text, i):
    """The alpha[.alpha...] token ending immediately before index i (the '.')."""
    m = re.search(r"([A-Za-z][A-Za-z.]*)$", text[:i])
    return m.group(1) if m else ""


def _content_words(unit):
    return sum(1 for t in unit.split() if _CONTENT_RE.search(t))


def _merge_short(units, min_words=2):
    """Fold sub-`min_words`-content-word units into a neighbour. With sentence-only
    splitting these are NOT mid-clause garbage (those can't occur here) but genuine
    1-word sentences ('Yes.') or a row-final audio truncation; folding them keeps the
    fragment rate at ~0 like the prior spaCy fix. First short unit folds FORWARD, the
    rest fold BACKWARD. min_words=1 disables."""
    units = [u for u in units if u and u.strip()]
    if min_words <= 1 or len(units) <= 1:
        return units
    while len(units) > 1 and _content_words(units[0]) < min_words:
        units = [re.sub(r"\s+", " ", (units[0] + " " + units[1]).strip())] + units[2:]
    out = [units[0]]
    for u in units[1:]:
        if _content_words(u) < min_words:
            out[-1] = re.sub(r"\s+", " ", (out[-1] + " " + u).strip())
        else:
            out.append(u)
    return out


def _split_punct(text, min_words=2):
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    units, start, i, n = [], 0, 0, len(text)
    while i < n:
        c = text[i]
        if c in _END:
            # consume a run of end-punct + closing quotes (handles '...', '?"', etc.)
            j = i + 1
            while j < n and text[j] in (_END + _CLOSERS):
                j += 1
            k = j
            while k < n and text[k] == " ":
                k += 1
            split_here = True
            if c == ".":
                # (a) decimal
                if i > 0 and text[i - 1].isdigit() and k < n and text[k].isdigit():
                    split_here = False
                # (b) abbreviation / single initial
                if split_here:
                    w = _prev_token(text, i).lower().rstrip(".")
                    if w and (w in _ABBR or len(w) == 1):
                        split_here = False
                # (c) lowercase continuation -> not a real sentence start
                if split_here and k < n and text[k].islower():
                    split_here = False
            if split_here and k < n:        # never split at the very end
                seg = text[start:j].strip()
                if seg:
                    units.append(seg)
                start, i = j, j
                continue
        i += 1
    seg = text[start:].strip()
    if seg:
        units.append(seg)
    units = [u for u in units if u and _CONTENT_RE.search(u)]
    return _merge_short(units, min_words=min_words)


# ---- clause-level mode (⑤b): match the old-asr LLM clause granularity --------------
# Old asr split mid-sentence at clause junctions, keeping the comma with the LEFT clause
# ("...he is the editor," | "and not the author of the fairy tales,"). We approximate
# that by cutting at a comma that immediately precedes a clause-introducing word
# (coordinating conj / relative / subordinator). The comma stays on the left; the new
# unit starts with the conjunction. Then _merge_short folds any sub-min_words fragment.
_CLAUSE_CONJ = (r"(?:and|but|or|nor|so|yet|for|which|who|whom|whose|where|when|while|"
                r"because|though|although|since|unless|whereas|that|as|if)")
_CLAUSE_RE = re.compile(r",[ \t]+(?=" + _CLAUSE_CONJ + r"\b)", re.IGNORECASE)


def _clause_split_sentence(sent):
    out, last = [], 0
    for m in _CLAUSE_RE.finditer(sent):
        cut = m.start() + 1                 # keep the comma with the left clause
        left = sent[last:cut].strip()
        if left:
            out.append(left)
        last = m.end()
    tail = sent[last:].strip()
    if tail:
        out.append(tail)
    return out if out else [sent]


def _split_clause(text, min_words=2):
    units = []
    for sent in _split_punct(text, min_words=1):   # sentences first (no merge yet)
        units.extend(_clause_split_sentence(sent))
    units = [u for u in units if u and _CONTENT_RE.search(u)]
    return _merge_short(units, min_words=min_words)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-tsv", required=True)
    p.add_argument("--output-tsv", required=True)
    p.add_argument("--src-column", default="asr",
                   help="Column holding the raw paragraph blob to split (read).")
    p.add_argument("--out-column", default="src_text_full",
                   help="Column the split unit-list is written to (the decoder reads this).")
    p.add_argument("--mode", choices=["sentence", "clause"], default="sentence",
                   help="sentence=split on sentence-final punct only (⑤); "
                        "clause=also split at clause junctions to match old-asr granularity (⑤b).")
    p.add_argument("--min-unit-words", type=int, default=2,
                   help="Fold units with fewer content words into a neighbour. 1 disables.")
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()

    n_split = n_empty = 0
    with open(args.input_tsv, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        out_fields = list(reader.fieldnames)
        for extra in (args.out_column, "src_text", "asr"):
            if extra not in out_fields:
                out_fields.append(extra)
        with open(args.output_tsv, "w", newline="", encoding="utf-8") as fout:
            w = csv.DictWriter(fout, fieldnames=out_fields, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for idx, row in enumerate(reader):
                if args.max_rows is not None and idx >= args.max_rows:
                    break
                blob = parse_blob(row.get(args.src_column))
                splitter = _split_clause if args.mode == "clause" else _split_punct
                units = splitter(blob, min_words=args.min_unit_words)
                if not units:
                    units = [blob] if blob else []
                    n_empty += 1
                else:
                    n_split += 1
                # write the split unit-list to the DECODER-read column (src_text_full),
                # independent of which column we read the blob from.
                row[args.out_column] = repr(units)
                row["src_text"] = " ".join(units).strip()
                row["asr"] = blob
                w.writerow(row)

    print(f"[done] split_ok={n_split} empty/fallback={n_empty}")
    print(f"[done] wrote {args.output_tsv}")


if __name__ == "__main__":
    main()
