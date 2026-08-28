#!/usr/bin/env python3
"""Re-segment a blob `src_text_full` manifest into clause-level units (spaCy).

The cleaned qwenasr manifest stores `src_text_full` as a single un-split string,
which collapses both the consensus decode's streaming source window and the
per-sentence MetricX QE to whole-document granularity. The original
`asr-filtered` manifest had `src_text_full` as a list of clause-level units
(LLM-segmented). We can't reproduce those exact boundaries, but spaCy
sentence-segmentation + a coordinating-conjunction (and/but/or/...) clause split
matches the original *granularity* (≈5.78 vs 5.70 units/doc on the old manifest),
which is what drives the QE survivor rate.

Output schema mirrors the old manifest: appends `src_text` (joined) and `asr`
(raw blob) and replaces `src_text_full` with the split list.
"""
import argparse, ast, csv, re, sys
csv.field_size_limit(sys.maxsize)


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


def make_splitter(model="en_core_web_sm"):
    import spacy
    nlp = spacy.load(model, disable=["ner", "lemmatizer"])

    return nlp, _split_doc


_CONTENT_RE = re.compile(r"[A-Za-z0-9]")


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _content_words(unit):
    """Count tokens carrying at least one alphanumeric char (so 'Editor,' -> 1,
    'and.' -> 1, 'In Arcadia,' -> 2). Punctuation-only tokens don't count."""
    return sum(1 for t in unit.split() if _CONTENT_RE.search(t))


def _merge_short(units, min_words=2):
    """Glue away sub-threshold fragments that the clause split leaves behind
    (e.g. a leading 'Editor,' or a trailing 'and.'/'so.'). A unit with fewer than
    `min_words` content words is merged into a neighbour: the first unit folds
    FORWARD into the next, every other short unit folds BACKWARD into the previous.
    Default min_words=2 removes 1-content-word garbage fragments while keeping
    legitimate 2-word clauses like 'In Arcadia,' (matches the old manifest's
    fragment rate). Set min_words=1 to disable."""
    units = [u for u in units if u and u.strip()]
    if min_words <= 1 or len(units) <= 1:
        return units
    # First unit too short -> fold forward into the second.
    while len(units) > 1 and _content_words(units[0]) < min_words:
        units = [_norm(units[0] + " " + units[1])] + units[2:]
    # Remaining short units -> fold backward into the previous (kept) unit.
    out = [units[0]]
    for u in units[1:]:
        if _content_words(u) < min_words:
            out[-1] = _norm(out[-1] + " " + u)
        else:
            out.append(u)
    return out


def _split_doc(doc, min_words=2):
    """Clause-level split: cut BEFORE a coordinating conjunction when it joins
    clauses (head is a VERB/AUX) or an explicit comma precedes it. Skips bare
    phrase coordination like 'monotonous and unavailing'. Trailing comma stays
    with the previous unit (matches old '...editor,' | 'and not...').

    Then `_merge_short` folds away the sub-`min_words` fragments the raw cc-split
    produces on the new qwenasr blobs ('Editor,', 'and.', 'so.') — the over-
    segmentation that broke streaming alignment vs the old manifest."""
    units = []
    for sent in doc.sents:
        toks = list(sent)
        start = 0
        for i, t in enumerate(toks):
            if t.pos_ == "CCONJ" and t.dep_ == "cc" and i > start:
                comma_before = i > 0 and toks[i - 1].text == ","
                if comma_before or t.head.pos_ in ("VERB", "AUX"):
                    seg = "".join(tt.text_with_ws for tt in toks[start:i]).strip()
                    if seg:
                        units.append(re.sub(r"\s+", " ", seg).strip())
                    start = i
        seg = "".join(tt.text_with_ws for tt in toks[start:]).strip()
        if seg:
            units.append(re.sub(r"\s+", " ", seg).strip())
    units = [u for u in units if u]
    return _merge_short(units, min_words=min_words)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-tsv", required=True)
    p.add_argument("--output-tsv", required=True)
    p.add_argument("--src-column", default="src_text_full")
    p.add_argument("--model", default="en_core_web_sm")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-process", type=int, default=8)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--min-unit-words", type=int, default=2,
                   help="Merge units with fewer than this many content words into a "
                        "neighbour (kills 'Editor,'/'and.' fragments). 1 disables.")
    args = p.parse_args()

    nlp, split = make_splitter(args.model)

    with open(args.input_tsv, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        in_fields = list(reader.fieldnames)
        out_fields = list(in_fields)
        for extra in ("src_text", "asr"):
            if extra not in out_fields:
                out_fields.append(extra)
        rows = []
        for i, row in enumerate(reader):
            if args.max_rows is not None and i >= args.max_rows:
                break
            rows.append(row)

    blobs = [parse_blob(r.get(args.src_column)) for r in rows]

    # spaCy pipe with multiprocessing for the parse
    docs = nlp.pipe(blobs, batch_size=args.batch_size, n_process=args.n_process)

    n_split = n_empty = 0
    with open(args.output_tsv, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=out_fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row, blob, doc in zip(rows, blobs, docs):
            units = _split_doc(doc, min_words=args.min_unit_words)
            if not units:
                units = [blob] if blob else []
                n_empty += 1
            else:
                n_split += 1
            row[args.src_column] = repr(units)        # Python-list string, like old manifest
            row["src_text"] = " ".join(units).strip()
            row["asr"] = blob
            w.writerow(row)

    print(f"[done] rows={len(rows)} split_ok={n_split} empty/fallback={n_empty}")
    print(f"[done] wrote {args.output_tsv}")


if __name__ == "__main__":
    main()
