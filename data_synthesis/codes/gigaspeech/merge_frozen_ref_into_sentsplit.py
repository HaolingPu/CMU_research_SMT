#!/usr/bin/env python3
"""Attach the frozen whole-document LLM reference onto a re-split sentsplit TSV.

The corrected `split_src_text_full_spacy.py` output has the fixed `src_text_full`
(+ regenerated `src_text`, `asr`) but no reference columns. `llm_reference_text`
is a WHOLE-document translation (independent of the sub-sentence boundaries,
reference_source=cache), so we just left-join it by `id` from the existing
decode-ready TSV — no re-translation needed.

Output columns match the decode-ready TSV the consensus pipeline consumes:
  id audio n_frames speaker src_text_full src_lang tgt_lang src_trajectory
  src_text asr llm_reference_text reference_source reference_chars
"""
import argparse, csv, sys
csv.field_size_limit(sys.maxsize)

REF_COLS = ("llm_reference_text", "reference_source", "reference_chars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-tsv", required=True, help="corrected sentsplit output (no ref cols)")
    ap.add_argument("--ref-tsv", required=True, help="existing decode TSV holding frozen llm_reference_text")
    ap.add_argument("--output-tsv", required=True)
    args = ap.parse_args()

    # index frozen reference by id
    ref = {}
    with open(args.ref_tsv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            ref[row["id"]] = {c: row.get(c, "") for c in REF_COLS}
    print(f"[ref] loaded {len(ref)} frozen references from {args.ref_tsv}")

    n = miss = 0
    with open(args.split_tsv, newline="", encoding="utf-8") as fin:
        rd = csv.DictReader(fin, delimiter="\t")
        out_fields = list(rd.fieldnames)
        for c in REF_COLS:
            if c not in out_fields:
                out_fields.append(c)
        with open(args.output_tsv, "w", newline="", encoding="utf-8") as fout:
            w = csv.DictWriter(fout, fieldnames=out_fields, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for row in rd:
                rc = ref.get(row["id"])
                if rc is None:
                    miss += 1
                    for c in REF_COLS:
                        row.setdefault(c, "")
                else:
                    row.update(rc)
                w.writerow(row)
                n += 1
    print(f"[done] wrote {n} rows ({miss} without a frozen ref) -> {args.output_tsv}")


if __name__ == "__main__":
    main()
