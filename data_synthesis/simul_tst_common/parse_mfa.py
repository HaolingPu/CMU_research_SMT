#!/usr/bin/env python3
"""Convert MFA JSON alignments into a per-sentence MuST-C-style YAML.

Replaces the repo's parse_gentle.py. For each talk we walk the aligned word
intervals and consume one sentence's worth of tokens at a time; offset = start
of first word, duration = end of last word - offset. Surface mismatches
(usually OOV -> <unk>) are tolerated but counted.
"""
import argparse
import json
import difflib
import re
import string

STRIP = string.punctuation.replace("'", "") + "“”‘’…"


def tokenize(sentence):
    toks = []
    for w in sentence.split():
        w = w.strip(STRIP)
        if not w:
            continue
        # MFA splits tokens on internal commas/hyphens/periods ("60,000" -> "60","000")
        for part in re.split(r"[,\-.]", w):
            if part:
                toks.append(part.lower())
    return toks


def load_words(mfa_json_path):
    data = json.load(open(mfa_json_path, encoding="utf-8"))
    entries = data["tiers"]["words"]["entries"]
    return [(s, e, w.lower()) for s, e, w in entries if w and w.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mfa-dir", required=True)
    ap.add_argument("--sentences", default="sentences_by_talk.json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    by_talk = json.load(open(args.sentences, encoding="utf-8"))
    total = sum(len(v) for v in by_talk.values())
    yaml_lines = [None] * total
    unmatched_sents = 0

    for talk_id, items in by_talk.items():
        words = load_words(f"{args.mfa_dir}/ted_{talk_id}/ted_{talk_id}.json")
        mfa_toks = [w for _, _, w in words]

        toks, sent_of_tok = [], []
        for local_i, (_, sentence) in enumerate(items):
            for t in tokenize(sentence):
                toks.append(t)
                sent_of_tok.append(local_i)

        # align token stream to MFA word stream; local extra/missing words
        # (OOV splits, disfluencies) become non-equal opcodes and are skipped
        sm = difflib.SequenceMatcher(a=toks, b=mfa_toks, autojunk=False)
        span = [[None, None] for _ in items]  # [min word idx, max word idx]
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            pairs = ()
            if tag == "equal":
                pairs = ((i1 + k, j1 + k) for k in range(i2 - i1))
            elif tag == "replace" and (i2 - i1) == (j2 - j1):
                # 1:1 substitutions are trustworthy (typically OOV -> <unk>)
                pairs = ((i1 + k, j1 + k) for k in range(i2 - i1))
            for ti, wj in pairs:
                s = span[sent_of_tok[ti]]
                if s[0] is None or wj < s[0]:
                    s[0] = wj
                if s[1] is None or wj > s[1]:
                    s[1] = wj

        for local_i, (global_idx, _) in enumerate(items):
            lo, hi = span[local_i]
            if lo is None:
                unmatched_sents += 1
                yaml_lines[global_idx] = (
                    f"- {{duration: null, offset: null, "
                    f"speaker_id: spk.{talk_id}, wav: ted_{talk_id}.wav}}"
                )
            else:
                offset = words[lo][0]
                end = words[hi][1]
                yaml_lines[global_idx] = (
                    f"- {{duration: {end - offset:.6f}, offset: {offset:.6f}, "
                    f"speaker_id: spk.{talk_id}, wav: ted_{talk_id}.wav}}"
                )

    assert all(l is not None for l in yaml_lines)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")
    print(f"wrote {total} entries to {args.output}; unmatched sentences: {unmatched_sents}")


if __name__ == "__main__":
    main()
