#!/usr/bin/env python3
"""Assign each final source sentence to its TED talk and build an MFA corpus.

Per-talk sentence lists are recovered by running split.py's logic per talk and
aligning the concatenation against the final (edited) source with difflib.
Writes:
  mfa_corpus/ted_XXXX/ted_XXXX.wav (symlink) + .lab (talk transcript)
  sentences_by_talk.json  {talk_id: [(global_line_idx, sentence), ...]}
"""
import difflib
import json
import os
import subprocess
import sys

WAV_DIR = "/data/group_data/li_lab/siqiouya/datasets/must-c/v2.0/en-zh/data/tst-COMMON/wav"
BUILD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BUILD, "repo", "scripts"))
from split import split_sentences, is_sentence_end  # noqa: E402


def split_talk(path):
    sentences, buffer = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        buffer.append(line)
        if is_sentence_end(line):
            sentences.extend(split_sentences(" ".join(buffer)))
            buffer = []
    if buffer:
        sentences.extend(split_sentences(" ".join(buffer)))
    return sentences


def main():
    talks = sorted(
        (f for f in os.listdir("asr_out") if f.endswith(".txt")),
        key=lambda f: int(f.split("_")[1].split(".")[0]),
    )
    pre_edit, boundaries = [], []  # boundaries[i] = (talk_id, start, end) in pre-edit lines
    for f in talks:
        talk_id = f.split("_")[1].split(".")[0]
        sents = split_talk(os.path.join("asr_out", f))
        boundaries.append((talk_id, len(pre_edit), len(pre_edit) + len(sents)))
        pre_edit.extend(sents)

    final = [l.strip() for l in open("Simul-tst-COMMON.en", encoding="utf-8") if l.strip()]
    print(f"pre-edit {len(pre_edit)} sentences, final {len(final)}")

    def talk_of(pre_idx):
        for talk_id, s, e in boundaries:
            if s <= pre_idx < e:
                return talk_id
        raise ValueError(pre_idx)

    # align final -> pre-edit; every final line inherits the talk of its matched
    # or replaced pre-edit region
    sm = difflib.SequenceMatcher(a=pre_edit, b=final, autojunk=False)
    assignment = [None] * len(final)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                assignment[j1 + k] = talk_of(i1 + k)
        elif tag in ("replace", "insert"):
            # attribute to the talk at the corresponding pre-edit position
            anchor = min(i1, len(pre_edit) - 1)
            for k in range(j1, j2):
                assignment[k] = talk_of(anchor)

    assert all(a is not None for a in assignment)
    # sanity: assignments must be non-decreasing in talk order
    order = {t: i for i, (t, _, _) in enumerate(boundaries)}
    assert all(order[assignment[i]] <= order[assignment[i + 1]] for i in range(len(final) - 1)), \
        "non-monotonic talk assignment"

    by_talk = {}
    for idx, (talk, sent) in enumerate(zip(assignment, final)):
        by_talk.setdefault(talk, []).append((idx, sent))
    json.dump(by_talk, open("sentences_by_talk.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    for talk_id, items in by_talk.items():
        d = os.path.join("mfa_corpus", f"ted_{talk_id}")
        os.makedirs(d, exist_ok=True)
        wav = os.path.join(d, f"ted_{talk_id}.wav")
        if not os.path.exists(wav):
            os.symlink(os.path.join(WAV_DIR, f"ted_{talk_id}.wav"), wav)
        with open(os.path.join(d, f"ted_{talk_id}.lab"), "w", encoding="utf-8") as f:
            f.write("\n".join(s for _, s in items) + "\n")
        print(f"ted_{talk_id}: {len(items)} sentences")


if __name__ == "__main__":
    main()
