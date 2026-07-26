---
title: MFA (Montreal Forced Aligner)
type: entity
tags: [tool, alignment, audio]
sources:
  - ../codes/gigaspeech/find_bad_json_gigaspeech.py
  - ../outputs/mfa_corpus/
created: 2026-06-01
updated: 2026-06-01
---

# MFA (Montreal Forced Aligner)

Forced aligner for matching audio to text, used in the [[synthesis-pipeline]] quality filter
(`find_bad_json_gigaspeech.py` checks LLM output aligns to the audio grid via TextGrids). Largely
phased out in favor of `src_trajectory` taken directly from the [[gigaspeech]] manifest, but still
used by [[salami]]'s `--allow-one-word` check. Outputs under `../outputs/mfa_corpus/`,
`../outputs/mfa_textgrids/`.

Distinct from [[segale-alignment]] (token-to-segment) — MFA is audio-to-text.

## Related
- [[synthesis-pipeline]], [[salami]], [[gigaspeech]].

## Sources
- code: `../codes/gigaspeech/find_bad_json_gigaspeech.py`; outputs `../outputs/mfa_*`
