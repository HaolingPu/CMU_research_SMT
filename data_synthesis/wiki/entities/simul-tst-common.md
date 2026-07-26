---
title: Simul-tst-COMMON (monotonic SMT eval set)
type: entity
tags: [dataset, eval, monotonic, references]
sources:
  - ../codes/../simul_tst_common/
  - https://github.com/naist-nlp/Simul-tst-COMMON
created: 2026-07-11
updated: 2026-07-11
---

# Simul-tst-COMMON (monotonic SMT eval set)

NAIST test set (COLI paper *"Rethinking Evaluation in Simultaneous Speech Translation"*,
Makinae et al.) built on MuST-C tst-COMMON audio with **interpreter-style monotonic
references** (Salami-technique GPT-4o translation, then professional-interpreter QC).
Purpose here: re-score existing zh checkpoints to test whether the consensus-vs-hibiki
BLEU gap on [[acl-6060]] is a **reference-style artifact** (offline references reward
reordering that ref-based methods like hibiki can mimic but future-blind
[[consensus-decoding]] cannot; see [[comet-vs-bleu-ranking]]).

## Our rebuild (2026-07-11)

The repo ships a recipe (whisper `medium.en` → `split.py` → hash-keyed src patch →
pinned `gpt-4o-2024-05-13` batch → hash-keyed tgt patch), not the data (MuST-C license).
Build lives in `data_synthesis/simul_tst_common/`; installed as
`datasets/simul_tst_common/{tst.en, tst.zh, tst.source, tst.yaml, tst.unpatched.zh}`
(2,853 sentences, 27 TED talks).

- **Source side reproduces well**: 103/107 src edits hash-matched; the 4 orphans were
  hand-reconciled against the official transcript (whisper stutter duplicates + one
  truncated line).
- **Target side has drifted**: final build applied **703/1,894 tgt edits (37%)**
  (pilot estimate was ~27%); ≈66% expected if byte-identical. Two years of OpenAI
  backend changes altered outputs despite pinned snapshot + seed 0 + top_p 0.
  ⇒ ~63% of the interpreter QC corrections orphan (raw GPT output kept on those lines,
  preserved separately as `tst.unpatched.zh`).
- **Consequence**: our copy is **diagnostic-grade** (genuinely monotonic references),
  but NOT comparable to the paper's published numbers. For paper-grade numbers, email
  the authors for the final file (draft: `data_synthesis/simul_tst_common/email_naist_draft.md`).
- Timing YAML built with [[mfa]] (english_us_arpa) replacing Gentle; difflib-aligned
  token↔word mapping (`parse_mfa.py`), 0 unmatched sentences.
- Tier-1 OpenAI orgs: Batch API caps at 90k **enqueued** tokens (prompt + max_tokens
  per request!) — chunked submission with adaptive max_tokens (`submit_chunked.py`);
  lowering max_tokens is safe under top_p=0 greedy (truncation-only). Full rebuild
  cost ≈ $2–3 (Batch, ~146 in / ~89 out tokens per sentence).

## Results (2026-07-11) — gap is real

Re-scoring done same day: [[2026-07-simul-tst-common-rescore]]. The consensus-vs-hibiki
BLEU gap **persisted at 6.5–8.3** at matched latency (COMET tied, consensus a hair ahead
at high seg) → reference-style-bias hypothesis rejected; treat the deficit as real.
The EAST-even canary did **not** crater on BLEU (still leader) while posting the worst
COMET — BLEU here still anti-correlates with adequacy. Next levers: paraphrased-oracle
diagnostic, Qwen-122B futures.

## Related
- [[acl-6060]], [[comet-vs-bleu-ranking]], [[consensus-decoding]], [[scoreboard]],
  [[salami]], [[mfa]], [[checkpoint-evaluation]].

## Sources
- build: `data_synthesis/simul_tst_common/` (scripts + logs); repo clone in `repo/`
- upstream: https://github.com/naist-nlp/Simul-tst-COMMON
