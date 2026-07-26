---
title: Consensus Post-Edit to Recover BLEU — a Triple Negative Result
type: experiment
tags: [synthesis, consensus, ref-free, bleu, negative-result, post-edit]
sources:
  - ../codes/gigaspeech/future_sampling/consensus_decoding_retranslate.py
  - ../codes/gigaspeech/future_sampling/consensus_decoding_token_id_level_instruct.py
  - ../codes/gigaspeech/future_sampling/run_J_smoke5_retranslate.sbatch
created: 2026-06-01
updated: 2026-06-01
---

# Consensus Post-Edit to Recover BLEU — a Triple Negative Result

**Question.** [[consensus-decoding]] scores ~5–8 BLEU below reference-based synthesis
([[east]], hibiki, word-align/`s_origin`) on [[acl-6060]] (see [[scoreboard]]). Can an LLM
**post-edit** step recover that BLEU without abandoning consensus's value (ref-free, no
[early-commitment](#why-conservatism-is-the-frontier))?

**Answer: no.** Three post-edit variants were tried; all regress vs raw consensus at
synthesis-time char-BLEU (vs the frozen LLM reference, 5-case smoke). The BLEU gap is
**structural**, not recoverable by editing.

## Why the gap is structural

[[east]] and [[salami]] (Simul-MuST-C) translate the **whole source offline** (the LLM sees
the future) and then segment that fluent translation into read/write steps. Consensus commits
incrementally and **never sees the future** — it only commits tokens that all sampled futures
agree on. Reference-based hibiki/word-align additionally see the gold target. So consensus is
the only policy that is both ref-free AND future-blind. Its lower BLEU is the honest cost of
that constraint. (Framing mirrors the memory note `project_consensus_reffree_vs_refbased`.)

## The four data points (synthesis-time char-BLEU vs frozen reference)

| variant | mean BLEU | vs consensus | failure mode |
|---|---|---|---|
| **raw consensus** | **56.7** | — | (the conservative frontier) |
| strict minimal-edit polish | 52.4 | −4 | ties/loses; injects bad punctuation |
| free re-translation | 46.2 | −11 | anticipation / early commitment |
| free polish | 12.8 | −44 | embellishment / hallucination (len-ratio 2.36×) |
| soft-vote (trained, ACL6060) | — | −2 to −4 | looser commit → worse (see below) |

1. **soft-vote** — loosen the hard top-k intersection to a `min_voters_ratio=0.75` majority
   (the `consensus-top5-axis5-sv` ckpt, [[min-p-sampling]] / [[majority-vote]] family). Trained
   + evaluated: lost ~2–4 BLEU vs hard `top5-axis5` on [[acl-6060]] (seg960 30.9 vs 34.9),
   COMET ~tied. **Loosening hurts → the gap is NOT "too conservative."**

2. **free re-translation** — use consensus only as a timing/boundary oracle, discard its tokens,
   let the instruct model re-translate the *revealed-so-far* source forward from the committed
   prefix (past-only). 46.2 vs 57.6. Smoking gun: after revealed source `"...inevitably both"`
   the LLM committed `"双重性。"` (read "both" as the noun "duality" and **closed the sentence**),
   while consensus stayed at the safe adverb `"不可避免地"` and correctly picked up
   `"既单调又无用"` two chunks later. Classic early commitment.

3. **free polish** — feed consensus's committed draft to the LLM: *"rewrite into fluent natural
   Chinese, don't add beyond the draft."* 12.8 vs 57.9, length-ratio 2.36×. "Make fluent"
   licensed literary invention (`"迫使编辑"` → `"迫使编辑不得不直面这一困境。"`; output became flowery
   prose with invented clauses).

4. **strict minimal-edit polish** — only reorder words already present / fix particles / delete
   duplicates; forbid new content words, clauses, sentence-final punctuation; output ≤ draft
   length. 52.4 vs 56.7, length-ratio back to 1.01. No longer hallucinates, but **still
   ties-to-loses** and occasionally injects a mid-thought period
   (`"重复自己，是编辑。而非作者"`).

## Why conservatism is the frontier

Loose post-edits re-introduce exactly the anticipation/embellishment that consensus's
token-level agreement suppresses; strict post-edits don't help and add noise. Consensus's
committed text is already at the fluency ceiling reachable **without future leakage**. The
~5–8 BLEU deficit vs ref-based methods is not a tuning problem — it is the structural price of
honest online translation. See [[comet-vs-bleu-ranking]] (rank by COMET; consensus reaches the
COMET top at high latency) and [[latency-quality-tradeoff]].

**Paper angle:** ref-free + no-future-leakage + COMET parity at high latency; the BLEU deficit
is the honest online cost, and we show post-editing cannot close it (loose → hallucinate,
strict → no gain).

## Caveats
- Synthesis-time BLEU is **not predictive** of trained-eval BLEU (memory:
  `feedback_synthesis_bleu_not_predictive`). A variant that does not even win at synthesis time
  is doubly not worth a training run.
- 5-case smoke only; the verbose dual-track logs (`consensus_prediction`/`consensus_metrics`
  kept alongside `prediction`/`metrics`) are the qualitative evidence, not corpus stats.

## Code
- `consensus_decoding_retranslate.py` — imports the 5-axis baseline
  `consensus_decoding_token_id_level_instruct.py` and monkeypatches `run_one_utterance` to run a
  dual track (consensus vs post-edit); currently holds the polish variant. The post-edit prompt
  is the only substantive lever (`build_polish_prompt`).
- `run_J_smoke5_retranslate.sbatch` — 5-case verbose smoke harness (clone of the axis5
  `run_J_smoke5_softvote.sbatch`, decoder filename swapped).

## Sources
- [[consensus-decoding]], [[future-sampling]], [[comet-vs-bleu-ranking]], [[scoreboard]],
  [[east]], [[salami]], [[acl-6060]], [[2026-06-consensus-axis5-vs-futures200]]
