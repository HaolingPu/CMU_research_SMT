---
title: Consensus Decoding
type: concept
tags: [synthesis, decoding, ref-free]
sources:
  - ../codes/gigaspeech/future_sampling/consensus_decoding.py
  - ../codes/gigaspeech/future_sampling/consensus_decoding_token_id_level_instruct.py
created: 2026-06-01
updated: 2026-07-11
---

# Consensus Decoding

Reference-free [[future-sampling]] variant: **two base models** (e.g. Gemma4-E2B + Qwen3-4B)
sample futures in parallel, an instruct model supplies next-token distributions, and consensus
selects the committed token from a top-k / [[min-p-sampling]] candidate set. Unlike ref-based
baselines (hibiki / word-align = `s_origin`, which see the reference, cf. [[infinisst-omni]]),
consensus does **not** — it trades surface-form BLEU for semantic quality (COMET). See
[[comet-vs-bleu-ranking]].

## Variants
- `consensus_decoding.py` — text-level selection.
- `consensus_decoding_token_id_level.py` — token-level consensus.
- `consensus_decoding_token_id_level_instruct.py` — instruct-model token ranking (+ per-sentence
  [[metricx]] QE filtering).
- `_gpt.py` — OpenAI backend.

Trained downstream as the `consensus-top5*` checkpoint family (see [[infinisst-omni]],
[[dataset-conversion-pipeline]]). Output: `../outputs/gigaspeech/consensus_decoding_prod/`.

## What actually improves consensus
The production winner is **5-axis** directed future sampling (`top5-axis5`, 20 futures): it beats
the naive **futures=200 baseline** (`topk5`, 200 undirected futures) on COMET *and* BLEU at every
latency, while looser commitment (soft-vote) and brute-force scaling (100 futures) do not help —
the lever is *directed diversity*, not sample count or commit threshold. See
[[2026-06-consensus-axis5-vs-futures200]]. NB: the `top5-axis5` flagship is trained on the **old
ASR** (`asr_filtered`, prod root `J_40k`) — it is the canonical old-asr+QE baseline.

## ASR data quality dominates decode-method gains
Re-running the audio through a **new ASR** (Qwen-ASR + spaCy sub-sentence split) *regresses* trained
eval by −4–6 BLEU / −0.05 COMET vs the old-asr 5-axis baseline, across *both* the win3 and 5-axis
decode methods. The new ASR's chunk-start 句号 artifact is a red herring: period-fix recovers only
~+1.2 BLEU. The regression is an ASR/segmentation **data-quality** penalty, not a decode-method or
period-handling issue. See [[2026-06-qwenasr-asr-regression-periodfix]].

**Refinement (2026-07-11):** sentence-level forensics ([[2026-07-simul-tst-common-rescore]])
show much of the vs-hibiki gap is **candidate-distribution style, not future-blindness**:
on short fully-revealed sentences (future-blindness not binding) consensus still emits
correct-but-marked phrasings where hibiki emits the canonical rendering (its targets are
word-aligned frozen offline LLM translations). Post-edit of committed text remains dead
(below), but a canonical-register prior *inside* the commit loop is untested.

**Root cause found (2026-07-11, [[2026-07-consensus-register-forensics]]):** the SAME
Qwen3-30B-Instruct model produces hibiki's targets (per-sub-sentence greedy), the frozen
reference (whole-doc greedy), AND consensus's token distributions — the gap is purely the
decode procedure. The ≥75 % soft-vote selects *future-proof formal* tokens (因此/而/将
survive more continuations than 所以/和/把; voted body has 3× the 因此 of the same
utterances' free-completion tails), axis futures skew conditioning formal, and the
incremental prefix drifts off the model's greedy manifold. Consensus targets sit at only
59 char-BLEU / 0.48 4-gram overlap vs the frozen ref. Proposed fix: **rank-by-present** —
soft-vote keeps gating eligibility (no early commitment), but eligible tokens are ranked
by the no-future observed-source distribution.

## The BLEU gap is structural, not fixable by post-edit
Consensus is the only policy that is both ref-free AND **future-blind** (it commits incrementally
without ever seeing the rest of the source), so its ~5–8 BLEU deficit vs ref-based
hibiki/word-align and vs [[east]]/[[salami]] (which translate the whole source offline then
segment) is the honest cost of online translation. Attempts to recover that BLEU by LLM
post-editing all regress — see [[2026-06-consensus-post-edit-bleu]] (soft-vote, free
re-translation, free polish, strict minimal-edit polish all ≤ raw consensus). Conservatism is a
feature already at its fluency frontier.

## Related
- [[future-sampling]], [[majority-vote]], [[thinking-policy]], [[synthesis-pipeline]],
  [[2026-06-consensus-axis5-vs-futures200]], [[2026-06-consensus-post-edit-bleu]],
  [[2026-06-qwenasr-asr-regression-periodfix]], [[scoreboard]],
  [[2026-07-consensus-register-forensics]], [[2026-07-anchor-smoke500-sweep]] (anchor-and-veto
  fixes the register drift; strict-gate variant A → anchor_40k prod run).

## Sources
- code: `../codes/gigaspeech/future_sampling/consensus_decoding*.py`
