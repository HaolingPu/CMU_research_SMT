---
title: Why consensus loses ~7 BLEU — register/canonicality forensics
type: experiment
tags: [consensus, bleu, register, forensics, word-level, synthesis]
sources:
  - ../codes/gigaspeech/future_sampling/consensus_decoding_token_id_level_instruct.py
  - ../codes/gigaspeech/future_sampling/build_llm_full_translation_cache.py
  - ../codes/gigaspeech/hibiki/code/translate_subsentences.py
  - ../outputs/gigaspeech/consensus_decoding_prod/J_40k/
created: 2026-07-11
updated: 2026-07-11
---

# Why consensus loses ~7 BLEU — register/canonicality forensics

Deep word-level analysis (test outputs on [[simul-tst-common]] + [[acl-6060]], training
targets in `J_40k`, full code trace of all consensus versions). Answers *where the BLEU
deficit vs hibiki physically comes from* and what to change.

## Headline: one model, three decoding modes

`Qwen3-30B-A3B-Instruct-2507-FP8` @ temperature 0 generates **all three** surface
distributions:

| data | decoding mode | style |
|---|---|---|
| frozen reference (`llm_reference_text`) | whole-doc greedy chat translation | plain, canonical |
| **hibiki targets** | **per-sub-sentence greedy** (rolling en context; `translate_subsentences.py`; alignment adds NO text) | plain, canonical |
| **consensus targets** | token-by-token soft-vote across ~40 adversarial futures, incremental prefix | **formal, verbose, non-canonical** |

So the gap is **entirely a property of the decode procedure**, not the translator model.
Consensus targets score only **59 char-BLEU vs the frozen reference** (median 59.6, p10
43.5; 4-gram overlap 0.48) — the two checkpoint families were trained toward surface
distributions ~40 BLEU apart, and the test-set gap is that divergence's shadow.

## Word-level evidence (trained-model outputs, seg3840)

- **n-gram compounding**: consensus's recall deficit vs hibiki is −4.7 % at char 1-grams
  but −9.9 % at 2-grams and **−19.2 % at 4-grams**. Content is preserved (COMET ties);
  sparse word swaps break every n-gram crossing them → big BLEU, no adequacy loss.
- **The substitution catalog is a spoken→written register shift**, mined automatically:
  所以→因此(×36), 是→正是, 可以→能够, 现在→如今, 而且→而, 和→与, 把→将, 去→前往,
  不是→并非, 可能→或许, 很多→许多, 说→回答. Consensus **never says 但是** (0 vs ref
  23.6/10k) and starves 所以/并且/这个/非常.
- **Replicates against human references**: ACL 6060 mean delta +4.96 sent-chrF (Simul-tst
  +4.93); length ratio consensus 1.05 vs hibiki 1.01 on both sets.
- Gap is tail-heavy (worst 10 % of sentences = 71 % of the mass) and worst on short
  sentences (+6.3 chrF at 9–15 chars), where hibiki verbatim-hits the canonical rendering.

## Mechanism (from the code trace)

All committed Chinese tokens come from the probe model's next-token distribution under a
bare "Translate the [INPUT] text into Chinese" prompt; English futures never emit Chinese.
Commit rule = two stages: an **eligibility gate** (flagship top5-axis5: strict 100 %
intersection — token must be in EVERY future's filtered top-K; the current file HEAD
post-c5647b7 defaults to the looser soft-vote `min_voters_ratio=0.75`, which is the `-sv`
variant that lost 2–4 BLEU) and a **winner rule** (argmax of mean/summed probability
ACROSS FUTURES). Register enters via:

1. **Both stages are future-driven.** Formal connectives (因此/而/将) stay grammatical
   under more continuations than colloquial ones, so they dominate the intersection (a
   fortiori under the strict 100 % gate), and the winner rule then picks the most
   future-averaged of them. Evidence: the voted body has 因此 12.5/10k vs only 4.4 in the
   *same utterance's* free-completion tail (same model, full visibility) — the
   vote machinery itself is the amplifier.
2. **Adversarial axis futures skew the conditioning text formal** — the 5-axis hints
   (named_actor / time_pivot / direct_speech / **factual encyclopedia-style aside**) push
   the English context toward narrative/encyclopedic register; the probe translates
   partly-hallucinated formal text.
3. **Off-manifold incremental prefix**: the committed prefix was assembled token-by-token
   under shifting futures — a string the model would never produce free-running; greedy
   continuation of it drifts to the model's "safe" formal basin.
4. The probe prompt has **no register guidance** (hibiki's per-sentence prompt implicitly
   yields the model's canonical rendering).
5. Final chunk = unconstrained greedy free-completion (~10 % of target chars) — measured
   NOT more formal than the body; not the main source.

## Fix plan (ranked)

1. **Anchor-and-veto (propose-from-present, futures-verify) — the root-cause fix.**
   Invert the roles. Per step: (a) greedy-decode an **anchor continuation** conditioned
   ONLY on observed source + committed prefix, using verbatim the sub-sentence translation
   prompt that generated hibiki's targets (`translate_subsentences.py:176-200`) — the
   anchor is on the canonical greedy manifold by construction, in hibiki's register;
   (b) teacher-force the anchor under each sampled future (one batched
   `prompt_logprobs` call per future, scores all anchor tokens at once); (c) commit the
   longest anchor prefix whose every token passes the gate (p ≥ min_p under 100 % of
   futures — keep the flagship's strictness); first failing token → READ. Futures never
   choose wording — they only decide **how far** to commit (the timing policy). This
   eliminates off-manifold drift (committed text is always the model's own greedy prefix),
   kills the future-averaged winner rule, and confines axis-hint style bleed to timing.
   Early commitment stays impossible (anticipatory tokens/。 get vetoed by continuing
   futures — the exact protection whose absence killed free re-translation in
   [[2026-06-consensus-post-edit-bleu]]). Also cheaper: ~1 anchor + N scoring calls per
   chunk vs up-to-12 sequential steps × N max_tokens=1 probes.
2. **Rank-by-present (fallback, ~20-line diff):** keep the existing intersection gate as
   eligibility but rank eligible tokens by the no-future distribution instead of the
   summed-over-futures score. Subset of 1; use if the anchor rewrite is too invasive.
3. **Prompt alignment is part of 1, not a separate hack:** the anchor's conditioning
   prompt is the sub-sentence translator prompt, so consensus draws from the SAME
   distribution hibiki's targets came from (same model + same prompt ⇒ same register).
4. **Validation protocol**: smoke 500 utts → leading indicators = char-BLEU vs frozen ref
   (expect 59→65+), 4-gram overlap (0.48→0.6+), marker-profile distance, len ratio →1.0,
   AND laal_text distribution (veto strictness must be tuned so latency matches top5-axis5,
   else the comparison confounds); then 40k + train + eval on [[acl-6060]] +
   [[simul-tst-common]]. Success = BLEU gap vs hibiki shrinks, COMET holds, latency matched.
5. Oracle-future diagnostic ([[2026-07-simul-tst-common-rescore]]) remains useful to bound
   any *residual* gap after 1.

## Related
- **Validated**: [[2026-07-anchor-smoke500-sweep]] executed fix plan #1's smoke protocol —
  strict-gate variant A wins +6.98 paired char-BLEU, register normalized; anchor_40k launched.
- [[consensus-decoding]], [[2026-07-simul-tst-common-rescore]],
  [[2026-06-consensus-post-edit-bleu]], [[comet-vs-bleu-ranking]], [[scoreboard]],
  [[gigaspeech]], [[qwen3-omni]].

## Sources
- probe/commit: `consensus_decoding_token_id_level_instruct.py:455-483` (probe prompt),
  `:1265-1323` (soft-vote), `:742-792` (axis hints), `_run_J_40k_common.sh:111-131` (prod args)
- frozen ref: `build_llm_full_translation_cache.py:105-111,161` → `..._frozen_llm_reference.tsv`
- hibiki text: `hibiki/code/translate_subsentences.py:176-200` → `llm_reference_text_list`
  → `contextual_alignment_final.py` (timing only) → `train_s_zh-hibiki.jsonl`
- analysis scripts: session scratchpad (`word_level.py`, `training_target_style.py`,
  `style_confound.py`, `register_analysis.py`)
