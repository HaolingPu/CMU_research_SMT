---
title: Ambiguity Future Set (coordinated plausible + contrastive futures)
type: concept
tags: [synthesis, future-sampling, prompt, ambiguity]
sources:
  - ../codes/gigaspeech/future_sampling/ambiguity_sampler_prompt.py
  - ../codes/gigaspeech/future_sampling/consensus_decoding_token_id_level_instruct.py
  - ../codes/gigaspeech/future_sampling/scripts/qwen38/serve_qwen38_gemma_colocated.sh
created: 2026-09-05
updated: 2026-09-05
---

# Ambiguity Future Set

Successor to the 5-axis sampler in [[consensus-decoding]] / [[future-sampling]]. Instead of
five hand-written narrative axes, one generic prompt (`future_set_v2_two_groups`) asks an
instruction-tuned sampler for a *coordinated* numbered list: N/2 **plausible** continuations
(the strongest local reading) and N/2 **contrastive** ones (realistic continuations that resolve
a lexical, syntactic, referential, discourse, or attachment uncertainty differently). Planning
the whole set in one response lets the model avoid duplicating itself. N = 20 per sampler;
two samplers (Gemma-4-E2B-it + Qwen3.8-27B) → ≤ 40 raw candidates per prefix.

## What the sampler sees
`future_source_prefix` = the current sentence unit only (`--future-source-window-chunks 1`),
plus the committed target text. The probe/translator always sees the full observed source.
At a sentence boundary the sampler prefix resets even if the previous clause is uncommitted —
the mechanism behind the case-71 gender error in [[2026-09-ambiguity-fsetv2-40k]].

## Filter after sampling (per model, per mode)
1. clean: strip think tags, markdown, leading `...`; strip a repeated observed prefix
   case-insensitively at a word boundary (prefix normalization).
2. validity: empty, >20 words, `*`/backtick, any CJK, or meta markers → `invalid_or_meta`.
3. `too_short` (<3 word tokens); `repeats_observed_prefix` (normalized source substring).
4. diversity: exact dup, ≤1 candidate per opening content word, Jaccard ≥ 0.65 → dropped.
The filter never checks grammaticality or on-topic-ness; those come only from the prompt.

## Measured behaviour (100-case bundle)
Keep rates: Gemma-4-E2B 70 % plausible / 72 % contrastive (mostly `too_short`), Qwen3.8
90 % / 88 %. Mean 32 selected futures per step. At ambiguity steps Qwen3.8 produced the
cue-carrying futures (e.g. 9 "her/she" futures vs 0 from Gemma in `AUD0000000003_1125`),
which is the capacity argument for a larger second sampler (Gemma-4 12B or 31B-it).

## Related
- [[2026-09-ambiguity-fsetv2-40k]] (production run), [[consensus-decoding]], [[future-sampling]],
  [[min-p-sampling]], [[majority-vote]], [[2026-06-consensus-axis5-vs-futures200]].

## Sources
- prompt: `../codes/gigaspeech/future_sampling/ambiguity_sampler_prompt.py`
- filter: `consensus_decoding_token_id_level_instruct.py` (`clean_future_text`,
  `is_valid_future_text`, `_sample_coordinated_future_set`, `select_diverse_futures`)
