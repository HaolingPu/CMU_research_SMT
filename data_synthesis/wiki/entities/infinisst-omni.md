---
title: InfiniSST-Omni (agent + checkpoints)
type: entity
tags: [model, agent, checkpoints]
sources:
  - scripts/infer/infinisst_omni.py
  - scripts/infer/ckpts.txt
created: 2026-06-01
updated: 2026-06-27
---

# InfiniSST-Omni (agent + checkpoints)

Two linked things under one name:

1. **The agent** `scripts/infer/infinisst_omni.py` — the [[simuleval]] `SpeechToTextAgent`
   wrapping [[vllm]] for [[streaming-inference]] (prompt types in [[east-prompt-handling]]).
2. **The checkpoint family** trained by [[megatron-swift]] from [[qwen3-omni]], stored at
   `ckpts/infinisst-omni/<exp>/v<N>-<date>-hf/` and listed in `scripts/infer/ckpts*.txt`.

Notable trained variants (en→zh): `s_origin` (= word-align baseline, sees the reference),
`hibiki`, EAST / refined-EAST / EAST-latency2mult, Simul-MuST-C, LA, PA, and consensus-top5
families. Latest zh as of 2026-06-01: `gigaspeech-zh-consensus-top5-axis5-fut100_n100-s-bsz4`.
New-asr regression runs (2026-06): `gigaspeech-zh-consensus-FULL40k-win3[-nopfix]-s-bsz4` and
`…-top5-axis5-qwenasr[-fixed]-s-bsz4` — all below the old-asr `top5-axis5` baseline, see
[[2026-06-qwenasr-asr-regression-periodfix]].

~22 trained experiments live under `ckpts/infinisst-omni/<exp>/v<N>-<date>-hf/`; each holds the
HF weights plus an `evaluation/acl_6060/` tree of `scores.tsv`. Measured performance for all of
them is tabulated in the **[[scoreboard]]**.

## Related
- [[scoreboard]], [[streaming-inference]], [[checkpoint-evaluation]], [[megatron-swift]],
  [[qwen3-omni]], [[la-n-vs-wait-k]], [[2026-06-qwenasr-asr-regression-periodfix]].

## Sources
- code: `scripts/infer/infinisst_omni.py`; checkpoints `ckpts/infinisst-omni/`; list `scripts/infer/ckpts*.txt`
