---
title: Qwen3.5-122B Future Sampler
type: entity
tags: [model, sampler, vllm, serving, scaling]
sources:
  - ../codes/gigaspeech/future_sampling/serve_qwen35_122b.sbatch
  - ../codes/gigaspeech/future_sampling/serve_qwen35_122b_a100.sbatch
  - ../codes/gigaspeech/future_sampling/serve_qwen35_122b_a100_dp.sbatch
  - ../codes/gigaspeech/future_sampling/scripts/qwen35/
  - measured DONE.txt + per-case usage, outputs/gigaspeech/consensus_decoding_debug/qwen35/ (2026-06-17..28, 2026-07-11)
created: 2026-07-10
updated: 2026-07-11
---

# Qwen3.5-122B Future Sampler

`Qwen3.5-122B-A10B-FP8` (MoE, 10B active, hybrid Gated-DeltaNet attention; weights ~116 GB)
served by [[vllm]] as the **local replacement for the paid GPT/DeepSeek future sampler** in
[[consensus-decoding]] / [[future-sampling]]. Decoder path: `--sampler-backend chat` in
`consensus_decoding_token_id_level_gpt.py`; run scripts in `scripts/qwen35/`.
**Thinking must stay ON** (the prompt asks for adversarial futures); server needs
`--reasoning-parser qwen3` so CoT lands in `reasoning_content` and `content` stays parseable.
`SAMPLER_PROMPT_FORMAT=numbered` (default) already cuts visible tokens ~4× vs `json`.

## Code map — where everything lives

All paths relative to `../codes/gigaspeech/future_sampling/`:

| What | Where |
|---|---|
| serve the 122B sampler | `serve_qwen35_122b.sbatch` (4×L40S TP=4) / `serve_qwen35_122b_a100.sbatch` (2×A100) — vLLM, port 8300, `--reasoning-parser qwen3`, MAX_LEN default 16384 |
| client sbatch (test) | `scripts/qwen35/run_qwen35_general.sbatch` — pass `LABEL`, `TOTAL_ROWS`, `SAMPLER_API_BASE` |
| client sbatch (40k) | `scripts/qwen35/run_qwen35_40k_array.sbatch` — 8-task array, endpoint auto-discovery |
| orchestration + all hyperparams | `scripts/qwen35/run_qwen35_common.sh` — env-var defaults at lines 42–71, serves the local 1-GPU instruct translator, preflights the sampler, builds the decoder CMD (lines 171–195), writes DONE.txt |
| decoder | `consensus_decoding_token_id_level_gpt.py` |
| **the LLM call** | `sample_source_futures_chat()` at `consensus_decoding_token_id_level_gpt.py:1102` — POSTs `{SAMPLER_API_BASE}/chat/completions`; futures prompt built by `build_gpt_future_sampling_input()`, numbered output parsed by `parse_method_a_output()` |
| max_tokens math | `compute_gpt_sampler_max_output_tokens()` at `:857` — reasoning budget by effort (minimal 1k / low 2k / **medium 6k** / high 20k) + visible `max(ft·nf·6, 200·nf+400)`; truncation retry doubles, clamped by `clamp_sampler_max_tokens()` (`:881`) to env `SAMPLER_MAX_MODEL_LEN` (default 16384) |

## Hyperparameters (validated c64 test = these defaults unless noted)

| Param | Value | Set where |
|---|---|---|
| `NUM_FUTURES` / `FUTURE_TOKENS` | 20 / 20 | run_qwen35_common.sh:42–43 |
| `MAX_CONSENSUS_STEPS` | 32 (→ ~24 sampler calls/row) | :44 |
| `CANDIDATE_TOP_K` / `MIN_P` / `TOP_P` | 5 / 0 / 0 | :46–48 |
| `NUM_CONCURRENT_CASES` | default 8; **test used 64** | :45, override at sbatch |
| `TARGET_LANG` | Chinese | :49 |
| thinking | ON — `chat_extra_body={"chat_template_kwargs":{"enable_thinking":true}}` (`ENABLE_THINKING=1`) | :58–66 |
| reasoning effort (chat path) | **hardcoded "medium"** → 6000-token CoT budget → per-call max_tokens = 6000+4400 = **10,400** | gpt.py:1140 |
| `SAMPLER_PROMPT_FORMAT` | numbered (4× fewer visible tokens than json) | :71 |
| temperature | `--sample-temperature` default 1.0, sent in payload (vLLM applies it; the "reasoners ignore it" comment is about DeepSeek) | gpt.py:201,1144 |
| `GPT_API_TIMEOUT` | default 600 s; test used 1800 | :75, override |
| instruct translator | Qwen3-30B-A3B-Instruct-2507-FP8, 1 L40S, local port 8200+task·10 | run_qwen35_common.sh |

## FOOTGUN: server max-model-len must be ≥ 16384

The chat decoder requests `max_tokens ≈ 10.4k` per futures call (6000 reasoning budget +
visible, nf=20/ft=20). If the server's `--max-model-len` is below that (old default was 8192),
**every sampler call 400s and the decoder silently falls back to futureless instruct-only
decoding** — outputs still look plausible and DONE.txt still writes, but
`gpt_sampler_usage.calls=0` (a 100-row "run" finished in 203 s this way on 2026-07-11).
Fixes now in place: all three serve scripts default `MAX_LEN=16384`, and
`scripts/qwen35/run_qwen35_common.sh` preflights a real request at the decoder's full
max_tokens and hard-fails on non-200. **Validity check for any run:** confirm
`gpt_sampler_usage.calls > 0` in the output JSONs before trusting it.

## Serving configs

All serve scripts request only **4 CPU / 32 GB** (GPUs are the scarce resource on babel;
fat CPU/mem requests stall scheduling) and **self-register their endpoint** under
`outputs/gigaspeech/qwen35_endpoints/<jobid>.txt`; the 40k client array auto-discovers and
health-filters that dir, so replicas can come and go (preemption-safe).

| Script | GPUs | Role |
|---|---|---|
| `serve_qwen35_122b_a100.sbatch` | 2× A100_80GB TP=2 (NVLink) | **primary scale-out unit** — submit N times; validated boot (job 8812769, CUDA graphs OK; died to preemption only) |
| `serve_qwen35_122b.sbatch` | 4× L40S TP=4 (PCIe) | **fallback scale-out unit** — L40S is the most available GPU (50 nodes); needs NCCL P2P off; compiled works, eager ~13 tok/s |
| `serve_qwen35_122b_a100_dp.sbatch` | 8× A100_80GB, TP=2 × DP=4 | whole-node variant (hard to schedule); `SERVE_MODE=multi` = 4 servers on ports 8300–8303 |

## Measured cost (10-row runs, old-ASR frozen TSV, nf=20, concurrency 8, 4×L40S server)

| Mode | s/row | sampler out tok/row | calls/row |
|---|---|---|---|
| thinking ON | **480–548** | **~150k** (max 208k) | ~25 |
| thinking OFF | **24.3** | ~6.4k | ~25 |

Thinking = **24× token blowup** (~6k CoT per futures call; chat path caps max_tokens at
6000 + visible ≈ 10.4k). Aggregate server throughput at concurrency 8 was only ~315 tok/s
— far below saturation for a 10B-active MoE; **client concurrency is the first free lever**.
nf=10 barely helps (469 s/row): thinking is per-call and the call count doesn't drop.

## VALIDATED: concurrency 64 saturation test (2026-07-11, jobs 9213060/9213089)

100 rows, thinking ON, nf=20, `NUM_CONCURRENT_CASES=64`, one 4×L40S TP=4 server
(MAX_LEN=16384). Output: `consensus_decoding_debug/qwen35/qwen35_sat_c64_100_16k/`.

| Metric | Value |
|---|---|
| wall / effective s/row | 20,510 s / **205 s/row** (2.7× faster than c=8's 546) |
| server generation tok/s | **888 median** (p90 960, max 1158); aggregate 780 incl. ramp/tail |
| sampler out tok/row | 160k mean (median 168k, max 205k); ~24 calls/row |
| quality | unchanged — same calls/steps per row as the c=8 run on the 8 overlapping rows; char-BLEU diffs are surface-form variance ([[comet-vs-bleu-ranking]]) |
| headroom | server pinned at Running=64 the whole run, KV cache peak 70.5% → **client concurrency was the binding limit**; c=96–128 worth a probe |

**Verdict (user, 2026-07-11): still not good enough.** 205 s/row → 95 replica-days for 40k
is dominated by the 6k-token "medium" CoT budget per call × ~24 calls/row. Open levers, in
order of expected payoff: (a) cut reasoning effort medium→low (6k→2k budget, gpt.py:857 —
needs a 10–100 row quality check per [[comet-vs-bleu-ranking]]); (b) raise client concurrency
past 64; (c) fewer sampler calls/row (lower MAX_CONSENSUS_STEPS or call every k-th step);
(d) more/faster replicas (A100 TP=2 untested).

## 40k-row scaling math (old-ASR data, TOTAL_ROWS=40000 of the frozen TSV)

40k × ~160k tok ≈ **6.4B sampler output tokens**. Measured: one 4×L40S replica (5 GPUs
with its 1-L40S client) does 205 s/row → 40k on 1 replica ≈ **95 days**; N replicas ≈ 95/N
days (4→24 d, 8→12 d, 16→6 d). L40S farm math is now measured, not estimated. A100 TP=2
replicas remain untested for throughput (job 9213010 never scheduled; all 4 A100 preempt
nodes allocated). Plan: **replica farm + saturated batching** —

1. Submit 8–12× `serve_qwen35_122b_a100.sbatch` (2-GPU jobs schedule far more easily than
   whole nodes; 16–24 GPUs ≤ preempt_qos cap of 24). If A100 pairs won't schedule, fall back
   to N× `serve_qwen35_122b.sbatch` on L40S (most available GPU: 50 nodes; 2 on general +
   up to 6 on preempt). Each replica self-registers its endpoint.
2. Clients: `scripts/qwen35/run_qwen35_40k_array.sbatch` — 8 array tasks (array_qos, 1 L40S
   instruct each), 5000 rows/task, `NUM_CONCURRENT_CASES=64`, live endpoints auto-discovered
   from the endpoint dir and sharded by task id. 512 in-flight cases ≈ 40–64 per replica.
3. At an estimated 1.5–3k tok/s per batched A100 TP=2 replica → 12–24k tok/s farm-wide →
   **~3–6 days wall**, preemption-safe (`--requeue` + `--skip-existing`, stateless servers).
4. Optional 2–3× multiplier: cap thinking (lower the 6000-token reasoning budget /
   "think concisely" prompt) — must pass a 10-row quality check first, since
   [[comet-vs-bleu-ranking]] rules apply to any quality comparison.

H100 (3×8, preempt, usually allocated) and H200 (1×4, 141 GB → TP=1 replicas possible) are
faster per GPU but too contended to plan around; grab opportunistically.

Follow [[babel-cluster]] rules: `--time` ≤ partition MaxTime, task-id-based ports, keep
short finalize steps off preempt.

## Sources
- [[future-sampling]], [[consensus-decoding]], [[vllm]], [[babel-cluster]], [[gigaspeech]]
- code: `../codes/gigaspeech/future_sampling/serve_qwen35_122b*.sbatch`, `scripts/qwen35/`
