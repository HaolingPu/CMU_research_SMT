# EAST-low-only / Qwen3.6 / multiplier 1-12

## Status: cancelled before training (2026-09-07 14:04 UTC)

Haoling cancelled this ablation after the streaming-turn audit below showed that
7,631 of the 12,500 low-only examples (61.0 %) collapse to a single input chunk,
i.e. offline-style supervision. Jobs 10345494 (train), 10345495 (eval launcher) and
10345496 (eval gate) were cancelled by `scancel`; prepare job 10345493 had already
completed, so the converted manifest and audio remain on disk for a future
re-design with a smaller multiplier range. The cancellation is appended to the
run manifest (`status=cancelled_before_training`). No checkpoint or scores exist.

## Question and interpretation

User-authorized ablation, submitted 2026-09-07. Can 12,500 low-latency
reference-based EAST trajectories, regrouped over audio multipliers 1-12,
outperform the completed Qwen3.6 EAST-even model?

This is worth testing, not an established improvement. The latency band specifies
the teacher's translation/commit trajectory. The multiplier specifies how many
adjacent 960 ms audio chunks and associated target deltas become one training turn.
It does not turn a low trajectory into a medium/high translation policy.
Each utterance gets one deterministic multiplier during conversion, not a new
multiplier each epoch. Medium/high trajectories are not necessarily harmful.

| Configuration | Latency trajectories | Audio multiplier |
| --- | --- | --- |
| Completed EAST-even | low 4,166; medium 4,166; high 4,168 | low 1; medium 2; high 3-12 |
| New EAST-low-only | low 12,500; medium/high 0 | uniform integer 1-12 per utterance |

This comparison changes trajectory mix, training utterance membership, and the
chunk-size distribution. It is not a pure teacher-only or latency-band-only causal
ablation. Retain every score, including regressions and empty outputs.

## Data and frozen training

- Teacher: `/data/user_data/haolingp/models/Qwen3.6-35B-A3B-FP8`.
- Reuse the completed r2 EAST synthesis and unchanged MetricX threshold 3.0.
- Filtered source: `/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/qwen36_teacher_ablation/20260906-qwen36-teacher-r2/east/final_jsonl_east`.
- Historical audio/template pool: `/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_xl_case_robust_asr-filtered_zh-EAST-latency2mult.jsonl`.
- Compute-node audit found 21,403 unique filtered low trajectories; 13,311 also match the historical ungrouped low audio/message pool.
- Select exactly 12,500 unique low IDs from that eligible pool using seed 42. Replace historical targets with the filtered Qwen3.6 targets before regrouping.
- Use the existing per-ID CRC32 plus seed 42 NumPy policy for multiplier 1-12. Do not generate fresh teacher targets or rerun MetricX.
- Regroup original PCM16, mono, 16 kHz audio sample-exactly, without resampling; concatenate the corresponding Chinese deltas without introducing text. Preserve the historical system prompt and message roles.
- Keep the original `scripts/train/train_consensus_s.sh` byte-identical; Qwen3-Omni-30B LoRA, batch size, learning rate, epoch count, split, export and all other training arguments are unchanged.
- Frozen training SHA256: `7bc6668c1522612b69ebbec7860c4544b3bc768c15e65f1f61dacc6c3d1dc9aa`.

New files: `scripts/train/build_east_lowonly_manifest.py`,
`scripts/train/run_build_east_lowonly.sbatch`, and
`scripts/train/submit_east_lowonly_ablation.py`. Existing converter, training,
generation, and completed results are not modified by this addition.

## Run and jobs

Run/variant: `east-lowonly-mult1to12-qwen36-20260907T134907Z-n12500-seed42`.

Home run directory:
`/home/haolingp/slurm_runs/east-lowonly-mult1to12-qwen36-20260907T134907Z-n12500-seed42`.
Read `run_manifest.txt` for the latest IDs, `submission_configuration.json` for
script hashes, and `manifest_build_report.json` for completed conversion evidence.
The `code/` directory contains the submission-time runtime copies, including the
unchanged training script and tested evaluation launcher.

| Stage | Initial job | Dependency |
| --- | --- | --- |
| Validate/convert 12,500 rows, CPU only | 10345493 | none |
| Train/export, four L40S GPUs | 10345494_2 | successful prepare and terminal pilot 10345238 |
| Launch both evaluation datasets | 10345495 | successful new training and existing Simul evaluation gate 10329394 |
| Wait for both new evaluations | 10345496 | launcher terminates; inspect its child score jobs |

The existing Simul generation uses at most 16 GPUs. Its downstream stages are
sequential (MetricX 8, training 4, inference up to 16). Pilot 10345238 uses at most
8; new training 4 cannot overlap that pilot. New inference waits for the existing
Simul evaluation gate and new training. This preserves the total 24-GPU ceiling.
Do not raise throttles or add repair work without recounting all job families.
All new stages use preempt with the appropriate CPU/GPU QoS and requeue; the full
28-node exclusions are recorded in the run manifest.

The new training manifest is
`/data/group_data/li_lab/siqiouya/datasets/gigaspeech/manifests/train_s_zh-consensus-east-lowonly-mult1to12-qwen36-20260907T134907Z-n12500-seed42.jsonl`.
Audio is under
`/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/east_lowonly/east-lowonly-mult1to12-qwen36-20260907T134907Z-n12500-seed42`.
Checkpoint experiment name is
`gigaspeech-zh-consensus-east-lowonly-mult1to12-qwen36-20260907T134907Z-n12500-seed42-s-bsz4`.

## Verification and pending results

Seven tests passed on Babel in the SMT Python environment. These cover exact
low-only seeded selection, multiplier bounds, no-overwrite behavior, actual WAV
regrouping, idempotent resume, corrupt-output rejection, dependency construction,
both evaluation requests, GPU-family guards, and submission dry-run behavior.
The original low audio format was also inspected on a compute node.

Preparation 10345493 completed successfully in 1 minute 24 seconds. A separate
compute-node read verified the actual manifest has exactly 12,500 unique low IDs,
in the recorded selection order, seed 42, consistent audio/message counts, and
the expected hash. The frozen training snapshot also matches the current Babel
repository byte-for-byte. Training/evaluation are dependency pending; training
still waits for pilot 10345238. No new BLEU or latency result exists yet.

| Multiplier | Utterances |
| --- | --- |
| 1 | 1,063 |
| 2 | 1,002 |
| 3 | 1,047 |
| 4 | 1,041 |
| 5 | 1,050 |
| 6 | 1,021 |
| 7 | 1,039 |
| 8 | 1,063 |
| 9 | 1,079 |
| 10 | 1,019 |
| 11 | 1,071 |
| 12 | 1,005 |

There are 25,599 regrouped audio clips. The converter checked every output
waveform against the original samples and preserved every concatenated target.
Manifest SHA256:
`6326e1fa701a2f252a58e051fd7502b3e580db9e91bb35b44f953e726be360c8`.
Local evidence copies are under
`data_synthesis/outputs/teacher_baseline_audits/2026-09-07-east-lowonly-12500/`.

## Follow-up: streaming-turn collapse audit

The user challenged the latency/multiplier coupling on 2026-09-07. A separate
compute-node read of the two actual Qwen3.6 training manifests found:

| Dataset band | Rows | One input audio chunk | Median input chunks | All nonempty translation in final turn |
| --- | --- | --- | --- | --- |
| EAST-even low, multiplier 1 | 4,166 | 39 (0.9%) | 5 | 864 (20.7%) |
| EAST-even medium, multiplier 2 | 4,166 | 431 (10.3%) | 3 | 825 (19.8%) |
| EAST-even high, multiplier 3-12 | 4,168 | 2,027 (48.6%) | 2 | 2,480 (59.5%) |
| New low-only, multiplier 1-12 | 12,500 | 7,631 (61.0%) | 1 | 8,112 (64.9%) |

These are training-example counts, not held-out evaluation metrics. One audio
chunk means the whole training utterance is supplied before any translation;
this provides offline-style supervision for that example, although the utterance
need not be exactly one linguistic sentence. Multiple-input examples with only a
final nonempty target additionally demonstrate waiting, so the final-only column
is not identical to the one-chunk column.

The historical `convert2swift_east-mult.py` at commit 9604507 already uses
low1/medium2/high3-12. It was preserved for the teacher control, not chosen anew
as an optimal diversity schedule. The current audit also exposes a limitation of
the newly submitted low-only1-12 ablation: most examples lose intermediate
streaming turns. The initial submission checks verified count, identity and
audio/target preservation but omitted this distribution audit. Do not claim this
ablation is already a better streaming recipe.

Training 10345494_2 was still dependency-pending at this audit. No multiplier,
training argument or scheduling change was made in response to this question.
Consider smaller multipliers and utterance-length-aware bounds that preserve
multiple turns; merely reversing the latency-to-multiplier mapping is not proven
optimal. Confirm the user's intended range before changing an authorized run.

Evaluate ACL6060 and Simul-tst-COMMON with the existing Standard prompt and
960/1920/2880/3840 ms operating points. Report BLEU, chrF, XCOMET-XL,
LongYAAL CU and CA distinctly, and empty-output counts. Compare against the
completed EAST-even/Qwen3.6, historical reference-based baselines, and corrected
12.5K ambiguity model without hiding regressions. Do not add invented or partial
metrics to Google Sheets. The existing three-hour heartbeat monitors this run
alongside the unfinished Simul control and suffix-ICL pilot.
