---
title: Ambiguity 12,500 - Simul-tst failure and training-filter audit
type: experiment
tags: [audit, consensus, repetition, filtering, domain, eval]
created: 2026-09-06
updated: 2026-09-06
---

# Ambiguity 12,500: Simul-tst failure audit

## Scope and conclusion

Read-only inspection of saved checkpoints, manifests, filter reports, raw
predictions and resegmented predictions on BABEL. CPU-only inspection allocation
10339976; no model inference, retraining, filtering rerun or score overwrite.

The observed failure is early erroneous output followed by failure to recover
from repetitive history. Long-form resegmentation amplifies the resulting
latency. The immediate mechanism is visible in outputs; the training change
that causes susceptibility is NOT established causally.

This is the ambiguity/future-consensus checkpoint, not either of the new
teacher-only EAST/Simul-MuST-C controls. Training used the 12,500-row seed-42
sample, with the original 1% validation split and unchanged Qwen3-Omni LoRA
recipe. The later sentence-boundary pilot was not part of its synthesis.

## Actual evaluation inputs

- ACL: 5 conference-presentation recordings, 468 reference segments. Actual
  filenames include `2022.acl-long.268.wav` and `2022.acl-long.367.wav`; references
  describe NLP papers. The existing [[acl-6060]] page calls them TED-derived;
  that description does not match these inputs. Both sets contain long audio:
  ACL recordings here span 577-737 seconds, so duration alone does not explain
  the failure.
- Simul-tst-COMMON: 27 TED recordings, 2,853 reference segments, our July 11
  diagnostic-grade monotonic-reference rebuild. This is not a paper-exact
  reproduction of the official references.
- All 27 raw output records exist at every chunk size. Raw/normalized prediction
  and timing lengths matched in the preceding audit, and CU emission times did
  not exceed recording duration. Missing jobs or a milliseconds/seconds mix-up
  do not explain the observed loops.

## Official full-set results

These are the stored `segmentation_output/scores.tsv` values, not filtered
diagnostic scores. Empty counts refer to resegmented reference sentences, not
missing talk files. A "shh-collapse talk" below means more than 100 occurrences
of U+5618 in the raw prediction; this detects one failure family, not all errors.

| Set | Chunk ms | BLEU | XCOMET-XL | LongYAAL CU ms | Empty segments | Shh-collapse talks |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 960 | 40.7779 | 0.7671 | 1431.1613 | 0 | 0 |
| ACL | 1920 | 47.4101 | 0.7906 | 2126.2125 | 0 | 0 |
| ACL | 2880 | 48.4794 | 0.8027 | 2756.1194 | 0 | 0 |
| ACL | 3840 | 48.4153 | 0.7987 | 3243.0008 | 0 | 0 |
| Simul-tst | 960 | 23.9180 | 0.8000 | 9976.2001 | 96 | 6 |
| Simul-tst | 1920 | 31.7957 | 0.8280 | 16478.1055 | 163 | 3 |
| Simul-tst | 2880 | 39.8659 | 0.8303 | 29549.3038 | 21 | 1 |
| Simul-tst | 3840 | 42.1952 | 0.8455 | 3774.5613 | 60 | 0 |

ACL has better BLEU than top5-axis5 but lower XCOMET at every chunk size; "works
well" must not be interpreted as improvement on every metric. The September
experiment wiki has older approximate ACL latency entries; this audit records
the files observed without replacing that historical table.

## Evidence for recovery failure

At 2880 ms on `ted_1375.wav`:

- Ambiguity emits the shh/exclamation pair at 2.88 s, then escalates to repeated
  30-character bursts. The full prediction is 10,000 characters, all shh pairs,
  through the 974-second recording.
- Hibiki also emits shh pairs initially, but switches to the translation of
  "Let's begin" at 20.16 s and continues translating the talk.
- EAST produces unrelated initial text, but also switches to "Let's begin"
  at 20.16 s and translates subsequent speech.
- Top5-axis5 produces initial filler, then recovers at 20.16 s.
- At 3840 ms, the ambiguity checkpoint itself produces an applause label at
  15.36 s and the translation of "Let's begin" at 19.2 s instead of the loop.

Thus the initial cue can confuse several models; persistence of the error is
the important difference. The acoustic identity of that initial cue has not
been manually annotated in this audit. Historical [[2026-07-anchor-smoke500-sweep]]
reports similar onomatopoeia loops on non-speech audio, but that is supporting
precedent, not proof of the precise trigger in this checkpoint.

The resegmented first reference in this talk has source duration 850 ms and five
Chinese characters ("Let's begin"). At 2880 ms it receives 9,256 hypothesis
characters with CU emission times reaching 907,040 ms. This explains extreme
alignment-derived latency without interpreting it as ordinary waiting.

`scripts/infer/infinisst_omni.py` appends every generated response to assistant
history and trims 60 cached chunks down to 30. The run uses 30 new tokens per
step and neutral repetition/frequency/presence penalties. Self-reinforcement is
a supported mechanism; a controlled history intervention is needed to establish
causality. No production inference settings were changed.

## Diagnostic subset, not a new result

For each chunk size, exclude the ambiguity model's shh-collapse talk IDs from
ALL compared models. Recompute sacreBLEU with `tokenize='zh'` from the saved
resegmented predictions/references. The all-sentence recomputation reproduces
every stored BLEU value. No re-alignment, file modification or retraining occurs.

Selection depends on the model's failures: these scores are post-hoc diagnostics
and MUST NOT replace the official full-set scores or be claimed as improvements.

| Chunk ms | Excluded talk IDs | Remaining sentences | Ambiguity diagnostic BLEU | Axis5 | EAST | Hibiki |
|---|---|---:|---:|---:|---:|---:|
| 960 | 0,1,7,8,10,13 | 2503 | 37.2918 | 26.9127 | 40.0622 | 38.1794 |
| 1920 | 5,14,23 | 2456 | 43.4849 | 31.8548 | 43.6125 | 40.1490 |
| 2880 | 23 | 2686 | 45.2845 | 34.0751 | 44.2480 | 40.6898 |
| 3840 | none | 2853 | 42.1952 | 34.2296 | 43.5802 | 41.0546 |

The failure is concentrated, not evidence that the model cannot translate TED
speech generally. BLEU can improve as fewer talks collapse while a single
extreme alignment worsens aggregate latency. The 3840-ms run still has 60 empty
resegmented sentences and other repetition; zero shh-collapse talks is not a
clean bill of health.

## Training and filtering audit

All 40,000 input rows are from the GigaSpeech `audio/audiobook/` directory.
This restriction exists BEFORE MetricX filtering: there are no podcast/YouTube
examples in this candidate pool for the filter to retain. All source transcripts
are nonempty; this audit did not identify explicit all-nonspeech training rows.

| Current saved training manifest | Audiobook IDs | Podcast IDs | YouTube IDs |
|---|---:|---:|---:|
| Ambiguity 12,500 | 12500 | 0 | 0 |
| Top5-axis5 | 12500 | 0 | 0 |
| EAST-even | 1522 | 4429 | 6549 |
| Simul-MuST-C-fixed-v2 | 3030 | 4411 | 5059 |

Historical Hibiki's currently referenced manifest is also audiobook-only, but
contains 12,872 rows today. Its historical training-time count was not verified
here, so do not silently relabel it as a confirmed exactly-12,500 control.

Domain coverage can contribute to EAST robustness, but it is not a sufficient
explanation: axis5 and Hibiki are also audiobook-only and do not exhibit these
large shh loops in this evaluation.

The complete MetricX report contains 40,000 scored utterances, zero missing
scores, 17,326 passes and 22,674 rejections. The length filter removes only 20
more (11 too short, 9 too long), leaving 17,306. Exactly 12,500 were sampled.

- The MAX-per-sentence filter keeps 53.19% of one-sentence utterances versus
  41.32% of five-sentence utterances. This is a measurable selection bias, not
  proof that it caused repetition.
- Mean source duration: QE passes 23.84 s; rejects 25.34 s. Both groups max out
  at 28.8 s. Training conversations have median 4 and maximum 30 assistant turns.
  Long-history error recovery is not directly exercised by these short episodes;
  this limitation also applies to the historical controls.
- Fixed-seed samples of 300 raw QE passes and 300 raw rejections have nearly
  identical empty-target fractions: 53.11% versus 53.05%. No repeated-shh loops
  were found in either sample or in the 20 length-filter rejections.
- Across all 40,000 aligned documents, a repeated-shh regex found zero matches.
  Keyword matches for music/laughter mostly describe narrated events, so those
  counts must not be presented as verified acoustic non-speech coverage.
- The actual 12,500 training manifest contains zero repeated-shh targets. It has
  19,545 empty assistant turns out of 84,896 (23.02%), versus axis5's
  13,737/85,205 (16.12%). Therefore "no WAIT examples" is contradicted by data.

For the 960-ms training subset only (matched chunk multiplier, not pooled across
different chunk sizes):

| Manifest | Empty turns | Mean chars per nonempty WRITE | Mean initial empty steps |
|---|---:|---:|---:|
| Ambiguity 12,500 | 53.36% | 9.04 | 2.73 |
| Ambiguity full 17,306 | 53.45% | 9.07 | 2.74 |
| Axis5 | 40.55% | 7.21 | 1.51 |
| EAST-even | 37.29% | 6.02 | 1.37 |

The new targets wait longer, then write larger bursts than axis5. This is a
training-distribution change worth testing, NOT proof that larger bursts cause
the loops. The 12,500 and full-pool distributions are almost identical on these
summaries; the random downsample did not introduce an obvious new domain or
chunk-size skew. Changed checkpoint robustness remains possible.

The current filters score textual translation quality and utterance-level
length ratio. They do not run the trained audio student through long non-speech
or recovery tests. Passing them does not certify streaming robustness.

## Existing reference-based evaluations

The wiki [[2026-07-simul-tst-common-rescore]] and actual checkpoint files agree:

| Historical method | Chunk ms | BLEU | chrF | XCOMET-XL | LongYAAL CU ms |
|---|---:|---:|---:|---:|---:|
| EAST-even | 960 | 40.1084 | 34.5336 | 0.7654 | 1106.8986 |
| EAST-even | 1920 | 43.7494 | 37.2448 | 0.8169 | 1899.7833 |
| EAST-even | 2880 | 44.2387 | 37.5477 | 0.8374 | 2619.9031 |
| EAST-even | 3840 | 43.5802 | 37.2173 | 0.8455 | 3242.9713 |
| Hibiki | 960 | 38.3151 | 33.6143 | 0.8376 | 1281.9657 |
| Hibiki | 1920 | 40.3642 | 34.6313 | 0.8608 | 1849.1934 |
| Hibiki | 2880 | 40.8242 | 35.0161 | 0.8655 | 2428.8514 |
| Hibiki | 3840 | 41.0546 | 35.0112 | 0.8691 | 2869.7214 |

No Simul-tst result was found for the historical
`gigaspeech-zh-Simul-MuST-C-fixed-v2-s_origin-bsz4/v1-20260601-005257-hf`:
all 15 weight shards exist, but its score files are ACL-only and it is absent from
the Simul-tst wiki table. This is scoped to the wiki and inspected checkpoint
root, not a claim that no copy exists anywhere. It needs inference/scoring only,
not retraining, if this missing historical comparison is requested.

The NEW Qwen3.6 teacher-only controls are separate. At this audit EAST training
10329385 was pending Priority after data preparation completed; Simul generation
10329388 was dependency-blocked. Launchers 10329386 and 10329393 already specify
both ACL and Simul-tst evaluations after their respective training stages.
Do not submit duplicate chains to obtain those forthcoming results.

## Next discriminating test

Keep training frozen at 12,500 rows, seed 42 and the existing LoRA recipe.
First run a separately labeled, small paired inference diagnostic on
`ted_1375`, one other failing TED talk and one ACL control, using one pinned
runtime for both checkpoints. Annotate the actual initial audio before calling
it silence/music. Test whether removing the erroneous assistant prefix permits
recovery, with all other inputs/settings fixed. Preserve original full-set
scores and never remove failing talks from the primary comparison.

Do not change production training, loosen the filter, add a repetition penalty,
or adopt sentence-boundary changes based on these correlations alone.

## Artifact locations

- Checkpoint root: `/data/user_data/haolingp/ckpts/infinisst-omni/`.
- Ambiguity checkpoint: `gigaspeech-zh-consensus-ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831-n12500-seed42-s-bsz4/v2-20260906-174211-hf`.
- Historical EAST: `gigaspeech-zh-EAST-even-s-bsz4/v0-20260601-001001-hf`.
- Historical Hibiki: `gigaspeech-zh-hibiki-s-bsz4/v0-20260601-121643-hf`.
- Per-checkpoint outputs: `evaluation/{acl_6060,simul_tst_common}/en-zh/seg<N>/instances.log` and `segmentation_output/{scores.tsv,instances.resegmented.jsonl}`.
- Postprocessing root: `/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod/ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831-segale`.
- Full filter reports: `metricx-aligned/filter_report_per_utt.jsonl` and `metricx-aligned/length_ratio_report.jsonl`; aligned texts: `aligned_all.jsonl`.
- Input TSV: `/data/group_data/li_lab/haolingp/consensus_handoff/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv`, rows 0-39999.
- Training recipe: `scripts/train/train_consensus_s.sh`; conversion: `scripts/train/convert2swift_consensus.py`; QE filter: `data_synthesis/codes/gigaspeech/future_sampling/filter_consensus_by_metricx_qe_per_sentence.py`.
