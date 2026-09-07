# Boundary pilot and experiment-sheet entry (2026-09-06)

## Verified trained-model result (not the new boundary pilot)

Spreadsheet: https://docs.google.com/spreadsheets/d/1tzMr2y4P4HKR_Lt6QHjFcIrtuRzmWcATKaSuyYv2vYI/edit

On inspection, `ACL 6060 dev En-Zh` has existing results through row 53.
Suggested new section at row 54: `Ambiguity future consensus (Q38 + Gemma / Q36, frozen fsetv2)`.
Use rows 55-58 only if still empty when actually editing. This document is a draft;
the shared spreadsheet has not been edited.

| A: Data | B: Size | C: Chunk Size | D: LongYAAL | E: LongYAAL-BLEU | F: LongYAAL-XCOMET-XL |
|---|---|---|---|---|---|
| Ambiguity-fsetv2-Q38-Gemma-Q36-n12500-seed42 | 12.5K | 960 | 1431.1613 | 40.7779 | 0.7671 |
| Ambiguity-fsetv2-Q38-Gemma-Q36-n12500-seed42 | 12.5K | 1920 | 2126.2125 | 47.4101 | 0.7906 |
| Ambiguity-fsetv2-Q38-Gemma-Q36-n12500-seed42 | 12.5K | 2880 | 2756.1194 | 48.4794 | 0.8027 |
| Ambiguity-fsetv2-Q38-Gemma-Q36-n12500-seed42 | 12.5K | 3840 | 3243.0008 | 48.4153 | 0.7987 |

- D is LongYAAL (CU), in milliseconds. Lower is better.
- E is resegmented longform BLEU, not the chunk-level BLEU in H.
- F uses Unbabel/XCOMET-XL. Do not copy these scores into G/H (StreamLAAL family).
- G/H should remain empty until that separate scoring family is verified.
- Supplemental chrF at the four settings: 36.4538 / 40.4499 / 41.3071 / 41.6074.
- Empty predictions: 0 / 468 resegmented instances at every setting.
- Size means the 12,500-row training input manifest, not 40,000 synthesized utterances
  or the 17,306-row survivor pool. The unchanged recipe holds out 1% internally.
- Training: seed 42; unchanged `scripts/train/train_consensus_s.sh`; pretrained
  Qwen3-Omni-30B-A3B-Instruct, LoRA rank/alpha 32, batch 4, one epoch, lr 1e-4.
- Checkpoint: `gigaspeech-zh-consensus-ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831-n12500-seed42-s-bsz4/v2-20260906-174211-hf`.
- ACL scoring job 10333149, OmniSTEval 0.1.7, ACL6060 dev, zh tokenizer, char-level,
  468 resegmented instances. Source: checkpoint `evaluation/acl_6060/en-zh/seg*/segmentation_output/scores.tsv`.
- Boundary features are OFF for these trained-model results.
- Compared with the earlier 17,306-row run, BLEU increased but LongYAAL (CU)
  worsened at all four settings. This is not a matched-actual-latency comparison.
- Simul-tst-COMMON is a different evaluation set and needs its own tab/table.
  Its matched-run scores are pending verification; do not fill them with ACL scores.

The existing `Metrics` tab is for synthetic-data diagnostics. The single-case
char-BLEU/word-based LAAL values from the pilot belong in a separately labelled
pilot section, not in the trained-model ACL table. Do not invent aggregate
MetricX averages or equate 17,306 / 40,000 (combined-filter survival) with the
sheet's QE-only pass rate.

## Three-case development pilot

Purpose: isolate sentence-boundary completion and a punctuation-based sampler
anchor without modifying the frozen 40k method, training, or existing outputs.

| TSV row (zero-based) | Utterance | Reason |
|---|---|---|
| 10 | AUD0000000003_1015 | Boundary carry-over / invented location detail |
| 70 | AUD0000000003_1125 | Prefix-reset/pronoun uncertainty; antecedent is absent from the input TSV |
| 33 | AUD0000000003_1059 | Successful READ for `sat deep`; protect correct uncertainty handling |

Five runs per case, in order: baseline; sentence_end; sentence_anchor; both;
baseline_repeat. The repeated baseline uses the same sampler seed 1015 and is
a reproducibility check, not a fifth method. Total: 15 decoded case outputs.

- `sentence_end`: `--sentence-end-completion --sentence-end-boundary-mode conservative`.
- `sentence_anchor`: `--future-source-window-mode sentence-anchor`.
- `both`: the preceding flags combined.
- All variants retain space-only future joining. The sentence-aware join and
  until-closed flags remain OFF. No prior sentence is added in this pilot.
- Anchor input uses only observed text, never unseen parts of `src_text_full`.
  Commas do not reset it. A 128-word suffix cap protects prompt length.
- The conservative detector defers abbreviations, initials, ellipses and
  digit-final periods. It deliberately misses some true boundaries; record
  this limitation rather than calling it an exact sentence parser.
- Same teacher and samplers as the existing single-case harness, 20 futures
  per sampler, strict min-voters-ratio 1.0, one decoding case at a time.
- Pin runtime files with SHA-256 hashes. Preserve completed per-case JSONs and
  verbose traces on resume. Model servers are loaded once for the whole job.

Review every case's full source, source delta, sampler prefix, committed target
prefix, Chinese delta, omissions, unsupported additions, READ/WRITE decisions,
output length, char-BLEU and word-based LAAL. No cherry-picking the best variant
per case. Compare the repeated baseline before attributing small differences.

These three cases are deliberately selected and all from the same recording.
They cannot establish generalization or an optimal method. Select a candidate,
then validate on a broader held-out development sample before full synthesis.
Keep final evaluation datasets out of prompt/rule tuning. Any subsequent
training must use exactly 12,500 input rows, seed 42, and the frozen recipe.

## Runtime record

Run tag: `boundary-pilot-3cases-20260906-1823`.
Runtime/log directory on BABEL:
`/home/haolingp/slurm_runs/boundary-pilot-3cases-20260906-1823`.
Output directory on compute nodes:
`/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_pilots/boundary-pilot-3cases-20260906-1823`.
Submitted as job **10335059** at 2026-09-06 14:25 EDT (2 L40S GPUs, preempt_qos,
requeue enabled, three-hour limit). Started at 14:25:34 EDT on `babel-q9-16`;
latest check: RUNNING (startup), not completed.
The runtime manifest is `run_manifest.json` in that run directory; it contains
the four runtime file SHA-256 hashes, variant flags, exclusions and row IDs.
The remote checkout was pulled with `--ff-only` before staging isolated runtime
copies. Local decoder changes are uncommitted on top of `1a31f57`; no production
decoder file on BABEL was overwritten.

GPU-cap guard: EAST eval launcher 10329386 retains its original dependencies
on 10329385 and 10329241, plus `afterany:10335059`. This prevents the new
2-GPU pilot from overlapping the later 16-GPU evaluation plus 8-GPU generation
phase. Current ambiguity evaluation is not delayed by this guard. Recorded in
the root and EAST baseline manifests. Account usage at submission was 4 GPUs;
the pilot adds 2. Remove/replace the guard only after a fresh cap audit.

Verification before submission: 19 local unit tests passed, shell syntax passed,
all three TSV row IDs were verified on a compute node, new flags imported with
the real BABEL environment, and runtime hashes matched the local files.
