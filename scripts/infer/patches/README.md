# OmniSTEval reference-normalization fix (BLEU scoring)

`omnisteval-0.1.7-nfkc-reference.patch` fixes an asymmetry in the longform scorer used by
`eval_all_ckpts.sh` and `eval_all_ckpts_simultst.sh` (OmniSTEval 0.1.7,
https://github.com/pe-trik/OmniSTEval, installed in editable mode from
`/data/user_data/haolingp/code/OmniSTEval` on BABEL).

## The bug

- OmniSTEval applies Unicode NFKC to every hypothesis (`omnisteval/data.py:338`; the repo's
  `normalize_instances.py` mirrors this so the per-character timing arrays stay aligned), but it reads
  the reference file raw (`data.py:146`, `io.py:49`).
- NFKC maps full-width punctuation `，？！：；（）` to ASCII `,?!:;()` (`。` is left alone). Every model
  writes full-width commas, so the scorer compared `,` in the hypothesis against `，` in the reference,
  and sacrebleu's `zh` tokenizer counts those as different tokens: each comma became a unigram miss plus
  broken 2/3/4-grams around it. One comma in a 14-character sentence costs 22 BLEU points on that
  sentence.
- The penalty scaled with how many commas a system writes, so it was unequal: once fixed, BLEU rises by
  about +6 for Hibiki, +3 to +5 for EAST-even and Simul-MuST-C, and +7 to +8 for the consensus models
  on Simul-tst COMMON (about +4.5 / +2 to +4 / +5.5 on ACL 60/60). XCOMET, chrF-independent latency
  metrics and the resegmentation itself are unaffected (the resegmenter already normalizes both sides).

## The fix

Reference lines are passed through the same `unicode_normalize` (NFKC) as the hypothesis in
`data.py` (`load_reference`, `load_text_segmentation`) and `io.py` (`load_shortform_instances`).

Apply it to the editable checkout (no reinstall needed):

```bash
git -C /data/user_data/haolingp/code/OmniSTEval apply /path/to/omnisteval-0.1.7-nfkc-reference.patch
```

Applied on BABEL on 2026-09-14 at 13:29 as a working-tree change on OmniSTEval `main` (5a6cc9b); every
evaluation run after that time writes symmetric BLEU into `segmentation_output/scores.tsv`. Verified on
Hibiki w0.2, Simul-tst COMMON seg1920: BLEU 40.53 -> 46.61, chrF 34.54 -> 40.20, every latency metric
byte-identical; a full re-score with XCOMET on both test sets reproduced XCOMET to 4 decimals.

## Re-scoring evaluations made before the fix

`segmentation_output/scores.tsv` files written before 2026-09-14 13:29 hold the asymmetric BLEU. No
inference is needed to correct them: re-run the scorer on the existing `instances.normalized.log` into a
separate folder (COMET omitted because it does not change), e.g. for Simul-tst COMMON:

```bash
omnisteval longform \
  --speech_segmentation /data/user_data/haolingp/datasets/simul_tst_common/tst.yaml \
  --source_sentences_file /data/user_data/haolingp/datasets/simul_tst_common/tst.en \
  --ref_sentences_file /data/user_data/haolingp/datasets/simul_tst_common/tst.zh \
  --hypothesis_file <seg dir>/instances.normalized.log --hypothesis_format jsonl \
  --lang zh --char_level --bleu_tokenizer zh \
  --output_folder <seg dir>/segmentation_output_nfkc
```

The 2026-09-14 checkpoints (Hibiki, EAST-even, Simul-MuST-C-v2, consensus ambiguity-v2, v3json-boundary
and oracle-probe, at no-fix / w0.5 / w0.2) were re-scored this way into `segmentation_output_nfkc/` on
both test sets; take BLEU and chrF from there and XCOMET from the original `segmentation_output/`.
