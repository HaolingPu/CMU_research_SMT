# Wiki Log

Append-only. See [[WIKI.md]] for format.

## [2026-06-01] init | wiki bootstrapped
- added: [[future-sampling]], [[consensus-decoding]], [[gigaspeech]], [[la-n-vs-wait-k]], [[comet-vs-bleu-ranking]]
- seeded from existing memory facts

## [2026-06-01] ingest | codes/gigaspeech (synthesis side)
- source: `../codes/gigaspeech/` (synthesis methods + pipeline), mapped via read-only scan
- added: [[synthesis-pipeline]], [[east]], [[salami]], [[thinking-policy]], [[majority-vote]], [[segale-alignment]], [[min-p-sampling]], [[metricx]], [[mfa]], [[vllm]]
- updated: [[future-sampling]] (b1/b2/final/thinking variants), [[consensus-decoding]] (dualbase, token-id-level variants), [[gigaspeech]] (manifest + downstream links)
- contradictions: none

## [2026-06-01] ingest | scripts/ (training/inference/eval side)
- source: `scripts/` (train/infer/debug), mapped via read-only scan
- added: [[streaming-inference]], [[latency-quality-tradeoff]], [[checkpoint-evaluation]], [[east-prompt-handling]], [[dataset-conversion-pipeline]], [[megatron-swift]], [[simuleval]], [[qwen3-omni]], [[acl-6060]], [[infinisst-omni]]
- updated: [[la-n-vs-wait-k]] (trained LA_s/PA_s counterparts), [[comet-vs-bleu-ranking]] (eval pipeline links)
- contradictions: none

## [2026-06-01] ingest | ckpts/ (trained models + performance)
- source: `ckpts/infinisst-omni/` — 22 trained experiments + their `evaluation/.../scores.tsv` (deterministic bash extract, not transcribed)
- added: [[scoreboard]] (COMET/BLEU/chrF/LongYAAL for ~20 evaluated checkpoints across zh/ja/de × 4 seg)
- updated: [[infinisst-omni]] (ckpts layout + scoreboard link), [[checkpoint-evaluation]] (results location), [[comet-vs-bleu-ranking]] (live BLEU≠COMET example), raw/README (ckpts as referenced source)
- contradictions: none. Note: LA-40k-s and LA-40k-seg13 have no eval output at the standard path.

## [2026-06-07] ingest | Consensus "other approaches" — 5-axis vs soft-vote vs scaling vs futures=200
- source: trained ACL6060 en-zh eval of `consensus-{topk5, top5-axis5, top5-axis5-sv, fut100, fut100_n100}` ckpts; `scripts/train/convert2swift_consensus.py` (baseline provenance: futures200-segale qe3-lr-aligned-full → topk5)
- added: [[2026-06-consensus-axis5-vs-futures200]]
- updated: [[consensus-decoding]], [[future-sampling]], [[scoreboard]], [[comet-vs-bleu-ranking]], [[min-p-sampling]], [[majority-vote]], [[latency-quality-tradeoff]], [[2026-06-consensus-post-edit-bleu]]; index.md Experiments section (also indexed the previously-unlisted post-edit page)
- contradictions: none. Finding: directed diversity (5-axis, 20 futures) is the consensus quality lever — beats 200 undirected futures; soft-vote (looser commit) and 100-future scaling both fail to help.

## [2026-06-20] ingest | Babel HPC Cluster (SLURM usage + job submission)
- source: https://wiki.babel.cs.cmu.edu/index.php/BABEL (user-pasted; SSO-gated) + live `sinfo`/`scontrol show partition`/`sacctmgr show qos`
- added: [[babel-cluster]] (partitions+MaxTime, QoS limits, GPU gres types, storage/quota, AutoFS gotcha, job-submission rules)
- updated: index.md (Entities/Tools); [[megatron-swift]], [[streaming-inference]], [[checkpoint-evaluation]], [[synthesis-pipeline]] (link [[babel-cluster]])
- contradictions: none. Motivated by two real mistakes this session — `--time=3d` on `general` (max 2d → PartitionTimeLimit stall) and preempt evictions breaking an afterok chain.

## [2026-06-27] ingest | new-asr regression + period-fix ablation
- source: ckpts `consensus-FULL40k-win3[-nopfix]`, `top5-axis5[-qwenasr[-fixed]]`; codes `period_fix_traj_nested.py`, `submit_J40k_post.sh`, `convert2swift_consensus.py`
- added: [[2026-06-qwenasr-asr-regression-periodfix]]
- updated: [[consensus-decoding]], [[2026-06-consensus-axis5-vs-futures200]], [[scoreboard]], [[segale-alignment]], [[metricx]], [[dataset-conversion-pipeline]], [[gigaspeech]], [[infinisst-omni]], [[index]]
- key facts: 5-axis `top5-axis5` = canonical OLD-asr+QE baseline; new qwenasr regresses -4..-6 BLEU / -0.05 COMET across win3 AND 5-axis methods; controlled period-fix ablation (identical 12,446 instances) = +1.2 BLEU avg only; QE-MAX rescues collapse but not the ASR gap
- contradictions: corrects the earlier working hypothesis that the chunk-start 。 artifact was the regression's main driver — ablation shows it is minor; ASR/segmentation data quality is the root cause

## [2026-06-28] ingest | split-fix (④) result + boundary-leakage root cause
- source: ckpt `consensus-FULL40k-win3-splitfix-s-bsz4` (eval 8857769); codes `split_src_text_full_spacy.py` (`_merge_short`); 401-utt old-vs-new ASR audit
- updated: [[2026-06-qwenasr-asr-regression-periodfix]], [[index]]
- key facts: split-fix (_merge_short min_words=2) recovers +0.7/+2.5 BLEU & +.019 COMET at seg2880/3840 (narrows seg3840 −5.9→−3.4) but leaves −3.4..−4.6 vs 5-axis baseline; residual gap is NOT transcription (qwenasr WER ~2.7% vs old, better casing) — it is segmentation boundary leakage: ~1.47x finer mid-clause cuts, HEAD-leak 15% / TAIL-leak 21% dangling fragments, 9.7% adjacent-pair 2-word boundary duplication; min_words=2 only kills visible 1-word frags, mid-clause cut+dup survive
- next: re-segment on sentence-final punctuation only (drop mid-clause cc-cuts) or merge ≤3-word boundary frags + de-dup overlap
- contradictions: none (extends the page; refines finding #4 from "audit data quality" to the specific boundary-leakage mechanism)

## [2026-06-30] ingest | clause-split ⑤b trained-eval result (negative)
- source: ckpts/.../gigaspeech-zh-consensus-FULL40k-win3-clausesplit-s-bsz4 (v0-20260630-060135), eval 8882005
- updated: [[2026-06-qwenasr-asr-regression-periodfix]] (added ⑤b row 29.4/34.3/34.2/33.1, clause-split section, refuted granularity hypothesis, findings 5-7 rewritten, new "Open: win1 vs win3" section), [[index]]
- finding: clause-split matched old-asr granularity (5.3 vs 5.1 units/utt) yet regressed −1..−3.6 BLEU vs spaCy split-fix and stayed −5.5..−7.0 vs ① → segmentation/granularity refuted as root cause; last confound is decode window win1(①) vs win3(new-asr)
- contradictions: corrects prior finding #6 (which said re-segment on sentence-final punct would close the gap) — it did not
## [2026-07-10] query+backfill | Qwen3.5-122B sampler: 40k scaling
- question: how to run the Qwen3.5-122B thinking future sampler over the 40k old-ASR rows, faster/scalable
- measured: thinking = ~150k sampler-output tok/row (~25 calls × ~6k CoT) vs 6.4k no-think; 4×L40S server only ~315 tok/s aggregate at concurrency 8 → 40k ≈ 6.1B tokens ≈ 225 server-days as-is
- added: [[qwen35-122b-sampler]] (serving configs, cost table, A100 TP=2×DP=4 farm plan, ~3–6 day estimate)
- new code: `serve_qwen35_122b_a100_dp.sbatch` (8×A100 farm, dp/multi modes), `scripts/qwen35/run_qwen35_40k_array.sbatch` (8-task array client, endpoint sharding)
- updated: [[future-sampling]], index
- contradictions: none
- 2026-07-11: added [[simul-tst-common]] (NAIST monotonic eval set rebuild: src 103/107, GPT drift → ~27% fidelity, MFA yaml); caveat added to [[comet-vs-bleu-ranking]] (hibiki COMET tied, not won)

## [2026-07-11] ingest | Simul-tst-COMMON re-scoring results (jobs 9214519-22/9214529)
- added: [[2026-07-simul-tst-common-rescore]]
- updated: [[simul-tst-common]] (results section replaces "how to read"), [[comet-vs-bleu-ranking]] (open question answered: deficit real), index.md (fidelity 27%→37%, new experiment line)
- contradictions: none — prediction "EAST-even should crater" was falsified and is noted as such, not overwritten

## [2026-07-11] update | qwen35 sampler c64 saturation test validated (jobs 9213060/9213089)
- updated: [[qwen35-122b-sampler]] — new FOOTGUN section (server max-model-len ≥ 16384, else silent futureless fallback with calls=0; preflight guard now in run_qwen35_common.sh), validated c64 measurements (205 s/row, 888 tok/s median, 160k out tok/row, quality unchanged), 40k projection now measured: 95/N days for N× 4-L40S replicas
- contradictions: none — old "225 days @315 tok/s" estimate superseded by measurement, kept as the c=8 baseline row

## [2026-07-11] update | qwen35 sampler: code map + hyperparameters + verdict
- updated: [[qwen35-122b-sampler]] — added code-map table (serve/client scripts, LLM call site sample_source_futures_chat at consensus_decoding_token_id_level_gpt.py:1102, max_tokens math at :857), full hyperparameter table (nf=20, ft=20, max_steps=32, top_k=5, thinking ON, effort=medium hardcoded, temp=1.0, numbered format), and the user's verdict that 205 s/row is still not good enough with ranked speed levers (effort medium→low first)

## [2026-07-11] ingest | Sentence-level forensics: consensus vs hibiki on Simul-tst-COMMON
- added: (section in) [[2026-07-simul-tst-common-rescore]]
- updated: [[consensus-decoding]] (refinement: gap ≈ candidate-style, not only future-blindness)
- contradictions: partially contradicts the pure "structural future-blind cost" framing — both claims kept; post-edit negative results stand, in-loop canonical prior is the new untested lever

## [2026-07-11] ingest | Deep register forensics: root cause of consensus BLEU deficit
- added: [[2026-07-consensus-register-forensics]]
- updated: [[consensus-decoding]] (root-cause section), index.md
- contradictions: refines (does not overturn) the "structural future-blind cost" claim — post-edit negatives stand; the deficit is now attributed to the soft-vote's formal-register selection + off-manifold prefix drift, with a concrete in-loop fix proposed (rank-by-present)

## [2026-07-12] ingest | Anchor-and-veto smoke sweep results (A/B/C/D)
- added: [[2026-07-anchor-smoke500-sweep]]
- updated: [[2026-07-consensus-register-forensics]], [[consensus-decoding]], [[future-sampling]], [[synthesis-pipeline]], [[latency-quality-tradeoff]], [[scoreboard]], index.md (6 content pages — fewer than 15 relevant exist for this narrow result)
- contradictions: none — confirms forensics fix plan #1's prediction (strict gate: char-BLEU 60→67, target 65+ met; 4-gram 0.58 vs target 0.6 near-miss); new caveat logged: anchor LAAL +1.8 vs consensus track, latency-matching deferred to trained-model eval
