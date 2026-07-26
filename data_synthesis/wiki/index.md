# Wiki Index

Master catalog. See [[WIKI.md]] for conventions.

## Concepts

### Synthesis methods
- [[synthesis-pipeline]] — end-to-end EAST/SALAMI/future synthesis stages
- [[east]] — LLM segmentation into low/med/high latency levels
- [[salami]] — segmented-pairs synthesis format
- [[future-sampling]] — online READ/WRITE policy via base+instruct+align+judge
- [[consensus-decoding]] — ref-free dual-base consensus across decodes
- [[thinking-policy]] — reasoning model decides READ/WRITE
- [[majority-vote]] — commit the most common candidate
- [[segale-alignment]] — token-to-segment alignment for prefix truncation
- [[min-p-sampling]] — min-p/top-k diversity-control ablations

### Training / inference / eval
- [[dataset-conversion-pipeline]] — synthesis JSONL → SWIFT/Megatron training instances
- [[streaming-inference]] — simuleval + vLLM streaming agent
- [[east-prompt-handling]] — EAST vs Standard prompt gotcha at infer time
- [[checkpoint-evaluation]] — simuleval → normalize → omnisteval scoring
- [[latency-quality-tradeoff]] — quality vs LongYAAL Pareto analysis

## Entities

### Datasets
- [[gigaspeech]] — primary speech corpus (synthesis source)
- [[acl-6060]] — held-out dev eval set (zh/ja/de refs)
- [[simul-tst-common]] — NAIST monotonic-reference SMT eval set (rebuilt 2026-07-11; GPT drift → diagnostic-grade, 37% tgt-edit fidelity)

### Tools / frameworks
- [[babel-cluster]] — CMU SLURM cluster: partitions/QoS/quotas + job-submission rules
- [[metricx]] — reference-free QE model + filter
- [[mfa]] — Montreal Forced Aligner (audio↔text)
- [[vllm]] — inference engine (local + served)
- [[megatron-swift]] — LoRA training framework
- [[simuleval]] — streaming speech-to-text eval harness

### Models
- [[qwen3-omni]] — multimodal base model fine-tuned for all checkpoints
- [[qwen35-122b-sampler]] — local Qwen3.5-122B thinking future sampler: serving configs, measured token cost, 40k scaling plan
- [[infinisst-omni]] — the streaming agent + trained checkpoint family

## Experiments
- [[2026-06-consensus-axis5-vs-futures200]] — 5-axis (20 directed futures) beats the futures=200 baseline at every latency; soft-vote and 100-future scaling don't help
- [[2026-06-consensus-post-edit-bleu]] — LLM post-edit (soft-vote / re-translate / polish) all regress vs raw consensus; the BLEU gap is structural
- [[2026-06-qwenasr-asr-regression-periodfix]] — new Qwen-ASR regresses −4–6 BLEU vs the old-asr 5-axis baseline; period-fix ~+1 BLEU, spaCy split-fix +0.7/2.5 at long latency. But a clause-split (⑤b) matching old-asr granularity exactly does NOT help (−1..−3.6 vs split-fix, still −5.5..−7.0 vs ①) → **segmentation/granularity refuted as the cause**; last uncontrolled confound is decode window (baseline win1 vs new-asr win3) — testing qwenasr@win1

- [[2026-07-simul-tst-common-rescore]] — monotonic-ref re-score: consensus-vs-hibiki BLEU gap persists (6.5–8.3, COMET tied) → real, not reference-style bias; EAST-even canary didn't crater on BLEU but has worst COMET
- [[2026-07-consensus-register-forensics]] — root cause of the ~7 BLEU: same Qwen3-30B translator, three decode modes; soft-vote selects future-proof formal register (所以→因此 catalog), −19% 4-gram recall with content preserved; fix = rank-by-present canonical anchor
- [[2026-07-anchor-smoke500-sweep]] — anchor-and-veto smoke A/B/C/D: strict gate (0.05/5/1.0) wins +6.98 paired char-BLEU, register normalized, quality monotone in strictness; LAAL +1.8 caveat; variant A promoted to anchor_40k prod decode (9239309/9239310)
- [[2026-07-refsel-bestof5]] — CLOSED NEGATIVE: bestof5 (+10.9 oracle → ~0 trained) AND bestof4 (negative every cell, tst timing blow-ups) — per-utt selection conclusively does not transfer
- [[2026-07-present-rank-winner]] — NEGATIVE (−2.15 paired char-BLEU, hygiene clean): strict gate leaves the winner rule no freedom; register bias enters at the GATE, not the winner
- [[2026-07-present-propose-gate]] — successor: present distribution proposes candidates, futures verify by majority; smoke500 vs J_40k

## Comparisons
- [[scoreboard]] — measured COMET/BLEU/chrF/latency of all trained checkpoints (ACL 6060 dev)
- [[la-n-vs-wait-k]] — rule-based LA/wait-k/PA offline policies + trained counterparts
- [[comet-vs-bleu-ranking]] — why we rank by COMET, not BLEU
