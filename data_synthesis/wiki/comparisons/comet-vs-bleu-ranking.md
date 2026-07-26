---
title: COMET vs BLEU for Ranking
type: comparison
tags: [eval, metric]
sources:
  - ../codes/metricx/
  - scripts/infer/eval_all_ckpts.sh
created: 2026-06-01
updated: 2026-07-11
---

# COMET vs BLEU for Ranking

When ranking synthesis configs / checkpoints, prefer **COMET** over BLEU/chrF.

- Synthesis-time BLEU does not predict trained-model eval BLEU.
- On ACL 6060 dev ([[acl-6060]]), BLEU/chrF is biased toward set-intersection's conservative
  surface forms, so it unfairly favors ref-based methods over [[consensus-decoding]] (ref-free).
- Rule of thumb: rank by COMET; ignore BLEU gaps of ≤2–3 points when COMET is tied.

In practice COMET (`Unbabel/XCOMET-XL`) is computed in [[checkpoint-evaluation]] alongside BLEU,
chrF, and [[metricx]] QE; the frontier is read off the [[latency-quality-tradeoff]] plots.

Live illustration: in the [[scoreboard]], zh `EAST-even` leads BLEU (~47) but trails COMET (~0.79)
while ref-free `consensus-top5-axis5` leads ref-free COMET (~0.817) at lower BLEU.

The BLEU gap is not a tuning artifact you can edit away: [[2026-06-consensus-post-edit-bleu]]
shows three LLM post-edit attempts (soft-vote, re-translation, polish) all fail to lift consensus
BLEU — it is structural (consensus never sees the future), so COMET is the metric that credits it.

**Caveat (2026-07-11): vs hibiki specifically, COMET is tied, not won.** At matched latency the
hibiki word-align checkpoint scores COMET 0.780/0.812/0.814/0.820 (seg960–3840) vs consensus
top5-axis5's 0.787/0.808/0.812/0.817 — a wash — while hibiki wins BLEU by 6–8. So "trades BLEU
for COMET" holds against EAST-even but **not** against hibiki; there consensus currently has no
measurable win. Whether the BLEU deficit is real quality or reference-style bias is exactly what
[[simul-tst-common]] (monotonic references) was built to decide.

**Answered (2026-07-11): the deficit is real.** On monotonic references the hibiki-vs-consensus
BLEU gap persisted unchanged (6.5–8.3 at matched latency) with COMET again tied — see
[[2026-07-simul-tst-common-rescore]]. Notably EAST-even kept its BLEU lead there too while
scoring the worst COMET, so the rank-by-COMET rule survives on a second, independent test set.

## Related
- [[scoreboard]], [[la-n-vs-wait-k]], [[consensus-decoding]], [[metricx]], [[checkpoint-evaluation]],
  [[2026-06-consensus-post-edit-bleu]], [[2026-06-consensus-axis5-vs-futures200]].

## Sources
- code: `../codes/metricx/`, `scripts/infer/eval_all_ckpts.sh`
