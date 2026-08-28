# Draft email to NAIST authors (send from your own address)

**To:** Mana Makinae — try `makinae.mana.mh2@is.naist.jp` (git commits show
`mana-ma@pine11.naist.jp`, which is an internal machine; the paper PDF header
has the canonical address — check https://direct.mit.edu/coli/article/doi/10.1162/COLI.a.622
or the Simul-MuST-C EMNLP 2024 paper)

**Subject:** Simul-tst-COMMON en-zh — GPT-4o reproduction drift; could you share the final references?

---

Dear Makinae-san,

I am a graduate student at CMU (LTI) working on simultaneous speech
translation data synthesis. Thank you for releasing Simul-tst-COMMON — the
interpreter-grounded monotonic references are exactly what our evaluation
needs, and the patch-based release is a clever way to handle the MuST-C
license.

I followed the reproduction workflow in naist-nlp/Simul-tst-COMMON for
en-zh and hit a reproducibility limit I wanted to report, and ask about:

- ASR + source side reproduces very well: 103/107 source edit operations
  applied cleanly (whisper medium.en, default settings).
- The GPT-4o step has drifted: using the pinned `gpt-4o-2024-05-13`
  snapshot with your exact request parameters (temperature 0.5, top_p 0.0,
  seed 0, JSON mode), only ~18% of output lines hash-match the target
  edits.json (≈66% expected if outputs were byte-identical). It appears
  two years of OpenAI backend changes have altered the sampled outputs,
  so roughly half of the interpreter-checked corrections can no longer be
  applied.

Since MuST-C is distributed under CC BY-NC-ND 4.0 and we hold the corpus,
would you be able to share the final Simul-tst-COMMON en-zh references
(and if possible en-ja / en-de) directly, e.g. for research use? That
would preserve comparability with the numbers in your COLI paper, which
the patch route can unfortunately no longer guarantee.

Happy to share our rebuilt (drifted) version or the hash-match diagnostics
if useful to you.

Best regards,
Haoling Pu
Language Technologies Institute, Carnegie Mellon University
haolingp@andrew.cmu.edu
