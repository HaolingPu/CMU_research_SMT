# Related Work — SiMT data synthesis methods only

**Your contribution:** future-aware sampling + consensus decoding from an LLM to generate prefix-aligned training data for simultaneous machine translation.

**Scope of this list:** ONLY methods for **constructing or synthesizing training data for SiMT**. Policies (wait-k as a policy), streaming model architectures (Hibiki/StreamSpeech as a *system*), and generic MBR/decoding work are *out of scope* for related work — they belong in the broader background or comparison-system sections of the paper, not here.

**Crawler ran 2026-05-01: 636 papers harvested, 14 strictly-on-topic, ~30 in adjacent useful categories.** ✅ marks crawler-confirmed entries (title/year/authors verified in `related_work_survey/results.json`); 🆕 marks entries surfaced by the crawler that I'd missed; **[VERIFY]** still means I couldn't confirm and you need to look it up.

## ⚠ Top crawler findings (read first)

1. **🚨 BPA paper not found in major venues.** Crawler searched 2018-2026 ACL/EMNLP/NAACL/ICLR/ICML/NeurIPS/ICASSP/Interspeech and got **0 hits** for "Bilingual Prefix Alignment" or "BPA". Your `prefix_alignment.py` references it as a known method — it's likely from IWSLT, a workshop, or arxiv-only. **Look it up directly** before citing as your baseline; you need the exact citation.
2. **🚨 "Future-Aware Distillation" is already a published phrase.** Fu et al. EMNLP 2023 *Adapting Offline ST Models for Streaming with Future-Aware Distillation and Inference*. You **must cite + clearly differentiate** or reviewers will flag terminology overlap.
3. **★ Closest competitor: Simul-MuST-C — Makinae et al. EMNLP 2024.** LLMs for simultaneous corpus construction. Your method is the next iteration. Read end-to-end before writing your method.
4. **★ Tailored Reference — Guo et al. EMNLP 2023.** Constructs custom prefix targets per source prefix. Methodologically near your "consensus output". Must cite + position.
5. **★ Glancing Future — Guo et al. ICASSP 2023.** Curriculum prefix-data construction.

These five drive your related-work positioning.

---

## A. Prefix-pair construction from full-sentence parallel data

The dominant pre-LLM paradigm. Take a full source-target sentence pair, generate (prefix_src, prefix_tgt) training pairs by some alignment / re-pairing rule.

- ✅ **STACL — Ma et al. ACL 2018** (309 cites). *STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework*. — **Cite for the data-recipe**: wait-k training pairs every source prefix of length `k+i` with the target prefix of length `i` from the gold full sentence. This is the simplest "implicit" prefix synthesis. Your method replaces this rule with LLM future-aware sampling. **(Year correction: 2018, not 2019.)**
- ✅ **Glancing Future for SiMT — Shoutao Guo et al. ICASSP 2023** (11 cites). — Explicit curriculum where prefix data goes from easy (long context) to hard (short context). Data-construction recipe; direct competitor.
- 🆕 **Simultaneous Machine Translation with Tailored Reference — Shoutao Guo et al. EMNLP 2023** (12 cites). ⭐ — Constructs a *tailored* reference per prefix instead of the gold full target. **Methodologically very close to your "consensus" output**. You must cite + position against.
- 🆕 **Reducing Position Bias in SiMT with Length-Aware Framework — Shaolei Zhang, Yang Feng. ACL 2022** (23 cites). — Identifies that wait-k naive prefix-pairing creates position bias; their fix is data-side (re-balancing prefix lengths in training).
- 🆕 **Better Simultaneous Translation with Monotonic Knowledge Distillation — Shushu Wang et al. ACL 2023** (14 cites). ⭐ — Distillation pipeline that produces monotonic-friendly target sequences from an offline MT teacher; direct data-construction work.
- ❌ **Future-Guided Incremental Transformer (Zhang 2021 AAAI)** — Crawler couldn't confirm in NLP/speech venues. May be AAAI-only. If you want to cite it, look it up directly. Likely safer to cite the data-construction works above instead.

## B. Word-alignment-based prefix construction

Use external word alignments to decide where to cut both source and target so the prefix-pair is semantically faithful (vs naive length-cut).

- **awesome-align — Dou & Neubig 2021** (ACL). — Toolkit cited by basically every alignment-based SiMT data-construction paper. Cite if you use it; cite as background otherwise.
- **SimAlign — Jalili Sabet, Dufter, Yvon, Schütze 2020** (Findings EMNLP). — Embedding-based alignment, alternative to fast_align. Often used in SiMT data prep.
- **fast_align — Dyer, Chahuneau, Smith 2013** (NAACL). — The pre-neural workhorse; many SiMT papers still use it.
- **Cross-lingual chunk alignment for SiMT — multiple recent papers** **[VERIFY]** — there's a 2022-2024 line that aligns source chunks to target chunks (not just words) using monotonic alignments specifically for SiMT data; need to look up exact citations.
- **Re-pairing parallel corpora for SiMT — Chen et al. ACL 2021 (or 2022)** **[VERIFY exact paper]** — *Improving Simultaneous Translation by Incorporating Pseudo-References with Fuzzy Contexts* or similar title. Generates fuzzy / multi-prefix targets per source prefix.

## C. LLM-driven SiMT training data generation (★ closest competitors)

This is your direct comparison set — papers from the last ~18 months that use LLMs to generate SiMT training data. Crawler confirmed several; the field is concentrated in 2024-2025 ACL/EMNLP/Interspeech.

- 🆕 **Simul-MuST-C: Simultaneous Multilingual Speech Translation Corpus Using Large Language Models — Mana Makinae et al. EMNLP 2024** (5 cites). ⭐⭐⭐ — **THE most directly competing paper**: uses LLMs to construct a simultaneous-friendly multilingual corpus from MuST-C. Your paper's method is the next iteration of this exact line. Read end-to-end before writing your method section.
- 🆕 **Adapting Offline ST Models for Streaming with Future-Aware Distillation and Inference — Biao Fu et al. EMNLP 2023** (8 cites). ⭐⭐⭐ — **The phrase "Future-Aware Distillation" is theirs**. You must cite and *clearly differentiate* (theirs is offline-ST→streaming-ST distillation; yours is LLM-prefix sampling). Otherwise reviewer will flag.
- 🆕 **Simultaneous Masking, Not Prompting Optimization (SimulMask) — Matthew Raffel et al. EMNLP 2024** (10 cites). ⭐⭐ — Fine-tunes LLMs for SiMT via masking-based training data construction. Direct LLM-for-SiMT precedent.
- 🆕 **InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model — Siqi Ouyang et al. ACL 2025** (8 cites). ⭐⭐ — LLM-based SiMT with custom training data for streaming. Recent direct competitor.
- 🆕 **SimulS2S-LLM: Unlocking Simultaneous Inference of Speech LLMs for S2S Translation — Keqi Deng et al. ACL 2025** (4 cites). ⭐ — Speech-LLM SiMT; data construction angle.
- 🆕 **Empowering LLMs for End-to-End Speech Translation Leveraging Synthetic Data — Yu Pu et al. Interspeech 2025** (4 cites). — LLM + synthetic ST data; cite for context if you have a speech experiment.
- 🆕 **Language Model Augmented Monotonic Attention for SiMT — Indurthi et al. NAACL 2022** (10 cites). — Earlier LM-augmented SiMT; cite as predecessor.
- ❌ **Simul-LLM / Agent-SiMT / Koshkin** — Crawler did not surface these under those names in 2018+ ACL/EMNLP/NAACL/ICLR/ICML/NeurIPS/ICASSP/Interspeech. Either workshop-only, arxiv-only, or named differently. **If you remember a specific paper here, look it up by author**; otherwise the entries above cover the space.

**Where your method fits this section:** you contribute (i) **future-aware sampling** — drawing samples conditioned on a probabilistic look at the source future — and (ii) **consensus decoding** — agreement across samples to pick the prefix translation. The closest precedents are Fu 2023 (term "future-aware" but used differently), Guo 2023 *Tailored Reference* (constructs custom prefix targets but no LLM/sampling), and Makinae 2024 *Simul-MuST-C* (LLM for corpus construction but no future-aware or consensus mechanism). Your novelty = the combination.

## D. Re-segmentation of speech-translation corpora for SiMT

- ✅ **Simul-MuST-C — Mana Makinae et al. EMNLP 2024** (5 cites). — Already covered in §C above. The data-construction recipe (LLM-driven re-pairing) is what to cite here too.
- ✅ **AlignAtt — Sara Papi et al. Interspeech 2023** (33 cites). *AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide for Simultaneous Speech Translation*. — Use attention from offline ST model to time-align translations to audio frames; alignment-as-policy, but the alignment artifact is also usable as training data.
- 🆕 **Attention as a Guide for Simultaneous Speech Translation (EDAtt) — Sara Papi et al. ACL 2022** (42 cites). — Predecessor of AlignAtt. Same Papi/FBK group; cite both if you discuss attention-alignment-based data construction.
- ❌ **MuST-C original** — Crawler didn't surface in your 2018-2026 venue list (likely 2019 paper at ACL: Di Gangi et al.). Cite if you describe your eval setup uses MuST-C.

## E. Streaming-system data recipes (cite the data-construction part only)

These are systems whose **training-data construction technique** is novel, even though the system itself is the headline contribution. Scope your citation to the data-recipe.

- ✅ **Hibiki — Tom Labiausse et al. ICML 2025** (21 cites). *High-Fidelity Simultaneous Speech-To-Speech Translation*. — Their training data is built by aligning offline-translated targets to source audio at fine granularity, producing a synchronized streaming corpus. **(Authors corrected: Labiausse et al., not Défossez first-author.)**
- 🆕 **Non-autoregressive Streaming Transformer for Simultaneous Translation — Zhengrui Ma et al. EMNLP 2023** (16 cites). — NAT for SiMT with custom data preparation.
- ❌ **SeamlessStreaming / StreamSpeech** — Crawler didn't surface in your venue list under those exact names (StreamSpeech may be ACL 2024 but didn't pass topic regex). Search Semantic Scholar directly if you need them.

## F. Pseudo-reference / multi-reference generation for SiMT

Generating multiple plausible translations per prefix (so the SiMT model isn't penalized for picking one valid continuation over another).

- **Multi-reference SiMT — pseudo-reference papers, Chen et al. or Zhang & Feng** **[VERIFY]** — Use offline MT to produce K alternatives per (source prefix), train against the best-fit one. Methodologically very close to your "consensus" framing if framed correctly.
- **Fuzzy-context training — Chen et al.** **[VERIFY]** — Same family.

## G. Quality filtering of synthetic translation data (relevant if you use QE in your pipeline)

Only cite if your method actually filters or scores generated samples with QE.

- **CometKiwi — Rei et al. 2022** (WMT). — Reference-free QE; standard tool for filtering synthetic translations.
- **MetricX-23 / MetricX-24 — Juraska, Finkelstein, Deutsch, Siddhant, Mirzazadeh, Freitag 2023/2024** (WMT). — Stronger QE; you mentioned using this. Cite the specific year you actually used.
- **Quality-Aware Decoding — Fernandes, Farinhas, Rei et al. 2022** (NAACL). — Sample-then-rerank with QE for *decoding* — but their reranker mechanism is the same one used in synthetic-data QE filtering. Borderline; cite if your "consensus decoding" reranks against QE.
- **xCOMET — Guerreiro, Rei et al. 2024** (TACL). **[VERIFY]** — Token-level QE; relevant if you do prefix-quality scoring at sub-sentence level.

## H. Sampling-as-data-synthesis for general MT (background context)

Cite if you frame your sampling component in this lineage. Be selective — these aren't SiMT-specific.

- **Sampling-Based MBR — Eikema & Aziz 2022** (EMNLP). — The "sample many, pick best" recipe that anchors much modern synthetic-MT-data work. If your "consensus decoding" is sample-and-vote, this is your antecedent.
- **High-Quality MBR with Neural Metrics — Freitag et al. 2022** (TACL). — MBR with COMET utility. Direct ancestor of QE-based sample-rank.
- **ALMA / ALMA-R — Xu et al. 2024** (ICLR). — LLM fine-tuning for MT; their training-data recipe (parallel + monolingual + preference pairs) is a useful contrast.
- **Distillation in MT — Kim & Rush 2016** (EMNLP). — Foundational sequence-level KD; cite only if you frame your synthetic data as KD from an LLM teacher.

## I. NOT in related work (background only — don't cite here)

Items I had in v1 that should be cut from the data-synthesis-related section. Move them to background / experimental setup if needed:

- ❌ wait-k as a *policy* (mention only the data-recipe interpretation in §A above)
- ❌ MMA, MILk, ITST, DiSeg, Hidden Markov Transformer — these are policy/architecture work
- ❌ Hibiki/StreamSpeech/SeamlessStreaming as *systems* — only their data recipes (§E) belong here
- ❌ Generic MBR theory — unless you specifically reuse it
- ❌ Speculative decoding, lookahead-for-speedup — these are inference-speedup, not data synthesis
- ❌ FUDGE / NeuroLogic A\*esque — generic controlled decoding, not MT data synthesis
- ❌ COMET (the original) — that's just the metric. Cite only the QE variants you use.

---

## Honest caveats

- The **C section (LLM-driven SiMT data)** is the most important for your paper and the section where I'm least confident on exact paper attributions. The 2024-2025 SiMT-with-LLM space has many concurrent submissions; the crawler is more reliable than my memory here. **Wait for crawler results before finalizing this section.**
- The **D/E sections (Simul-MuST-C, Hibiki)** — I have uncertainty on titles/years because Papi et al. and the Kyutai team have published several papers each. The crawler will pin these down.
- The **F section (pseudo-references)** — I remember the line of work exists but not which specific paper to anchor on. Pull-up by SemScholar query.

After the crawler finishes, the workflow is:
1. Open `related_work_survey/results.md`
2. Filter to entries where the abstract mentions "training data", "data construction", "synthetic", "pseudo-reference", "prefix-aligned", "alignment-based" — not policies/architectures
3. Add anything new to sections A-F above; ignore the rest
