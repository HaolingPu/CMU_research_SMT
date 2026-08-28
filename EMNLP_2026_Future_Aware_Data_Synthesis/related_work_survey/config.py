"""Topic configuration for a literature survey pipeline.

Imported by survey.py, filter_results_llm.py, and synthesize_field.py.
Every constant here is "topic-specific" — the rest of the toolkit
(S2 client, caching, clustering, citation linker) is generic.

Topic: future-aware sampling, lookahead, and consensus/MBR decoding for
data synthesis in simultaneous (and speech) machine translation.
"""

from __future__ import annotations
import re

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

TOPIC_SHORT = ("future-aware sampling, lookahead, and consensus decoding for "
               "data synthesis in simultaneous (and speech) machine translation")

YEAR_FROM = 2018


# ---------------------------------------------------------------------------
# Venues
# ---------------------------------------------------------------------------

S2_VENUE_FILTER: list[str] = [
    "ACL", "EMNLP", "NAACL",
    "ICLR", "ICML",
    "NeurIPS", "Neural Information Processing Systems",
    "ICASSP", "Interspeech", "INTERSPEECH",
]

VENUE_GROUPS: dict[str, list[str]] = {
    "ACL": ["annual meeting of the association for computational linguistics",
            "acl-", " acl ", "acl 20", "acl 19", "acl 18",
            "findings of the association for computational linguistics"],
    "EMNLP": ["empirical methods in natural language processing", "emnlp",
              "findings of emnlp"],
    "NAACL": ["north american chapter of the association for computational linguistics",
              "naacl", "findings of naacl"],
    "ICLR": ["international conference on learning representations", "iclr"],
    "ICML": ["international conference on machine learning", "icml"],
    "NeurIPS": ["neural information processing systems", "neurips", "nips",
                "advances in neural information processing"],
    "ICASSP": ["icassp", "acoustics, speech and signal", "acoustics speech and signal",
               "ieee international conference on acoustics"],
    "Interspeech": ["interspeech", "eurospeech",
                    "international speech communication association"],
}

VENUE_ORDER: list[str] = list(VENUE_GROUPS.keys())

VENUE_DESCRIPTION = ("NLP (ACL, EMNLP, NAACL), ML (ICLR, ICML, NeurIPS), "
                     "and speech (ICASSP, Interspeech) venues")


# ---------------------------------------------------------------------------
# Crawler (survey.py)
# ---------------------------------------------------------------------------

SEED_QUERIES: list[str] = [
    "simultaneous machine translation",
    "simultaneous neural machine translation",
    "streaming machine translation",
    "online machine translation",
    "wait-k simultaneous translation",
    "prefix-to-prefix translation",
    "read write policy simultaneous translation",
    "adaptive policy simultaneous translation",
    "monotonic attention translation",
    "incremental decoding translation",
    "anticipation simultaneous translation",
    "lookahead simultaneous translation",
    "future prediction simultaneous translation",
    "latency quality simultaneous translation",
    "data augmentation simultaneous translation",
    "synthetic data simultaneous translation",
    "streaming speech translation",
    "simultaneous speech translation",
    "LLM data synthesis machine translation",
    "distillation simultaneous machine translation",
    "minimum bayes risk decoding translation",
    "MBR decoding neural machine translation",
    "consensus decoding machine translation",
    "sample then rerank translation",
    "COMET quality estimation translation",
    "MetricX quality estimation translation",
    "self-training simultaneous translation",
    "back-translation simultaneous",
]

QUERY_TASK_SUFFIXES: list[str] = ["simultaneous translation",
                                  "machine translation",
                                  "speech translation"]

ANCHOR_TERMS: list[str] = [
    "simultaneous", "streaming", "online", "incremental",
    "wait-k", "prefix-to-prefix", "monotonic", "anticipation", "latency",
    "machine translation", "translation", "nmt",
    "synthetic", "data synthesis", "data augmentation",
    "mbr", "minimum bayes risk", "consensus",
    "lookahead", "future", "rerank",
    "comet", "metricx", "quality estimation",
    "distillation", "back-translation", "self-training",
    "speech translation",
]

# Topic gate: a paper must match BOTH regexes on title+abstract before the
# crawler absorbs it.
#
#   TOPIC_RE — the *method/problem* keywords (simultaneous, lookahead, MBR,
#              data synthesis, distillation, sampling, …)
#   TASK_RE  — the *task surface* (simultaneous/streaming/online translation,
#              speech translation, machine translation)
TOPIC_RE = re.compile(
    r"\b(simultaneous|streaming|online|incremental|"
    r"wait[- ]?k|prefix[- ]to[- ]prefix|prefix2prefix|"
    r"read[- ]write|read/write|monotonic|adaptive policy|"
    r"anticipat\w*|lookahead|look[- ]ahead|future[- ]aware|future sampling|"
    r"latency|low[- ]latency|"
    r"synthetic|synthesi[sz]\w*|distill\w*|data augmentation|"
    r"minimum bayes risk|mbr|consensus|"
    r"speculative|value[- ]guided|rerank\w*|sample[- ]?then[- ]?rerank|"
    r"back[- ]translation|forward[- ]translation|self[- ]training|pseudo[- ]?label\w*|"
    r"quality estimation|comet|metricx|qe filter\w*|"
    r"sampling|nucleus|top[- ]p|min[- ]p|contrastive decoding)\b",
    re.IGNORECASE,
)
TASK_RE = re.compile(
    r"\b(simultaneous (machine )?translation|simultaneous interpretation|"
    r"streaming translation|online translation|"
    r"speech translation|s2t|"
    r"machine translation|neural machine translation|nmt|"
    r"\btranslation\b|parallel (data|corpus|corpora))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# LLM judge (filter_results_llm.py)
# ---------------------------------------------------------------------------

TOPIC_RUBRIC = """You are a senior researcher screening papers for inclusion in a literature survey.

Survey topic: **simultaneous machine translation (SiMT) — including streaming/online translation and simultaneous speech translation — with a focus on (a) data synthesis for training SiMT models (LLM-based forward translation, distillation, prefix-data construction, back-translation), (b) future-aware / lookahead / anticipation methods, (c) READ/WRITE policy learning (wait-k, adaptive, monotonic attention, RL agents), (d) decoding strategies (MBR, consensus, lookahead, sample-then-rerank), and (e) quality-aware filtering of synthetic translation data (COMET-QE, MetricX-QE).**

A paper is RELATED if and only if BOTH conditions hold:
  1. Its primary task is simultaneous (or streaming/online) MT, simultaneous speech translation, OR offline MT methodology that is directly applicable to SiMT (e.g., MBR/consensus decoding, QE-based filtering, LLM data synthesis for translation, sample-then-rerank).
  2. A central contribution is one of: SiMT policies (wait-k, adaptive, RL agent, monotonic), prefix-to-prefix training, anticipation / future prediction, lookahead / value-guided decoding, MBR / consensus decoding, QE-based filtering of synthetic translation data, self-training / iterative refinement / distillation for translation, or LLM-driven generation of synthetic translation training data.

Examples of RELATED:
- Wait-k or prefix-to-prefix simultaneous neural MT
- Adaptive READ/WRITE policies (RL agents, monotonic attention) for SiMT
- Anticipation losses, future-context prediction for simultaneous translation
- MBR decoding for NMT (transferable to SiMT/data synthesis)
- LLM forward-translation generating synthetic parallel data with QE filtering
- COMET-QE / MetricX-QE filtering of synthetic translation data
- Knowledge distillation from a teacher MT/LLM into a SiMT student
- Lookahead / value-guided / speculative decoding evaluated on translation
- Streaming / re-translation simultaneous speech translation
- Sample-then-rerank or QE-rerank for translation
- Latency–quality trade-off methodology for SiMT
- Self-training / iterative refinement loops for NMT or SiMT
- Prefix-data construction or alignment for SiMT training

Examples of NOT RELATED:
- Generic NMT architecture / regularization / training improvements without a simultaneous, data-synthesis, decoding-strategy, or QE-filtering angle
- ASR-only papers (no translation component)
- Offline speech translation papers with no streaming, no data-synthesis, and no decoding-strategy contribution
- LLM distillation / synthesis for math, code, instruction tuning, or chat — non-translation tasks
- Generic prompt engineering or LLM evaluation / benchmark papers
- Pure decoding speedup with no translation-quality angle
- Standard back-translation papers (pre-2018) with no sampling, policy, or filtering innovation
- QE method papers that don't apply QE to filter synthetic data or rerank candidates
- Multilingual NMT or low-resource MT papers without simultaneous/data-synthesis focus
- Generic RLHF or alignment papers
- Non-translation data augmentation (vision, dialogue, summarization)
- Pure tokenization / subword / large-vocab tricks without a SiMT or data-synthesis focus

Be strict. When in doubt, mark NOT RELATED — the survey author would rather lose a borderline paper than include an off-topic one.

Output `related` (true/false) and `reason` (≤ 30 words). In the reason, cite the specific signal (or its absence) from the title/abstract — e.g., "wait-k SiMT with anticipation loss", or "generic transformer NMT, no simultaneous/synthesis/decoding angle"."""


# ---------------------------------------------------------------------------
# Synthesis (synthesize_field.py)
# ---------------------------------------------------------------------------

TOPIC_HEADER = """We are surveying a research area: **simultaneous machine translation (SiMT) — including streaming/online translation and simultaneous speech translation — with a focus on data synthesis, future-aware / lookahead methods, READ/WRITE policy learning, decoding strategies (MBR, consensus, sample-then-rerank), and quality-aware filtering of synthetic translation data.**

The corpus spans 2018–2026 across NLP (ACL, EMNLP, NAACL), ML (ICLR, ICML, NeurIPS), and speech (ICASSP, Interspeech) venues. Citation-graph expansion may have pulled in some off-topic papers (e.g., a generic transformer paper cited by on-topic seeds). Synthesis must be tolerant of this.

Sub-areas you should look for include (but are not limited to):
- SiMT policies: wait-k, adaptive, RL-trained READ/WRITE agents, monotonic attention
- Prefix-to-prefix training and prefix data construction for SiMT
- Anticipation, future prediction, and lookahead in simultaneous translation
- Latency–quality trade-off methodology and evaluation
- Streaming / re-translation simultaneous speech translation
- LLM-based forward translation for synthetic SiMT / MT data
- Self-training, iterative refinement, distillation for SiMT
- MBR and consensus decoding for translation (offline, transferable to SiMT)
- Sample-then-rerank, QE-rerank, value-guided decoding for translation
- COMET-QE / MetricX-QE filtering of synthetic translation data
- Sampling strategies (temperature, top-p, min-p, diverse beam) for translation generation
- Knowledge distillation from LLM/teacher into SiMT student models

Use precise, technique-level terminology. Cite specific papers by author + year inline."""
