"""Topic configuration for a literature survey pipeline.

Imported by survey.py, filter_results_llm.py, and synthesize_field.py.
Every constant here is "topic-specific" — the rest of the toolkit
(S2 client, caching, clustering, citation linker) is generic.

The example values below are from a survey on "rare/new/OOV words and
named entities in ASR and MT". Replace them with values for your topic.
"""

from __future__ import annotations
import re

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

# Short topic identifier used in markdown titles. Sentence-style, no period.
TOPIC_SHORT = "rare/new/OOV words & named entities in ASR and MT"

# Earliest publication year. Set lower for foundational / historical surveys.
YEAR_FROM = 2017


# ---------------------------------------------------------------------------
# Venues
# ---------------------------------------------------------------------------

# Server-side prefilter for the S2 search endpoint. A loose union of the
# conference names and aliases — final classification happens client-side
# in VENUE_GROUPS so off-spelling forms aren't dropped here.
S2_VENUE_FILTER: list[str] = [
    "ICASSP", "Interspeech", "INTERSPEECH", "ASRU", "SLT",
    "ACL", "EMNLP", "NAACL", "ICLR", "ICML",
    "NeurIPS", "Neural Information Processing Systems",
]

# Client-side venue classification. For each S2 paper, the `venue` field
# (lowercased) is checked against every substring; the first VENUE_GROUPS
# key whose substring matches wins. Papers with no match are DROPPED.
# Add aliases generously: "proc.", "ieee", expanded names, etc.
VENUE_GROUPS: dict[str, list[str]] = {
    "ICASSP": ["icassp", "acoustics, speech and signal", "acoustics speech and signal",
               "ieee international conference on acoustics"],
    "Interspeech": ["interspeech", "eurospeech",
                    "international speech communication association"],
    "ASRU": ["asru", "automatic speech recognition and understanding"],
    "SLT": ["spoken language technology", " slt ", "ieee slt"],
    "ACL": ["annual meeting of the association for computational linguistics",
            "acl-", " acl "],
    "EMNLP": ["empirical methods in natural language processing", "emnlp"],
    "NAACL": ["north american chapter of the association for computational linguistics",
              "naacl"],
    "ICLR": ["international conference on learning representations", "iclr"],
    "ICML": ["international conference on machine learning", "icml"],
    "NeurIPS": ["neural information processing systems", "neurips", "nips"],
}

# Display order for venue-grouped output. Defaults to insertion order.
VENUE_ORDER: list[str] = list(VENUE_GROUPS.keys())

# One-line human-readable description used in synthesis prompts and headers.
VENUE_DESCRIPTION = ("speech (ICASSP, Interspeech, ASRU, SLT), "
                     "NLP (ACL, EMNLP, NAACL), and ML (ICLR, ICML, NeurIPS) venues")


# ---------------------------------------------------------------------------
# Crawler (survey.py)
# ---------------------------------------------------------------------------

# Initial S2 search queries. The crawler mines new queries from each
# round's accepted papers (bigrams/trigrams), so SEED_QUERIES just needs
# to seed the space. Aim for 15-30 queries spanning the breadth of the
# topic — alternative phrasings and sub-areas matter more than count.
SEED_QUERIES: list[str] = [
    "rare word speech recognition",
    "out-of-vocabulary speech recognition",
    "OOV speech recognition",
    "new word speech recognition",
    "named entity speech recognition",
    "contextual biasing speech recognition",
    "hotword speech recognition",
    "personalization speech recognition rare",
    "long-tail speech recognition",
    "rare entity ASR",
    "vocabulary expansion speech recognition",
    "subword speech recognition rare",
    "rare word machine translation",
    "out-of-vocabulary machine translation",
    "OOV neural machine translation",
    "named entity machine translation",
    "low-frequency words neural machine translation",
    "terminology machine translation",
    "lexically constrained translation",
    "copy mechanism translation rare",
    "subword translation rare words",
    "transliteration named entity translation",
    "rare token language model",
    "long-tail named entity recognition",
    "unseen entity translation",
    "domain terminology speech translation",
]

# Suffixes appended to mined phrases when the crawler generates new queries.
# Typically the dominant task names in the corpus.
QUERY_TASK_SUFFIXES: list[str] = ["speech recognition", "machine translation"]

# Anchor terms used by keyword-expansion scoring. Phrases mined from
# previous-round papers get a score bonus if they co-occur with any of
# these in the same document. Keep short (5-15 lowercased substrings).
ANCHOR_TERMS: list[str] = [
    "speech recognition", "machine translation", "asr", "nmt", "translation",
    "rare", "oov", "unseen", "named entity", "entity", "vocabulary",
    "subword", "biasing", "hotword", "terminology",
]

# Topic gate: a paper must match BOTH regexes on title+abstract before the
# crawler absorbs it. Used during citation-graph expansion to keep cited
# works of seed papers from polluting the corpus.
#
# Pattern:
#   TOPIC_RE — the *problem* keywords (rare, OOV, named entity, hotword, ...)
#   TASK_RE  — the *task surface* (speech recognition, MT, translation, ...)
# A generic NER paper fails TASK_RE; a generic NMT paper fails TOPIC_RE.
TOPIC_RE = re.compile(
    r"\b(rare|oov|out[- ]of[- ]vocabulary|unseen|new word|novel word|"
    r"named entit\w*|entity|hotword|biasing|long[- ]tail|low[- ]frequency|"
    r"terminolog\w*|transliterat\w*|copy(ing)? mechanism|"
    r"lexical(ly)?[- ]constrained|placeholder|vocabulary|subword)\b",
    re.IGNORECASE,
)
TASK_RE = re.compile(
    r"\b(speech recognition|asr|automatic speech|"
    r"machine translation|nmt|translation|speech translation|st)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# LLM judge (filter_results_llm.py)
# ---------------------------------------------------------------------------

# Sonnet 4.6 prompt that decides whether a paper is on-topic. Cached behind
# a `cache_control` breakpoint, so verbosity is fine — paid for once per
# run and read at ~0.1× cost on every subsequent paper.
#
# Structure that works well:
#   1. One-line role priming
#   2. Survey topic in **bold**
#   3. Two-condition rubric (BOTH must hold)
#   4. RELATED examples (5-10 specific kinds of work)
#   5. NOT RELATED examples (5-10 adjacent-but-out-of-scope kinds — these
#      are the ones a regex gate would mistakenly admit)
#   6. "Be strict. When in doubt, mark NOT RELATED."
#   7. Output format reminder
TOPIC_RUBRIC = """You are a senior researcher screening papers for inclusion in a literature survey.

Survey topic: **handling new words, rare words, and named entities that are unseen during training or under-trained, in automatic speech recognition (ASR) and machine translation (MT)** (including speech translation).

A paper is RELATED if and only if BOTH conditions hold:
  1. Its primary task is ASR, MT, or speech translation.
  2. A central focus of the work is improving how the model handles vocabulary it has limited or no training exposure to — rare words, OOV words, hotwords, named entities, terminology, lexically constrained content, long-tail distribution items, transliterated foreign names, or new words added post-training.

Examples of RELATED:
- Contextual biasing / hotword boosting in end-to-end ASR
- OOV or rare-word recovery in ASR via lexical / subword / shallow-fusion methods specifically targeting rare items
- Named-entity-aware NMT; transliteration of NEs in translation
- Lexically-constrained or terminology-controlled MT
- Methods to inject new vocabulary into a deployed ASR/MT model
- Long-tail entity translation; rare-word translation evaluation
- Personalization or contextual adaptation specifically aimed at rare names/words (contacts, hotwords, biasing lists)

Examples of NOT RELATED:
- Generic ASR/MT architecture or training improvements (RNN-T variants, transformer tricks, regularization) without a rare-word focus
- Multilingual NMT methods that don't specifically address rare items
- Audio-visual ASR, generic code-switching ASR, low-resource MT — UNLESS the paper specifically targets rare-word / NE handling
- Pure text NER (no ASR, no MT) — out of scope
- Generic subword tokenization papers (BPE, SentencePiece, character models) without an explicit rare-word focus
- Personalization or domain adaptation that doesn't center on rare words/NEs
- Large-vocabulary LM training tricks (softmax approximation, hierarchical softmax) without a rare-word focus
- Speech recognition data selection / language-model adaptation papers without an explicit rare-word focus
- Generic decoding / lattice generation / WFST work for large vocab
- Open-vocabulary translation methods that focus on robustness to noisy text rather than rare-word handling

Be strict. When in doubt, mark NOT RELATED — the survey author would rather lose a borderline paper than include an off-topic one.

Output `related` (true/false) and `reason` (≤ 30 words). In the reason, cite the specific signal (or its absence) from the title/abstract — e.g., "explicitly targets rare-word ASR via biasing list", or "RNN-T architecture paper, no rare-word mechanism mentioned"."""


# ---------------------------------------------------------------------------
# Synthesis (synthesize_field.py)
# ---------------------------------------------------------------------------

# Opus 4.7 synthesis system prompt. Includes:
#   - Topic statement
#   - Venue families (so the model can recognize cross-pollination)
#   - Bullet list of 8-12 expected sub-areas (helps clustering quality)
TOPIC_HEADER = """We are surveying a research area: **handling new words, rare words, and named entities that are unseen during training or under-trained, in automatic speech recognition (ASR) and machine translation (MT)** — including speech translation.

The corpus spans 2000–2026 across speech (ICASSP, Interspeech, ASRU, SLT), NLP (ACL, EMNLP, NAACL), and ML (ICLR, ICML, NeurIPS) venues. Citation-graph expansion may have pulled in some off-topic papers (e.g., a generic GPT-3 paper cited by on-topic seeds). Synthesis must be tolerant of this.

Sub-areas you should look for include (but are not limited to):
- Contextual biasing / hotword boosting in E2E ASR (RNN-T, attention, biasing-list encoders, deep / shallow fusion)
- Subword and character methods originally motivated by rare-word handling in NMT (BPE, SentencePiece, hybrid word-character)
- Named-entity-aware NMT and entity translation (NE retrieval, entity-aware encoders, denoising entity pre-training)
- Lexically-constrained / terminology-controlled MT
- Transliteration of foreign / cross-script names
- Personalization (contacts, user-specific vocabulary, on-device adaptation)
- LM rescoring / shallow fusion targeting rare items in ASR
- Retrieval-augmented and in-context learning for rare items (LLM-era)
- Pronunciation learning and grapheme-to-phoneme for new words
- OOV detection and recovery in legacy / hybrid ASR
- Long-tail / low-frequency word handling in NMT

Use precise, technique-level terminology. Cite specific papers by author + year inline."""
