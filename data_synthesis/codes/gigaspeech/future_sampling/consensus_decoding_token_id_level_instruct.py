#!/usr/bin/env python3
"""Consensus decoding with future sampling via vLLM serve APIs.

Each sampler model generates one coordinated response containing distinct
plausible and contrastive continuations of the observed source prefix. A
separate translation/probe model provides next-token distributions for each
hypothesized full source, and consensus selects the committed target token.
"""
import argparse
import ast
import glob
import http.client
import json
import math
import os
import re
import sys
import time
import threading
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer

from ambiguity_sampler_prompt import (
    build_open_word_messages, build_plausible_messages, build_reading_messages, open_word_schema, plausible_schema,
    reading_schema,
    MAX_SUFFIX_CHARS,
    PROMPT_VERSION, PROMPT_VERSIONS, SUFFIX_ICL_PROMPT_VERSION,
    build_coordinated_future_messages, parse_grouped_future_output, sample_grouped_futures,
    structured_output_extras,
)
from ambiguity_list import (
    check_messages, check_schema, extend_to_sentence, natural_messages, natural_schema, open_entries, request as list_request, sample_messages,
    sample_schema,
)
from transformers import PreTrainedTokenizerBase
from sentence_boundary_helpers import (
    close_translation_delta, observed_sentence_is_complete,
    sentence_anchored_prefix, source_terminal_mark,
)


DEFAULT_TSV_PATH = "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference_subsentence_ref.tsv"
DEFAULT_INSTRUCT_API_BASE = os.environ.get("INSTRUCT_API_BASE", "")
DEFAULT_INSTRUCT_API_MODEL = os.environ.get("INSTRUCT_API_MODEL", "qwen3-instruct")
TOP_K = 6
MIN_P = 0.0
# Probe tokens with ASCII letters used to be dropped before the vote and at commit, so an English term (CDC, Kubernetes,
# iPhone) could never be voted in and the decoder waited until the unfiltered sentence-end completion or final flush
# (wiki experiments/2026-09-latin-hold-ascii-filter.md). Since 2026-09-29 they are allowed; --block-latin-tokens
# restores the old rule. Set once from the command line in main().
BLOCK_LATIN_TOKENS = False


def setup_env() -> None:
    os.environ.setdefault("HF_HOME", "/data/user_data/haolingp/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/data/user_data/haolingp/hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data/user_data/haolingp/hf_cache/transformers")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consensus decoding with future sampling via vLLM serve.")
    p.add_argument("--input-tsv", default=DEFAULT_TSV_PATH)
    p.add_argument("--id-column", default="id")
    # primary base model (future sampling)
    p.add_argument("--base-model-path", default="")
    p.add_argument("--base-api-base", default="",
                   help="Required unless --use-targeted-instruct-sampling is set.")
    p.add_argument("--base-api-model", default="",
                   help="Required unless --use-targeted-instruct-sampling is set.")
    p.add_argument("--base-api-timeout", type=float, default=120.0)
    # secondary base model (future sampling)
    p.add_argument("--secondary-base-model-path", default="")
    p.add_argument("--secondary-base-api-base", default="")
    p.add_argument("--secondary-base-api-model", default="")
    p.add_argument("--secondary-base-api-timeout", type=float, default=120.0)
    # instruct model (next-token distribution)
    p.add_argument("--instruct-tokenizer-path", required=True)
    p.add_argument("--instruct-api-base", required=True)
    p.add_argument("--instruct-api-model", default=DEFAULT_INSTRUCT_API_MODEL)
    p.add_argument("--instruct-api-timeout", type=float, default=120.0)
    # sampling / decoding
    p.add_argument("--num-futures", type=int, default=20)
    p.add_argument("--secondary-num-futures", type=int, default=10)
    p.add_argument("--future-tokens", type=int, default=10)
    p.add_argument("--sample-temperature", type=float, default=1.0)
    # Coordinated instruct future sampling bypasses independent base-LM sampling.
    p.add_argument("--use-targeted-instruct-sampling", action="store_true",
                   help="Use one or two instruct samplers to generate a jointly planned, "
                        "numbered continuation set instead of independent base-LM samples. "
                        "When set, base_specs are ignored for future sampling.")
    p.add_argument("--targeted-num-futures", type=int, default=20,
                   help="Continuations per sampler model (an upper limit for v3); must be even, "
                        "with half the budget for each group.")
    p.add_argument("--targeted-prompt-version", choices=PROMPT_VERSIONS, default=PROMPT_VERSION,
                   help="Opt-in v3 adds suffix examples and variable-size groups. "
                        "Default v2 preserves the historical fixed-count prompt and parser.")
    p.add_argument("--targeted-sample-temperature", type=float, default=1.0,
                   help="Sampling temperature for each coordinated future-set response.")
    p.add_argument("--targeted-top-p", type=float, default=0.98)
    p.add_argument("--targeted-max-tokens", type=int, default=40,
                   help="Minimum output budget; coordinated-list generation scales it by candidate count.")
    p.add_argument("--targeted-sampler-tokenizer-path", default="",
                   help="HF path/name of the tokenizer matching the sampler model. "
                        "Required because the sampler may use a different chat template "
                        "than the probe (e.g. gemma vs qwen). Defaults to --instruct-tokenizer-path.")
    # Optional split: route the sampler to a different endpoint than the probe so
    # the model that *generates* futures is not the same model being *probed* for
    # next-token distributions (mitigates shared-blindspot confirmation bias).
    # Defaults are empty → fall back to --instruct-api-* (same-model mode).
    p.add_argument("--targeted-sampler-api-base", default="",
                   help="If set, route the targeted-instruct sampler to this endpoint "
                        "(cross-family setup, e.g. gemma-4-it). Defaults to --instruct-api-base.")
    p.add_argument("--targeted-sampler-api-model", default="",
                   help="Served-model-name of the sampler endpoint. Defaults to --instruct-api-model.")
    p.add_argument("--targeted-sampler-api-timeout", type=float, default=0.0,
                   help="Timeout for sampler API. 0 or negative → fall back to --instruct-api-timeout.")
    # Optional second sampler (ensemble across families). Each configured model
    # returns one coordinated set; both models remain independent consensus voters.
    p.add_argument("--targeted-sampler2-tokenizer-path", default="",
                   help="HF path/name of the 2nd sampler's tokenizer (different family). "
                        "Enables dual-model ensemble when all 3 sampler2-* flags are set.")
    p.add_argument("--targeted-sampler2-api-base", default="",
                   help="API endpoint of the 2nd sampler (e.g. http://localhost:8100/v1 to "
                        "reuse the probe's Qwen3 server).")
    p.add_argument("--targeted-sampler2-api-model", default="",
                   help="Served-model-name of the 2nd sampler endpoint.")
    p.add_argument("--targeted-sampler2-api-timeout", type=float, default=0.0,
                   help="Timeout for 2nd sampler API. 0 or negative → fall back to --instruct-api-timeout.")
    p.add_argument("--max-consensus-steps", type=int, default=12)
    p.add_argument("--vote-match", choices=["exact", "prefix", "prefix_semantic"], default="exact",
                   help="2026-10-09 vote test. exact: every voter must list the same token id. prefix: a voter also "
                        "supports token A when one of its candidates' text starts with A's text (理由是 supports 理由). "
                        "prefix_semantic: in addition, a candidate whose LaBSE cosine with A is >= --semantic-threshold "
                        "supports A (女子 / 妇女).")
    p.add_argument("--semantic-model", default="/data/user_data/haolingp/models/LaBSE")
    p.add_argument("--list-natural", choices=["qwen", "gemma", "none"], default="qwen",
                   help="--future-roles list: which sampler writes the natural continuations (qwen = sampler 2).")
    p.add_argument("--list-natural-n", type=int, default=6)
    p.add_argument("--list-per-reading", type=int, default=2)
    p.add_argument("--list-targeted-thinking", action="store_true",
                   help="--future-roles list: targeted continuations with thinking, --list-think-budget reasoning tokens "
                        "then the answer is forced.")
    p.add_argument("--list-think-budget", type=int, default=2000)
    p.add_argument("--anchor-reference", action="store_true",
                   help="Write the row's reference translation; the futures only decide when (see patch_anchor.py).")
    p.add_argument("--anchor-min-p", type=float, default=0.02)
    p.add_argument("--anchor-ratio-floor", type=float, default=0.0,
                   help="--anchor-ratio yardstick = max(P(token | true continuation), this floor).")
    p.add_argument("--anchor-ratio", type=float, default=0.0,
                   help="With --anchor-reference: > 0 writes a reference token iff every sampled future gives it at least "
                        "this fraction of the true continuation's probability (exact, prompt_logprobs; see patch_ratio.py).")
    p.add_argument("--list-reading-check", action="store_true",
                   help="--future-roles list: a separate call per open word keeps only the readings still possible.")
    p.add_argument("--list-probe-hint", action="store_true",
                   help="--future-roles list: the translator probe gets each targeted future's reading as a [NOTE].")
    p.add_argument("--list-skip-ruled-out", action="store_true",
                   help="--future-roles list: the targeted sampler skips readings the prefix already rules out.")
    p.add_argument("--future-roles", choices=["off", "two_layer", "given_word", "list"], default="off",
                   help="2026-10-09 vote test. two_layer: sampler 1 (Gemma) returns --role-plausible-n natural "
                        "continuations; sampler 2 (Qwen3.8) first lists the observed words whose meaning is still open "
                        "with their readings, then writes --role-per-reading continuations per reading. given_word: "
                        "sampler 2 skips the first step and uses the row's ambiguous_word / ambiguous_readings once the "
                        "word has been heard.")
    p.add_argument("--role-plausible-n", type=int, default=10)
    p.add_argument("--role-per-reading", type=int, default=2)
    p.add_argument("--role-max-words", type=int, default=2)
    p.add_argument("--role-max-readings", type=int, default=3)
    p.add_argument("--semantic-threshold", type=float, default=0.93)
    p.add_argument("--single-token-filter", action="store_true",
                   help="2026-10-09 vote test, with --min-consensus-horizon 1: a lone agreed token is committed unless it "
                        "is punctuation only or its lowest voter probability is below --single-token-min-p.")
    p.add_argument("--single-token-min-p", type=float, default=0.05)
    p.add_argument("--min-consensus-horizon", type=int, default=2,
                   help="Minimum number of consensus-confirmed tokens required before committing. "
                        "If the pending buffer ends up shorter than this (after consensus breaks), "
                        "discard all pending tokens and READ instead. Default 2: avoid single-token "
                        "commits that the translator may misinterpret (reduces early lock-in).")
    p.add_argument("--future-join-mode", choices=["space", "sentence-aware"], default="space",
                   help="How a sampled future is appended to the observed source for the probe. "
                        "'sentence-aware': if the observed prefix is mid-sentence and the future starts a "
                        "new capitalized sentence, join with '. ' (and drop a trailing comma) so the probe "
                        "sees a well-formed 'sentence ends here' hypothesis instead of a run-on.")
    p.add_argument("--targeted-sampler-seed", type=int, default=None,
                   help="If set, pass a deterministic per-prefix seed to the coordinated future samplers "
                        "(vLLM 'seed'), so reruns of the same row are comparable.")
    p.add_argument("--targeted-sampler-context", choices=["source-and-target", "source-only"],
                   default="source-and-target",
                   help="Opt-in source-only removes committed target text from both sampler prompt and seed.")
    p.add_argument("--max-dropped-rows", type=int, default=0,
                   help="Rows whose decode raises (e.g. the sentence-end guard refusing a degenerate "
                        "completion) are recorded and skipped instead of failing the task, as long as "
                        "no more than this many are dropped. 0 keeps the historical fail-fast behaviour.")
    p.add_argument("--dropped-rows-log", default="",
                   help="JSONL file receiving one record per dropped row (default: "
                        "<verbose-dir>/dropped_rows.jsonl when --verbose-dir is set).")
    p.add_argument("--contrastive-notes", action="store_true",
                   help="v3 JSON only: each contrastive item must also name the reading it resolves "
                        "(object with suffix and resolves); the note is kept in the audit, not in the future.")
    p.add_argument("--block-latin-tokens", action="store_true",
                   help="Drop probe tokens that contain ASCII letters, in the vote and at commit (the rule before "
                        "2026-09-29). Default: English terms can be voted in on time.")
    p.add_argument("--oracle-future-vote", action="store_true",
                   help="Let the true source continuation cast a vote as future N+1 (implies "
                        "--oracle-future-audit). NOTE: with --min-voters-ratio 1.0 this gives one "
                        "draw the same veto power as the whole sampled set.")
    p.add_argument("--probe-temperature", type=float, default=0.0,
                   help="Sampling temperature on the probe /completions request. 0.0 (default) "
                        "keeps the historical greedy/unscaled distribution.")
    p.add_argument("--probe-top-p", type=float, default=0.0,
                   help="top_p on the probe request; 0 = omit.")
    p.add_argument("--probe-thinking", choices=["off", "all", "oracle"], default="off",
                   help="Probe thinking (2026-10-06): once per chunk, generate a reasoning (Qwen3.6 thinking mode, THINKING_SAMPLING) for "
                        "every probe source (all) or only the true continuation (oracle); every probe step of the "
                        "chunk then reads the next-token distribution after that reasoning. off = historical probe.")
    p.add_argument("--probe-thinking-max-tokens", type=int, default=8192,
                   help="Token cap of one reasoning (probe and completion); a reasoning cut there is used as it is.")
    p.add_argument("--completion-thinking", action="store_true",
                   help="The sentence-end and final completions reason first (thinking mode), then complete greedily.")
    p.add_argument("--probe-top-k", type=int, default=0,
                   help="top_k on the probe request; 0 = omit.")
    p.add_argument("--oracle-future-audit", action="store_true",
                   help="Audit only, never changes a commit: probe the TRUE source continuation "
                        "alongside the sampled futures and record whether it would have vetoed the "
                        "token the sampled futures unanimously committed. Measures how often the "
                        "sampler misses the real ambiguity.")
    p.add_argument("--probe-prompt-order", choices=["historical", "shared-first"], default="historical",
                   help="Order of the probe prompt sections. 'shared-first' puts the fixed [IMPORTANT] instruction "
                        "before [INPUT] so vLLM prefix caching covers everything up to the future; wording is identical.")
    p.add_argument("--targeted-parse-retries", type=int, default=2,
                   help="Resample a sampler whose grouped (v3) response is malformed, up to this many "
                        "extra attempts with a shifted seed; every malformed response is logged verbatim.")
    p.add_argument("--malformed-response-log", default="",
                   help="JSONL file receiving every malformed sampler response (default: "
                        "<verbose-dir>/malformed_sampler_responses.jsonl when --verbose-dir is set).")
    p.add_argument("--targeted-fail-on-api-error", action="store_true",
                   help="Fail a pilot case on sampler API errors instead of treating them as empty futures.")
    p.add_argument("--future-source-window-mode", choices=["fixed", "until-closed", "sentence-anchor"], default="fixed",
                   help="'fixed': always --future-source-window-chunks units. 'until-closed': additionally "
                        "include the previous sentence unit while the Chinese committed for it does not "
                        "yet end with sentence-final punctuation; drop it once it is closed. "
                        "'sentence-anchor': use only the observed current sentence, not source-unit boundaries.")
    p.add_argument("--future-source-anchor-max-words", type=int, default=128,
                   help="Word cap for sentence-anchor only; 0 disables the cap.")
    p.add_argument("--sentence-end-completion", action="store_true",
                   help="When the current source chunk ends a sentence, skip consensus and let the "
                        "translator finish that sentence (same call as the final chunk).")
    p.add_argument("--sentence-end-boundary-mode", choices=["literal", "conservative"], default="literal",
                   help="Boundary detector for sentence-end completion. Conservative defers ambiguous "
                        "abbreviations, initials, ellipses and number-final periods. Default preserves prior pilots.")
    p.add_argument("--sentence-end-punctuation", choices=["off", "match-source"], default="off",
                   help="Opt-in Chinese boundary completion: request and normalize only the new delta's terminal mark.")
    p.add_argument("--final-max-tokens", type=int, default=128,
                   help="Maximum tokens for the final tail-completion step.")
    p.add_argument("--candidate-top-k", type=int, default=TOP_K)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.0,
                   help="Nucleus (top-p) candidate selection: keep smallest set with cumulative prob >= top-p.")
    p.add_argument("--soft-vote-top-k", type=int, default=20,
                   help="Per-future top-N cap (safety) before voting.")
    p.add_argument("--soft-vote-min-p", type=float, default=0.1,
                   help="Per-future per-token min probability for entering the vote. "
                        "Tokens with p < this are dropped from that future before voting.")
    p.add_argument("--min-voters-ratio", type=float, default=0.75,
                   help="A token is eligible to win iff at least this fraction of futures "
                        "contributed it (after min-p + top-K filter). I.e., ≥ ceil(N*ratio) "
                        "futures must list the token in their filtered top-K. Looser-than-"
                        "intersection: ratio=1.0 ↔ strict intersection; <1.0 lets majority "
                        "agreement override a minority that ranks the token lower.")
    p.add_argument("--soft-vote-threshold", type=float, default=0.0,
                   help="DEPRECATED — kept for CLI backward compat, ignored.")
    # target language
    p.add_argument("--target-lang", default="Chinese",
                   help="Target language name for prompts (e.g. Chinese, Japanese, German)")
    p.add_argument(
        "--future-source-window-chunks",
        type=int,
        default=0,
        help="For future sampling, keep only the most recent N observed source chunks. "
             "Use 0 to keep the full observed prefix.",
    )
    # output
    p.add_argument("--output-jsonl", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-dir", default=None)
    p.add_argument(
        "--compact-verbose",
        action="store_true",
        help="Keep candidate/filter and consensus summaries, but omit the repeated "
             "per-future next-token probability rows.",
    )
    p.add_argument("--row-idx", type=int, default=0)
    p.add_argument("--utt-id", default=None)
    p.add_argument("--max-rows", type=int, default=1)
    p.add_argument("--test-one", action="store_true")
    p.add_argument("--num-concurrent-cases", type=int, default=8)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip rows whose per-utterance output JSON already exists.")
    p.add_argument(
        "--skip-existing-root",
        default="",
        help="Also skip utterance filenames already present under "
             "ROOT/task_*/per_utt. This supports safe repartitioning on resume.",
    )
    args = p.parse_args()
    if args.sentence_end_punctuation == "match-source" and (
        not args.sentence_end_completion or args.sentence_end_boundary_mode != "conservative"
        or args.target_lang.lower() != "chinese"
    ):
        p.error("match-source requires Chinese, --sentence-end-completion and conservative boundaries")
    return args


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def parse_trajectory(raw: str) -> List[str]:
    return ast.literal_eval(raw)


def join_source_chunks(chunks: List[str]) -> str:
    text = ""
    for raw_piece in chunks:
        piece = str(raw_piece or "")
        if not piece:
            continue
        if not text:
            text = piece
            continue
        if text[-1].isspace() or piece[0].isspace():
            text += piece
        elif piece[0] in ",.!?;:)]}\"'":
            text += piece
        elif text[-1] in "([{\"'":
            text += piece
        else:
            text += " " + piece
    return text.strip()


def build_source_observed(chunks: List[str], t: int) -> str:
    return join_source_chunks(chunks[: t + 1])


def parse_source_units(raw: Any) -> List[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [str(x or "") for x in parsed]
    return [str(parsed)]


def build_source_observed_recent_units(
    source_units: List[str],
    observed_full: str,
    num_units: int,
) -> str:
    if not observed_full:
        return ""
    if not source_units or num_units <= 0:
        return observed_full

    prev_full = ""
    for unit_idx, unit in enumerate(source_units):
        full_through_unit = append_text_continuation(prev_full, unit)
        if observed_full == full_through_unit or full_through_unit.startswith(observed_full):
            start_idx = max(0, unit_idx - num_units + 1)
            prefix = ""
            for prior_unit in source_units[start_idx:unit_idx]:
                prefix = append_text_continuation(prefix, prior_unit)
            partial_current = observed_full[len(prev_full):] if observed_full.startswith(prev_full) else observed_full
            return append_text_continuation(prefix, partial_current)
        prev_full = full_through_unit

    return observed_full


def oracle_future_suffix(source_units: List[str], observed_full: str,
                         max_chars: int = MAX_SUFFIX_CHARS) -> str:
    """The TRUE continuation of the source, on the same budget a sampled future gets.

    A sampled future is a suffix of up to MAX_SUFFIX_CHARS that the sampler is told to
    carry past the end of the open grammatical structure. The oracle has to span the
    same horizon or the comparison is rigged: about half the units of `src_text_full`
    end at a comma rather than a sentence end, so stopping at the unit boundary would
    hand the oracle a systematically shorter view than the futures it is being
    compared against, and it would under-veto. So this walks THROUGH unit boundaries
    and cuts on the same character budget, backing off to a word boundary rather than
    splitting a token.

    This is for the audit in `extend_pending_tokens`. It is never a voter: the decoder
    must stay a function of what the student can see, or the targets teach a policy the
    student cannot reproduce. Returns "" when the prefix already covers everything.
    """
    if not source_units or not observed_full:
        return ""
    full = ""
    for unit in source_units:
        full = append_text_continuation(full, unit)
    if not full.startswith(observed_full):
        return ""
    remainder = full[len(observed_full):].strip()
    if len(remainder) <= max_chars:
        return remainder
    cut = remainder[:max_chars]
    space = cut.rfind(" ")
    return (cut[:space] if space > max_chars // 2 else cut).rstrip()


def current_unit_index(source_units: List[str], observed_full: str) -> int:
    """Index of the sentence unit that the observed prefix currently ends inside (0 if unknown)."""
    if not source_units or not observed_full:
        return 0
    prev_full = ""
    for unit_idx, unit in enumerate(source_units):
        full_through_unit = append_text_continuation(prev_full, unit)
        if observed_full == full_through_unit or full_through_unit.startswith(observed_full):
            return unit_idx
        prev_full = full_through_unit
    return len(source_units) - 1


_TARGET_CLOSED_RE = re.compile(r"[。！？.!?][\"\u201d\u2019'\u300d\u300f)\]]*\s*$")


def target_is_closed(committed_text: str) -> bool:
    """True when the committed target text ends a sentence."""
    return bool(committed_text.strip()) and bool(_TARGET_CLOSED_RE.search(committed_text.rstrip()))


def get_full_source_text(row: Dict[str, Any]) -> str:
    raw = row.get("src_text")
    if raw is None or pd.isna(raw):
        raise ValueError("src_text is missing from input row")
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        raise ValueError("src_text is empty in input row")
    return text


def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))


def append_text_continuation(prefix: str, continuation: str) -> str:
    if not prefix:
        return continuation
    if not continuation:
        return prefix
    if prefix[-1].isspace() or continuation[0].isspace():
        return prefix + continuation
    if continuation[0] in ",.!?;:)]}\"'":
        return prefix + continuation
    return prefix + " " + continuation


_SENTENCE_INITIAL_WORDS = {
    "he", "she", "it", "they", "the", "a", "an", "but", "and", "then", "his", "her", "we", "i",
    "you", "this", "that", "there", "when", "as", "so", "at", "in", "on", "after", "before",
    "with", "now", "suddenly", "however", "meanwhile", "everyone", "no", "yes", "all", "his",
    "their", "its", "my", "our", "some", "every", "each", "nothing", "nobody", "one",
}
_SENTENCE_END_RE = re.compile(r"[.!?][\"\u201d\u2019')\]]*\s*$")


_CANNOT_END_SENTENCE = {
    "the", "a", "an", "of", "to", "and", "but", "or", "in", "on", "at", "for", "with", "by",
    "from", "as", "than", "that", "which", "who", "whose", "whom", "is", "are", "was", "were",
    "be", "been", "being", "had", "has", "have", "will", "would", "could", "should", "can",
    "may", "might", "must", "not", "very", "so", "into", "onto", "about", "over", "under",
    "his", "her", "their", "its", "my", "our", "your", "this", "these", "those", "some", "any",
    "every", "each", "no", "if", "when", "while", "because", "though", "although", "until",
}


def looks_like_sentence_restart(prefix: str, future: str) -> bool:
    """True when the prefix could end a sentence here and the future begins a new capitalized one."""
    if not prefix or not future:
        return False
    stripped = prefix.rstrip()
    if _SENTENCE_END_RE.search(stripped):
        return False
    last = re.findall(r"[A-Za-z']+", stripped)
    if last and last[-1].lower() in _CANNOT_END_SENTENCE:
        return False
    match = re.match(r"[\"\u201c(]?([A-Za-z][A-Za-z']*)", future.strip())
    if not match:
        return False
    word = match.group(1)
    return word[0].isupper() and word.lower() in _SENTENCE_INITIAL_WORDS


def join_future_to_source(prefix: str, future: str, mode: str = "space") -> str:
    """Append a sampled future to the observed source for the probe prompt."""
    if mode == "sentence-aware" and looks_like_sentence_restart(prefix, future):
        base = prefix.rstrip()
        base = base[:-1].rstrip() if base and base[-1] in ",;:" else base
        return base + ". " + future.strip()
    return append_text_continuation(prefix, future)


# ---------------------------------------------------------------------------
# Verbose logging (lightweight)
# ---------------------------------------------------------------------------

def _vlog(f: Optional[Any], msg: str) -> None:
    if f is None:
        return
    line = str(msg) if msg.endswith("\n") else str(msg) + "\n"
    f.write(line)


def _vlog_pretty_value(f: Optional[Any], label: str, value: Any) -> None:
    if f is None:
        return
    if isinstance(value, (list, dict)):
        pretty = json.dumps(value, ensure_ascii=False, indent=2)
        lines = pretty.splitlines()
        if not lines:
            _vlog(f, f"# {label}: []")
            return
        _vlog(f, f"# {label}: {lines[0]}")
        for line in lines[1:]:
            _vlog(f, f"#   {line}")
        return
    _vlog(f, f"# {label}: {value}")


class _TeeWriter:
    """Write to both a file and stdout (used only for --test-one)."""
    def __init__(self, fobj: Any):
        self._f = fobj
    def write(self, msg: str) -> None:
        self._f.write(msg); sys.stdout.write(msg)
    def flush(self) -> None:
        self._f.flush(); sys.stdout.flush()
    def close(self) -> None:
        self._f.close()


def write_pretty_json(path: str, data: Dict[str, Any]) -> None:
    """Write atomically (temp file + rename) so a preempted worker never leaves a truncated
    JSON that --skip-existing would later treat as complete."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_model_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.split("<|im_end|>")[0]
    text = text.split("<|endoftext|>")[0]
    return text.strip()


def strip_markdown_wrappers(text: str) -> str:
    text = str(text or "").strip()
    # Gemma-style completions sometimes wrap continuations in markdown bold.
    # The continuation itself can still be useful, but the markup must never
    # enter either the source future or the committed Chinese token stream.
    text = re.sub(r"^\*{1,3}\s*(.*?)\s*\*{1,3}$", r"\1", text)
    text = re.sub(r"^`+\s*(.*?)\s*`+$", r"\1", text)
    return text.strip()


def clean_future_text(observed_source: str, raw_text: str) -> str:
    def strip_prefix_ci(value: str, prefix: str) -> Optional[str]:
        """Strip an English prefix case-insensitively at word boundaries."""
        prefix = prefix.strip()
        if not prefix or len(value) < len(prefix):
            return None
        if value[:len(prefix)].casefold() != prefix.casefold():
            return None
        if len(value) > len(prefix) and prefix[-1].isalnum() and value[len(prefix)].isalnum():
            return None
        return value[len(prefix):].lstrip()

    text = strip_markdown_wrappers(clean_model_text(raw_text))
    stripped = strip_prefix_ci(text, observed_source)
    if stripped is not None:
        text = stripped
    text = text.splitlines()[0].strip() if text else ""
    text = strip_markdown_wrappers(text)
    # Strip leading placeholders like "..." or "…" that instruct models sometimes prepend.
    text = re.sub(r"^[\.\s…\-]+", "", text)
    # If the cleaned continuation still starts by repeating the trailing words of the
    # observed source (instruct-model habit), strip the longest matching suffix-prefix.
    obs_trailing = observed_source.strip()
    if obs_trailing and text:
        # Try the longest word-boundary suffix that matches, ignoring capitalization.
        for k in range(min(len(obs_trailing), len(text)), 2, -1):
            if k < len(obs_trailing) and obs_trailing[-k - 1].isalnum():
                continue
            tail = obs_trailing[-k:]
            stripped = strip_prefix_ci(text, tail)
            if stripped is not None:
                text = stripped
                break
    return text


def is_valid_future_text(text: str) -> bool:
    if not text:
        return False
    if any(ch in text for ch in ("*", "`")):
        return False
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text):
        return False
    lowered = text.lower()
    banned = [
        "translate", "translation", "grammar analysis", "analyze", "analysis",
        "<think>", "</think>", "assistant turn", "user prompt", "the prompt",
        "sampling mode", "unfinished english prefix", "4-15 words", "let's think",
        "这句话", "翻译", "语法", "句子结构",
    ]
    if len(text.split()) > 20:
        return False
    return not any(b in lowered for b in banned)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Method A: instruct-model future sampling via /chat/completions
# Generates N diverse continuations in a single call as a numbered list.
# ---------------------------------------------------------------------------

def build_future_sampling_chat_messages(observed_source: str, num_futures: int) -> List[Dict[str, str]]:
    """Method A: ask the model to produce a numbered list of `num_futures` continuations in one call."""
    return [
        {"role": "system", "content": (
            "You are an expert linguist predicting how an incomplete spoken English sentence "
            "might continue. Stay on the same domain and style as the input. Generate diverse "
            "continuations that vary in topic emphasis and sentence structure but remain plausible."
        )},
        {"role": "user", "content": (
            f"Generate exactly {num_futures} different continuations of this English text. "
            f"Each should be 15-30 words. Output English only (no Chinese, no analysis).\n\n"
            f"IMPORTANT: each numbered item must contain ONLY the new words that come AFTER the "
            f"input — do NOT repeat any words from the input text, do NOT prepend '...' or any "
            f"placeholder. Start each item with the very next word.\n\n"
            f"Text: {observed_source}\n\n"
            f"Format strictly as:\n1. <new words after input>\n2. <new words after input>\n"
            f"...\n{num_futures}. <new words after input>"
        )},
    ]


def build_future_sampling_chat_messages_single(observed_source: str) -> List[Dict[str, str]]:
    """Method B: ask the model for ONE continuation; caller invokes with n=num_futures for independent samples."""
    return [
        {"role": "system", "content": (
            "You are an expert linguist predicting how an incomplete spoken English sentence "
            "might continue. Stay on the same domain and style as the input."
        )},
        {"role": "user", "content": (
            f"Continue this English text with one plausible continuation of 15-30 words. "
            f"Output English only (no Chinese, no analysis).\n\n"
            f"IMPORTANT: output ONLY the new words that come AFTER the input — do NOT repeat any "
            f"words from the input text, do NOT prepend '...' or any placeholder. Start with the "
            f"very next word.\n\n"
            f"Text: {observed_source}\n\n"
            f"Continuation:"
        )},
    ]


_NUMBERED_LIST_RE = re.compile(r"^\s*(\d+)[\.\)]\s*(.+?)\s*$", re.MULTILINE)


def parse_method_a_output(raw_text: str, num_expected: int) -> List[str]:
    """Parse a numbered list response (1. ... 2. ... N. ...) into a list of continuations."""
    if not raw_text:
        return []
    items: List[Tuple[int, str]] = []
    for match in _NUMBERED_LIST_RE.finditer(raw_text):
        idx = int(match.group(1))
        text = match.group(2).strip()
        # Some models output "1. ... so it was in <continuation>" — strip leading "..." or "so it was in"
        text = re.sub(r"^[\.\s\-…]+", "", text)
        if 1 <= idx <= num_expected and text:
            items.append((idx, text))
    items.sort(key=lambda x: x[0])
    seen_idx: set = set()
    out: List[str] = []
    for idx, text in items:
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        out.append(text)
    return out


def _is_chat_endpoint_model(api_model: str) -> bool:
    """Decide whether to route to /chat/completions (instruct-style) or /completions (raw LM).

    Heuristic: model name contains an instruct suffix ('-it', 'instruct', 'chat').
    Examples:
      gemma4-e2b-it, qwen3-instruct, llama-3-chat → /chat/completions
      gemma4-e2b, qwen3-4b-base                    → /completions (raw text continuation)
    """
    name = str(api_model or "").lower()
    return any(tag in name for tag in ("-it", "instruct", "chat"))


def build_translation_probe_prompt_prefix_token_ids(
    tokenizer: Any,
    full_source: str,
    has_target_prefix: bool,
    target_lang: str = "Chinese",
    shared_first: bool = False,
    enable_thinking: bool = False,
    note: str = "",
) -> List[int]:
    task = f"[TASK]\nTranslate the [INPUT] text into {target_lang}."
    source = f"[INPUT]\n{full_source}"
    rule = (
        f"[IMPORTANT]\nA partial {target_lang} translation is already committed "
        "at the start of the assistant reply. You must continue from that "
        "exact prefix and produce only the continuation."
        if has_target_prefix else
        f"[IMPORTANT]\nStart the {target_lang} translation from the beginning "
        "and output only the next continuation token(s)."
    )
    # Historical order is [TASK][INPUT][IMPORTANT]. shared_first moves the fixed
    # instruction ahead of the source so vLLM prefix caching covers everything up
    # to the future-specific tail; the wording is unchanged.
    parts = (task, rule, source) if shared_first else (task, source, rule)
    if note:   # --list-probe-hint: the reading the sampler wrote this future for
        parts = parts + (f"[NOTE]\n{note}",)
    else:
        pass
    messages = [{"role": "user", "content": "\n\n".join(parts)}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=enable_thinking,
    )
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids.get("input_ids", [])
    elif hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    return list(prompt_ids)


def build_final_completion_prompt(tokenizer: Any, full_source: str, committed_text: str, target_lang: str = "Chinese", terminal_mark: str = "",
                                  enable_thinking: bool = False, append_committed: bool = True) -> str:
    if not str(committed_text or "").strip():
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nOutput the complete {target_lang} translation only.\n"
            "Do not add explanations, summaries, examples, lists, numbering, markdown, or background knowledge."
        )}]
    else:
        messages = [{"role": "user", "content": (
            f"[TASK]\nTranslate the [INPUT] text into {target_lang}.\n\n"
            f"[INPUT]\n{full_source}\n\n"
            f"[IMPORTANT]\nA partial {target_lang} translation is already committed "
            "at the start of the assistant reply.\n"
            f"Continue ONLY with the source content that is not yet covered by the committed {target_lang} prefix.\n"
            "Do not repeat the committed prefix.\n"
            "Do not add explanations, summaries, examples, lists, numbering, markdown, or background knowledge.\n"
            "Do not translate beyond the source.\n"
            "If there is no remaining source content to translate, output nothing.\n"
            f"Output only the remaining {target_lang} continuation."
        )}]
    if terminal_mark:
        messages[0]["content"] += (
            f"\nThe observed English sentence is finished. End its Chinese translation with {terminal_mark} "
            "before any closing quotation mark. Do not invent a following sentence or any unobserved content. "
            "If only the terminal punctuation is missing, output that punctuation only."
        )
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )
    if append_committed and str(committed_text or "").strip():
        prompt += committed_text
    else:
        pass
    return prompt


# ---------------------------------------------------------------------------
# Model / API helpers
# ---------------------------------------------------------------------------

def load_tokenizer(path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def normalize_api_base(api_base: str) -> str:
    base = str(api_base or "").strip().rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _http_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode('utf-8', errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e}") from e
    except (TimeoutError, OSError, http.client.HTTPException) as e:
        # A read timeout or a server dropping mid-response surfaces as a bare OS error;
        # name the endpoint so a failed row can be attributed to a server.
        raise RuntimeError(f"Request to {url} failed after {timeout}s: {type(e).__name__}: {e}") from e


def _http_get_json(url: str, timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer dummy"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode('utf-8', errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e}") from e
    except (TimeoutError, OSError, http.client.HTTPException) as e:
        raise RuntimeError(f"Request to {url} failed after {timeout}s: {type(e).__name__}: {e}") from e


def verify_api(api_base: str, timeout: float) -> List[str]:
    data = _http_get_json(f"{normalize_api_base(api_base)}/models", timeout=timeout)
    return [str(item.get("id", "")) for item in data.get("data", []) if item.get("id")]


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _single_token_text(tokenizer: Any, tok_id: int) -> str:
    return tokenizer.decode([tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _disallowed_generation_token_reason(tokenizer: Any, tok_id: int) -> Optional[str]:
    if tok_id is None:
        return "missing_token_id"
    if tok_id in set(getattr(tokenizer, "all_special_ids", []) or []):
        return "special_token_id"
    token_text = _single_token_text(tokenizer, tok_id)
    for frag in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        if frag in token_text:
            return "special_token_text"
    if any(ch in token_text for ch in ("*", "`")):
        return "markdown_token"
    # Qwen's added tag tokens (<think>, </tool_call>, <|fim_prefix|> ...) are not in all_special_ids; the ASCII rule used
    # to drop them by accident, so they need their own rule now that English tokens are allowed.
    if re.fullmatch(r"\s*(</?[A-Za-z_]+>|<\|[A-Za-z_]+\|>)\s*", token_text):
        return "special_token_text"
    if BLOCK_LATIN_TOKENS and re.search(r"[A-Za-z]", token_text):
        return "ascii_letters"
    # NOTE: Do NOT filter replacement_char (U+FFFD) here. Byte-level BPE tokens
    # for rare characters (e.g. Japanese kanji "繰") decode to U+FFFD individually
    # but combine into valid characters when decoded together in the pending buffer.
    if any(ch in {"\u200d", "\ufe0f"} for ch in token_text):
        return "zero_width_or_variation_selector"
    if any(unicodedata.category(ch) in {"Cc", "Cs"} for ch in token_text):
        return "control_or_surrogate"
    return None


def filter_distribution_token_ids(
    tokenizer: Any,
    id_distribution: Dict[int, float],
    dist_debug: Dict[str, Any],
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    filtered: Dict[int, float] = {}
    removed: List[Dict[str, Any]] = []
    for tok_id, prob in id_distribution.items():
        reason = _disallowed_generation_token_reason(tokenizer, tok_id)
        if reason is not None:
            removed.append({"token_id": int(tok_id), "token_text": _single_token_text(tokenizer, tok_id),
                            "prob": round(float(prob), 6), "reason": reason})
            continue
        filtered[int(tok_id)] = float(prob)

    out_debug = dict(dist_debug)
    out_debug["filtered_disallowed_tokens"] = removed
    if filtered:
        out_debug["topk_token_ids"] = list(filtered.keys())
        out_debug["topk_token_texts"] = [_single_token_text(tokenizer, t) for t in filtered]
        out_debug["topk_true_probs"] = [round(float(filtered[t]), 6) for t in filtered]
    elif removed and out_debug.get("reason") == "ok":
        out_debug["reason"] = "all_top_tokens_filtered_as_disallowed"
    return filtered, out_debug


# ---------------------------------------------------------------------------
# Future sampling (API only)
# ---------------------------------------------------------------------------

def sample_source_futures_api(
    observed_source: str,
    num_futures: int,
    future_tokens: int,
    sample_temperature: float,
    api_base: str,
    api_model: str,
    api_timeout: float,
) -> List[str]:
    if not observed_source.strip():
        return []

    base = normalize_api_base(api_base)

    if _is_chat_endpoint_model(api_model):
        # Choose Method A (numbered-list, one call) or Method B (n independent calls in one batched request).
        # Set FUTURE_SAMPLING_METHOD=B to switch; default is A.
        method = os.environ.get("FUTURE_SAMPLING_METHOD", "A").strip().upper()

        if method == "B":
            # Method B: one /chat/completions call with n=num_futures. vLLM samples each completion
            # independently and returns them as separate `choices` (semantically equivalent to N
            # independent calls but batched on the GPU for speed).
            messages = build_future_sampling_chat_messages_single(observed_source)
            payload = {
                "model": api_model,
                "messages": messages,
                "max_tokens": future_tokens,
                "temperature": sample_temperature,
                "top_p": 0.95,
                "n": num_futures,
            }
            data = _http_json(f"{base}/chat/completions", payload=payload, timeout=api_timeout)
            futures: List[str] = []
            for choice in data.get("choices", []):
                msg = choice.get("message", {}) if isinstance(choice, dict) else {}
                raw = str(msg.get("content", "")) if isinstance(msg, dict) else ""
                cleaned = clean_future_text(observed_source, raw)
                if cleaned and is_valid_future_text(cleaned):
                    futures.append(cleaned)
            return futures

        # Method A (default): one /chat/completions call returning a numbered list of N continuations.
        messages = build_future_sampling_chat_messages(observed_source, num_futures)
        # Allow enough output for ~30 words per item × num_futures + numbering overhead.
        max_tokens = max(future_tokens * num_futures, 40 * num_futures + 50)
        payload = {
            "model": api_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sample_temperature,
            "top_p": 0.95,
            "n": 1,
        }
        data = _http_json(f"{base}/chat/completions", payload=payload, timeout=api_timeout)
        choices = data.get("choices", [])
        if not choices:
            return []
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        raw_text = str(msg.get("content", "")) if isinstance(msg, dict) else ""
        items = parse_method_a_output(raw_text, num_expected=num_futures)
        futures: List[str] = []
        for raw in items:
            cleaned = clean_future_text(observed_source, raw)
            if cleaned and is_valid_future_text(cleaned):
                futures.append(cleaned)
        return futures

    # Default: base model raw text continuation via /completions.
    payload = {
        "model": api_model,
        "prompt": observed_source,
        "max_tokens": future_tokens,
        "temperature": sample_temperature,
        "top_p": 0.95,
        "n": num_futures,
        "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    }
    data = _http_json(f"{base}/completions", payload=payload, timeout=api_timeout) #call base model，用observed source作为prompt，采样n条可能的未来源语言续写
    futures: List[str] = []
    for choice in data.get("choices", []):
        raw = str(choice.get("text", "")) if isinstance(choice, dict) else ""
        cleaned = clean_future_text(observed_source, raw)
        if cleaned and is_valid_future_text(cleaned):
            futures.append(cleaned)
    return futures

#Step1: primary base model采样未来续写，Step2: secondary base model采样未来续写，合并去重后得到所有候选未来续写列
def sample_source_futures_multi(
    base_specs: List[Dict[str, Any]],
    observed_source: str,
    future_tokens: int,
    sample_temperature: float,
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    merged: List[str] = []
    merged_info: List[Dict[str, Any]] = []
    seen: set = set()
    for spec in base_specs:
        requested = int(spec.get("num_futures", 0) or 0)
        if requested <= 0:
            continue
        futures = sample_source_futures_api(
            observed_source=observed_source,
            num_futures=requested,
            future_tokens=future_tokens,
            sample_temperature=sample_temperature,
            api_base=spec["api_base"],
            api_model=spec["api_model"],
            api_timeout=spec["api_timeout"],
        )
        for future in futures:
            key = future.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(future)
            merged_info.append({"source": spec["name"], "path": spec["path"], "future": future})
    return merged, merged_info


MALFORMED_RESPONSE_LOG_PATH: Optional[str] = None
_MALFORMED_LOG_LOCK = threading.Lock()


class _SamplerUnavailable(Exception):
    """Sentinel for a failed sampler request when the caller opted for silent empties."""


def _record_sampler_event(**event: Any) -> None:
    """Append one JSON line per malformed/recovered/exhausted sampler response.

    Called synchronously from the retry loop, so a raw response is on disk before
    the next request is sent. Events are rare (about 1 per 1,000 calls), so the
    per-event open/append/close under a lock is negligible at 40k-utterance scale.
    """
    event["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(
        f"[SamplerResponse] event={event['event']} model={event['model']} attempt={event['attempt']} "
        f"finish_reason={event.get('finish_reason')} error={event.get('error')}",
        file=sys.stderr, flush=True,
    )
    if MALFORMED_RESPONSE_LOG_PATH:
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with _MALFORMED_LOG_LOCK, open(MALFORMED_RESPONSE_LOG_PATH, "a", encoding="utf-8") as stream:
            stream.write(line)


def _sample_coordinated_future_set(
    sampler_tokenizer: Any,
    observed_source: str,
    committed_text: str,
    target_lang: str,
    num_futures: int,
    api_base: str,
    api_model: str,
    api_timeout: float,
    sample_temperature: float,
    top_p: float,
    max_tokens: int,
    sampler_seed: Optional[int] = None,
    sampler_context: str = "source-and-target",
    fail_on_api_error: bool = False,
    prompt_version: str = PROMPT_VERSION,
    parse_retries: int = 2,
    contrastive_notes: bool = False,
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate jointly planned plausible and contrastive lists in one response."""
    if not observed_source.strip() or num_futures <= 0:
        return [], [], []

    sampler_committed = "" if sampler_context == "source-only" else committed_text
    messages = build_coordinated_future_messages(
        observed_source=observed_source,
        target_lang=target_lang,
        committed_text=sampler_committed,
        num_candidates=num_futures,
        prompt_version=prompt_version,
        contrastive_notes=contrastive_notes,
    )

    # One response contains the whole list, allowing the instruction-tuned model
    # to compare candidates and avoid duplicating its own likely completion.
    prompt = sampler_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    base = normalize_api_base(api_base)
    payload: Dict[str, Any] = {
        "model": api_model,
        "prompt": prompt,
        "max_tokens": max(max_tokens, 32 * num_futures),
        "temperature": sample_temperature,
        "top_p": top_p,
        "n": 1,
        "presence_penalty": 0.15,
        "stop": ["<|im_end|>", "<end_of_turn>", "<|endoftext|>", "<|eot_id|>"],
    }
    if sampler_seed is not None:
        import zlib
        payload["seed"] = (int(sampler_seed) + zlib.crc32(f"{observed_source}|{sampler_committed}|{api_model}".encode("utf-8"))) % (2**31)
    if prompt_version == SUFFIX_ICL_PROMPT_VERSION:
        payload.update(structured_output_extras(num_futures, contrastive_notes))
    def request(attempt: int) -> Tuple[str, Optional[str]]:
        attempt_payload = payload
        if attempt and "seed" in payload:
            attempt_payload = {**payload, "seed": (payload["seed"] + 7919 * attempt) % (2**31)}
        try:
            data = _http_json(f"{base}/completions", payload=attempt_payload, timeout=api_timeout)
        except Exception as exc:
            if fail_on_api_error:
                raise
            print(f"[SamplerUnavailable] model={api_model} attempt={attempt + 1} error={exc}", file=sys.stderr, flush=True)
            raise _SamplerUnavailable() from None
        choices = data.get("choices") if isinstance(data, dict) else None
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        return str(choice.get("text", "")), choice.get("finish_reason")

    def record(event: str, attempt: int, raw_text: str, finish_reason: Optional[str], error: Optional[str]) -> None:
        _record_sampler_event(
            event=event, model=api_model, prompt_version=prompt_version, attempt=attempt + 1,
            observed_source=observed_source, committed_text=sampler_committed,
            finish_reason=finish_reason, error=error, raw_response=raw_text,
        )

    try:
        if prompt_version == SUFFIX_ICL_PROMPT_VERSION:
            grouped_candidates, raw_response = sample_grouped_futures(request, num_futures, parse_retries, record)
        else:
            raw_response, _ = request(0)
    except _SamplerUnavailable:
        return [], [], []
    except ValueError as exc:
        raise ValueError(f"Invalid {prompt_version} response from {api_model}: {exc}") from exc
    if prompt_version != SUFFIX_ICL_PROMPT_VERSION:
        parsed = parse_method_a_output(raw_response, num_expected=num_futures)
        # Preserve the historical fixed-position interpretation, including missing slots.
        grouped_candidates = [
            ("plausible" if i < num_futures // 2 else "contrastive", parsed[i] if i < len(parsed) else "", "")
            for i in range(num_futures)
        ]

    # Build a normalized form of the partial (lowercased, punctuation/space
    # stripped) so the (c) regurgitation check can catch capitalized/reformatted
    # restatements like "They held on for some time, ..." that don't startswith()
    # match the raw partial.
    def _norm_for_dup(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())
    partial_norm = _norm_for_dup(observed_source)
    # Use a min length to avoid false-positives on very short partials.
    partial_norm_min_len = max(8, len(partial_norm) // 2)

    candidates_by_mode: Dict[str, List[str]] = {"plausible": [], "contrastive": []}
    audit: List[Dict[str, Any]] = []
    for choice_index, (mode, raw, note) in enumerate(grouped_candidates):
        cleaned = clean_future_text(observed_source, raw)
        if not cleaned or not is_valid_future_text(cleaned):
            audit.append({
                "choice": choice_index, "model": api_model, "mode": mode,
                "raw": raw.strip(), "future": cleaned, "accepted": False, "note": note,
                "reason": "missing_or_invalid" if not raw else "invalid_or_meta",
            })
            continue
        # A one-word fragment rarely carries enough evidence to test whether a
        # translation commitment is safe. Require a short phrase while keeping
        # genuinely concise continuations available.
        if len(re.findall(r"[A-Za-z0-9']+", cleaned)) < 3:
            audit.append({
                "choice": choice_index, "model": api_model, "mode": mode,
                "raw": raw.strip(), "future": cleaned, "accepted": False, "note": note,
                "reason": "too_short",
            })
            continue
        # (c) Hard-drop continuations that contain the partial (or a long
        # substring of it) anywhere — handles restart/capitalization cases
        # where the model rewrote the partial before its actual continuation.
        if partial_norm and len(partial_norm) >= partial_norm_min_len:
            cleaned_norm = _norm_for_dup(cleaned)
            if partial_norm in cleaned_norm:
                audit.append({
                    "choice": choice_index, "model": api_model, "mode": mode,
                    "raw": raw.strip(), "future": cleaned, "accepted": False, "note": note,
                    "reason": "repeats_observed_prefix",
                })
                continue
        candidates_by_mode[mode].append(cleaned)
        audit.append({
            "choice": choice_index, "model": api_model, "mode": mode,
            "raw": raw.strip(), "future": cleaned, "accepted": None, "note": note,
            "reason": "pending_diversity_filter",
        })

    futures_by_mode = {
        mode: select_diverse_futures(candidates)
        for mode, candidates in candidates_by_mode.items()
    }
    futures = futures_by_mode["plausible"] + futures_by_mode["contrastive"]
    selected_counts = {
        mode: Counter(re.sub(r"\s+", " ", text.strip().lower()) for text in selected)
        for mode, selected in futures_by_mode.items()
    }
    for item in audit:
        if item["accepted"] is not None:
            continue
        key = re.sub(r"\s+", " ", item["future"].strip().lower())
        mode_counts = selected_counts[str(item["mode"])]
        if mode_counts[key] > 0:
            mode_counts[key] -= 1
            item["accepted"] = True
            item["reason"] = "accepted"
        else:
            item["accepted"] = False
            item["reason"] = "duplicate_or_repetitive"
    items_info: List[Dict[str, Any]] = []
    for mode in ("plausible", "contrastive"):
        for cleaned in futures_by_mode[mode]:
            items_info.append({
                "source": "coordinated_future_set",
                "path": api_model,
                "model": api_model,
                "mode": mode,
                "future": cleaned,
                "prompt_version": prompt_version,
            })
    return futures, items_info, audit


def select_diverse_futures(
    candidates: List[str],
    *,
    max_same_opening: int = 1,
    similarity_threshold: float = 0.65,
) -> List[str]:
    """Remove lexical mode collapse within one model/mode sample batch."""
    selected: List[str] = []
    selected_tokens: List[set[str]] = []
    opening_counts: Dict[str, int] = {}
    seen: set[str] = set()
    opening_stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "from", "has", "have", "he", "her", "his", "in", "is", "it", "its",
        "of", "on", "or", "she", "that", "the", "their", "they", "this", "to",
        "was", "were", "which", "who", "with", "would",
    }

    for candidate in candidates:
        normalized = re.sub(r"\s+", " ", candidate.strip().lower())
        if normalized in seen:
            continue
        tokens = re.findall(r"[a-z0-9']+", normalized)
        if not tokens:
            continue
        opening = next((token for token in tokens if token not in opening_stopwords), tokens[0])
        if opening_counts.get(opening, 0) >= max_same_opening:
            continue
        token_set = set(tokens)
        if any(
            len(token_set & prior) / len(token_set | prior) >= similarity_threshold
            for prior in selected_tokens
            if token_set | prior
        ):
            continue
        seen.add(normalized)
        opening_counts[opening] = opening_counts.get(opening, 0) + 1
        selected.append(candidate)
        selected_tokens.append(token_set)
    return selected


def sampler_display_name(model: str) -> str:
    return {
        "gemma4-sampler": "Gemma 4",
        "qwen38-sampler": "Qwen 3.8",
    }.get(model, model)


def format_raw_future_groups(audit: List[Dict[str, Any]]) -> List[str]:
    """Render raw samples as compact model/mode sections."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for item in audit:
        model = str(item.get("model", "?"))
        mode = str(item.get("mode", "?"))
        groups.setdefault((model, mode), []).append(item)

    lines: List[str] = []
    for (model, mode), items in groups.items():
        lines.append(
            f"[Raw candidates] {sampler_display_name(model)} | {mode} | "
            f"model={model} mode={mode} count={len(items)}"
        )
        for index, item in enumerate(items, 1):
            shown = item.get("future") or item.get("raw") or ""
            note = item.get("note") or ""
            lines.append(f"  {index:02d}. {shown!r}" + (f"  resolves: {note!r}" if note else ""))
        reasons = Counter(
            str(item.get("reason", "unknown"))
            for item in items
            if not item.get("accepted")
        )
        kept = sum(bool(item.get("accepted")) for item in items)
        dropped = ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()))
        lines.append(f"  Filter summary: kept={kept}/{len(items)}; dropped: {dropped or 'none'}")
    return lines


def format_selected_future_groups(
    futures: List[str],
    future_infos: List[Dict[str, Any]],
) -> List[str]:
    """Render candidates used by consensus without repeated per-line labels."""
    groups: Dict[Tuple[str, str], List[str]] = {}
    for index, future in enumerate(futures):
        info = future_infos[index] if index < len(future_infos) else {}
        model = str(info.get("model") or info.get("path") or "?")
        mode = str(info.get("mode") or "?")
        groups.setdefault((model, mode), []).append(future)

    lines: List[str] = []
    for (model, mode), items in groups.items():
        lines.append(
            f"[Selected candidates] {sampler_display_name(model)} | {mode} | "
            f"model={model} mode={mode} count={len(items)}"
        )
        lines.extend(f"  {index:02d}. {future!r}" for index, future in enumerate(items, 1))
    tagged = [(future, info) for future, info in zip(futures, future_infos) if info.get("word")]
    lines.extend(f"  [tag] {info['word']} = {info['reading']}: {future!r}" for future, info in tagged)
    return lines


def request_role_json(tokenizer: Any, messages: List[Dict[str, str]], schema: Dict[str, Any], api_base: str,
                      api_model: str, api_timeout: float, temperature: float, top_p: float, max_tokens: int,
                      retries: int = 2) -> Optional[Dict[str, Any]]:
    """One structured-output request of the role-split sampler; None if every attempt fails to parse."""
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    payload = {"model": api_model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
               "n": 1, "stop": ["<|im_end|>", "<end_of_turn>", "<|endoftext|>", "<|eot_id|>"],
               "structured_outputs": {"json": schema}}
    for attempt in range(1 + retries):
        try:
            data = _http_json(f"{normalize_api_base(api_base)}/completions", payload=payload, timeout=api_timeout)
            return json.loads(data["choices"][0]["text"])
        except (RuntimeError, ValueError, KeyError, IndexError) as exc:   # _http_json wraps transport errors in RuntimeError
            print(f"[RoleSampler] model={api_model} attempt={attempt + 1} error={exc}", file=sys.stderr, flush=True)
            continue
    return None


def filter_role_futures(observed_source: str, items: List[Tuple[str, str, Dict[str, Any]]], api_model: str,
                        ) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The coordinated sampler's checks (cleaning, validity, >= 3 words, no restated prefix) plus its diversity filter
    within each (mode, word, reading) group."""
    partial_norm = re.sub(r"[^a-z0-9]+", "", observed_source.lower())
    groups: Dict[Tuple[str, str, str], List[str]] = {}
    audit: List[Dict[str, Any]] = []
    for mode, raw, extra in items:
        cleaned = clean_future_text(observed_source, raw)
        if not cleaned or not is_valid_future_text(cleaned):
            reason = "missing_or_invalid"
        elif len(re.findall(r"[A-Za-z0-9']+", cleaned)) < 3:
            reason = "too_short"
        elif len(partial_norm) >= max(8, len(partial_norm) // 2) and partial_norm in re.sub(r"[^a-z0-9]+", "", cleaned.lower()):
            reason = "repeats_observed_prefix"
        else:
            reason = "kept"
            groups.setdefault((mode, extra.get("word", ""), extra.get("reading", "")), []).append(cleaned)
        audit.append({"model": api_model, "mode": mode, "raw": raw.strip(), "future": cleaned, "accepted": reason == "kept",
                      "reason": reason, **extra})
    futures: List[str] = []
    infos: List[Dict[str, Any]] = []
    for (mode, word, reading), candidates in groups.items():
        for future in select_diverse_futures(candidates):
            futures.append(future)
            infos.append({"source": "role_future", "path": api_model, "model": api_model, "mode": mode, "future": future,
                          "word": word, "reading": reading, "prompt_version": "role_split_20261009"})
    return futures, infos, audit


def sample_role_futures(observed_source: str, plausible_spec: Tuple[Any, str, str, float], targeted_spec: Tuple[Any, str, str, float],
                        roles: str, ambiguity_hint: Optional[Dict[str, Any]], plausible_n: int, per_reading: int,
                        max_words: int, max_readings: int, temperature: float, top_p: float,
                        ) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Gemma: natural continuations. Qwen3.8: open words (layer 1, or the given word) -> continuations per reading."""
    def plausible() -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        tokenizer, base, model, timeout = plausible_spec
        reply = request_role_json(tokenizer, build_plausible_messages(observed_source, plausible_n), plausible_schema(plausible_n),
                                  base, model, timeout, temperature, top_p, max_tokens=60 * plausible_n)
        suffixes = (reply or {}).get("plausible", [])
        return filter_role_futures(observed_source, [("plausible", x, {}) for x in suffixes if isinstance(x, str)], model)

    def targeted() -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        tokenizer, base, model, timeout = targeted_spec
        if roles == "given_word":
            heard = ambiguity_hint is not None and re.search(rf"\b{re.escape(ambiguity_hint['word'])}\b", observed_source, re.I)
            open_words = [{"word": ambiguity_hint["word"], "readings": ambiguity_hint["readings"]}] if heard else []
            layer1 = {"given": bool(heard)}
        else:
            reply = request_role_json(tokenizer, build_open_word_messages(observed_source, max_words, max_readings),
                                      open_word_schema(max_words, max_readings), base, model, timeout, temperature, top_p,
                                      max_tokens=400)
            open_words = [w for w in (reply or {}).get("open_words", [])
                          if isinstance(w, dict) and w.get("word") and len(w.get("readings", [])) >= 2][:max_words]
            layer1 = {"open_words": open_words}
        if not open_words:
            return [], [], [{"model": model, "mode": "layer1", "accepted": None, **layer1}]
        else:
            max_items = per_reading * sum(len(w["readings"]) for w in open_words)
            reply = request_role_json(tokenizer, build_reading_messages(observed_source, open_words, per_reading),
                                      reading_schema(max_items), base, model, timeout, temperature, top_p,
                                      max_tokens=80 * max_items)
            items = [("contrastive", f["suffix"], {"word": f.get("word", ""), "reading": f.get("reading", "")})
                     for f in (reply or {}).get("futures", []) if isinstance(f, dict) and isinstance(f.get("suffix"), str)]
            futures, infos, audit = filter_role_futures(observed_source, items, model)
            return futures, infos, [{"model": model, "mode": "layer1", "accepted": None, **layer1}] + audit

    with ThreadPoolExecutor(max_workers=2) as pool:
        plausible_job, targeted_job = pool.submit(plausible), pool.submit(targeted)
        batches = [plausible_job.result(), targeted_job.result()]
    return ([f for b in batches for f in b[0]], [i for b in batches for i in b[1]], [a for b in batches for a in b[2]])


SamplerSpec = Tuple[PreTrainedTokenizerBase, str, str, float]


def sample_list_futures(observed_source: str, list_prefix: str, list_open: List[Dict[str, Any]], natural_spec: SamplerSpec,
                        targeted_spec: SamplerSpec, natural: str, natural_n: int, per_reading: int, targeted_thinking: bool,
                        think_budget: int, skip_ruled_out: bool = False,
                        reading_check: bool = False, observed_full: str = "") -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """design v3: natural continuations of the sentence-anchored prefix, plus continuations of the (extended) prefix per
    reading of every open listed word. Both go through the coordinated sampler's filter; each request's status goes to
    the audit (a failed request would otherwise leave no trace)."""
    def _natural() -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if natural == "none":
            return [], [], []
        else:
            tokenizer, base, model, timeout = natural_spec
        results, _ = list_request(base, model, tokenizer, natural_messages(observed_source, natural_n), False, 1,
                                  natural_schema(natural_n), 60 * natural_n, timeout)
        suffixes = (results[0]["answer"] or {"plausible": []})["plausible"]
        futures, infos, audit = filter_role_futures(observed_source, [("natural", suffix, {}) for suffix in suffixes], model)
        return futures, infos, [{"model": model, "mode": "natural_request", "status": results[0]["status"], "accepted": None}] + audit

    def _targeted() -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not list_open:
            return [], [], []
        else:
            tokenizer, base, model, timeout = targeted_spec
        words = [(entry["word"], [f"{r['gloss']} ({r['zh']})" for r in entry["readings"]]) for entry in list_open]
        check_audit = []
        if reading_check:   # keep only the readings still possible given the whole prefix; close a word left with < 2
            checked = []
            for word, readings in words:
                results, _ = list_request(base, model, tokenizer, check_messages(observed_full, word, readings), False, 1,
                                          check_schema(len(readings)), 60, timeout, temperature=0.0)
                answer = results[0]["answer"]
                possible = sorted({n for n in answer.get("possible", []) if 1 <= n <= len(readings)}) if isinstance(answer, dict) else list(range(1, len(readings) + 1))
                check_audit.append({"model": model, "mode": "check_request", "status": results[0]["status"], "accepted": None,
                                    "word": word, "possible": possible})
                if len(possible) >= 2:
                    checked.append((word, [readings[n - 1] for n in possible]))
                else:
                    continue
            words = checked
        else:
            pass
        if not words:
            return [], [], check_audit
        else:
            pass
        max_items = per_reading * sum(len(readings) for _, readings in words)
        results, _ = list_request(base, model, tokenizer, sample_messages(list_prefix, words, per_reading, skip_ruled_out), targeted_thinking, 1,
                                  sample_schema(max_items, max(len(readings) for _, readings in words)),
                                  think_budget if targeted_thinking else 80 * max_items, timeout)
        answer = results[0]["answer"]
        by_word = {word.lower(): (word, readings) for word, readings in words}
        items = []
        for future in answer.get("futures", []) if isinstance(answer, dict) else []:
            word, readings = by_word.get(str(future.get("word", "")).lower(), ("", []))
            number = future.get("reading")
            if isinstance(future.get("suffix"), str) and isinstance(number, int) and 1 <= number <= len(readings):
                items.append(("targeted", future["suffix"], {"word": word, "reading": readings[number - 1]}))
            else:
                continue
        futures, infos, audit = filter_role_futures(list_prefix, items, model)
        request_audit = {"model": model, "mode": "targeted_request", "status": results[0]["status"], "accepted": None,
                         "returned": len(answer.get("futures", [])) if isinstance(answer, dict) else 0, "valid": len(items)}
        return futures, infos, check_audit + [request_audit] + audit

    with ThreadPoolExecutor(max_workers=2) as pool:
        natural_job, targeted_job = pool.submit(_natural), pool.submit(_targeted)
        batches = [natural_job.result(), targeted_job.result()]
    return ([f for b in batches for f in b[0]], [i for b in batches for i in b[1]], [a for b in batches for a in b[2]])


def sample_source_futures_targeted_prefill(
    sampler_tokenizer: Any,
    observed_source: str,
    committed_text: str,
    target_lang: str,
    num_futures: int,
    api_base: str,
    api_model: str,
    api_timeout: float,
    sample_temperature: float = 1.0,
    top_p: float = 0.98,
    max_tokens: int = 40,
    sampler2_tokenizer: Any = None,
    sampler2_api_base: str = "",
    sampler2_api_model: str = "",
    sampler2_api_timeout: float = 0.0,
    return_audit: bool = False,
    sampler_seed: Optional[int] = None,
    sampler_context: str = "source-and-target",
    fail_on_api_error: bool = False,
    prompt_version: str = PROMPT_VERSION,
    parse_retries: int = 2,
    contrastive_notes: bool = False,
    future_roles: str = "off",
    ambiguity_hint: Optional[Dict[str, Any]] = None,
    role_plausible_n: int = 10,
    role_per_reading: int = 2,
    role_max_words: int = 2,
    role_max_readings: int = 3,
    list_open: Optional[List[Dict[str, Any]]] = None,
    list_prefix: str = "",
    list_natural: str = "qwen",
    list_natural_n: int = 6,
    list_per_reading: int = 2,
    list_targeted_thinking: bool = False,
    list_think_budget: int = 2000,
    list_skip_ruled_out: bool = False,
    list_reading_check: bool = False,
    list_observed_full: str = "",
) -> Any:
    """Generate a coordinated set per model with equal group budgets (or, with future_roles, the role-split set)."""
    if not observed_source.strip():
        return ([], [], []) if return_audit else ([], [])
    elif future_roles == "list":
        s2_timeout = sampler2_api_timeout if sampler2_api_timeout and sampler2_api_timeout > 0 else api_timeout
        result = sample_list_futures(
            observed_source=observed_source, list_prefix=list_prefix, list_open=list_open or [],
            natural_spec=((sampler2_tokenizer, sampler2_api_base, sampler2_api_model, s2_timeout) if list_natural == "qwen"
                          else (sampler_tokenizer, api_base, api_model, api_timeout)),
            targeted_spec=(sampler2_tokenizer, sampler2_api_base, sampler2_api_model, s2_timeout), natural=list_natural,
            natural_n=list_natural_n, per_reading=list_per_reading, targeted_thinking=list_targeted_thinking,
            think_budget=list_think_budget, skip_ruled_out=list_skip_ruled_out, reading_check=list_reading_check,
            observed_full=list_observed_full,
        )
        return result if return_audit else result[:2]
    elif future_roles != "off":
        s2_timeout = sampler2_api_timeout if sampler2_api_timeout and sampler2_api_timeout > 0 else api_timeout
        result = sample_role_futures(
            observed_source=observed_source,
            plausible_spec=(sampler_tokenizer, api_base, api_model, api_timeout),
            targeted_spec=(sampler2_tokenizer, sampler2_api_base, sampler2_api_model, s2_timeout),
            roles=future_roles, ambiguity_hint=ambiguity_hint, plausible_n=role_plausible_n, per_reading=role_per_reading,
            max_words=role_max_words, max_readings=role_max_readings, temperature=sample_temperature, top_p=top_p,
        )
        return result if return_audit else result[:2]
    else:
        pass
    if num_futures <= 0 or num_futures % 2:
        raise ValueError("num_futures must be a positive even number")
    use_ensemble = bool(sampler2_tokenizer and sampler2_api_base and sampler2_api_model)
    batches = []
    if use_ensemble:
        s2_to = sampler2_api_timeout if sampler2_api_timeout and sampler2_api_timeout > 0 else api_timeout
        model_specs = (
            (sampler_tokenizer, api_base, api_model, api_timeout),
            (sampler2_tokenizer, sampler2_api_base, sampler2_api_model, s2_to),
        )
        def sample_one(spec):
            model_tokenizer, model_base, model_name, model_timeout = spec
            return _sample_coordinated_future_set(
                sampler_tokenizer=model_tokenizer,
                observed_source=observed_source, committed_text=committed_text,
                target_lang=target_lang, num_futures=num_futures,
                api_base=model_base, api_model=model_name, api_timeout=model_timeout,
                sample_temperature=sample_temperature, top_p=top_p, max_tokens=max_tokens,
                sampler_seed=sampler_seed,
                sampler_context=sampler_context, fail_on_api_error=fail_on_api_error,
                prompt_version=prompt_version, parse_retries=parse_retries,
                contrastive_notes=contrastive_notes,
            )
        # The two samplers sit on different servers; overlap their generation.
        with ThreadPoolExecutor(max_workers=len(model_specs)) as pool:
            batches.extend(pool.map(sample_one, model_specs))
    else:
        batches.append(_sample_coordinated_future_set(
            sampler_tokenizer=sampler_tokenizer,
            observed_source=observed_source, committed_text=committed_text,
            target_lang=target_lang, num_futures=num_futures,
            api_base=api_base, api_model=api_model, api_timeout=api_timeout,
            sample_temperature=sample_temperature, top_p=top_p, max_tokens=max_tokens,
            sampler_seed=sampler_seed,
            sampler_context=sampler_context, fail_on_api_error=fail_on_api_error,
            prompt_version=prompt_version, parse_retries=parse_retries,
            contrastive_notes=contrastive_notes,
        ))
    # Preserve cross-model agreement as two independent votes. Each model's set
    # has already removed exact and near-duplicate candidates.
    merged_f: List[str] = []
    merged_i: List[Dict[str, Any]] = []
    merged_audit: List[Dict[str, Any]] = []
    for futs, infos, audit in batches:
        for f, info in zip(futs, infos):
            merged_f.append(f)
            merged_i.append(info)
        merged_audit.extend(audit)
    if return_audit:
        return merged_f, merged_i, merged_audit
    return merged_f, merged_i


# ---------------------------------------------------------------------------
# Next-token distribution (API only, batch)
# ---------------------------------------------------------------------------

def _parse_token_id_string(raw: str) -> Optional[int]:
    match = re.fullmatch(r"token_id:(\d+)", str(raw or "").strip())
    return int(match.group(1)) if match else None


def _parse_completion_top_logprobs(
    # The raw "top_logprobs" from the API is a list of dicts, 
    # one per generated token, where each dict maps
    # token_id strings like "token_id:1234" to logprobs. 
    #We only care about the first step (next token).
    top_logprobs: Optional[List[Optional[Dict[str, float]]]],
    tokenizer: Any,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    if not top_logprobs:
        return {}, {"reason": "missing_top_logprobs"}
    step = top_logprobs[0]
    if not step:
        return {}, {"reason": "empty_top_logprobs"}
    id_distribution: Dict[int, float] = {}
    unknown_tokens: List[str] = []
    for raw_token, logprob in step.items():
        tok_id = _parse_token_id_string(raw_token)
        if tok_id is None:
            unknown_tokens.append(str(raw_token))
            continue
        id_distribution[tok_id] = float(math.exp(float(logprob))) #logprob是对数概率，exp转回真实概率
    token_ids = list(id_distribution.keys())
    # {
    #     123: 0.82,
    #     456: 0.21,
    #     789: 0.07
    # }
    return id_distribution, {
        "reason": "ok" if id_distribution else "no_token_ids_in_top_logprobs",
        "topk_token_ids": token_ids,
        "topk_token_texts": [_single_token_text(tokenizer, t) for t in token_ids],
        "topk_true_probs": [round(float(id_distribution[t]), 6) for t in token_ids],
        "unknown_top_logprob_tokens": unknown_tokens,
    }


def batch_get_next_token_distributions(
    tokenizer: Any,
    full_sources: List[str],
    target_prefix_token_ids: List[int],
    top_k: int = TOP_K,
    min_p: float = 0.0,
    top_p: float = 0.0,
    api_base: str = "",
    api_model: str = "",
    api_timeout: float = 120.0,
    target_lang: str = "Chinese",
    min_logprobs: int = 0,
    probe_temperature: float = 0.0,
    probe_top_p: float = 0.0,
    probe_top_k: int = 0,
    shared_first: bool = False,
    reasonings: Optional[List[Optional[str]]] = None,
    notes: Optional[List[str]] = None,
) -> List[Tuple[Dict[int, float], Dict[str, Any]]]:
# Step 1: 用 chat template 把翻译指令编码成 token IDs
#         → [<|im_start|>, user, \n, [TASK], Translate..., <|im_end|>]
        
# Step 2: 拼上 assistant 前缀的 token IDs
#         → 上面 + encode("<|im_start|>assistant\n")
        
# Step 3: 拼上已提交翻译的 token IDs
#         → 上面 + [32108, 45621]
    # Probe thinking (2026-10-06): a source with a reasoning gets the thinking template, its reasoning and the closing
    # </think> before the committed prefix, so the next-token distribution reads the reasoning.
    prompts = []
    for i, src in enumerate(full_sources):
        if reasonings is not None and reasonings[i] is not None:
            prompt_ids = build_translation_probe_prompt_prefix_token_ids(
                tokenizer, src, has_target_prefix=bool(target_prefix_token_ids), target_lang=target_lang,
                shared_first=shared_first, enable_thinking=True,
            ) + tokenizer.encode(reasonings[i] + "\n</think>\n\n", add_special_tokens=False)
        else:
            prompt_ids = build_translation_probe_prompt_prefix_token_ids(
                tokenizer, src, has_target_prefix=bool(target_prefix_token_ids), target_lang=target_lang,
                shared_first=shared_first, note=notes[i] if notes else "",
            )
        prompts.append(prompt_ids + list(target_prefix_token_ids))  # Step 3
    # [user task tokens] + [assistant prefix tokens] + [already committed target prefix tokens]
    if not prompts:
        return []
    logprobs_n = max(top_k, 100, min_logprobs) if (min_p > 0 or top_p > 0) else max(top_k, min_logprobs)
    payload = {
        "model": api_model,
        "prompt": prompts,
        "max_tokens": 1,
        "temperature": probe_temperature,
        "logprobs": logprobs_n,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
    }
    # Only sent when explicitly asked for, so the historical request shape is unchanged
    # by default. These reshape the *reported distribution* the consensus vote reads --
    # they do not make the probe "sample", because max_tokens=1 and the code consumes
    # top_logprobs, never the drawn token.
    if probe_top_p and probe_top_p > 0:
        payload["top_p"] = probe_top_p
    if probe_top_k and probe_top_k > 0:
        payload["top_k"] = probe_top_k
    data = _http_json(f"{normalize_api_base(api_base)}/completions", payload=payload, timeout=api_timeout) #call instruct model，传token ID列表，只生成1个token，拿top-k logprobs分布
    choices = data.get("choices", []) #vllm做batch处理，每个choice对应一个future的next-token分布
    results: List[Tuple[Dict[int, float], Dict[str, Any]]] = []
    for i in range(len(prompts)):
        if i >= len(choices):
            results.append(({}, {"reason": "missing_choice", "raw_response": data}))
            continue
        choice = choices[i] 
        # choice = {
        #     "text": "而",
        #     "logprobs": {
        #         "tokens": ["token_id:123"],
        #         "top_logprobs": [
        #             {
        #                 "token_id:123": -0.1,
        #                 "token_id:456": -1.2,
        #                 "token_id:789": -2.0,
        #                 "token_id:999": -3.1,
        #                 "token_id:555": -4.0,
        #             }
        #         ]
        #     }
        # }
        logprobs = choice.get("logprobs", {}) if isinstance(choice, dict) else {} #每个choice里有一个"logprobs"字段，里面的"top_logprobs"是一个列表，每个元素是一个dict，表示对应生成token的top-k logprobs分布；我们只关心第一个生成token的分布，所以取top_logprobs[0]来解析出token ID分布

        dist, dist_debug = _parse_completion_top_logprobs(logprobs.get("top_logprobs"), tokenizer=tokenizer) #解析出token ID分布，并记录debug信息
        dist_debug["api_backend"] = "vllm_completion"
        if min_p > 0:
            dist_debug["candidate_policy"] = "min_p"
            dist_debug["min_p"] = min_p
        else:
            dist_debug["candidate_policy"] = "top_k"
            dist_debug["candidate_top_k"] = top_k
        results.append(filter_distribution_token_ids(tokenizer, dist, dist_debug)) #过滤掉不合规的token ID，并在debug信息里记录被过滤掉的token ID和原因
    return results


# Thinking sampling (Haoling 2026-10-06: temperature 0.6 for every reasoning, so the voters do not drift apart): the
# Qwen3.6 model card's thinking setting for precise tasks, the setting of the translator check (translator_check_20261006).
THINKING_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def generate_reasonings(prompts: List[Any], api_base: str, api_model: str, max_tokens: int) -> List[Dict[str, Any]]:
    """One reasoning per prompt (token ids or text ending in the thinking template's <think>), stopped at </think>; a
    reasoning cut at max_tokens is kept as it is and marked."""
    payload = {"model": api_model, "prompt": prompts, "max_tokens": max_tokens, **THINKING_SAMPLING, "stop": ["</think>"]}
    data = _http_json(f"{normalize_api_base(api_base)}/completions", payload=payload, timeout=3600.0)
    return [{"text": choice["text"].strip(), "cut": choice["finish_reason"] == "length"} for choice in data["choices"]]


def generate_probe_reasonings(
    tokenizer: Any,
    full_sources: List[str],
    has_target_prefix: bool,
    api_base: str,
    api_model: str,
    target_lang: str,
    shared_first: bool,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """Probe thinking (Haoling 2026-10-06): one reasoning per probe source, generated once per chunk with thinking on and
    reused by every probe step of that chunk. The user turn is the probe's own, with the has_target_prefix of the chunk
    start: if the first token of an utterance commits in this chunk, the later steps of the chunk ask with the
    'partial translation committed' rule after a reasoning made under the 'from the beginning' rule."""
    prompts = [
        build_translation_probe_prompt_prefix_token_ids(
            tokenizer, src, has_target_prefix=has_target_prefix, target_lang=target_lang, shared_first=shared_first,
            enable_thinking=True,
        )
        for src in full_sources
    ]
    return generate_reasonings(prompts, api_base, api_model, max_tokens)


# ---------------------------------------------------------------------------
# Consensus logic
# ---------------------------------------------------------------------------

def topk_token_ids(dist: Dict[int, float], k: int = TOP_K) -> List[int]:
    # Return the top-k token IDs by probability, sorted in descending order of probability.
    return [tok_id for tok_id, _ in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def minp_token_ids(dist: Dict[int, float], min_p: float) -> List[int]:
    """Return token IDs whose probability >= min_p (absolute threshold)."""
    if not dist:
        return []
    return [tok_id for tok_id, prob in sorted(dist.items(), key=lambda kv: kv[1], reverse=True) if prob >= min_p]


def topp_token_ids(dist: Dict[int, float], top_p: float) -> List[int]:
    """Return the smallest set of token IDs whose cumulative probability >= top_p."""
    if not dist:
        return []
    sorted_items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    result: List[int] = []
    cumsum = 0.0
    for tok_id, prob in sorted_items:
        result.append(tok_id)
        cumsum += prob
        if cumsum >= top_p:
            break
    return result


def _select_candidates(dist: Dict[int, float], top_p: float = 0.0,
                       min_p: float = 0.0, top_k: int = TOP_K) -> List[int]:
    """Select candidate token IDs using the active policy (top-p > min-p > top-k)."""
    if top_p > 0:
        return topp_token_ids(dist, top_p)
    if min_p > 0:
        return minp_token_ids(dist, min_p)
    return topk_token_ids(dist, top_k)


def make_semantic_matcher(model_path: str, threshold: float) -> Any:
    """Near-synonym test for the tolerant vote (2026-10-09): LaBSE pooled embeddings on CPU, cached per string; both
    strings must contain a CJK character."""
    import torch
    from transformers import AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).eval()
    cache: Dict[str, Any] = {}
    lock = threading.Lock()

    def vector(text: str) -> Any:
        if text not in cache:
            batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=16)
            with torch.no_grad():
                pooled = model(**batch).pooler_output[0]
            cache[text] = torch.nn.functional.normalize(pooled, dim=-1)
        else:
            pass
        return cache[text]

    def similar(a: str, b: str) -> bool:
        if not (re.search(r"[\u4e00-\u9fff]", a) and re.search(r"[\u4e00-\u9fff]", b)):
            return False
        else:
            with lock:
                return float(vector(a) @ vector(b)) >= threshold

    return similar


def choose_consensus_token_tolerant(
    distributions: List[Dict[int, float]],
    token_text: Any,
    similar: Any,
    soft_vote_top_k: int,
    soft_vote_min_p: float,
    min_voters_ratio: float,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """Tolerant vote (2026-10-09 test). A voter supports candidate A when one of its own candidates (p >= soft_vote_min_p,
    top K) is A, starts with A's text, or (similar given) is a near-synonym of A. Its support is the highest probability
    among those. Eligibility (enough supporting voters), ranking (summed support) and the unique-winner rule are the
    exact vote's."""
    voter_lists = []
    for dist in distributions:
        filtered = sorted(((tok, p) for tok, p in dist.items() if p >= soft_vote_min_p), key=lambda kv: kv[1], reverse=True)
        voter_lists.append([(tok, p, token_text(tok).strip()) for tok, p in filtered[:soft_vote_top_k]])
    pool = {tok: text for candidates in voter_lists for tok, _, text in candidates}
    own = Counter()
    for candidates in voter_lists:
        for tok, p, _ in candidates:
            own[tok] += p
    support: Dict[int, List[float]] = {}
    for tok, text in pool.items():
        per_voter = []
        for candidates in voter_lists:
            per_voter.append(max((p for other, p, other_text in candidates
                                  if other == tok
                                  or (text and other_text.startswith(text))
                                  or (similar is not None and text and other_text and similar(text, other_text))),
                                 default=0.0))
        support[tok] = per_voter
    num_futures = len(distributions)
    min_voters = max(1, math.ceil(num_futures * min_voters_ratio))
    eligible = {tok: (sum(v), own[tok]) for tok, v in support.items() if sum(1 for p in v if p > 0) >= min_voters}
    base_meta = {"num_futures": num_futures, "min_voters": min_voters, "soft_vote_min_p": soft_vote_min_p, "vote_match": "tolerant"}
    if not eligible:
        close = sorted(support.items(), key=lambda kv: (sum(1 for p in kv[1] if p > 0), sum(kv[1])), reverse=True)
        return None, {"reason": "no_token_with_enough_voters", "intersection": [t for t, _ in close[:soft_vote_top_k]], **base_meta}
    else:
        ranked = sorted(eligible.items(), key=lambda kv: kv[1], reverse=True)
        top_tok, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else (0.0, 0.0)
        base_meta = {**base_meta, "intersection": [tok for tok, _ in ranked[:soft_vote_top_k]],
                     "min_voter_p": min(support[top_tok])}
        if top_score <= second_score:
            return None, {"reason": "tied_top", **base_meta}
        else:
            return top_tok, {"reason": "ok", "avg_score": top_score[0] / num_futures, **base_meta}


def choose_consensus_token(
    distributions: List[Dict[int, float]],
    min_p: float = MIN_P,
    candidate_top_k: int = TOP_K,
    top_p: float = 0.0,
    soft_vote_top_k: int = 20,
    soft_vote_min_p: float = 0.1,
    soft_vote_threshold: float = 0.0,  # deprecated, unused
    min_voters_ratio: float = 0.75,
) -> Tuple[Optional[int], Dict[str, Any]]:
    # Looser-than-intersection majority vote.
    # Per future:  filter dist to {tok : p >= soft_vote_min_p}, then take top-K.
    # For each token in that filtered set, increment per_token_voters[tok] and
    # accumulate agg_score[tok] += p.
    # A token is eligible to win iff per_token_voters[tok] >= ceil(N*ratio).
    # Winner = argmax_{eligible} agg_score, must be strictly unique.
    del soft_vote_threshold  # silently ignored — deprecated argument
    if not distributions:
        return None, {"reason": "no_distributions"}
    candidate_lists = [_select_candidates(dist, top_p=top_p, min_p=min_p, top_k=soft_vote_top_k)
                       for dist in distributions]
    num_futures = len(distributions)
    per_token_voters: Counter = Counter()
    agg_score: Counter = Counter()
    for dist in distributions:
        filtered = [(tok, p) for tok, p in dist.items() if p >= soft_vote_min_p]
        filtered.sort(key=lambda kv: kv[1], reverse=True)
        filtered = filtered[:soft_vote_top_k]
        for tok, p in filtered:
            per_token_voters[tok] += 1
            agg_score[tok] += p
    min_voters = max(1, math.ceil(num_futures * min_voters_ratio))
    eligible = {tok: agg_score[tok] for tok, v in per_token_voters.items() if v >= min_voters}
    base_meta = {
        "candidate_lists": candidate_lists,
        "num_futures": num_futures,
        "min_voters": min_voters,
        "soft_vote_min_p": soft_vote_min_p,
    }
    if not eligible:
        top_close = per_token_voters.most_common(soft_vote_top_k)
        return None, {"reason": "no_token_with_enough_voters",
                      "intersection": [t for t, _ in top_close],
                      "close_voters": top_close, **base_meta}
    ranked = sorted(eligible.items(), key=lambda kv: kv[1], reverse=True)
    top_tok, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    base_meta = {
        **base_meta,
        "intersection": [tok for tok, _ in ranked[:soft_vote_top_k]],
        "top_voters": per_token_voters[top_tok],
    }
    if top_score <= second_score:
        return None, {"reason": "tied_top", **base_meta}
    return top_tok, {
        "reason": "ok",
        "avg_score": top_score / num_futures,
        "min_voter_p": min(dist.get(top_tok, 0.0) for dist in distributions),
        **base_meta,
    }


# ---------------------------------------------------------------------------
# Token buffer management
# ---------------------------------------------------------------------------

def decode_token_ids_to_text(tokenizer: Any, token_ids: List[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def inspect_token_ids(tokenizer: Any, token_ids: List[int]) -> Dict[str, Any]:
    # Decode the entire token ID buffer to text for debugging and logging.  This is NOT used for any model input or state management, to avoid drift from re-encoding incomplete token sequences.
    decoded_text = decode_token_ids_to_text(tokenizer, token_ids)
    last_token_id = token_ids[-1] if token_ids else None
    last_token_text = _single_token_text(tokenizer, last_token_id) if last_token_id is not None else ""
    return {"decoded_text": decoded_text, "last_token_id": last_token_id, "last_token_text": last_token_text}


def has_suspicious_content(text: str) -> bool:
    if not text:
        return False
    # Check the ENTIRE decoded text for replacement characters, not just tail.
    # Byte-level BPE tokens may form broken sequences anywhere in the buffer
    # when consensus picks a byte token but fails to pick its pair.
    if "\ufffd" in text or "�" in text:
        return True
    last_char = text[-1]
    if last_char in {"\u200d", "\ufe0f"}:
        return True
    if unicodedata.category(last_char) in {"Mn", "Mc", "Me", "Cc", "Cs"}:
        return True
    return False


def sanitize_pending_token_ids(
    tokenizer: Any, pending_token_ids: List[int],
) -> Tuple[List[int], List[Dict[str, Any]]]:
# Filter out any disallowed tokens from the pending token ID buffer, and log the removed tokens with reasons.  This ensures that we never commit or feed back into the model any token that could cause decoding issues or model confusion.
    kept: List[int] = []
    removed: List[Dict[str, Any]] = []
    for idx, tok_id in enumerate(pending_token_ids):
        reason = _disallowed_generation_token_reason(tokenizer, tok_id)
        if reason is None:
            kept.append(tok_id)
        else:
            removed.append({"position": idx, "token_id": int(tok_id),
                            "token_text": _single_token_text(tokenizer, tok_id), "reason": reason})
    return kept, removed


def trim_pending_tokens_to_complete_boundary(
    tokenizer: Any, committed_text: str, pending_token_ids: List[int],
) -> Tuple[List[int], List[Dict[str, Any]], List[int], Dict[str, Any]]:
    work, removed_disallowed = sanitize_pending_token_ids(tokenizer, pending_token_ids)
    if not work:
        return [], removed_disallowed, [], {
            "decoded_text": "", "last_token_id": None, "last_token_text": "", "full_text": committed_text,
        }
    # Commit only an exact prefix of the original token-id buffer.
    # This preserves the true model state and avoids drift from re-encoding text
    # such as ["而", "且"] into a merged token like ["而且"].
    last_committable_idx = -1 #从前往后逐个decode，找到最后一个"解码后没有乱码"的位置
    for i in range(len(work)):
        partial = decode_token_ids_to_text(tokenizer, work[:i + 1])
        if not has_suspicious_content(partial): #没有U+FFFD等异常字符，说明到这里的token组合是完整的
            last_committable_idx = i
    if last_committable_idx >= 0:
        clean_ids = work[:last_committable_idx + 1]
        removed_tail = work[last_committable_idx + 1:]
        view = inspect_token_ids(tokenizer, clean_ids)
        view["full_text"] = committed_text + view["decoded_text"]
        return clean_ids, removed_disallowed, removed_tail, view
    return [], removed_disallowed, work, {
        "decoded_text": "", "last_token_id": None, "last_token_text": "", "full_text": committed_text,
    }


def finalize_external_commit(
    tokenizer: Any, committed_text: str, pending_token_ids: List[int],
) -> Tuple[str, str, List[int], Dict[str, Any]]:
    trimmed, removed_disallowed, removed_tail, view = trim_pending_tokens_to_complete_boundary(
        tokenizer, committed_text, pending_token_ids)
    commit_text = view["decoded_text"]
    return committed_text + commit_text, commit_text, trimmed, {
        "pending_before_trim": decode_token_ids_to_text(tokenizer, pending_token_ids),
        "pending_after_trim": commit_text,
        "removed_disallowed_tokens": removed_disallowed,
        "removed_tail_text": decode_token_ids_to_text(tokenizer, removed_tail),
    }


# ---------------------------------------------------------------------------
# Core consensus loop
# ---------------------------------------------------------------------------

def anchor_ratio_commit(tokenizer: Any, source_observed: str, futures: List[str], oracle_future: str, future_join_mode: str,
                        committed_token_ids: List[int], anchor_ids: List[int], ratio: float, max_tokens: int,
                        future_notes: Optional[List[str]], api_base: str, api_model: str, api_timeout: float,
                        target_lang: str, shared_first: bool, floor: float = 0.0) -> Tuple[List[int], List[Dict[str, Any]]]:
    """--anchor-ratio: the longest prefix of the remaining reference (at most max_tokens tokens) whose every token gets,
    under every sampled future, at least `ratio` times its probability under the true continuation."""
    remaining = list(anchor_ids[len(committed_token_ids):len(committed_token_ids) + max_tokens])
    if not remaining or not oracle_future:
        return [], [{"step": 0, "stop": "reference_end" if not remaining else "no_oracle"}]
    else:
        sources = [join_future_to_source(source_observed, f, future_join_mode) for f in futures]
        sources.append(join_future_to_source(source_observed, oracle_future, future_join_mode))
    notes = list(future_notes or []) + [""] * (len(sources) - len(future_notes or []))
    prompts = [build_translation_probe_prompt_prefix_token_ids(tokenizer, src, has_target_prefix=bool(committed_token_ids),
                                                              target_lang=target_lang, shared_first=shared_first, note=note)
               + list(committed_token_ids) + remaining
               for src, note in zip(sources, notes)]
    started = time.perf_counter()
    data = _http_json(f"{normalize_api_base(api_base)}/completions",
                      payload={"model": api_model, "prompt": prompts, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0},
                      timeout=api_timeout)
    logprobs = []   # per voter, the log-probability of each remaining reference token
    for choice in data["choices"]:
        tail = choice["prompt_logprobs"][-len(remaining):]
        logprobs.append([entry[str(token)]["logprob"] for entry, token in zip(tail, remaining)])
    oracle_lp, future_lps = logprobs[-1], logprobs[:-1]
    if floor > 0:   # --anchor-ratio-floor: the yardstick never drops below the floor
        oracle_lp = [max(lp, math.log(floor)) for lp in oracle_lp]
    else:
        pass
    threshold = math.log(ratio)
    accepted = 0
    for j in range(len(remaining)):
        if all(lp[j] - oracle_lp[j] >= threshold for lp in future_lps):
            accepted = j + 1
        else:
            break
    elapsed = round(time.perf_counter() - started, 3)
    logs: List[Dict[str, Any]] = []
    for j in range(accepted):
        logs.append({"step": j, "elapsed_s": elapsed if j == 0 else 0.0, "accepted_token_id": remaining[j],
                     "accepted_token_text": _single_token_text(tokenizer, remaining[j]),
                     "pending_text": decode_token_ids_to_text(tokenizer, remaining[:j + 1]), "per_future": [],
                     "meta": {"anchor_ratio": True, "min_voter_p": min(math.exp(lp[j]) for lp in future_lps),
                              "oracle_p": round(math.exp(oracle_lp[j]), 5),
                              "min_ratio": round(min(math.exp(lp[j] - oracle_lp[j]) for lp in future_lps), 5)}})
    if accepted < len(remaining):
        j = accepted
        logs.append({"step": j, "stop": "ratio_veto", "elapsed_s": elapsed if j == 0 else 0.0, "per_future": [],
                     "meta": {"candidate": _single_token_text(tokenizer, remaining[j]), "oracle_p": round(math.exp(oracle_lp[j]), 5),
                              "ratios": [round(math.exp(lp[j] - oracle_lp[j]), 5) for lp in future_lps],
                              "vetoing": [futures[i] for i, lp in enumerate(future_lps) if lp[j] - oracle_lp[j] < threshold][:4],
                              "intersection": []}})
    else:
        pass
    return remaining[:accepted], logs


def extend_pending_tokens(
    instruct_tokenizer: Any,
    source_observed: str,
    futures: List[str],
    committed_text: str,
    committed_token_ids: List[int],
    max_consensus_steps: int,
    candidate_top_k: int = TOP_K,
    instruct_api_base: str = "",
    instruct_api_model: str = "",
    instruct_api_timeout: float = 120.0,
    min_p: float = 0.0,
    top_p: float = 0.0,
    target_lang: str = "Chinese",
    soft_vote_top_k: int = 20,
    soft_vote_min_p: float = 0.1,
    soft_vote_threshold: float = 0.8,
    min_voters_ratio: float = 0.75,
    future_join_mode: str = "space",
    probe_shared_first: bool = False,
    oracle_future: str = "",
    oracle_votes: bool = False,
    probe_temperature: float = 0.0,
    probe_top_p: float = 0.0,
    probe_top_k: int = 0,
    probe_thinking: str = "off",
    probe_thinking_max_tokens: int = 8192,
    vote_match: str = "exact",
    semantic_matcher: Any = None,
    future_notes: Optional[List[str]] = None,
    anchor_ids: Optional[List[int]] = None,
    anchor_min_p: float = 0.02,
    anchor_ratio: float = 0.0,
    anchor_ratio_floor: float = 0.0,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    pending_token_ids: List[int] = []
    grow_logs: List[Dict[str, Any]] = []
    if anchor_ids is not None and anchor_ratio > 0:
        return anchor_ratio_commit(
            instruct_tokenizer, source_observed, futures, oracle_future, future_join_mode, committed_token_ids,
            anchor_ids, anchor_ratio, max_consensus_steps, future_notes, instruct_api_base, instruct_api_model,
            instruct_api_timeout, target_lang, probe_shared_first, anchor_ratio_floor)
    else:
        pass

    # 2a) 每条 future 拼成完整 source（本 chunk 内不变），问 instruct 模型："给定这个完整英文 + 已有中文前缀，下一个中文 token 是啥？"
    full_sources = [join_future_to_source(source_observed, f, future_join_mode) for f in futures]
    # The oracle rides along in the same batch (no extra round trip) but is sliced
    # off before the vote. It only ever answers "would the true continuation have
    # rejected what we just committed?".
    n_voting = len(full_sources)
    if oracle_future:
        full_sources.append(join_future_to_source(source_observed, oracle_future, future_join_mode))
        if oracle_votes:
            # The oracle is future N+1: it counts toward num_futures, so under
            # min_voters_ratio=1.0 the true continuation must also agree before a
            # token can commit.
            n_voting = len(full_sources)

    # Probe thinking (2026-10-06): reasoning once per chunk, for every probe source (all) or only the true
    # continuation (oracle), reused by every probe step below.
    if probe_thinking == "all":
        thinking_indices = list(range(len(full_sources)))
    elif probe_thinking == "oracle" and oracle_future:
        thinking_indices = [len(full_sources) - 1]
    else:
        thinking_indices = []
    reasonings: Optional[List[Optional[str]]] = None
    thinking_log: Dict[str, Any] = {}
    if thinking_indices:
        thinking_started = time.perf_counter()
        generated = generate_probe_reasonings(
            instruct_tokenizer, [full_sources[i] for i in thinking_indices], has_target_prefix=bool(committed_token_ids),
            api_base=instruct_api_base, api_model=instruct_api_model, target_lang=target_lang,
            shared_first=probe_shared_first, max_tokens=probe_thinking_max_tokens,
        )
        reasonings = [None] * len(full_sources)
        for i, g in zip(thinking_indices, generated):
            reasonings[i] = g["text"]
        thinking_log = {
            "mode": probe_thinking,
            "elapsed_s": round(time.perf_counter() - thinking_started, 3),
            "reasoning_chars": [len(g["text"]) for g in generated],
            "cut": sum(g["cut"] for g in generated),
            "oracle_reasoning": reasonings[-1] if oracle_future else None,
        }
    else:
        pass

    for step_idx in range(max_consensus_steps): #最多32 步
        target_prefix_token_ids = list(committed_token_ids) + list(pending_token_ids)
        probe_started = time.perf_counter()

        batch_results = batch_get_next_token_distributions(
            tokenizer=instruct_tokenizer,
            full_sources=full_sources,
            target_prefix_token_ids=target_prefix_token_ids,
            top_k=candidate_top_k,
            min_p=min_p,
            top_p=top_p,
            api_base=instruct_api_base,
            api_model=instruct_api_model,
            api_timeout=instruct_api_timeout,
            target_lang=target_lang,
            min_logprobs=soft_vote_top_k,
            shared_first=probe_shared_first,
            probe_temperature=probe_temperature,
            probe_top_p=probe_top_p,
            probe_top_k=probe_top_k,
            reasonings=reasonings,
            notes=(future_notes + [""] * (len(full_sources) - len(future_notes))) if future_notes else None,
        )

        if oracle_future and oracle_votes:
            oracle_result = batch_results[n_voting - 1] if len(batch_results) >= n_voting else None
        else:
            oracle_result = batch_results[n_voting] if oracle_future and len(batch_results) > n_voting else None
        batch_results = batch_results[:n_voting]

        distributions: List[Dict[int, float]] = []
        per_future: List[Dict[str, Any]] = []
        # When the oracle votes it is voter N+1, so batch_results is one longer than
        # `futures`. Label by position against a list that includes it, or `futures[i]`
        # runs off the end on the last voter.
        voter_labels = list(futures)
        if oracle_future and oracle_votes:
            voter_labels.append(f"[oracle] {oracle_future}")
        for i, (dist, dist_debug) in enumerate(batch_results):
            label = voter_labels[i] if i < len(voter_labels) else f"[voter {i}]"
            if not dist:
                grow_logs.append({"step": step_idx, "stop": "empty_distribution",
                                  "future": label, "dist_debug": dist_debug})
                grow_logs[0].update({"thinking": thinking_log} if thinking_log else {})
                return pending_token_ids, grow_logs
            distributions.append(dist)
            # select 3 policies: topk, topp, minp
            candidate_ids = _select_candidates(dist, top_p=top_p, min_p=min_p, top_k=soft_vote_top_k)
            per_future.append({
                "future": label,
                "candidate_texts": [_single_token_text(instruct_tokenizer, t) for t in candidate_ids],
                "candidate_probs": [dist.get(t, 0.0) for t in candidate_ids],
                "num_candidates": len(candidate_ids),
            })

        # 共识选 token
        if anchor_ids is not None:
            position = len(committed_token_ids) + len(pending_token_ids)
            if position >= len(anchor_ids):
                consensus_token_id, meta = None, {"anchor": True, "stop_reason": "reference_end", "min_voter_p": 0.0, "intersection": []}
            else:
                voter_probs = [dist.get(anchor_ids[position], 0.0) for dist in distributions]
                accepted = min(voter_probs) >= anchor_min_p
                consensus_token_id = anchor_ids[position] if accepted else None
                meta = {"anchor": True, "candidate": _single_token_text(instruct_tokenizer, anchor_ids[position]),
                        "voter_probs": [round(p, 4) for p in voter_probs], "min_voter_p": min(voter_probs),
                        "intersection": [anchor_ids[position]] if accepted else []}
        elif vote_match == "exact":
            consensus_token_id, meta = choose_consensus_token(
                distributions,
                min_p=min_p,
                candidate_top_k=candidate_top_k,
                top_p=top_p,
                soft_vote_top_k=soft_vote_top_k,
                soft_vote_min_p=soft_vote_min_p,
                soft_vote_threshold=soft_vote_threshold,
                min_voters_ratio=min_voters_ratio,
            )
        else:
            consensus_token_id, meta = choose_consensus_token_tolerant(
                distributions,
                token_text=lambda t: _single_token_text(instruct_tokenizer, t),
                similar=semantic_matcher if vote_match == "prefix_semantic" else None,
                soft_vote_top_k=soft_vote_top_k,
                soft_vote_min_p=soft_vote_min_p,
                min_voters_ratio=min_voters_ratio,
            )
        probe_elapsed = round(time.perf_counter() - probe_started, 3)

        oracle_audit: Optional[Dict[str, Any]] = None
        if oracle_result is not None:
            odist = oracle_result[0] or {}
            oracle_audit = {"oracle_future": oracle_future, "has_dist": bool(odist),
                            "voted": bool(oracle_votes)}
            if odist:
                ocands = set(_select_candidates(odist, top_p=top_p, min_p=min_p, top_k=soft_vote_top_k))
                otop = max(odist, key=odist.get)
                oracle_audit["oracle_top_token"] = _single_token_text(instruct_tokenizer, otop)
                if consensus_token_id is not None:
                    oracle_audit["committed_token"] = _single_token_text(instruct_tokenizer, consensus_token_id)
                    # would_veto: the sampled futures were unanimous, but the true
                    # continuation would not have voted for this token.
                    oracle_audit["would_veto"] = consensus_token_id not in ocands
                    oracle_audit["committed_prob_under_oracle"] = round(odist.get(consensus_token_id, 0.0), 5)

        if consensus_token_id is None: #没有共识token，停止本轮生长
            grow_logs.append({"step": step_idx, "stop": "no_consensus_token",
                              "per_future": per_future, "meta": meta, "elapsed_s": probe_elapsed,
                              **({"oracle_audit": oracle_audit} if oracle_audit else {})})
            break

        pending_token_ids.append(consensus_token_id) #共识成功，追加到pending buffer，继续下一步
        view = inspect_token_ids(instruct_tokenizer, pending_token_ids) #解码当前pending buffer，准备日志
        log_entry = {
            "step": step_idx,
            "elapsed_s": probe_elapsed,
            "accepted_token_id": consensus_token_id,
            "accepted_token_text": view["last_token_text"],
            "pending_text": view["decoded_text"],
            "llm_prefix": decode_token_ids_to_text(instruct_tokenizer, target_prefix_token_ids),
            "llm_prefix_token_ids": target_prefix_token_ids,
            "per_future": per_future,
            "meta": meta,
        }
        if oracle_audit:
            log_entry["oracle_audit"] = oracle_audit
        grow_logs.append(log_entry)

    grow_logs[0].update({"thinking": thinking_log} if thinking_log else {})
    return pending_token_ids, grow_logs


# ---------------------------------------------------------------------------
# Force-complete last chunk
# ---------------------------------------------------------------------------

def force_complete_translation(
    tokenizer: Any,
    full_source: str,
    committed_text: str,
    api_base: str,
    api_model: str,
    api_timeout: float = 120.0,
    target_lang: str = "Chinese",
    max_tokens: int = 128,
    terminal_mark: str = "",
    thinking: bool = False,
    thinking_max_tokens: int = 8192,
) -> str:
    if thinking:
        # Completion thinking (Haoling 2026-10-06): reason first under the same user turn, then complete the committed
        # prefix after the reasoning, greedy as before.
        head = build_final_completion_prompt(tokenizer, full_source, committed_text, target_lang=target_lang,
                                             terminal_mark=terminal_mark, enable_thinking=True, append_committed=False)
        reasoning = generate_reasonings([head], api_base, api_model, thinking_max_tokens)[0]["text"]
        prompt = head + reasoning + "\n</think>\n\n" + str(committed_text or "")
    else:
        prompt = build_final_completion_prompt(tokenizer, full_source, committed_text, target_lang=target_lang, terminal_mark=terminal_mark)
    payload = {
        "model": api_model,
        "prompt": prompt,
        "max_tokens": max(1, int(max_tokens)),
        "temperature": 0.0,
        "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    }
    data = _http_json(f"{normalize_api_base(api_base)}/completions", payload=payload, timeout=api_timeout) #call instruct model，带上已提交翻译前缀，让模型把剩余翻译补完
    choices = data.get("choices", [])
    if terminal_mark and (not choices or choices[0].get("finish_reason") == "length"):
        raise RuntimeError("Boundary completion missing or token-truncated; refusing to fabricate a closed sentence")
    if not choices:
        return ""
    raw_completion = str(choices[0].get("text", ""))
    completion = clean_model_text(raw_completion)
    # A commit can now end on an English word ("我们用 Visual"); keep the space the model put before the next English
    # word, which clean_model_text strips, so the target does not read "VisualStudio Code".
    if (completion and raw_completion[:1].isspace() and re.search(r"[A-Za-z0-9]$", committed_text)
            and re.match(r"[A-Za-z0-9]", completion)):
        return " " + completion
    else:
        return completion


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _extract_reference_text_from_row(row: Dict[str, Any], target_lang: str = "Chinese") -> Optional[str]:
    lang_suffix_map = {"Japanese": "ja", "German": "de", "French": "fr", "Spanish": "es"}
    lang_suffix = lang_suffix_map.get(target_lang, "")
    keys = []
    if lang_suffix:
        keys.extend([f"target_full_{lang_suffix}", f"tgt_text_full_{lang_suffix}", f"llm_reference_text_{lang_suffix}"])
    keys.extend(["llm_reference_text", "tgt_text_full", "tgt_text", "target_text", "translation", "ref_text", "reference"])
    for key in keys:
        raw = row.get(key)
        if raw is None or pd.isna(raw):
            continue
        text = str(raw).strip()
        if text and text.lower() != "nan":
            return text
    return None


def compute_laal(
    source_chunks: List[str], target_deltas: List[str], actions: List[str], reference: str,
) -> float:
    timeline: List[int] = []
    source_read = 0
    for chunk, delta, action in zip(source_chunks, target_deltas, actions):
        source_read += len(str(chunk).strip().split()) if str(chunk).strip() else 0
        if action == "WRITE" and str(delta).strip():
            for _ in str(delta).strip():
                timeline.append(source_read)
    y_len = len("".join(d for d in target_deltas if d))
    yref_len = len(str(reference).replace(" ", ""))
    x_len = sum(len(str(c).strip().split()) for c in source_chunks if str(c).strip())
    if y_len == 0 or x_len == 0 or yref_len == 0:
        return float("nan")
    denom = max(y_len, yref_len)
    if denom <= 0 or not timeline:
        return float("nan")
    total = 0.0
    for i in range(1, denom + 1):
        d_i = timeline[i - 1] if i <= len(timeline) else x_len
        total += d_i - (i - 1) * x_len / denom
    return total / denom


def compute_bleu_char(hypothesis: str, reference: str, max_order: int = 4, smooth: bool = True) -> float:
    hyp = [c for c in str(hypothesis) if not c.isspace()]
    ref = [c for c in str(reference) if not c.isspace()]
    hyp_len, ref_len = len(hyp), len(ref)
    if hyp_len == 0 or ref_len == 0:
        return float("nan")
    eff_order = min(max_order, hyp_len, ref_len)
    if eff_order <= 0:
        return float("nan")
    precisions: List[float] = []
    for n in range(1, eff_order + 1):
        hyp_ngrams = Counter(tuple(hyp[i:i + n]) for i in range(hyp_len - n + 1))
        ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(ref_len - n + 1))
        total = sum(hyp_ngrams.values())
        if total <= 0:
            return float("nan")
        clipped = sum(min(cnt, ref_ngrams.get(ng, 0)) for ng, cnt in hyp_ngrams.items())
        if smooth:
            precisions.append((clipped + 1.0) / (total + 1.0))
        else:
            if clipped == 0:
                return 0.0
            precisions.append(clipped / total)
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len))
    return bp * math.exp(sum(math.log(p) for p in precisions) / eff_order) * 100.0


def _nonspace_char_count(text: str) -> int:
    return sum(1 for c in str(text or "") if not c.isspace())


def compute_length_ratio_ref(hypothesis: str, reference: str) -> float:
    hyp_len = _nonspace_char_count(hypothesis)
    ref_len = _nonspace_char_count(reference)
    if hyp_len == 0 or ref_len == 0:
        return float("nan")
    return hyp_len / ref_len


def compute_length_ratio_src(hypothesis: str, source: str) -> float:
    hyp_len = _nonspace_char_count(hypothesis)
    src_word_count = len(str(source or "").split())
    if hyp_len == 0 or src_word_count == 0:
        return float("nan")
    return hyp_len / src_word_count


# ---------------------------------------------------------------------------
# Run one utterance
# ---------------------------------------------------------------------------

def run_one_utterance(
    row: Dict[str, Any],
    args: argparse.Namespace,
    base_specs: List[Dict[str, Any]],
    instruct_tokenizer: Any,
    sampler_tokenizer: Any = None,
    sampler2_tokenizer: Any = None,
    verbose_log_file: Optional[Any] = None,
) -> Dict[str, Any]:
    utt_id = str(row.get(args.id_column, row.get("id", f"row_{args.row_idx}")))
    # 2026-10-09 vote test: the known ambiguous word and its readings, for --future-roles given_word only
    if args.future_roles == "given_word":
        ambiguity_hint = {"word": str(row["ambiguous_word"]), "readings": json.loads(row["ambiguous_readings"])}
    else:
        ambiguity_hint = None
    # 2026-10-10 design v3: the utterance's ambiguity list, built before decoding (build_ambiguity_tsv.py)
    ambiguity_list = json.loads(row["ambiguity_list"]) if args.future_roles == "list" else []
    chunks = parse_trajectory(row["src_trajectory"])
    source_units = parse_source_units(row.get("src_text_full"))
    full_source_text = get_full_source_text(row)
    if args.anchor_reference:
        anchor_text = _extract_reference_text_from_row(row, target_lang=args.target_lang)
        anchor_ids = instruct_tokenizer.encode(anchor_text, add_special_tokens=False)
        anchor_prefix_lengths = [len(instruct_tokenizer.decode(anchor_ids[:n])) for n in range(len(anchor_ids) + 1)]
        reference_units = parse_source_units(row.get("llm_reference_text_list"))
        anchor_sentence_ends = []   # token count at the end of reference sentence k (one per source unit)
        search_from = 0
        for unit in reference_units if len(reference_units) == len(source_units) else []:
            found = anchor_text.find(unit.strip(), search_from)
            search_from = found + len(unit.strip()) if found >= 0 else search_from
            anchor_sentence_ends.append(next((n for n, length in enumerate(anchor_prefix_lengths) if length >= search_from), len(anchor_ids)))
    else:
        anchor_ids, anchor_prefix_lengths, anchor_sentence_ends = None, [], []

    committed_text = ""
    committed_token_ids: List[int] = []
    target_deltas: List[str] = []
    actions: List[str] = []
    boundary_audit: List[Dict[str, Any]] = []
    sampling_audit: List[Dict[str, Any]] = []

    if args.top_p > 0:
        candidate_policy = f"top_p({args.top_p})"
    elif args.min_p > 0:
        candidate_policy = f"min_p({args.min_p})"
    else:
        candidate_policy = f"top_k({args.candidate_top_k})"

    # ── verbose header ──
    _vlog(verbose_log_file, "############################################################")
    _vlog(verbose_log_file, f"# utt_id: {utt_id}")
    _vlog(verbose_log_file, f"# source_full_text: {full_source_text}")
    _vlog_pretty_value(verbose_log_file, "src_text_full", source_units)
    _vlog_pretty_value(verbose_log_file, "src_trajectory", chunks)
    _vlog(verbose_log_file, f"# Chunks: {len(chunks)}")
    _vlog(verbose_log_file, f"# num_futures={args.num_futures}, top_k={args.candidate_top_k}, min_p={args.min_p}")
    for si, bs in enumerate(base_specs):
        label = bs.get("name", f"spec_{si}")
        _vlog(verbose_log_file, f"# base_model[{label}]: api model={bs.get('api_model','')} base={bs.get('api_base','')} num_futures={bs.get('num_futures','')}")
    _vlog(verbose_log_file, "# instruct_backend: vllm_completion")
    _vlog(verbose_log_file, f"# targeted_sampler_context: {args.targeted_sampler_context}")
    _vlog(verbose_log_file, f"# targeted_prompt_version: {args.targeted_prompt_version}")
    _vlog(verbose_log_file, f"# probe_prompt_order: {args.probe_prompt_order} contrastive_notes: {args.contrastive_notes}")
    _vlog(verbose_log_file, f"# sentence_end_punctuation: {args.sentence_end_punctuation}")
    _vlog(verbose_log_file, "############################################################")

    last_unit_idx = -1
    unit_start_committed = ""
    for t in range(len(chunks)):
        current_source_chunk = str(chunks[t] or "")
        source_observed_full = build_source_observed(chunks, t)
        window_units = args.future_source_window_chunks
        if getattr(args, "future_source_window_mode", "fixed") == "until-closed" and window_units > 0:
            unit_idx = current_unit_index(source_units, source_observed_full)
            if unit_idx != last_unit_idx:
                unit_start_committed = committed_text
                last_unit_idx = unit_idx
            if unit_idx > 0 and not target_is_closed(unit_start_committed):
                window_units = max(window_units, 2)
        source_observed = build_source_observed_recent_units(
            source_units=source_units,
            observed_full=source_observed_full,
            num_units=window_units,
        )
        if getattr(args, "future_source_window_mode", "fixed") == "sentence-anchor":
            source_observed = sentence_anchored_prefix(
                source_observed_full, getattr(args, "future_source_anchor_max_words", 128)
            )
        _vlog(verbose_log_file, f"\n{'='*60}")
        _vlog(verbose_log_file, f"Chunk {t + 1}/{len(chunks)}")
        _vlog(verbose_log_file, f"source_observed: {current_source_chunk!r}")
        _vlog(verbose_log_file, f"future_source_prefix: {source_observed!r}")
        if getattr(args, "future_source_window_mode", "fixed") == "until-closed":
            _vlog(verbose_log_file, f"future_source_window_units: {window_units}")
        if getattr(args, "future_source_window_mode", "fixed") == "sentence-anchor":
            _vlog(verbose_log_file, "future_source_window_mode: sentence-anchor")
        if source_observed != source_observed_full:
            _vlog(verbose_log_file, f"source_observed_full: {source_observed_full!r}")
        _vlog(verbose_log_file, f"committed_before: {committed_text!r}")

        def log_sampling(status: str, reason: str) -> None:
            context = args.targeted_sampler_context if args.use_targeted_instruct_sampling else "source-only"
            event = {
                "chunk": t + 1, "status": status, "reason": reason,
                "prepared_source_prefix": source_observed,
                "input_source_prefix": source_observed if status == "invoked" else None,
                "context_mode": context,
                "prompt_version": args.targeted_prompt_version if args.use_targeted_instruct_sampling else None,
                "window_mode": args.future_source_window_mode,
                "committed_target_context": (
                    committed_text.strip() if status == "invoked" and context == "source-and-target" else ""
                ),
                "evidence": "decoder_dispatch",
            }
            sampling_audit.append(event)
            _vlog(verbose_log_file, f"[FutureSampling] {json.dumps(event, ensure_ascii=False)}")
            # One flush per chunk exposes the input before slow model calls.
            if verbose_log_file is not None:
                verbose_log_file.flush()

        if t == len(chunks) - 1: #最后一个chunk，不再做共识，直接让instruct model把翻译补完
            log_sampling("skipped", "final_chunk")
            if args.anchor_reference:   # anchor mode: the rest of the reference
                done_tokens = len(committed_token_ids)
                anchor_delta = instruct_tokenizer.decode(anchor_ids)[anchor_prefix_lengths[done_tokens]:]
                committed_text += anchor_delta
                committed_token_ids = list(anchor_ids)
                target_deltas.append(anchor_delta)
                actions.append("WRITE" if anchor_delta else "READ")
                _vlog(verbose_log_file, f"  [AnchorCommit] final delta={anchor_delta!r}")
                continue
            else:
                pass
            terminal = source_terminal_mark(source_observed_full) if args.sentence_end_punctuation == "match-source" else ""
            completion_started = time.perf_counter()
            final_delta = force_complete_translation(
                tokenizer=instruct_tokenizer,
                full_source=source_observed_full,
                committed_text=committed_text,
                api_base=args.instruct_api_base,
                api_model=args.instruct_api_model,
                api_timeout=args.instruct_api_timeout,
                target_lang=args.target_lang,
                max_tokens=args.final_max_tokens,
                terminal_mark=terminal,
                thinking=args.completion_thinking,
                thinking_max_tokens=args.probe_thinking_max_tokens,
            )
            if terminal:
                raw_delta = final_delta
                final_delta, reason = close_translation_delta(raw_delta, committed_text, terminal)
                audit = {"chunk": t + 1, "terminal": terminal, "raw_delta": raw_delta,
                         "delta": final_delta, "reason": reason, "source": source_observed_full}
                boundary_audit.append(audit)
                _vlog(verbose_log_file, f"[BoundaryClose] {json.dumps(audit, ensure_ascii=False)}")
            if final_delta:
                committed_text += final_delta
                target_deltas.append(final_delta)
                actions.append("WRITE")
            else:
                target_deltas.append("")
                actions.append("READ")
            _vlog(verbose_log_file, f"  [Final] delta={final_delta!r}")
            continue

        if args.sentence_end_punctuation == "match-source" and not current_source_chunk.strip():
            log_sampling("skipped", "no_new_source")
            target_deltas.append("")
            actions.append("READ")
            _vlog(verbose_log_file, "[NoNewSource] -> READ; no future sampling")
            continue

        sentence_end = bool(_SENTENCE_END_RE.search(current_source_chunk))
        if getattr(args, "sentence_end_boundary_mode", "literal") == "conservative":
            sentence_end = bool(current_source_chunk.strip()) and observed_sentence_is_complete(source_observed_full)
        if getattr(args, "sentence_end_completion", False) and sentence_end:
            log_sampling("skipped", "source_sentence_end")
            if args.anchor_reference:   # anchor mode: the reference to the end of the source unit just completed
                done_tokens = len(committed_token_ids)
                unit_index = current_unit_index(source_units, source_observed_full)
                through = ""
                for unit in source_units[:unit_index + 1]:
                    through = append_text_continuation(through, unit)
                unit_done = bool(anchor_sentence_ends) and through.strip() == source_observed_full.strip()
                target_tokens = max(done_tokens, anchor_sentence_ends[unit_index]) if unit_done else done_tokens
                anchor_delta = instruct_tokenizer.decode(anchor_ids[:target_tokens])[anchor_prefix_lengths[done_tokens]:]
                committed_text += anchor_delta
                committed_token_ids = list(anchor_ids[:target_tokens])
                target_deltas.append(anchor_delta)
                actions.append("WRITE" if anchor_delta else "READ")
                _vlog(verbose_log_file, f"  [AnchorCommit] sentence end unit={unit_index} done={unit_done} delta={anchor_delta!r}")
                continue
            else:
                pass
            # The source sentence is complete: let the translator finish it instead of voting.
            terminal = source_terminal_mark(source_observed_full) if args.sentence_end_punctuation == "match-source" else ""
            completion_started = time.perf_counter()
            sentence_delta = force_complete_translation(
                tokenizer=instruct_tokenizer,
                full_source=source_observed_full,
                committed_text=committed_text,
                api_base=args.instruct_api_base,
                api_model=args.instruct_api_model,
                api_timeout=args.instruct_api_timeout,
                target_lang=args.target_lang,
                max_tokens=args.final_max_tokens,
                terminal_mark=terminal,
                thinking=args.completion_thinking,
                thinking_max_tokens=args.probe_thinking_max_tokens,
            )
            if terminal:
                raw_delta = sentence_delta
                sentence_delta, reason = close_translation_delta(raw_delta, committed_text, terminal)
                audit = {"chunk": t + 1, "terminal": terminal, "raw_delta": raw_delta,
                         "delta": sentence_delta, "reason": reason, "source": source_observed_full}
                boundary_audit.append(audit)
                _vlog(verbose_log_file, f"[BoundaryClose] {json.dumps(audit, ensure_ascii=False)}")
            if sentence_delta:
                committed_text += sentence_delta
                committed_token_ids.extend(
                    instruct_tokenizer.encode(sentence_delta, add_special_tokens=False)
                )
                target_deltas.append(sentence_delta)
                actions.append("WRITE")
            else:
                target_deltas.append("")
                actions.append("READ")
            _vlog(verbose_log_file, f"  [SentenceEnd] delta={sentence_delta!r}")
            _vlog(verbose_log_file, f"[Timing] completion={time.perf_counter() - completion_started:.2f}s")
            _vlog(verbose_log_file, f"-> {'WRITE' if sentence_delta else 'READ'} delta={sentence_delta!r}")
            _vlog(verbose_log_file, f"committed_after: {committed_text!r}")
            continue

        if not source_observed.strip():
            log_sampling("skipped", "empty_prefix")
        elif not args.use_targeted_instruct_sampling and not any(int(spec.get("num_futures", 0) or 0) > 0 for spec in base_specs):
            log_sampling("skipped", "no_samplers")
        else:
            # Dispatch is not a claim that the HTTP request succeeded.
            log_sampling("invoked", "future_sampling")

        # design v3: listed words heard by this chunk whose sentence is unfinished (sentence-end completion fixes the
        # word's translation when its sentence ends), and the sampler prefix extended back to the start of the sentence
        # holding the earliest of them (only differs when a chunk spans a sentence end)
        list_open = open_entries(source_observed_full, ambiguity_list, t + 1)
        if list_open:
            earliest = min(len(join_source_chunks(chunks[:entry["chunk"] - 1]).strip()) for entry in list_open)
            list_prefix = extend_to_sentence(source_observed_full, source_observed, earliest)
            _vlog(verbose_log_file, "[ListOpen] " + json.dumps({"open": list_open, "prefix": list_prefix}, ensure_ascii=False))
        else:
            list_prefix = source_observed

        if getattr(args, "use_targeted_instruct_sampling", False):
            # Each instruction-tuned sampler sees the observed prefix and plans
            # its full numbered candidate set in one response.
            sampler_api_base = args.targeted_sampler_api_base or args.instruct_api_base
            sampler_api_model = args.targeted_sampler_api_model or args.instruct_api_model
            sampler_api_timeout = (
                args.targeted_sampler_api_timeout
                if args.targeted_sampler_api_timeout and args.targeted_sampler_api_timeout > 0
                else args.instruct_api_timeout
            )
            sampling_started = time.perf_counter()
            futures, future_infos, future_audit = sample_source_futures_targeted_prefill(
                sampler_tokenizer=sampler_tokenizer,
                observed_source=source_observed,
                committed_text=committed_text,
                target_lang=args.target_lang,
                num_futures=args.targeted_num_futures,
                api_base=sampler_api_base,
                api_model=sampler_api_model,
                api_timeout=sampler_api_timeout,
                sample_temperature=args.targeted_sample_temperature,
                top_p=args.targeted_top_p,
                max_tokens=args.targeted_max_tokens,
                sampler2_tokenizer=sampler2_tokenizer,
                sampler2_api_base=args.targeted_sampler2_api_base,
                sampler2_api_model=args.targeted_sampler2_api_model,
                sampler2_api_timeout=args.targeted_sampler2_api_timeout,
                return_audit=True,
                sampler_seed=getattr(args, "targeted_sampler_seed", None),
                sampler_context=args.targeted_sampler_context,
                fail_on_api_error=args.targeted_fail_on_api_error,
                prompt_version=args.targeted_prompt_version,
                parse_retries=args.targeted_parse_retries,
                contrastive_notes=args.contrastive_notes,
                future_roles=args.future_roles,
                ambiguity_hint=ambiguity_hint,
                role_plausible_n=args.role_plausible_n,
                role_per_reading=args.role_per_reading,
                role_max_words=args.role_max_words,
                role_max_readings=args.role_max_readings,
                list_open=list_open,
                list_prefix=list_prefix,
                list_natural=args.list_natural,
                list_natural_n=args.list_natural_n,
                list_per_reading=args.list_per_reading,
                list_targeted_thinking=args.list_targeted_thinking,
                list_think_budget=args.list_think_budget,
                list_skip_ruled_out=args.list_skip_ruled_out,
                list_reading_check=args.list_reading_check,
                list_observed_full=source_observed_full,
            )
        else:
            futures, future_infos = sample_source_futures_multi( #call base models，用当前observed source采样多条未来续写
                base_specs=base_specs,
                observed_source=source_observed,
                future_tokens=args.future_tokens,
                sample_temperature=args.sample_temperature,
            )
            future_audit = []

        if args.use_targeted_instruct_sampling:
            _vlog(verbose_log_file, f"[Timing] sampling={time.perf_counter() - sampling_started:.2f}s")
        # ── verbose: list futures ──
        request_items = [item for item in future_audit if str(item.get("mode", "")).endswith("_request")]
        for item in request_items:
            _vlog(verbose_log_file, "[ListRequest] " + json.dumps(item, ensure_ascii=False))
        future_audit = [item for item in future_audit if not str(item.get("mode", "")).endswith("_request")]
        if future_audit:
            accepted_raw = sum(bool(item.get("accepted")) for item in future_audit)
            _vlog(verbose_log_file,
                  f"[Step 1-1] raw_future_sampling total={len(future_audit)} accepted={accepted_raw}")
            for line in format_raw_future_groups(future_audit):
                _vlog(verbose_log_file, line)
        _vlog(verbose_log_file, f"[Step 1-2] future_sampling total={len(futures)}")
        # The true source continuation. Computed here (not inline at the probe call) so the
        # verbose log can list it alongside the sampled futures -- the review bundle builds
        # its future list by parsing these group lines, so anything not logged here is
        # invisible downstream even when it voted.
        oracle_future_text = (
            oracle_future_suffix(source_units, source_observed_full)
            if (getattr(args, "oracle_future_audit", False)
                or getattr(args, "oracle_future_vote", False))
            else ""
        )
        if verbose_log_file is not None:
            for line in format_selected_future_groups(futures, future_infos):
                _vlog(verbose_log_file, line)
            if oracle_future_text:
                mode = "oracle"
                _vlog(verbose_log_file,
                      f"[Selected candidates] [ORACLE] source continuation | {mode} | "
                      f"model=oracle mode={mode} count=1")
                _vlog(verbose_log_file, f"  01. {oracle_future_text!r}")

        if len(futures) <= 3: #future太少无法做共识，等待更多源语言输入
            target_deltas.append("")
            actions.append("READ")
            _vlog(verbose_log_file, "  -> READ (too few futures)")
            continue

        pending_token_ids, grow_logs = extend_pending_tokens(
            instruct_tokenizer=instruct_tokenizer,
            source_observed=source_observed_full,
            futures=futures,
            future_join_mode=getattr(args, "future_join_mode", "space"),
            committed_text=committed_text,
            committed_token_ids=committed_token_ids,
            max_consensus_steps=args.max_consensus_steps,
            candidate_top_k=args.candidate_top_k,
            instruct_api_base=args.instruct_api_base,
            instruct_api_model=args.instruct_api_model,
            instruct_api_timeout=args.instruct_api_timeout,
            min_p=args.min_p,
            top_p=args.top_p,
            target_lang=args.target_lang,
            soft_vote_top_k=args.soft_vote_top_k,
            soft_vote_min_p=args.soft_vote_min_p,
            soft_vote_threshold=args.soft_vote_threshold,
            min_voters_ratio=args.min_voters_ratio,
            probe_shared_first=args.probe_prompt_order == "shared-first",
            oracle_future=oracle_future_text,
            oracle_votes=getattr(args, "oracle_future_vote", False),
            probe_temperature=args.probe_temperature,
            probe_top_p=args.probe_top_p,
            probe_top_k=args.probe_top_k,
            probe_thinking=args.probe_thinking,
            probe_thinking_max_tokens=args.probe_thinking_max_tokens,
            vote_match=args.vote_match,
            semantic_matcher=args.semantic_matcher,
            anchor_ids=anchor_ids,
            anchor_min_p=args.anchor_min_p,
            anchor_ratio=args.anchor_ratio,
            anchor_ratio_floor=args.anchor_ratio_floor,
            future_notes=([f'In this text, "{info["word"]}" means: {info["reading"].rsplit(" (", 1)[0]}.' if info.get("word") else ""
                           for info in future_infos] if args.list_probe_hint else None),
        )
        thinking_log = grow_logs[0].get("thinking") if grow_logs else None
        if thinking_log:
            _vlog(verbose_log_file, f"[Thinking] mode={thinking_log['mode']} elapsed={thinking_log['elapsed_s']}s "
                                    f"reasoning_chars={thinking_log['reasoning_chars']} cut={thinking_log['cut']}")
            _vlog(verbose_log_file, f"[Thinking] oracle reasoning: {thinking_log['oracle_reasoning']!r}")
        else:
            pass
        _vlog(verbose_log_file, f"[Timing] probe_batches={sum(1 for g in grow_logs if 'elapsed_s' in g)} "
                                f"total={sum(g.get('elapsed_s', 0.0) for g in grow_logs):.2f}s")

        # ── min-consensus-horizon filter ──
        # If the consensus only carried us a few tokens before breaking, the path is fragile.
        # Discard the whole pending buffer and READ instead, to avoid early-commit lock-in.
        if (
            args.min_consensus_horizon > 1
            and 0 < len(pending_token_ids) < args.min_consensus_horizon
        ):
            pending_text_dropped = decode_token_ids_to_text(instruct_tokenizer, pending_token_ids)
            _vlog(verbose_log_file,
                  f"[Step 5.5] horizon_filter: dropped pending={pending_text_dropped!r} "
                  f"(len={len(pending_token_ids)} < min_horizon={args.min_consensus_horizon}) -> READ")
            grow_logs.append({"step": "filter", "stop": "below_min_horizon",
                              "horizon": len(pending_token_ids),
                              "min_horizon": args.min_consensus_horizon})
            pending_token_ids = []
        elif args.single_token_filter and len(pending_token_ids) == 1:
            first = next(gl for gl in grow_logs if "accepted_token_id" in gl)
            single_text = first["accepted_token_text"].strip()
            single_min_p = first["meta"]["min_voter_p"]
            if not single_text or all(unicodedata.category(c)[0] in "PSZ" for c in single_text) or single_min_p < args.single_token_min_p:
                _vlog(verbose_log_file, f"[Step 5.5] single_token_filter: dropped {single_text!r} (min voter p {single_min_p:.3f}) -> READ")
                grow_logs.append({"step": "filter", "stop": "single_token_filter", "token": single_text, "min_voter_p": single_min_p})
                pending_token_ids = []
            else:
                pass
        else:
            pass

        # ── verbose: consensus steps ──
        if verbose_log_file is not None and grow_logs:
            _vlog(verbose_log_file, "[Step 4-5] consensus summary:")
            for gl in grow_logs:
                step = gl.get("step", "?")
                stop = gl.get("stop", "")
                meta = gl.get("meta", {})
                intersection = meta.get("intersection", [])
                if stop:
                    intersection_texts = [_single_token_text(instruct_tokenizer, tid) for tid in intersection] if intersection else []
                    _vlog(verbose_log_file, f"  step={step} stop={stop} intersection={intersection_texts} pending={decode_token_ids_to_text(instruct_tokenizer, pending_token_ids)!r}")
                else:
                    accepted = gl.get("accepted_token_text", "?")
                    pending = gl.get("pending_text", "")
                    _vlog(verbose_log_file, f"  step={step} accepted={accepted!r} pending={pending!r}")
                if not args.compact_verbose:
                    per_future = gl.get("per_future", [])
                    for idx, pf in enumerate(per_future):
                        texts = pf.get("candidate_texts", [])
                        probs = pf.get("candidate_probs", [])
                        pairs = ", ".join(f"{t!r}:{p:.3f}" for t, p in zip(texts, probs))
                        _vlog(verbose_log_file, f"    future[{idx}] candidates={len(texts)}: [{pairs}]")
                oa = gl.get("oracle_audit")
                if oa:
                    # One machine-parseable line per step so the audit can be aggregated
                    # without re-parsing the human-readable block above.
                    _vlog(verbose_log_file, f"  [ORACLE] {json.dumps({'step': step, **oa}, ensure_ascii=False)}")

        new_committed, delta, committed_delta_token_ids, finalize_meta = finalize_external_commit( #修剪pending tokens，只提交解码后没有乱码的前缀部分
            tokenizer=instruct_tokenizer,
            committed_text=committed_text,
            pending_token_ids=pending_token_ids,
        )

        # ── verbose: finalize ──
        pending_text = decode_token_ids_to_text(instruct_tokenizer, pending_token_ids)
        _vlog(verbose_log_file, f"[Step 6-7] pending_before_trim={pending_text!r}")
        _vlog(verbose_log_file, f"[Step 6-7] commit_after_trim={delta!r}")

        action = "WRITE" if delta else "READ"
        target_deltas.append(delta)
        actions.append(action)
        _vlog(verbose_log_file, f"-> {action} delta={delta!r}")
        _vlog(verbose_log_file, f"committed_after: {new_committed!r}")
        committed_text = new_committed
        committed_token_ids.extend(committed_delta_token_ids)

    result: Dict[str, Any] = {
        "utt_id": utt_id,
        "source_full_text": full_source_text,
        "src_text_full": source_units,
        "src_trajectory": chunks,
        "target_trajectory": target_deltas,
        "actions": actions,
        "prediction": committed_text,
        "decoder_impl": {"candidate_policy": candidate_policy, "backend": "vllm_completion"},
        "decoder_settings": {
            "future_roles": args.future_roles,
            "list_natural": args.list_natural if args.future_roles == "list" else None,
            "list_natural_n": args.list_natural_n if args.future_roles == "list" else None,
            "list_per_reading": args.list_per_reading if args.future_roles == "list" else None,
            "list_targeted_thinking": args.list_targeted_thinking if args.future_roles == "list" else None,
            "list_think_budget": args.list_think_budget if args.future_roles == "list" and args.list_targeted_thinking else None,
            "list_sampling": ("no thinking 1.0 / 0.98; thinking 0.6 / 0.95 / top-k 20 with a forced answer"
                              if args.future_roles == "list" else None),
            "list_probe_hint": args.list_probe_hint if args.future_roles == "list" else None,
            "list_skip_ruled_out": args.list_skip_ruled_out if args.future_roles == "list" else None,
            "list_reading_check": args.list_reading_check if args.future_roles == "list" else None,
            "anchor_reference": args.anchor_reference,
            "anchor_min_p": args.anchor_min_p if args.anchor_reference and not args.anchor_ratio else None,
            "anchor_ratio": args.anchor_ratio if args.anchor_reference else None,
            "anchor_ratio_floor": args.anchor_ratio_floor if args.anchor_reference and args.anchor_ratio else None,
            "list_reading_check_mode": "greedy, whole observed text" if args.list_reading_check else None,
            "soft_vote_min_p": args.soft_vote_min_p,
            "min_voters_ratio": args.min_voters_ratio,
            "role_plausible_n": args.role_plausible_n if args.future_roles != "off" else None,
            "role_per_reading": args.role_per_reading if args.future_roles != "off" else None,
            "vote_match": args.vote_match,
            "semantic_threshold": args.semantic_threshold if args.vote_match == "prefix_semantic" else None,
            "min_consensus_horizon": args.min_consensus_horizon,
            "single_token_filter": args.single_token_filter,
            "single_token_min_p": args.single_token_min_p if args.single_token_filter else None,
            "future_source_window_mode": args.future_source_window_mode,
            "future_source_window_chunks": args.future_source_window_chunks,
            "future_source_anchor_max_words": args.future_source_anchor_max_words,
            "future_join_mode": args.future_join_mode,
            "targeted_sampler_context": args.targeted_sampler_context,
            "targeted_prompt_version": args.targeted_prompt_version,
            "probe_prompt_order": args.probe_prompt_order,
            "contrastive_notes": args.contrastive_notes,
            "targeted_sampler_seed": args.targeted_sampler_seed,
            "targeted_fail_on_api_error": args.targeted_fail_on_api_error,
            "sentence_end_completion": args.sentence_end_completion,
            "sentence_end_boundary_mode": args.sentence_end_boundary_mode,
            "sentence_end_punctuation": args.sentence_end_punctuation,
            "oracle_future_audit": bool(getattr(args, "oracle_future_audit", False)),
            "oracle_future_vote": bool(getattr(args, "oracle_future_vote", False)),
            "probe_temperature": args.probe_temperature,
            "probe_top_p": args.probe_top_p,
            "probe_top_k": args.probe_top_k,
            "probe_thinking": args.probe_thinking,
            "probe_thinking_max_tokens": args.probe_thinking_max_tokens,
            "completion_thinking": args.completion_thinking,
            "thinking_sampling": THINKING_SAMPLING,
            "block_latin_tokens": args.block_latin_tokens,
        },
        "boundary_audit": boundary_audit,
        "sampling_audit": sampling_audit,
    }

    reference_text = _extract_reference_text_from_row(row, target_lang=args.target_lang)
    laal_value = float("nan")
    bleu_char_value = float("nan")
    length_ratio_ref = float("nan")
    try:
        if not reference_text:
            raise ValueError("reference_text_unavailable")
        laal_value = compute_laal(chunks, target_deltas, actions, reference_text)
        bleu_char_value = compute_bleu_char(committed_text, reference_text)
        length_ratio_ref = compute_length_ratio_ref(committed_text, reference_text)
    except Exception:
        pass
    length_ratio_src = compute_length_ratio_src(committed_text, full_source_text)

    result["reference_text"] = reference_text or ""
    result["metrics"] = {
        "laal_text": laal_value,
        "bleu_char": bleu_char_value,
        "length_ratio_ref": length_ratio_ref,
        "length_ratio_src": length_ratio_src,
        "pred_chars": _nonspace_char_count(committed_text),
        "ref_chars": _nonspace_char_count(reference_text or ""),
        "src_words": len(str(full_source_text or "").split()),
    }
    _vlog(
        verbose_log_file,
        f"  prediction={committed_text!r} bleu={bleu_char_value:.2f} laal={laal_value:.2f} "
        f"len_ratio_ref={length_ratio_ref:.2f} len_ratio_src={length_ratio_src:.2f}",
    )
    return result


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------

def select_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.utt_id is not None:
        selected = df[df[args.id_column].astype(str) == str(args.utt_id)]
        if selected.empty:
            raise ValueError(f"utt_id not found: {args.utt_id}")
        return selected.iloc[:1] if args.test_one else selected
    if args.test_one:
        return df.iloc[[args.row_idx]]
    start = max(0, int(args.row_idx))
    end = min(len(df), start + max(1, int(args.max_rows)))
    return df.iloc[start:end]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global BLOCK_LATIN_TOKENS
    setup_env()
    args = parse_args()
    BLOCK_LATIN_TOKENS = args.block_latin_tokens
    # one LaBSE matcher shared by every concurrent row (2026-10-09 vote test)
    if args.vote_match == "prefix_semantic":
        args.semantic_matcher = make_semantic_matcher(args.semantic_model, args.semantic_threshold)
    else:
        args.semantic_matcher = None

    if args.use_targeted_instruct_sampling:
        # Targeted instruct sampling bypasses base-LM future sampling entirely.
        primary_num_futures = 0
    else:
        if not args.base_api_base or not args.base_api_model:
            raise ValueError(
                "--base-api-base and --base-api-model are required when "
                "--use-targeted-instruct-sampling is not set."
            )
        primary_num_futures = args.num_futures - args.secondary_num_futures # 两个base 各10个 future sampling
        if primary_num_futures <= 0:
            raise ValueError("primary base model must keep at least 1 future")

    df = pd.read_csv(args.input_tsv, sep="\t")
    rows = select_rows(df, args)

    # Build base model specs (API only). Skipped entirely when targeted instruct
    # sampling is enabled — the instruct endpoint generates futures directly.
    base_specs: List[Dict[str, Any]] = []
    if not args.use_targeted_instruct_sampling:
        for name, api_base, api_model, api_timeout, model_path, num_futures in [
            ("primary", args.base_api_base, args.base_api_model,
             args.base_api_timeout, args.base_model_path, primary_num_futures),
            ("secondary", args.secondary_base_api_base, args.secondary_base_api_model,
             args.secondary_base_api_timeout, args.secondary_base_model_path, args.secondary_num_futures),
        ]:
            if not api_base or num_futures <= 0:
                continue
            models = verify_api(api_base, api_timeout)
            if api_model not in models:
                raise RuntimeError(f"{name} api_model '{api_model}' not found; available={models}")
            base_specs.append({
                "name": name, "path": model_path, "num_futures": num_futures,
                "api_base": api_base, "api_model": api_model, "api_timeout": api_timeout,
            })
            print(f"[Base] {name}: model={api_model} api={normalize_api_base(api_base)} futures={num_futures}")
    else:
        # If the sampler is routed to a separate endpoint, verify it now.
        if args.targeted_sampler_api_base:
            sampler_models = verify_api(
                args.targeted_sampler_api_base,
                args.targeted_sampler_api_timeout if args.targeted_sampler_api_timeout > 0
                else args.instruct_api_timeout,
            )
            sampler_model_name = args.targeted_sampler_api_model or args.instruct_api_model
            if sampler_model_name not in sampler_models:
                raise RuntimeError(
                    f"targeted_sampler_api_model '{sampler_model_name}' not found at "
                    f"{args.targeted_sampler_api_base}; available={sampler_models}"
                )
            print(f"[Sampler] targeted_instruct (cross-family): "
                  f"model={sampler_model_name} api={normalize_api_base(args.targeted_sampler_api_base)}")
        else:
            print(f"[Sampler] targeted_instruct (same-model as probe): "
                  f"model={args.instruct_api_model} api={normalize_api_base(args.instruct_api_base)}")
        print(f"[Sampler] mode=coordinated_list per_model={args.targeted_num_futures} "
              f"prompt_version={args.targeted_prompt_version} "
              f"temp={args.targeted_sample_temperature} top_p={args.targeted_top_p} "
              f"max_tokens={args.targeted_max_tokens}")

    # Verify instruct API
    models = verify_api(args.instruct_api_base, args.instruct_api_timeout)
    if args.instruct_api_model not in models:
        raise RuntimeError(f"instruct_api_model '{args.instruct_api_model}' not found; available={models}")
    instruct_tokenizer = load_tokenizer(args.instruct_tokenizer_path)
    print(f"[Instruct] model={args.instruct_api_model} api={normalize_api_base(args.instruct_api_base)}")

    # Sampler tokenizer for the coordinated-list chat-template path. Defaults to the
    # probe's tokenizer (same-model setup); set --targeted-sampler-tokenizer-path
    # for cross-family (e.g. gemma sampler + qwen probe).
    sampler_tokenizer = None
    sampler2_tokenizer = None
    if args.use_targeted_instruct_sampling:
        sampler_tok_path = args.targeted_sampler_tokenizer_path or args.instruct_tokenizer_path
        sampler_tokenizer = load_tokenizer(sampler_tok_path)
        print(f"[Sampler tokenizer] {sampler_tok_path}")
        if (args.targeted_sampler2_tokenizer_path
                and args.targeted_sampler2_api_base
                and args.targeted_sampler2_api_model):
            sampler2_tokenizer = load_tokenizer(args.targeted_sampler2_tokenizer_path)
            print(f"[Sampler2 tokenizer] {args.targeted_sampler2_tokenizer_path}")
            print(f"[Sampler2] model={args.targeted_sampler2_api_model} "
                  f"api={normalize_api_base(args.targeted_sampler2_api_base)}")

    output_dir: Optional[str] = None
    if args.output_jsonl:
        output_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
        os.makedirs(output_dir, exist_ok=True)
    if args.verbose and args.verbose_dir:
        os.makedirs(args.verbose_dir, exist_ok=True)
    global MALFORMED_RESPONSE_LOG_PATH
    if args.malformed_response_log:
        MALFORMED_RESPONSE_LOG_PATH = args.malformed_response_log
    elif args.verbose_dir:
        MALFORMED_RESPONSE_LOG_PATH = os.path.join(args.verbose_dir, "malformed_sampler_responses.jsonl")
    if MALFORMED_RESPONSE_LOG_PATH:
        os.makedirs(os.path.dirname(MALFORMED_RESPONSE_LOG_PATH) or ".", exist_ok=True)

    existing_output_names: set[str] = set()
    if args.skip_existing and args.skip_existing_root:
        pattern = os.path.join(
            os.path.abspath(args.skip_existing_root), "task_*", "per_utt", "*.json"
        )
        existing_output_names = {os.path.basename(path) for path in glob.glob(pattern)}
        print(
            f"[Resume index] {len(existing_output_names)} existing utterances under "
            f"{args.skip_existing_root}"
        )

    def _process_one_row(row_idx: int, series_dict: Dict[str, Any]) -> Dict[str, Any]:
        row = series_dict
        utt_id = str(row.get(args.id_column, row.get("id", f"row_{row_idx}")))
        out_path = (
            os.path.join(output_dir, f"{sanitize_filename(utt_id)}.json")
            if output_dir is not None else None
        )

        output_name = os.path.basename(out_path) if out_path is not None else ""
        if (
            args.skip_existing
            and out_path is not None
            and (os.path.exists(out_path) or output_name in existing_output_names)
        ):
            print(f"[SKIP existing] {output_name}")
            return {"utt_id": utt_id, "skipped_existing": True}

        verbose_log_file: Optional[Any] = None
        if args.verbose:
            if args.verbose_dir:
                vpath = os.path.join(args.verbose_dir, f"verbose_{sanitize_filename(utt_id)}.log")
                raw_file = open(vpath, "w", encoding="utf-8")
                verbose_log_file = _TeeWriter(raw_file) if args.test_one else raw_file
            elif args.test_one:
                verbose_log_file = sys.stdout

        try:
            result = run_one_utterance(
                row=row, args=args, base_specs=base_specs,
                instruct_tokenizer=instruct_tokenizer,
                sampler_tokenizer=sampler_tokenizer,
                sampler2_tokenizer=sampler2_tokenizer,
                verbose_log_file=verbose_log_file,
            )
        finally:
            if verbose_log_file is not None and args.verbose_dir:
                verbose_log_file.close()

        if output_dir is not None:
            write_pretty_json(out_path or os.path.join(output_dir, f"{sanitize_filename(result['utt_id'])}.json"), result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    num_concurrent = max(1, args.num_concurrent_cases)
    row_items = [(idx, s.to_dict()) for idx, (_, s) in enumerate(rows.iterrows())]
    failures: List[str] = []
    drops_path = args.dropped_rows_log or (
        os.path.join(args.verbose_dir, "dropped_rows.jsonl") if args.verbose_dir else None)

    def record_failure(row_idx: int, row_dict: Dict[str, Any], exc: BaseException) -> None:
        """One unusable row must not cost the other rows in this task."""
        utt = str(row_dict.get(args.id_column, row_dict.get("id", f"row_{row_idx}")))
        print(f"[ERROR] Row {row_idx} ({utt}) raised: {exc}", file=sys.stderr, flush=True)
        failures.append(f"Row {row_idx} ({utt}): {exc}")
        if drops_path:
            os.makedirs(os.path.dirname(drops_path) or ".", exist_ok=True)
            with open(drops_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "row_idx": row_idx, "utt_id": utt, "reason": str(exc),
                    "error_type": type(exc).__name__,
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }, ensure_ascii=False) + "\n")

    if num_concurrent <= 1:
        for row_idx, row_dict in row_items:
            try:
                _process_one_row(row_idx, row_dict)
            except Exception as exc:
                record_failure(row_idx, row_dict, exc)
    else:
        print(f"[Concurrent] {len(row_items)} rows, {num_concurrent} workers")
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futs = {executor.submit(_process_one_row, ri, rd): (ri, rd) for ri, rd in row_items}
            for fut in as_completed(futs):
                row_idx, row_dict = futs[fut]
                try:
                    fut.result()
                except Exception as exc:
                    record_failure(row_idx, row_dict, exc)

    if failures:
        allowed = max(0, int(args.max_dropped_rows))
        where = f" (recorded in {drops_path})" if drops_path else ""
        if len(failures) > allowed:
            print(f"[FATAL] {len(failures)} row(s) failed, more than --max-dropped-rows={allowed}{where}",
                  file=sys.stderr)
            for item in failures[:20]:
                print(f"  {item}", file=sys.stderr)
            raise SystemExit(1)
        print(f"[DROPPED] {len(failures)} row(s) dropped, within --max-dropped-rows={allowed}{where}",
              file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)


if __name__ == "__main__":
    main()
