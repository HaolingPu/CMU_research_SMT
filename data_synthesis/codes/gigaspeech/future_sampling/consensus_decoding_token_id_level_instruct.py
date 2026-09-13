#!/usr/bin/env python3
"""Consensus decoding with future sampling via vLLM serve APIs.

For each source chunk, two sampler models each return one JSON set of plausible and
contrastive continuations of the observed English sentence. The translator model is probed
for the next Chinese token under every hypothesized full source (the observed source plus one
future; the true source continuation votes as one more future), and a token is committed only
when enough futures agree on it. A chunk that ends an English sentence, and the last chunk,
are completed by the translator directly.
"""
from __future__ import annotations

import argparse
import ast
import http.client
import json
import math
import os
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TextIO

import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ambiguity_sampler_prompt import (
    MAX_SUFFIX_CHARS, PROMPT_VERSION, build_future_set_messages, future_set_response_format, parse_future_set,
)
from sentence_boundary_helpers import (
    close_translation_delta, observed_sentence_is_complete, sentence_anchored_prefix, source_terminal_mark,
)


REQUEST_TIMEOUT_S = 120.0
# Each future votes for its VOTE_TOP_K most likely next tokens that have probability >= VOTE_MIN_PROB.
VOTE_TOP_K = 20
VOTE_MIN_PROB = 0.1
MAX_CONSENSUS_STEPS = 12
SENTENCE_ANCHOR_MAX_WORDS = 128
MALFORMED_LOG_LOCK = threading.Lock()


@dataclass
class ModelEndpoint:
    """A vLLM server: its OpenAI-compatible base URL (ending in /v1), served model name and tokenizer."""
    api_base: str
    model: str
    tokenizer: PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Source text
# ---------------------------------------------------------------------------

def append_text_continuation(prefix: str, continuation: str) -> str:
    """Join two English pieces with one space unless whitespace or closing punctuation already separates them."""
    if not prefix:
        return continuation
    elif (not continuation or prefix[-1].isspace() or continuation[0].isspace()
          or continuation[0] in ",.!?;:)]}\"'"):
        return prefix + continuation
    else:
        return prefix + " " + continuation


def join_source_chunks(chunks: list[str]) -> str:
    """The observed source: chunks joined like append_text_continuation, never spaced after an opening bracket or quote."""
    text = ""
    for chunk in chunks:
        if text and text[-1] in "([{\"'":
            text += chunk
        else:
            text = append_text_continuation(text, chunk)
    return text.strip()


def oracle_future_suffix(source_units: list[str], observed_source: str) -> str:
    """The true continuation of the source after the observed prefix, on a sampled future's budget.

    It walks through unit boundaries (about half the units of src_text_full end at a comma)
    and cuts at MAX_SUFFIX_CHARS, backing off to a word boundary, so it sees the same horizon
    as the futures it votes with. "" when the observed prefix already covers the source or
    the trajectory text does not line up with src_text_full.
    """
    full_source = ""
    for unit in source_units:
        full_source = append_text_continuation(full_source, unit)
    remainder = full_source[len(observed_source):].strip()
    cut = remainder[:MAX_SUFFIX_CHARS]
    if not full_source.startswith(observed_source):
        return ""
    elif len(remainder) <= MAX_SUFFIX_CHARS:
        return remainder
    elif cut.rfind(" ") > MAX_SUFFIX_CHARS // 2:
        return cut[:cut.rfind(" ")].rstrip()
    else:
        return cut.rstrip()


# ---------------------------------------------------------------------------
# Sampled futures
# ---------------------------------------------------------------------------

def clean_model_text(text: str) -> str:
    """Model output without <think> blocks and without anything after an end-of-turn marker."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()


def clean_future_text(observed_source: str, raw_text: str) -> str:
    """A sampled suffix without markdown, a restated observed prefix, leading ellipses or repeated trailing words."""
    def _strip_markdown(text: str) -> str:
        # Gemma sometimes wraps a continuation in bold or backticks; the markup must not reach the probe.
        text = re.sub(r"^\*{1,3}\s*(.*?)\s*\*{1,3}$", r"\1", text.strip())
        return re.sub(r"^`+\s*(.*?)\s*`+$", r"\1", text).strip()

    def _strip_prefix_casefold(text: str, prefix: str) -> str | None:
        """`text` after `prefix` matched case-insensitively at a word boundary; None when it does not start with it."""
        prefix = prefix.strip()
        if not prefix or len(text) < len(prefix) or text[:len(prefix)].casefold() != prefix.casefold():
            return None
        elif len(text) > len(prefix) and prefix[-1].isalnum() and text[len(prefix)].isalnum():
            return None
        else:
            return text[len(prefix):].lstrip()

    text = _strip_markdown(clean_model_text(raw_text))
    without_prefix = _strip_prefix_casefold(text, observed_source)
    text = text if without_prefix is None else without_prefix
    text = _strip_markdown(text.splitlines()[0]) if text else ""
    text = re.sub(r"^[\.\s…\-]+", "", text)
    # An instruct model sometimes restarts with the last words of the prefix: strip the longest such overlap.
    observed = observed_source.strip()
    overlaps = (_strip_prefix_casefold(text, observed[-size:])
                for size in range(min(len(observed), len(text)), 2, -1)
                if size == len(observed) or not observed[-size - 1].isalnum())
    return next((stripped for stripped in overlaps if stripped is not None), text)


def is_valid_future_text(text: str) -> bool:
    """False for an empty future, markdown, Chinese, an echo of the prompt, or more than 20 words."""
    prompt_echoes = ("translate", "translation", "grammar analysis", "analyze", "analysis", "<think>", "</think>",
                     "assistant turn", "user prompt", "the prompt", "sampling mode", "unfinished english prefix",
                     "4-15 words", "let's think")
    return (bool(text) and "*" not in text and "`" not in text
            and re.search(r"[㐀-䶿一-鿿豈-﫿]", text) is None
            and len(text.split()) <= 20
            and not any(echo in text.lower() for echo in prompt_echoes))


def select_diverse_futures(candidates: list[str]) -> list[str]:
    """Keep the first future per opening content word and drop near-duplicates (word-set Jaccard >= 0.65)."""
    opening_stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "from", "has", "have", "he", "her", "his", "in", "is", "it", "its",
        "of", "on", "or", "she", "that", "the", "their", "they", "this", "to",
        "was", "were", "which", "who", "with", "would",
    }
    selected, selected_word_sets, openings = [], [], set()
    for candidate in candidates:
        words = re.findall(r"[a-z0-9']+", re.sub(r"\s+", " ", candidate.strip().lower()))
        word_set = set(words)
        opening = next((word for word in words if word not in opening_stopwords), words[0] if words else "")
        if (not words or opening in openings
                or any(len(word_set & prior) / len(word_set | prior) >= 0.65 for prior in selected_word_sets)):
            continue
        else:
            selected.append(candidate)
            selected_word_sets.append(word_set)
            openings.add(opening)
    return selected


def http_json(url: str, payload: dict | None = None) -> dict:
    """GET (no payload) or POST JSON to a vLLM endpoint; a failed request raises RuntimeError naming the endpoint."""
    request = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {url}: {exc.read().decode('utf-8', errors='replace')}") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"Request to {url} failed: {type(exc).__name__}: {exc}") from exc


def sample_future_set(sampler: ModelEndpoint, observed_source: str, num_futures: int, seed: int | None,
                      fail_on_api_error: bool, malformed_log_path: str) -> tuple[dict[str, list[str]], list[dict]]:
    """One sampler's filtered futures by group, and an audit record for every suffix it returned.

    A malformed reply is requested again up to twice (with a shifted seed when seeded) and
    logged verbatim. An API error raises when fail_on_api_error is set; otherwise this
    sampler contributes no futures for the chunk.
    """
    prompt = sampler.tokenizer.apply_chat_template(
        build_future_set_messages(observed_source, num_futures),
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    payload = {
        "model": sampler.model, "prompt": prompt, "max_tokens": 32 * num_futures, "temperature": 1.0,
        "top_p": 0.98, "n": 1, "presence_penalty": 0.15,
        "stop": ["<|im_end|>", "<end_of_turn>", "<|endoftext|>", "<|eot_id|>"],
    }
    if seed is not None:
        # The empty middle field held the committed target, which the sampler no longer sees;
        # keeping it keeps seeds identical to earlier runs.
        payload["seed"] = (seed + zlib.crc32(f"{observed_source}||{sampler.model}".encode("utf-8"))) % (2**31)
    payload.update(future_set_response_format(num_futures))

    def _log_event(event: str, attempt: int, raw_text: str, finish_reason: str | None, error: str | None) -> None:
        print(f"[SamplerResponse] event={event} model={sampler.model} attempt={attempt + 1} "
              f"finish_reason={finish_reason} error={error}", file=sys.stderr, flush=True)
        record = {"event": event, "model": sampler.model, "prompt_version": PROMPT_VERSION, "attempt": attempt + 1,
                  "observed_source": observed_source, "finish_reason": finish_reason, "error": error,
                  "raw_response": raw_text, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with MALFORMED_LOG_LOCK, open(malformed_log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    for attempt in range(3):
        if attempt and seed is not None:
            attempt_payload = {**payload, "seed": (payload["seed"] + 7919 * attempt) % (2**31)}
        else:
            attempt_payload = payload
        try:
            choice = http_json(f"{sampler.api_base}/completions", attempt_payload)["choices"][0]
        except (RuntimeError, ValueError) as exc:
            if fail_on_api_error:
                raise
            else:
                print(f"[SamplerUnavailable] model={sampler.model} attempt={attempt + 1} error={exc}",
                      file=sys.stderr, flush=True)
                return {"plausible": [], "contrastive": []}, []
        try:
            suffixes = parse_future_set(choice["text"], choice["finish_reason"], num_futures)
        except ValueError as exc:
            error = str(exc)
            _log_event("malformed", attempt, choice["text"], choice["finish_reason"], error)
        else:
            if attempt:
                _log_event("recovered", attempt, choice["text"], choice["finish_reason"], None)
            break
    else:
        _log_event("exhausted", attempt, choice["text"], choice["finish_reason"], error)
        raise ValueError(f"Invalid {PROMPT_VERSION} response from {sampler.model}: "
                         f"malformed grouped response after {attempt + 1} attempt(s): {error}")

    observed_letters = re.sub(r"[^a-z0-9]+", "", observed_source.lower())
    candidates = {"plausible": [], "contrastive": []}
    audit = []
    for choice_index, (group, suffix) in enumerate(suffixes):
        future = clean_future_text(observed_source, suffix)
        if not is_valid_future_text(future):
            reason = "invalid_or_meta"
        elif len(re.findall(r"[A-Za-z0-9']+", future)) < 3:
            # A one- or two-word fragment rarely tests whether a translation commitment is safe.
            reason = "too_short"
        elif len(observed_letters) >= 8 and observed_letters in re.sub(r"[^a-z0-9]+", "", future.lower()):
            # The sampler restated the observed prefix (possibly recapitalized) before continuing it.
            reason = "repeats_observed_prefix"
        else:
            reason = "pending_diversity_filter"
            candidates[group].append(future)
        audit.append({"choice": choice_index, "model": sampler.model, "mode": group, "raw": suffix,
                      "future": future, "accepted": False, "reason": reason})

    futures_by_group = {group: select_diverse_futures(texts) for group, texts in candidates.items()}
    unclaimed = {group: Counter(re.sub(r"\s+", " ", text.strip().lower()) for text in texts)
                 for group, texts in futures_by_group.items()}
    for item in audit:
        key = re.sub(r"\s+", " ", item["future"].strip().lower())
        if item["reason"] != "pending_diversity_filter":
            continue
        elif unclaimed[item["mode"]][key] > 0:
            unclaimed[item["mode"]][key] -= 1
            item["accepted"], item["reason"] = True, "accepted"
        else:
            item["reason"] = "duplicate_or_repetitive"
    return futures_by_group, audit


# ---------------------------------------------------------------------------
# Translator probe and consensus
# ---------------------------------------------------------------------------

def decode_tokens(tokenizer: PreTrainedTokenizerBase, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def is_disallowed_token(tokenizer: PreTrainedTokenizerBase, token_id: int) -> bool:
    """Special, markdown, Latin-letter, zero-width/variation-selector and control tokens are never committed.

    U+FFFD is allowed: a byte-level token of a rare character decodes to it alone and becomes
    a valid character together with the token that follows.
    """
    text = decode_tokens(tokenizer, [token_id])
    return (token_id in tokenizer.all_special_ids
            or "*" in text or "`" in text or "‍" in text or "️" in text
            or re.search(r"[A-Za-z]", text) is not None
            or any(unicodedata.category(char) in ("Cc", "Cs") for char in text))


def translation_probe_prompt_ids(tokenizer: PreTrainedTokenizerBase, full_source: str, has_target_prefix: bool) -> list[int]:
    if has_target_prefix:
        instruction = ("A partial Chinese translation is already committed at the start of the assistant reply. "
                       "You must continue from that exact prefix and produce only the continuation.")
    else:
        instruction = "Start the Chinese translation from the beginning and output only the next continuation token(s)."
    content = f"[TASK]\nTranslate the [INPUT] text into Chinese.\n\n[INPUT]\n{full_source}\n\n[IMPORTANT]\n{instruction}"
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                         tokenize=True, return_dict=False, enable_thinking=False)


def probe_next_token_distributions(translator: ModelEndpoint, full_sources: list[str], target_prefix_ids: list[int],
                                   args: argparse.Namespace) -> list[dict[int, float]]:
    """The translator's top-VOTE_TOP_K next-token probabilities for each full source, disallowed tokens removed."""
    prompts = [translation_probe_prompt_ids(translator.tokenizer, source, bool(target_prefix_ids)) + target_prefix_ids
               for source in full_sources]
    payload = {
        "model": translator.model, "prompt": prompts, "max_tokens": 1, "temperature": args.probe_temperature,
        "logprobs": VOTE_TOP_K, "return_tokens_as_token_ids": True, "return_token_ids": True,
        "top_p": args.probe_top_p, "top_k": args.probe_top_k,
    }
    distributions = []
    for choice in http_json(f"{translator.api_base}/completions", payload)["choices"]:
        # One generated token, so top_logprobs holds one {"token_id:<id>": logprob} map.
        token_logprobs = {int(key.removeprefix("token_id:")): logprob
                          for key, logprob in choice["logprobs"]["top_logprobs"][0].items()}
        distributions.append({token_id: math.exp(logprob) for token_id, logprob in token_logprobs.items()
                              if not is_disallowed_token(translator.tokenizer, token_id)})
    return distributions


def choose_consensus_token(distributions: list[dict[int, float]], min_voters_ratio: float) -> tuple[int | None, list[int]]:
    """The committed token, or None, and the ranked candidate tokens shown in the log.

    A token is eligible when at least ceil(num_futures * min_voters_ratio) futures vote for
    it; the eligible token with the highest summed probability wins and must be strictly ahead.
    """
    voters, summed_prob = Counter(), Counter()
    for distribution in distributions:
        votes = sorted(((token_id, prob) for token_id, prob in distribution.items() if prob >= VOTE_MIN_PROB),
                       key=lambda item: item[1], reverse=True)[:VOTE_TOP_K]
        for token_id, prob in votes:
            voters[token_id] += 1
            summed_prob[token_id] += prob
    min_voters = max(1, math.ceil(len(distributions) * min_voters_ratio))
    ranked = sorted(((token_id, summed_prob[token_id]) for token_id, count in voters.items() if count >= min_voters),
                    key=lambda item: item[1], reverse=True)
    if not ranked:
        return None, [token_id for token_id, _ in voters.most_common(VOTE_TOP_K)]
    elif len(ranked) > 1 and ranked[0][1] <= ranked[1][1]:
        return None, [token_id for token_id, _ in ranked[:VOTE_TOP_K]]
    else:
        return ranked[0][0], [token_id for token_id, _ in ranked[:VOTE_TOP_K]]


def extend_pending_tokens(translator: ModelEndpoint, source_observed: str, voter_futures: list[str],
                          committed_token_ids: list[int], args: argparse.Namespace) -> tuple[list[int], list[dict]]:
    """Grow a pending token buffer while the futures agree on the next token; one record per probe step."""
    full_sources = [append_text_continuation(source_observed, future) for future in voter_futures]
    pending_token_ids: list[int] = []
    steps: list[dict] = []
    for step in range(MAX_CONSENSUS_STEPS):
        started = time.perf_counter()
        distributions = probe_next_token_distributions(translator, full_sources, committed_token_ids + pending_token_ids, args)
        token_id, ranked = choose_consensus_token(distributions, args.min_voters_ratio)
        elapsed_s = round(time.perf_counter() - started, 3)
        if not all(distributions):
            # Every top token of some future was disallowed.
            steps.append({"step": step, "stop": "empty_distribution", "intersection": [], "distributions": [],
                          "elapsed_s": elapsed_s})
            break
        elif token_id is None:
            steps.append({"step": step, "stop": "no_consensus_token", "intersection": ranked,
                          "distributions": distributions, "elapsed_s": elapsed_s})
            break
        else:
            pending_token_ids.append(token_id)
            steps.append({"step": step, "accepted": decode_tokens(translator.tokenizer, [token_id]),
                          "pending": decode_tokens(translator.tokenizer, pending_token_ids),
                          "distributions": distributions, "elapsed_s": elapsed_s})
    return pending_token_ids, steps


def committable_prefix(tokenizer: PreTrainedTokenizerBase, token_ids: list[int]) -> list[int]:
    """The longest prefix of the token buffer that decodes to clean text.

    Committing token ids (never re-encoded text) keeps the translator state exact; a trailing
    byte token whose partner was not agreed on decodes to U+FFFD and is left out.
    """
    end = 0
    for size in range(1, len(token_ids) + 1):
        text = decode_tokens(tokenizer, token_ids[:size])
        if not text or ("�" not in text and text[-1] not in "‍️"
                        and unicodedata.category(text[-1]) not in ("Mn", "Mc", "Me", "Cc", "Cs")):
            end = size
    return token_ids[:end]


def force_complete_translation(translator: ModelEndpoint, full_source: str, committed_text: str,
                               terminal_mark: str, max_tokens: int) -> str:
    """The translator's greedy translation of the rest of the observed source after the committed Chinese."""
    if committed_text.strip():
        instruction = ("A partial Chinese translation is already committed at the start of the assistant reply.\n"
                       "Continue ONLY with the source content that is not yet covered by the committed Chinese prefix.\n"
                       "Do not repeat the committed prefix.\n"
                       "Do not add explanations, summaries, examples, lists, numbering, markdown, or background knowledge.\n"
                       "Do not translate beyond the source.\n"
                       "If there is no remaining source content to translate, output nothing.\n"
                       "Output only the remaining Chinese continuation.")
        prefill = committed_text
    else:
        instruction = ("Output the complete Chinese translation only.\n"
                       "Do not add explanations, summaries, examples, lists, numbering, markdown, or background knowledge.")
        prefill = ""
    if terminal_mark:
        sentence_end_rule = (f"\nThe observed English sentence is finished. End its Chinese translation with {terminal_mark} "
                             "before any closing quotation mark. Do not invent a following sentence or any unobserved content. "
                             "If only the terminal punctuation is missing, output that punctuation only.")
    else:
        sentence_end_rule = ""
    content = f"[TASK]\nTranslate the [INPUT] text into Chinese.\n\n[INPUT]\n{full_source}\n\n[IMPORTANT]\n{instruction}{sentence_end_rule}"
    prompt = translator.tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                      add_generation_prompt=True, tokenize=False, enable_thinking=False)
    payload = {"model": translator.model, "prompt": prompt + prefill, "max_tokens": max_tokens, "temperature": 0.0,
               "stop": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]}
    choice = http_json(f"{translator.api_base}/completions", payload)["choices"][0]
    if terminal_mark and choice["finish_reason"] == "length":
        raise RuntimeError("Boundary completion missing or token-truncated; refusing to fabricate a closed sentence")
    else:
        return clean_model_text(choice["text"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def count_nonspace_chars(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def compute_laal(chunks: list[str], target_deltas: list[str], actions: list[str], reference: str) -> float:
    """Length-adaptive average lagging, in source words, over the characters of the committed Chinese."""
    words_read_per_char: list[int] = []
    words_read = 0
    for chunk, delta, action in zip(chunks, target_deltas, actions):
        words_read += len(chunk.split())
        if action == "WRITE":
            words_read_per_char.extend([words_read] * len(delta.strip()))
    target_chars = len("".join(target_deltas))
    reference_chars = len(reference.replace(" ", ""))
    source_words = sum(len(chunk.split()) for chunk in chunks)
    denominator = max(target_chars, reference_chars)
    if not (target_chars and reference_chars and source_words and words_read_per_char):
        return float("nan")
    else:
        # A plain += loop, not sum(): Python 3.12+ sum() compensates float error and would shift the value.
        lagging = 0.0
        for i in range(denominator):
            delay = words_read_per_char[i] if i < len(words_read_per_char) else source_words
            lagging += delay - i * source_words / denominator
        return lagging / denominator


def compute_bleu_char(hypothesis: str, reference: str) -> float:
    """Sentence-level character BLEU up to 4-grams with add-one smoothing, whitespace ignored."""
    hyp = [char for char in hypothesis if not char.isspace()]
    ref = [char for char in reference if not char.isspace()]
    order = min(4, len(hyp), len(ref))
    if order == 0:
        return float("nan")
    else:
        precisions = []
        for n in range(1, order + 1):
            hyp_ngrams = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
            ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
            matches = sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())
            precisions.append((matches + 1.0) / (sum(hyp_ngrams.values()) + 1.0))
        brevity_penalty = 1.0 if len(hyp) > len(ref) else math.exp(1.0 - len(ref) / len(hyp))
        return brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / order) * 100.0


# ---------------------------------------------------------------------------
# Decode one utterance
# ---------------------------------------------------------------------------

def run_one_utterance(row: dict, args: argparse.Namespace, translator: ModelEndpoint, samplers: list[ModelEndpoint],
                      malformed_log_path: str, log: TextIO) -> dict:
    """Decode one TSV row chunk by chunk; returns the per-utterance record and writes the verbose trace to `log`."""
    utt_id = row["id"]
    chunks = ast.literal_eval(row["src_trajectory"])
    source_units = ast.literal_eval(row["src_text_full"])
    full_source_text = row["src_text"].strip()
    reference_text = row["llm_reference_text"].strip()
    sampler_names = {"gemma4-sampler": "Gemma 4", "qwen38-sampler": "Qwen 3.8"}

    print("#" * 60, file=log)
    print(f"# utt_id: {utt_id}", file=log)
    print(f"# source_full_text: {full_source_text}", file=log)
    for label, value in (("src_text_full", source_units), ("src_trajectory", chunks)):
        first_line, *other_lines = json.dumps(value, ensure_ascii=False, indent=2).splitlines()
        print(f"# {label}: {first_line}", file=log)
        for line in other_lines:
            print(f"#   {line}", file=log)
    print(f"# Chunks: {len(chunks)}", file=log)
    print("#" * 60, file=log)

    committed_text = ""
    committed_token_ids: list[int] = []
    target_deltas: list[str] = []
    actions: list[str] = []
    boundary_audit: list[dict] = []
    sampling_audit: list[dict] = []
    for t, chunk in enumerate(chunks):
        source_observed_full = join_source_chunks(chunks[:t + 1])
        source_observed = sentence_anchored_prefix(source_observed_full, SENTENCE_ANCHOR_MAX_WORDS)
        is_last_chunk = t == len(chunks) - 1
        print(f"\n{'=' * 60}", file=log)
        print(f"Chunk {t + 1}/{len(chunks)}", file=log)
        print(f"source_observed: {chunk!r}", file=log)
        print(f"future_source_prefix: {source_observed!r}", file=log)
        if source_observed != source_observed_full:
            print(f"source_observed_full: {source_observed_full!r}", file=log)
        print(f"committed_before: {committed_text!r}", file=log)

        def _record_sampling(status: str, reason: str) -> None:
            event = {"chunk": t + 1, "status": status, "reason": reason, "prepared_source_prefix": source_observed,
                     "input_source_prefix": source_observed if status == "invoked" else None,
                     "context_mode": "source-only", "prompt_version": PROMPT_VERSION, "window_mode": "sentence-anchor",
                     "committed_target_context": "", "evidence": "decoder_dispatch"}
            sampling_audit.append(event)
            print(f"[FutureSampling] {json.dumps(event, ensure_ascii=False)}", file=log)
            log.flush()  # the sampler input is on disk before the slow model calls

        if is_last_chunk or (chunk.strip() and observed_sentence_is_complete(source_observed_full)):
            # The English sentence (or the whole source) is complete: the translator finishes it instead of voting.
            _record_sampling("skipped", "final_chunk" if is_last_chunk else "source_sentence_end")
            terminal = source_terminal_mark(source_observed_full)
            started = time.perf_counter()
            raw_delta = force_complete_translation(translator, source_observed_full, committed_text, terminal,
                                                   args.final_max_tokens)
            if terminal:
                delta, reason = close_translation_delta(raw_delta, committed_text, terminal)
                closure = {"chunk": t + 1, "terminal": terminal, "raw_delta": raw_delta, "delta": delta,
                           "reason": reason, "source": source_observed_full}
                boundary_audit.append(closure)
                print(f"[BoundaryClose] {json.dumps(closure, ensure_ascii=False)}", file=log)
            else:
                delta = raw_delta
            if is_last_chunk:
                print(f"  [Final] delta={delta!r}", file=log)
            else:
                print(f"  [SentenceEnd] delta={delta!r}", file=log)
                print(f"[Timing] completion={time.perf_counter() - started:.2f}s", file=log)
                print(f"-> {'WRITE' if delta else 'READ'} delta={delta!r}", file=log)
                print(f"committed_after: {committed_text + delta!r}", file=log)
            committed_text += delta
            committed_token_ids.extend(translator.tokenizer.encode(delta, add_special_tokens=False))
            target_deltas.append(delta)
            actions.append("WRITE" if delta else "READ")
        elif not chunk.strip():
            _record_sampling("skipped", "no_new_source")
            target_deltas.append("")
            actions.append("READ")
            print("[NoNewSource] -> READ; no future sampling", file=log)
        else:
            _record_sampling("invoked", "future_sampling")
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=len(samplers)) as pool:
                sampled = list(pool.map(
                    lambda sampler: sample_future_set(sampler, source_observed, args.targeted_num_futures,
                                                      args.targeted_sampler_seed, args.targeted_fail_on_api_error,
                                                      malformed_log_path),
                    samplers,
                ))
            print(f"[Timing] sampling={time.perf_counter() - started:.2f}s", file=log)
            audit = [item for _, sampler_audit in sampled for item in sampler_audit]
            if audit:
                print(f"[Step 1-1] raw_future_sampling total={len(audit)} "
                      f"accepted={sum(item['accepted'] for item in audit)}", file=log)
            for sampler, (_, sampler_audit) in zip(samplers, sampled):
                for group in ("plausible", "contrastive"):
                    items = [item for item in sampler_audit if item["mode"] == group]
                    if items:
                        dropped = Counter(item["reason"] for item in items if not item["accepted"])
                        print(f"[Raw candidates] {sampler_names.get(sampler.model, sampler.model)} | {group} | "
                              f"model={sampler.model} mode={group} count={len(items)}", file=log)
                        for index, item in enumerate(items, 1):
                            print(f"  {index:02d}. {(item['future'] or item['raw'])!r}", file=log)
                        print(f"  Filter summary: kept={sum(item['accepted'] for item in items)}/{len(items)}; dropped: "
                              f"{', '.join(f'{reason}={count}' for reason, count in sorted(dropped.items())) or 'none'}",
                              file=log)
            futures = [future for futures_by_group, _ in sampled for group in ("plausible", "contrastive")
                       for future in futures_by_group[group]]
            print(f"[Step 1-2] future_sampling total={len(futures)}", file=log)
            for sampler, (futures_by_group, _) in zip(samplers, sampled):
                for group in ("plausible", "contrastive"):
                    if futures_by_group[group]:
                        print(f"[Selected candidates] {sampler_names.get(sampler.model, sampler.model)} | {group} | "
                              f"model={sampler.model} mode={group} count={len(futures_by_group[group])}", file=log)
                        for index, future in enumerate(futures_by_group[group], 1):
                            print(f"  {index:02d}. {future!r}", file=log)
            oracle_future = oracle_future_suffix(source_units, source_observed_full) if args.oracle_future_vote else ""
            if oracle_future:
                print("[Selected candidates] [ORACLE] source continuation | oracle | model=oracle mode=oracle count=1",
                      file=log)
                print(f"  01. {oracle_future!r}", file=log)

            if len(futures) <= 3:
                # Too few futures to test a commitment: wait for more source.
                target_deltas.append("")
                actions.append("READ")
                print("  -> READ (too few futures)", file=log)
            else:
                voter_futures = futures + [oracle_future] if oracle_future else futures
                pending_token_ids, steps = extend_pending_tokens(translator, source_observed_full, voter_futures,
                                                                 committed_token_ids, args)
                print(f"[Timing] probe_batches={len(steps)} total={sum(step['elapsed_s'] for step in steps):.2f}s",
                      file=log)
                if 0 < len(pending_token_ids) < args.min_consensus_horizon:
                    # A too-short agreement is an irreversible commit on thin evidence: READ and wait for more source.
                    print(f"[Step 5.5] horizon_filter: dropped pending="
                          f"{decode_tokens(translator.tokenizer, pending_token_ids)!r} (len={len(pending_token_ids)} "
                          f"< min_horizon={args.min_consensus_horizon}) -> READ", file=log)
                    steps.append({"step": "filter", "stop": "below_min_horizon", "intersection": [], "distributions": []})
                    kept_token_ids = []
                else:
                    kept_token_ids = pending_token_ids
                kept_text = decode_tokens(translator.tokenizer, kept_token_ids)
                print("[Step 4-5] consensus summary:", file=log)
                for record in steps:
                    if "stop" in record:
                        candidates = [decode_tokens(translator.tokenizer, [token_id]) for token_id in record["intersection"]]
                        print(f"  step={record['step']} stop={record['stop']} intersection={candidates} "
                              f"pending={kept_text!r}", file=log)
                    else:
                        print(f"  step={record['step']} accepted={record['accepted']!r} pending={record['pending']!r}",
                              file=log)
                    for index, distribution in enumerate([] if args.compact_verbose else record["distributions"]):
                        top = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:VOTE_TOP_K]
                        pairs = ", ".join(f"{decode_tokens(translator.tokenizer, [token_id])!r}:{prob:.3f}"
                                          for token_id, prob in top)
                        print(f"    future[{index}] candidates={len(top)}: [{pairs}]", file=log)
                delta_token_ids = committable_prefix(translator.tokenizer, kept_token_ids)
                delta = decode_tokens(translator.tokenizer, delta_token_ids)
                print(f"[Step 6-7] pending_before_trim={kept_text!r}", file=log)
                print(f"[Step 6-7] commit_after_trim={delta!r}", file=log)
                print(f"-> {'WRITE' if delta else 'READ'} delta={delta!r}", file=log)
                print(f"committed_after: {committed_text + delta!r}", file=log)
                committed_text += delta
                committed_token_ids.extend(delta_token_ids)
                target_deltas.append(delta)
                actions.append("WRITE" if delta else "READ")

    pred_chars = count_nonspace_chars(committed_text)
    ref_chars = count_nonspace_chars(reference_text)
    src_words = len(full_source_text.split())
    metrics = {
        "laal_text": compute_laal(chunks, target_deltas, actions, reference_text),
        "bleu_char": compute_bleu_char(committed_text, reference_text),
        "length_ratio_ref": pred_chars / ref_chars if pred_chars and ref_chars else float("nan"),
        "length_ratio_src": pred_chars / src_words if pred_chars and src_words else float("nan"),
        "pred_chars": pred_chars,
        "ref_chars": ref_chars,
        "src_words": src_words,
    }
    print(f"  prediction={committed_text!r} bleu={metrics['bleu_char']:.2f} laal={metrics['laal_text']:.2f} "
          f"len_ratio_ref={metrics['length_ratio_ref']:.2f} len_ratio_src={metrics['length_ratio_src']:.2f}", file=log)
    return {
        "utt_id": utt_id,
        "source_full_text": full_source_text,
        "src_text_full": source_units,
        "src_trajectory": chunks,
        "target_trajectory": target_deltas,
        "actions": actions,
        "prediction": committed_text,
        "decoder_settings": {
            # Modes fixed in this implementation are recorded under their historical keys, so these
            # records stay comparable with earlier decodes in the review bundle and pilot_case_guard.
            "future_source_window_mode": "sentence-anchor",
            "future_source_window_chunks": 1,
            "future_source_anchor_max_words": SENTENCE_ANCHOR_MAX_WORDS,
            "future_join_mode": "space",
            "targeted_sampler_context": "source-only",
            "targeted_prompt_version": PROMPT_VERSION,
            "probe_prompt_order": "historical",
            "contrastive_notes": False,
            "targeted_sampler_seed": args.targeted_sampler_seed,
            "targeted_fail_on_api_error": args.targeted_fail_on_api_error,
            "sentence_end_completion": True,
            "sentence_end_boundary_mode": "conservative",
            "sentence_end_punctuation": "match-source",
            "oracle_future_audit": False,
            "oracle_future_vote": args.oracle_future_vote,
            "probe_temperature": args.probe_temperature,
            "probe_top_p": args.probe_top_p,
            "probe_top_k": args.probe_top_k,
        },
        "boundary_audit": boundary_audit,
        "sampling_audit": sampling_audit,
        "reference_text": reference_text,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consensus decoding with future sampling via vLLM serve.")
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--row-idx", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=1)
    parser.add_argument("--num-concurrent-cases", type=int, default=8)
    parser.add_argument("--output-dir", required=True, help="Receives one <utt_id>.json per decoded row.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip rows whose <utt_id>.json already exists.")
    parser.add_argument("--max-dropped-rows", type=int, default=0,
                        help="A row that raises is recorded in <verbose-dir>/dropped_rows.jsonl and skipped; "
                             "exit 1 when more rows than this raise.")
    parser.add_argument("--verbose-dir", required=True,
                        help="Receives verbose_<utt_id>.log, malformed_sampler_responses.jsonl and dropped_rows.jsonl.")
    parser.add_argument("--compact-verbose", action="store_true",
                        help="Leave the per-future next-token probabilities out of the verbose logs.")
    parser.add_argument("--instruct-tokenizer-path", required=True, help="Translator/probe model.")
    parser.add_argument("--instruct-api-base", required=True)
    parser.add_argument("--instruct-api-model", required=True)
    parser.add_argument("--targeted-sampler-tokenizer-path", required=True, help="First future sampler.")
    parser.add_argument("--targeted-sampler-api-base", required=True)
    parser.add_argument("--targeted-sampler-api-model", required=True)
    parser.add_argument("--targeted-sampler2-tokenizer-path", required=True, help="Second future sampler.")
    parser.add_argument("--targeted-sampler2-api-base", required=True)
    parser.add_argument("--targeted-sampler2-api-model", required=True)
    parser.add_argument("--targeted-num-futures", type=int, default=20,
                        help="Most futures one sampler may return, half per group; must be even.")
    parser.add_argument("--targeted-sampler-seed", type=int, default=None,
                        help="Deterministic per-prefix sampler seed, so reruns of a row draw the same futures.")
    parser.add_argument("--targeted-fail-on-api-error", action="store_true",
                        help="Raise on a sampler API error instead of continuing without that sampler's futures.")
    parser.add_argument("--oracle-future-vote", action=argparse.BooleanOptionalAction, default=True,
                        help="The true source continuation votes as one more future "
                             "(with --min-voters-ratio 1.0 it can veto a token).")
    parser.add_argument("--min-voters-ratio", type=float, default=1.0,
                        help="A token needs votes from at least ceil(num_voters * ratio) futures.")
    parser.add_argument("--min-consensus-horizon", type=int, default=2,
                        help="Agreed tokens needed to commit; a shorter agreement is dropped and the chunk READs.")
    parser.add_argument("--probe-temperature", type=float, default=0.6)
    parser.add_argument("--probe-top-p", type=float, default=0.95)
    parser.add_argument("--probe-top-k", type=int, default=20)
    parser.add_argument("--final-max-tokens", type=int, default=128,
                        help="Token budget of the translator completion at a sentence end or the last chunk.")
    args = parser.parse_args()
    if args.targeted_num_futures <= 0 or args.targeted_num_futures % 2:
        parser.error("--targeted-num-futures must be a positive even number")
    else:
        return args


def main() -> None:
    args = parse_args()
    endpoints = []
    for api_base, model, tokenizer_path in (
        (args.instruct_api_base, args.instruct_api_model, args.instruct_tokenizer_path),
        (args.targeted_sampler_api_base, args.targeted_sampler_api_model, args.targeted_sampler_tokenizer_path),
        (args.targeted_sampler2_api_base, args.targeted_sampler2_api_model, args.targeted_sampler2_tokenizer_path),
    ):
        served = [item["id"] for item in http_json(f"{api_base}/models")["data"]]
        if model not in served:
            raise RuntimeError(f"{model} is not served at {api_base}; available={served}")
        else:
            endpoints.append(ModelEndpoint(api_base, model, AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)))
            print(f"[Endpoint] model={model} api={api_base} tokenizer={tokenizer_path}")
    translator, *samplers = endpoints

    rows = pd.read_csv(args.input_tsv, sep="\t").iloc[args.row_idx:args.row_idx + args.max_rows]
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.verbose_dir, exist_ok=True)
    malformed_log_path = os.path.join(args.verbose_dir, "malformed_sampler_responses.jsonl")
    dropped_rows_path = os.path.join(args.verbose_dir, "dropped_rows.jsonl")

    def _decode_row(row: dict) -> None:
        file_stem = "".join(char if char.isalnum() or char in "-_." else "_" for char in row["id"])
        output_path = os.path.join(args.output_dir, f"{file_stem}.json")
        if args.skip_existing and os.path.exists(output_path):
            print(f"[SKIP existing] {file_stem}.json")
        else:
            with open(os.path.join(args.verbose_dir, f"verbose_{file_stem}.log"), "w", encoding="utf-8") as log:
                result = run_one_utterance(row, args, translator, samplers, malformed_log_path, log)
            # Write then rename, so a preempted worker never leaves a truncated JSON that --skip-existing trusts.
            temporary_path = f"{output_path}.tmp.{os.getpid()}"
            with open(temporary_path, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary_path, output_path)

    failures = []
    with ThreadPoolExecutor(max_workers=args.num_concurrent_cases) as executor:
        jobs = {executor.submit(_decode_row, row.to_dict()): (row_idx, row["id"])
                for row_idx, (_, row) in enumerate(rows.iterrows())}
        for job in as_completed(jobs):
            row_idx, utt_id = jobs[job]
            try:
                job.result()
            except Exception as exc:
                # One unusable row must not cost the other rows of the task.
                print(f"[ERROR] Row {row_idx} ({utt_id}) raised: {exc}", file=sys.stderr, flush=True)
                failures.append(f"Row {row_idx} ({utt_id}): {exc}")
                with open(dropped_rows_path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"row_idx": row_idx, "utt_id": utt_id, "reason": str(exc),
                                             "error_type": type(exc).__name__,
                                             "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                            ensure_ascii=False) + "\n")

    if len(failures) > args.max_dropped_rows:
        print(f"[FATAL] {len(failures)} row(s) failed, more than --max-dropped-rows={args.max_dropped_rows} "
              f"(recorded in {dropped_rows_path})", file=sys.stderr)
        for failure in failures[:20]:
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)
    elif failures:
        print(f"[DROPPED] {len(failures)} row(s) dropped, within --max-dropped-rows={args.max_dropped_rows} "
              f"(recorded in {dropped_rows_path})", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
    else:
        print(f"[DONE] {len(rows)} row(s) decoded without errors")


if __name__ == "__main__":
    main()
