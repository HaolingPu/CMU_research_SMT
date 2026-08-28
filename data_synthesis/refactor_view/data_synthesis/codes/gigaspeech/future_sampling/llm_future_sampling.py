#!/usr/bin/env python3
"""
Future Source Sampling + Token Consensus + Logprob Safety for Streaming MT.

Algorithm (per 960ms chunk, flat utterance stream):
  1. Accumulate observed source words across the whole utterance.
  2. Sample M future source continuations (LLM, temp > 0).
  3. Translate (observed + each future) -> M Chinese translations (forced prefix = committed).
  4. Pick t_star = most central of M (embedding medoid).
  5. Token-level majority vote across M translations -> consensus extension.
  6. Score consensus extension tokens under observed-only prompt (1 forward pass, logprobs).
  7. Commit only the prefix of consensus that passes logprob + margin thresholds.
  8. At utterance end, force-commit remainder.

Usage:
  python llm_future_sampling.py \\
    --input-tsv MANIFEST.tsv \\
    --output-root OUTPUT_DIR \\
    --task-id $SLURM_ARRAY_TASK_ID \\
    --num-tasks 8

  Test / verbose (one utterance, print every step to stdout AND log file):
  python llm_future_sampling.py --input-tsv MANIFEST.tsv --output-root OUT \\
    --test-one [--utt-id AUD0000000003_0]
"""

import argparse
import ast
import csv
import itertools
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams


# ===================================================================
# CLI Arguments
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Future source sampling + majority vote for streaming MT data synthesis."
    )

    # I/O
    p.add_argument("--input-tsv", required=True, help="Path to GigaSpeech manifest TSV.")
    p.add_argument("--output-root", required=True, help="Directory to write per-utterance JSON results.")
    p.add_argument("--model-path", default="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel size for vLLM.")

    # Data-parallel sharding (for SLURM array jobs)
    p.add_argument("--task-id", type=int, default=0, help="Worker id (0-indexed).")
    p.add_argument("--num-tasks", type=int, default=1, help="Total number of workers.")

    # Future sampling hyper-parameters
    p.add_argument("--num-candidates", type=int, default=10,
                   help="M: number of future source samples per chunk.")
    p.add_argument("--future-tokens", type=int, default=8,
                   help="Max tokens for each sampled future continuation.")
    p.add_argument("--sample-temperature", type=float, default=0.8,
                   help="Temperature for source continuation sampling (higher = more diverse).")

    # Commit-policy hyper-parameters
    p.add_argument("--min-commit-chars", type=int, default=2,
                   help="Minimum Chinese characters to commit in one WRITE.")
    p.add_argument("--tau", type=float, default=0.7,
                   help="Token consensus threshold (fraction of M that must agree, 0.6-0.8).")
    p.add_argument("--logp-min", type=float, default=-2.0,
                   help="Minimum per-token logprob under observed-only prompt to allow commit.")
    p.add_argument("--margin-max", type=float, default=1.5,
                   help="Maximum logprob margin (top1_logp - chosen_logp) for safe commit.")
    p.add_argument("--no-think-prefix", default="",
                   help="Prefix to disable thinking mode (e.g. '<think>\\n\\n</think>\\n' for Qwen3).")

    # Misc
    p.add_argument("--max-rows", type=int, default=None, help="Process first N assigned rows (debug).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    p.add_argument("--save-details", action="store_true",
                   help="Save per-chunk debug info (sampled futures, candidates, etc.).")
    p.add_argument("--id-column", default="id", help="Column name for utterance id.")

    # Test / verbose
    p.add_argument("--test-one", action="store_true",
                   help="Process only one utterance and print step-by-step (implies --verbose).")
    p.add_argument("--utt-id", default=None,
                   help="With --test-one: process this utterance id only (scan TSV to find it). Else: first assigned row.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-chunk details: observed, M futures, M full translations, agreed prefix, cap, action.")

    return p.parse_args()


# ===================================================================
# Environment Setup
# ===================================================================

def setup_env() -> None:
    os.environ["HF_HOME"] = "/data/user_data/haolingp/hf_cache"
    os.environ["HF_HUB_CACHE"] = "/data/user_data/haolingp/hf_cache/hub"
    os.environ["TRANSFORMERS_CACHE"] = "/data/user_data/haolingp/hf_cache/transformers"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===================================================================
# Prompt Templates
# ===================================================================

def build_continuation_prompt(observed_source: str, n_words: int = 8) -> str:
    """Prompt the LLM to predict what the speaker will say next."""
    return (
        f"Continue this English speech naturally with about {n_words} more words. "
        f"Output ONLY the continuation words, nothing else.\n\n"
        f'"{observed_source}"'
    )


def build_translate_prompt(
    full_source: str, prev_context: str = "", committed: str = "",
) -> str:
    """Prompt the LLM to translate English -> Chinese."""
    ctx = f"Previous context: {prev_context}\n\n" if prev_context else ""
    if committed:
        return (
            f"{ctx}"
            f"Translate the following English to Chinese. "
            f"The translation must begin exactly with: \"{committed}\". "
            f"Continue from where this prefix ends.\n\n"
            f'English: "{full_source}"\n'
            f"Chinese:"
        )
    return (
        f"{ctx}"
        f"Translate the following English to Chinese. "
        f"Output ONLY the Chinese translation, no explanation.\n\n"
        f'English: "{full_source}"\n'
        f"Chinese:"
    )


def build_observed_only_prompt(
    observed_source: str, committed: str = "", prev_context: str = "",
) -> str:
    """Observed-only scoring prompt (no future source). For logprob safety verification."""
    ctx = f"Previous context: {prev_context}\n\n" if prev_context else ""
    if committed:
        return (
            f"{ctx}"
            f"Translate only the English text shown below to Chinese. "
            f"Do not guess any words not yet shown. "
            f"The translation so far is: \"{committed}\". "
            f"Continue translating.\n\n"
            f'English: "{observed_source}"\n'
            f"Chinese:"
        )
    return (
        f"{ctx}"
        f"Translate only the English text shown below to Chinese. "
        f"Do not guess any words not yet shown.\n\n"
        f'English: "{observed_source}"\n'
        f"Chinese:"
    )


def build_complete_prompt(
    full_source: str, committed: str, prev_context: str = ""
) -> str:
    """Prompt the LLM to complete a partial translation at utterance end."""
    ctx = f"Previous context: {prev_context}\n\n" if prev_context else ""
    return (
        f"{ctx}"
        f"Complete the Chinese translation of this English text.\n\n"
        f'English: "{full_source}"\n'
        f'Chinese so far: "{committed}"\n\n'
        f"Continue the translation from where it left off. "
        f"Output ONLY the remaining Chinese text:"
    )


# ===================================================================
# Text Utilities
# ===================================================================

def parse_list_column(raw: Any) -> List[str]:
    """Parse a Python list stored as a string in a TSV column."""
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return [raw] if raw else []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)] if str(parsed).strip() else []


def normalize_zh(text: str) -> str:
    """Normalize Chinese text for comparison: NFC + remove whitespace."""
    text = unicodedata.normalize("NFC", text.strip())
    text = re.sub(r"\s+", "", text)
    return text


def clean_llm_output(text: str) -> str:
    """Strip Qwen3 <think> blocks, surrounding quotes (EN + ZH), and whitespace."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip().strip('"').strip("'")
    text = text.strip("\u201c\u201d\u2018\u2019")
    text = text.strip('"').strip("'")
    return text


def clean_continuation(observed: str, raw_output: str, max_words: int = 12) -> str:
    """Clean up a source continuation: remove repeated input, truncate."""
    text = clean_llm_output(raw_output)

    obs_lower = observed.lower().strip()
    text_lower = text.lower()
    if text_lower.startswith(obs_lower):
        text = text[len(obs_lower):].strip()

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])

    return text.strip()


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe[:200] if safe else "unknown"


class _TeeWriter:
    """Write to both a file and stdout (used for --test-one verbose output)."""
    def __init__(self, file_obj):
        self._f = file_obj
    def write(self, msg):
        self._f.write(msg)
        sys.stdout.write(msg)
    def flush(self):
        self._f.flush()
        sys.stdout.flush()
    def close(self):
        self._f.close()


def _vlog(log_file: Optional[Any], msg: str) -> None:
    """Write msg to verbose log file (under output_root). No-op if log_file is None."""
    if log_file is not None:
        log_file.write(msg)
        if not msg.endswith("\n"):
            log_file.write("\n")
        log_file.flush()


# ===================================================================
# Voting: medoid selection, token consensus, logprob scoring
# ===================================================================

def pick_most_central_translation(texts: List[str], encoder: Any) -> str:
    """
    From M translations, pick the one most central (min sum of cosine distances).

    Embedding similarity is computed on normalized text for robust comparison,
    but we return the *original* ``texts[idx]`` so downstream token operations
    stay consistent with the actual model output.
    """
    valid = [t for t in texts if t.strip()]
    if not valid:
        return ""
    normalized = [normalize_zh(t) for t in valid]
    try:
        embs = encoder.encode(normalized, convert_to_numpy=True)
    except Exception:
        embs = encoder.encode(normalized)
    if hasattr(embs, "numpy"):
        embs = embs.numpy()
    import numpy as np
    embs = np.asarray(embs)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_n = embs / norms
    sim = np.dot(embs_n, embs_n.T)
    sum_dist = (1.0 - sim).sum(axis=1)
    idx = int(np.argmin(sum_dist))
    return valid[idx]


def longest_common_prefix_norm(a: str, b: str) -> str:
    """LCP of two strings (normalized). Kept for backward compat / debugging."""
    an, bn = normalize_zh(a), normalize_zh(b)
    i = 0
    while i < len(an) and i < len(bn) and an[i] == bn[i]:
        i += 1
    return an[:i]


def token_vote_consensus(
    all_token_ids: List[List[int]],
    prefix_len: int,
    tau: float,
    M_total: int,
) -> List[int]:
    """
    Token-level majority vote for consensus extension beyond committed prefix.

    Denominator = M_total (shorter translations implicitly vote against extending).
    Returns list of consensus token IDs (extension only, prefix excluded).
    """
    consensus: List[int] = []
    pos = prefix_len
    while True:
        votes: List[int] = []
        for tids in all_token_ids:
            if pos < len(tids):
                votes.append(tids[pos])
        if not votes:
            break
        best_tok, best_count = Counter(votes).most_common(1)[0]
        if best_count / M_total >= tau:
            consensus.append(best_tok)
            pos += 1
        else:
            break
    return consensus


def _apply_chat_template(
    tokenizer: Any, user_content: str, no_think_prefix: str = "",
) -> str:
    """Apply chat template up to assistant turn start, with optional no-think prefix."""
    msgs = [{"role": "user", "content": user_content}]
    text = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False,
    )
    return text + no_think_prefix


def score_continuation_logprobs(
    llm: LLM,
    tokenizer: Any,
    prompt_text: str,
    committed: str,
    continuation: str,
    no_think_prefix: str = "",
    top_k: int = 20,
    logprob_offset: int = 0,
) -> List[Tuple[int, str, float, float]]:
    """
    Score each token of ``continuation`` under observed-only conditions.

    Base and continuation are tokenized **separately** then concatenated
    so that BPE never re-merges tokens across the boundary.

    Sub-character byte tokens (whose individual decode contains U+FFFD)
    are auto-passed with logp=0, margin=0 because their isolated logprobs
    are not meaningful for character-level safety assessment.

    ``logprob_offset``: 0 means plp[pos] scores token[pos] (standard).
    Set to -1 if your vLLM build aligns one position earlier.

    Returns list of (token_id, token_str, logprob, margin)
    where margin = logp(top1) - logp(chosen).
    """
    base = _apply_chat_template(tokenizer, prompt_text, no_think_prefix) + committed

    base_ids = tokenizer.encode(base, add_special_tokens=False)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if not cont_ids:
        return []

    full_ids = base_ids + cont_ids
    ext_start = len(base_ids)

    sp = SamplingParams(max_tokens=1, prompt_logprobs=top_k, temperature=0.0)
    outs = llm.generate([{"prompt_token_ids": full_ids}], sampling_params=sp)
    plp = outs[0].prompt_logprobs

    results: List[Tuple[int, str, float, float]] = []
    for i, tid in enumerate(cont_ids):
        tok_str = tokenizer.decode([tid], skip_special_tokens=True)

        # Sub-character byte tokens (e.g. first byte of a multi-byte CJK char)
        # decode to U+FFFD when examined in isolation.  Their individual
        # logprobs are meaningless — auto-pass them.
        if "\ufffd" in tok_str:
            results.append((tid, tok_str, 0.0, 0.0))
            continue

        pos = ext_start + i + logprob_offset
        if 0 <= pos < len(plp) and plp[pos] is not None:
            entry = plp[pos]
            chosen_lp = entry[tid].logprob if tid in entry else -100.0
            top1_lp = max(v.logprob for v in entry.values()) if entry else chosen_lp
            margin = top1_lp - chosen_lp
        else:
            chosen_lp, margin = -100.0, 100.0
        results.append((tid, tok_str, chosen_lp, margin))
    return results


def calibrate_logprob_offset(llm: LLM, tokenizer: Any, no_think_prefix: str = "") -> int:
    """
    Auto-detect whether prompt_logprobs is aligned at pos or pos-1.

    Encodes a trivial prompt + known continuation, checks which offset
    yields a sane logprob for the first continuation token.  Result is
    cached for the lifetime of the process.
    """
    test_user = "Translate to Chinese.\n\nEnglish: \"Hello\"\nChinese:"
    test_cont = "你好"

    base = _apply_chat_template(tokenizer, test_user, no_think_prefix)
    base_ids = tokenizer.encode(base, add_special_tokens=False)
    cont_ids = tokenizer.encode(test_cont, add_special_tokens=False)
    if not cont_ids:
        return 0

    full_ids = base_ids + cont_ids
    ext_start = len(base_ids)
    first_tid = cont_ids[0]

    sp = SamplingParams(max_tokens=1, prompt_logprobs=20, temperature=0.0)
    outs = llm.generate([{"prompt_token_ids": full_ids}], sampling_params=sp)
    plp = outs[0].prompt_logprobs

    for candidate_offset in (0, -1, 1):
        pos = ext_start + candidate_offset
        if 0 <= pos < len(plp) and plp[pos] is not None:
            entry = plp[pos]
            if first_tid in entry and entry[first_tid].logprob > -10.0:
                return candidate_offset
    return 0


# ===================================================================
# LLM Interaction
# ===================================================================

def sample_source_futures(
    llm: LLM,
    observed_source: str,
    num_candidates: int,
    sample_params: SamplingParams,
) -> List[str]:
    """
    Sample M diverse future source continuations via vLLM ``n=M``.

    Single prompt, M independent samples -- more efficient than M copies
    and produces the same stochastic diversity.  ``top_p`` and
    ``presence_penalty`` further reduce duplicate continuations.
    """
    prompt = build_continuation_prompt(observed_source)
    messages = [{"role": "user", "content": prompt}]

    params = SamplingParams(
        temperature=sample_params.temperature,
        max_tokens=sample_params.max_tokens,
        n=num_candidates,
        top_p=0.95,
        presence_penalty=0.6,
    )

    outputs = llm.chat(messages=[messages], sampling_params=params)

    futures: List[str] = []
    for out in outputs[0].outputs:
        cleaned = clean_continuation(observed_source, out.text)
        futures.append(cleaned)

    return futures


def translate_with_futures(
    llm: LLM,
    observed_source: str,
    futures: List[str],
    prev_context: str,
    translate_params: SamplingParams,
    committed: str = "",
    tokenizer: Any = None,
    no_think_prefix: str = "",
) -> List[str]:
    """
    Translate (observed + each sampled future) to Chinese.

    When ``committed`` is non-empty **and** a tokenizer is provided, generation
    is forced to start with ``committed`` via prompt_token_ids (immutable prefix).
    Otherwise falls back to normal chat generation.
    """
    if not committed or tokenizer is None:
        messages_list = []
        for future in futures:
            full_source = (observed_source + " " + future).strip() if future else observed_source
            prompt = build_translate_prompt(full_source, prev_context)
            messages_list.append([{"role": "user", "content": prompt}])

        outputs = llm.chat(messages=messages_list, sampling_params=translate_params)
        return [clean_llm_output(o.outputs[0].text) for o in outputs]

    committed_norm = normalize_zh(committed)
    prompt_token_ids_list = []
    for future in futures:
        full_source = (observed_source + " " + future).strip() if future else observed_source
        prompt = build_translate_prompt(full_source, prev_context, committed=committed_norm)
        chat_text = _apply_chat_template(tokenizer, prompt, no_think_prefix) + committed_norm
        token_ids = tokenizer.encode(chat_text, add_special_tokens=False)
        prompt_token_ids_list.append(token_ids)

    outputs = llm.generate(
        [{"prompt_token_ids": tids} for tids in prompt_token_ids_list],
        sampling_params=translate_params,
    )
    translations: List[str] = []
    for output in outputs:
        continuation = clean_llm_output(output.outputs[0].text)
        translations.append(normalize_zh(committed_norm + continuation))
    return translations


def translate_final(
    llm: LLM,
    full_source: str,
    committed: str,
    translate_params: SamplingParams,
) -> str:
    """
    Translate the full utterance at the end (all words known).

    If we already have some committed text, ask the model to continue
    from where it left off. Otherwise, translate from scratch.
    """
    if committed:
        prompt = build_complete_prompt(full_source, committed)
    else:
        prompt = build_translate_prompt(full_source)

    messages = [[{"role": "user", "content": prompt}]]
    outputs = llm.chat(messages=messages, sampling_params=translate_params)
    raw = outputs[0].outputs[0].text
    cleaned = clean_llm_output(raw)

    if committed:
        return normalize_zh(committed) + normalize_zh(cleaned)
    else:
        return normalize_zh(cleaned)


# ===================================================================
# Core Processing: Per-utterance (flat -- no sentence grouping)
# ===================================================================

def process_one_utterance(
    llm: LLM,
    utt_id: str,
    sentences: List[str],
    trajectory: List[str],
    row: Dict[str, str],
    args: argparse.Namespace,
    sample_params: SamplingParams,
    translate_params: SamplingParams,
    verbose_log_file: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Process one full utterance by iterating over *all* trajectory chunks
    in a single flat stream (no sentence-level grouping).

    ``accumulated_source`` and ``committed_norm`` grow monotonically
    across the whole utterance, so the LLM always sees the full context
    of everything translated so far.
    """
    full_source_text = " ".join(sentences)
    n_chunks = len(trajectory)

    _vlog(verbose_log_file, f"\n{'#'*60}")
    _vlog(verbose_log_file, f"# Utterance: {utt_id}")
    _vlog(verbose_log_file, f"# Full text (EN): {full_source_text}")
    _vlog(verbose_log_file, f"# Trajectory chunks: {n_chunks}")
    _vlog(verbose_log_file, f"{'#'*60}")

    committed_norm = ""
    accumulated_source = ""
    decisions: List[Tuple[str, str]] = []
    all_details: List[Optional[Dict]] = []

    tokenizer = getattr(args, "_tokenizer", None)
    no_think = getattr(args, "no_think_prefix", "")
    encoder = getattr(args, "_embedding_encoder", None)

    for chunk_pos, chunk in enumerate(trajectory):
        stripped = chunk.strip()
        if stripped:
            accumulated_source = (accumulated_source + " " + stripped).strip()

        is_last = (chunk_pos == n_chunks - 1)

        _vlog(verbose_log_file, f"\n{'='*60}")
        _vlog(verbose_log_file,
              f"  Chunk {chunk_pos + 1}/{n_chunks}")
        _vlog(verbose_log_file,
              f"  accumulated observed source (EN): \"{accumulated_source}\"")
        _vlog(verbose_log_file, f"{'='*60}")

        # ----- Case 1: Empty chunk (no new words) -> READ -----
        if not stripped and not is_last:
            decisions.append(("READ", ""))
            _vlog(verbose_log_file,
                  "  [Case 1] Empty chunk -> READ (no new words)")
            if args.save_details:
                all_details.append({"reason": "empty_chunk"})
            continue

        # ----- Case 2: Last chunk -> force commit remainder -----
        if is_last:
            full_translation = translate_final(
                llm, accumulated_source, committed_norm, translate_params
            )
            if len(full_translation) > len(committed_norm):
                remaining = full_translation[len(committed_norm):]
                _vlog(verbose_log_file,
                      "  [Case 2] End of utterance -> force commit")
                _vlog(verbose_log_file,
                      f"  full_translation: \"{full_translation}\"")
                _vlog(verbose_log_file,
                      f"  committed so far: \"{committed_norm}\"")
                _vlog(verbose_log_file,
                      f"  remaining written: \"{remaining}\"")
                committed_norm = full_translation
                decisions.append(("WRITE", remaining))
            else:
                _vlog(verbose_log_file,
                      "  [Case 2] End of utterance -> nothing new to commit")
                _vlog(verbose_log_file,
                      f"  full_translation: \"{full_translation}\"")
                decisions.append(("READ", ""))
            if args.save_details:
                all_details.append({
                    "reason": "end_of_utterance",
                    "full_translation": full_translation,
                })
            continue

        # ----- Case 3: future sampling + consensus + logprob safety -----

        # Step 1: Sample M future source continuations
        futures = sample_source_futures(
            llm, accumulated_source, args.num_candidates, sample_params
        )
        _vlog(verbose_log_file,
              "  [Step 1] Sampled M future source continuations:")
        for fi, f in enumerate(futures):
            _vlog(verbose_log_file, f"    future[{fi}]: \"{f}\"")

        # Step 2: Translate (observed + each future), forced committed prefix
        full_translations = translate_with_futures(
            llm, accumulated_source, futures, "", translate_params,
            committed=committed_norm, tokenizer=tokenizer,
            no_think_prefix=no_think,
        )
        valid_translations = [
            normalize_zh(t) for t in full_translations
            if normalize_zh(t).startswith(committed_norm)
        ]
        _vlog(verbose_log_file,
              f"  [Step 2] Translations (valid "
              f"{len(valid_translations)}/{len(full_translations)}):")
        for ti, t in enumerate(valid_translations):
            _vlog(verbose_log_file, f"    trans[{ti}]: \"{t}\"")

        if len(valid_translations) < 2:
            decisions.append(("READ", ""))
            _vlog(verbose_log_file,
                  f"  Too few valid translations "
                  f"({len(valid_translations)}) -> READ")
            if args.save_details:
                all_details.append({"reason": "too_few_valid",
                                    "observed": accumulated_source})
            continue

        # Step 3: Pick medoid (most central translation)
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                args._embedding_encoder = SentenceTransformer(
                    "sentence-transformers/"
                    "paraphrase-multilingual-MiniLM-L12-v2")
                encoder = args._embedding_encoder
            except ImportError as e:
                raise ImportError(
                    "pip install sentence-transformers") from e
        candidate = pick_most_central_translation(
            valid_translations, encoder)
        _vlog(verbose_log_file,
              f"  [Step 3] Medoid of {len(valid_translations)}"
              f" -> t_star = \"{candidate}\"")

        # Step 4: Token-level consensus on extension tokens only.
        # Tokenize each translation's extension *separately* to avoid
        # BPE merges across the committed/extension boundary.
        all_ext_token_ids: List[List[int]] = []
        for t in valid_translations:
            ext_text = t[len(committed_norm):]
            ext_ids = tokenizer.encode(ext_text, add_special_tokens=False)
            all_ext_token_ids.append(ext_ids)

        cons_ext_ids = token_vote_consensus(
            all_ext_token_ids, 0, args.tau,
            M_total=len(valid_translations),
        )
        if not cons_ext_ids:
            decisions.append(("READ", ""))
            _vlog(verbose_log_file,
                  "  [Step 4] No token consensus -> READ")
            if args.save_details:
                all_details.append({
                    "reason": "no_consensus",
                    "observed": accumulated_source,
                    "candidate": candidate,
                })
            continue

        ext_cons = normalize_zh(
            tokenizer.decode(cons_ext_ids, skip_special_tokens=True))
        _vlog(verbose_log_file,
              f"  [Step 4] Token consensus: ext_cons = \"{ext_cons}\" "
              f"({len(cons_ext_ids)} tokens)")

        # Ensure consensus is a prefix of t_star
        t_star_ext = candidate[len(committed_norm):]
        if not t_star_ext.startswith(ext_cons):
            trimmed = ""
            for ca, cb in zip(ext_cons, t_star_ext):
                if ca == cb:
                    trimmed += ca
                else:
                    break
            ext_cons = trimmed
            if not ext_cons:
                decisions.append(("READ", ""))
                _vlog(verbose_log_file,
                      "  [Step 4] Consensus not prefix of t_star -> READ")
                if args.save_details:
                    all_details.append({
                        "reason": "consensus_tstar_mismatch",
                        "observed": accumulated_source,
                    })
                continue
            cons_ext_ids = tokenizer.encode(
                ext_cons, add_special_tokens=False)
            _vlog(verbose_log_file,
                  f"  [Step 4] Trimmed to t_star prefix: "
                  f"ext_cons = \"{ext_cons}\"")

        # Step 5: Observed-only logprob safety (1 forward pass)
        obs_prompt = build_observed_only_prompt(
            accumulated_source, committed_norm)
        logprob_offset = getattr(args, "_logprob_offset", 0)
        token_scores = score_continuation_logprobs(
            llm, tokenizer, obs_prompt, committed_norm, ext_cons,
            no_think_prefix=no_think,
            logprob_offset=logprob_offset,
        )

        safe_count = 0
        stop_reason = "all_safe"
        for _tid, tok_str, logp, margin in token_scores:
            if logp < args.logp_min:
                stop_reason = (
                    f"logprob_fail: '{tok_str}' "
                    f"logp={logp:.3f} < {args.logp_min}")
                break
            if margin > args.margin_max:
                stop_reason = (
                    f"margin_fail: '{tok_str}' "
                    f"margin={margin:.3f} > {args.margin_max}")
                break
            safe_count += 1

        _vlog(verbose_log_file,
              f"  [Step 5] Logprob safety: "
              f"{safe_count}/{len(token_scores)} tokens safe  "
              f"({stop_reason})")
        for _tid, tok_str, logp, margin in token_scores:
            if "\ufffd" in tok_str:
                flag = "SKIP"
            elif logp >= args.logp_min and margin <= args.margin_max:
                flag = "PASS"
            else:
                flag = "FAIL"
            _vlog(verbose_log_file,
                  f"    [{flag}] {tok_str!r:>6s}  logp={logp:+.3f}  "
                  f"margin={margin:.3f}")

        # Step 6: Commit safe extension
        if safe_count > 0:
            safe_ids = cons_ext_ids[:safe_count]
            ext_safe = normalize_zh(
                tokenizer.decode(safe_ids, skip_special_tokens=True))
        else:
            ext_safe = ""

        if ext_safe and len(ext_safe) >= args.min_commit_chars:
            committed_norm = committed_norm + ext_safe
            decisions.append(("WRITE", ext_safe))
            _vlog(verbose_log_file,
                  f"  [Step 6] WRITE  extension=\"{ext_safe}\"  "
                  f"committed=\"{committed_norm}\"")
        else:
            decisions.append(("READ", ""))
            reason = ("too_short" if ext_safe
                      else "logprob_unsafe" if safe_count == 0
                      else "empty")
            _vlog(verbose_log_file,
                  f"  [Step 6] READ ({reason})  "
                  f"committed=\"{committed_norm}\"")

        if args.save_details:
            all_details.append({
                "observed": accumulated_source,
                "futures": futures,
                "translations": valid_translations,
                "candidate": candidate,
                "ext_cons": ext_cons,
                "ext_safe": ext_safe,
                "token_scores": [
                    (s[1], s[2], s[3]) for s in token_scores],
                "stop_reason": stop_reason,
                "committed_after": committed_norm,
                "action": decisions[-1][0],
            })

    # ---- Assemble output ----
    source_traj = trajectory
    target_traj = [d[1] for d in decisions]
    actions = [d[0] for d in decisions]

    result: Dict[str, Any] = {
        "utt_id": utt_id,
        "original_text": full_source_text,
        "input_sentences": sentences,
        "source_future_sampling": source_traj,
        "target_future_sampling": target_traj,
        "actions": actions,
        "config": {
            "num_candidates": args.num_candidates,
            "future_tokens": args.future_tokens,
            "tau": args.tau,
            "logp_min": args.logp_min,
            "margin_max": args.margin_max,
            "min_commit_chars": args.min_commit_chars,
        },
    }

    for k in ["audio", "n_frames", "speaker", "src_lang", "tgt_lang"]:
        if k in row:
            result[k] = row[k]

    if args.save_details:
        result["details"] = all_details

    return result


# ===================================================================
# Data-Parallel I/O (same sharding as existing scripts)
# ===================================================================

def iter_assigned_rows(input_tsv: str, task_id: int, num_tasks: int):
    """Yield rows assigned to this worker (row_idx % num_tasks == task_id)."""
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if row_idx % num_tasks != task_id:
                continue
            yield row_idx, row


def count_assigned_rows(input_tsv: str, task_id: int, num_tasks: int) -> int:
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        total = sum(1 for _ in f) - 1  # minus header
    if total <= task_id:
        return 0
    return int(math.ceil((total - task_id) / num_tasks))


def get_one_row_by_id(input_tsv: str, utt_id: str, id_column: str = "id") -> Optional[Tuple[int, Dict[str, str]]]:
    """Scan TSV for first row with id_column == utt_id. Return (row_idx, row) or None."""
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if str(row.get(id_column, "")).strip() == str(utt_id).strip():
                return row_idx, row
    return None


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    args = parse_args()
    setup_env()

    os.makedirs(args.output_root, exist_ok=True)

    # ----- Load vLLM model -----
    print(f"[Task {args.task_id}] Loading model TP={args.tp} from {args.model_path}")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        max_model_len=16384,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )
    args._tokenizer = llm.get_tokenizer()

    print(f"[Task {args.task_id}] Loading embedding model (paraphrase-multilingual-MiniLM-L12-v2)")
    try:
        from sentence_transformers import SentenceTransformer
        args._embedding_encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    except ImportError as e:
        raise ImportError("pip install sentence-transformers") from e

    # ----- Calibrate prompt_logprobs offset (once per process) -----
    no_think = getattr(args, "no_think_prefix", "")
    args._logprob_offset = calibrate_logprob_offset(llm, args._tokenizer, no_think)
    print(f"[Task {args.task_id}] prompt_logprobs offset = {args._logprob_offset}")

    # ----- Sampling parameters -----
    sample_params = SamplingParams(
        temperature=args.sample_temperature,
        max_tokens=args.future_tokens,
    )
    translate_params = SamplingParams(
        temperature=0.0,
        max_tokens=500,
        repetition_penalty=1.1,
    )

    # ----- Test-one / row iterator -----
    if getattr(args, "test_one", False):
        args.verbose = True
    if getattr(args, "test_one", False) and getattr(args, "utt_id", None):
        one = get_one_row_by_id(args.input_tsv, args.utt_id, args.id_column)
        if one is None:
            print(f"[--test-one] utt-id '{args.utt_id}' not found in {args.input_tsv}")
            return
        row_iter: Any = [one]
        total = 1
    elif getattr(args, "test_one", False):
        row_iter = list(itertools.islice(
            iter_assigned_rows(args.input_tsv, args.task_id, args.num_tasks), 1
        ))
        total = 1 if row_iter else 0
        if total == 0:
            print("[--test-one] no assigned rows for this task.")
            return
    else:
        row_iter = iter_assigned_rows(args.input_tsv, args.task_id, args.num_tasks)
        total = count_assigned_rows(args.input_tsv, args.task_id, args.num_tasks)
        if args.max_rows is not None:
            total = min(total, args.max_rows)

    print(
        f"[Task {args.task_id}] Processing {total} rows from {args.input_tsv}\n"
        f"  output={args.output_root}\n"
        f"  M={args.num_candidates}, tau={args.tau}, logp_min={args.logp_min}, "
        f"margin_max={args.margin_max}, future_tokens={args.future_tokens}"
    )
    use_tee = getattr(args, "test_one", False)
    if getattr(args, "verbose", False):
        where = "stdout + log file" if use_tee else "verbose_<utt_id>.log"
        print(f"  [verbose] output to: {where}\n")

    written = 0
    skipped = 0
    failed = 0
    processed = 0

    pbar = tqdm(total=total, desc=f"task_{args.task_id}")

    for row_idx, row in row_iter:
        if not getattr(args, "test_one", False) and args.max_rows is not None and processed >= args.max_rows:
            break
        processed += 1

        utt_id = str(row.get(args.id_column, "")).strip()
        if not utt_id:
            utt_id = f"row_{row_idx:09d}"

        out_path = os.path.join(args.output_root, f"{sanitize_filename(utt_id)}.json")
        output_exists = os.path.exists(out_path)
        if output_exists and not args.overwrite and not getattr(args, "verbose", False):
            skipped += 1
            pbar.update(1)
            continue

        try:
            sentences = parse_list_column(row.get("src_text_full"))
            trajectory = parse_list_column(row.get("src_trajectory"))

            if not sentences:
                raise ValueError("Empty src_text_full")
            if not trajectory:
                raise ValueError("Empty src_trajectory column")

            verbose_log_path = None
            if getattr(args, "verbose", False):
                verbose_log_path = os.path.join(
                    args.output_root,
                    f"verbose_{sanitize_filename(utt_id)}.log",
                )
            verbose_log_file = None
            if verbose_log_path is not None:
                raw_file = open(verbose_log_path, "w", encoding="utf-8")
                verbose_log_file = (_TeeWriter(raw_file) if use_tee else raw_file)

            try:
                result = process_one_utterance(
                    llm, utt_id, sentences, trajectory, row,
                    args, sample_params, translate_params,
                    verbose_log_file=verbose_log_file,
                )
            finally:
                if verbose_log_file is not None:
                    verbose_log_file.close()
                    print(f"  [verbose] log written to {verbose_log_path}")

            if getattr(args, "verbose", False) and not args.overwrite and output_exists:
                skipped += 1
            else:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                written += 1

        except Exception as e:
            failed += 1
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "utt_id": utt_id,
                    "error": str(e),
                    "row_index": row_idx,
                }, f, ensure_ascii=False, indent=2)

        pbar.update(1)

    pbar.close()
    print(
        f"\n[Task {args.task_id}] Done. "
        f"written={written}, skipped={skipped}, failed={failed}"
    )


if __name__ == "__main__":
    main()
