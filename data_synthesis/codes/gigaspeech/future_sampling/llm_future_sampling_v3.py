#!/usr/bin/env python3
"""
Future Source Sampling + LLM-as-Judge for Streaming MT.

v3: Instead of char trie vote (v2) or token consensus + logprob (v1),
we ask the LLM itself to judge how many characters of the *greedy
translation* can be safely committed, given the M candidate translations
from future source sampling.

Key insight: The LLM judges on the greedy path (observed-only translation),
so the output is always a prefix of an actual greedy decode — no risk of
the judge producing semantically-correct-but-different characters.

Algorithm (per 960ms chunk, flat utterance stream):
  1. Accumulate observed source words across the whole utterance.
  2. Sample M future source continuations (LLM, temp > 0).
  3. Translate (observed + each future) -> M Chinese translations (forced prefix = committed).
  4. Greedy-translate observed-only source -> greedy extension beyond committed.
  5. Ask LLM judge: how many chars of greedy extension are safe given M candidates?
  6. Commit that many chars (if >= min_commit_chars), else READ.
  7. At utterance end, force-commit remainder.

Usage:
  python llm_future_sampling_v3.py \\
    --input-tsv MANIFEST.tsv \\
    --output-root OUTPUT_DIR \\
    --task-id $SLURM_ARRAY_TASK_ID \\
    --num-tasks 8

  Test / verbose (one utterance, print every step to stdout AND log file):
  python llm_future_sampling_v3.py --input-tsv MANIFEST.tsv --output-root OUT \\
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
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams


# ===================================================================
# CLI Arguments
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Future source sampling + LLM-as-judge for streaming MT data synthesis (v3)."
    )

    p.add_argument("--input-tsv", required=True, help="Path to GigaSpeech manifest TSV.")
    p.add_argument("--output-root", required=True, help="Directory to write per-utterance JSON results.")
    p.add_argument("--model-path", default="/data/user_data/haolingp/models/Qwen3-30B-A3B-Instruct-2507-FP8")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel size for vLLM.")

    p.add_argument("--task-id", type=int, default=0, help="Worker id (0-indexed).")
    p.add_argument("--num-tasks", type=int, default=1, help="Total number of workers.")

    p.add_argument("--num-candidates", type=int, default=10,
                   help="M: number of future source samples per chunk.")
    p.add_argument("--future-tokens", type=int, default=8,
                   help="Max tokens for each sampled future continuation.")
    p.add_argument("--sample-temperature", type=float, default=0.8,
                   help="Temperature for source continuation sampling.")

    p.add_argument("--min-commit-chars", type=int, default=2,
                   help="Minimum Chinese characters to commit in one WRITE.")
    p.add_argument("--min-observed-words", type=int, default=5,
                   help="Skip future sampling when observed source has fewer words than this.")
    p.add_argument("--no-think-prefix", default="",
                   help="Prefix to disable thinking mode (e.g. '<think>\\n\\n</think>\\n' for Qwen3).")

    p.add_argument("--max-rows", type=int, default=None, help="Process first N assigned rows (debug).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    p.add_argument("--save-details", action="store_true",
                   help="Save per-chunk debug info (sampled futures, candidates, etc.).")
    p.add_argument("--id-column", default="id", help="Column name for utterance id.")

    p.add_argument("--test-one", action="store_true",
                   help="Process only one utterance and print step-by-step (implies --verbose).")
    p.add_argument("--utt-id", default=None,
                   help="With --test-one: process this utterance id only.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-chunk details.")

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
    return (
        f"Continue this English speech naturally with about {n_words} more words. "
        f"Output ONLY the continuation words, nothing else.\n\n"
        f'"{observed_source}"'
    )


def build_translate_prompt(
    full_source: str, prev_context: str = "",
) -> str:
    ctx = f"Previous context: {prev_context}\n\n" if prev_context else ""
    return (
        f"{ctx}"
        f"Translate the following English to Chinese. "
        f"Output ONLY the Chinese translation, no explanation.\n\n"
        f'English: "{full_source}"\n'
        f"Chinese:"
    )


def build_complete_prompt(
    full_source: str, committed: str, prev_context: str = ""
) -> str:
    ctx = f"Previous context: {prev_context}\n\n" if prev_context else ""
    return (
        f"{ctx}"
        f"Complete the Chinese translation of this English text.\n\n"
        f'English: "{full_source}"\n'
        f'Chinese so far: "{committed}"\n\n'
        f"Continue the translation from where it left off. "
        f"Output ONLY the remaining Chinese text:"
    )


def build_judge_prompt(
    observed_source: str,
    committed: str,
    greedy_ext: str,
    future_translations: List[str],
) -> str:
    """
    Ask LLM to judge how many characters of greedy_ext can be safely
    committed, given only observed source and the candidate translations
    from future sampling.
    """
    candidates_str = "\n".join(
        f'  {i+1}. "{t}"' for i, t in enumerate(future_translations)
    )
    return f"""You are judging a simultaneous translation decision.

Observed English so far: "{observed_source}"
Chinese committed so far: "{committed}"

Proposed next Chinese characters (from greedy translation): "{greedy_ext}"

Other candidate translations (from future source sampling):
{candidates_str}

Task: Decide how many characters of the proposed text can be safely committed right now, given only the observed English.

Rules:
- Only commit characters that are semantically consistent with the MAJORITY of candidate translations.
- Stop before any character where the candidates diverge in meaning.
- If nothing is safe to commit, output 0.

Output a JSON object with exactly these fields:
{{
  "safe_chars": <integer, how many characters of proposed text to commit>,
  "reasoning": "<one sentence explanation>"
}}"""


# ===================================================================
# Text Utilities
# ===================================================================

def parse_list_column(raw: Any) -> List[str]:
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
    text = unicodedata.normalize("NFC", text.strip())
    text = re.sub(r"\s+", "", text)
    return text


def clean_llm_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip().strip('"').strip("'")
    text = text.strip("\u201c\u201d\u2018\u2019")
    text = text.strip('"').strip("'")
    return text


def clean_continuation(observed: str, raw_output: str, max_words: int = 12) -> str:
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
    if log_file is not None:
        log_file.write(msg)
        if not msg.endswith("\n"):
            log_file.write("\n")
        log_file.flush()


def parse_judge_response(raw: str) -> Tuple[int, str]:
    """
    Parse the JSON response from the judge LLM.

    Robust against markdown fences, extra text around JSON, etc.
    Returns (safe_chars, reasoning). Falls back to (0, error_msg) on failure.
    """
    text = clean_llm_output(raw)

    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not json_match:
        return 0, f"no JSON found in judge output: {text[:200]}"

    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        return 0, f"JSON parse error: {e}; raw: {json_match.group()[:200]}"

    safe = obj.get("safe_chars", 0)
    if not isinstance(safe, int):
        try:
            safe = int(safe)
        except (ValueError, TypeError):
            safe = 0

    reasoning = str(obj.get("reasoning", ""))
    return max(0, safe), reasoning


# ===================================================================
# LLM Interaction
# ===================================================================

def _apply_chat_template(
    tokenizer: Any, user_content: str, no_think_prefix: str = "",
) -> str:
    msgs = [{"role": "user", "content": user_content}]
    text = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False,
    )
    return text + no_think_prefix


def sample_source_futures(
    llm: LLM,
    observed_source: str,
    num_candidates: int,
    sample_params: SamplingParams,
) -> List[str]:
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
        prompt = build_translate_prompt(full_source, prev_context)
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


def greedy_translate_observed(
    llm: LLM,
    observed_source: str,
    committed: str,
    translate_params: SamplingParams,
    tokenizer: Any = None,
    no_think_prefix: str = "",
) -> str:
    """
    Greedy-translate observed-only source and return the full translation.

    When committed is non-empty and tokenizer is available, force the output
    to start with committed via prompt_token_ids.
    """
    prompt = build_translate_prompt(observed_source)

    if committed and tokenizer is not None:
        committed_norm = normalize_zh(committed)
        chat_text = _apply_chat_template(tokenizer, prompt, no_think_prefix) + committed_norm
        token_ids = tokenizer.encode(chat_text, add_special_tokens=False)
        outputs = llm.generate(
            [{"prompt_token_ids": token_ids}],
            sampling_params=translate_params,
        )
        continuation = clean_llm_output(outputs[0].outputs[0].text)
        return normalize_zh(committed_norm + continuation)

    messages = [[{"role": "user", "content": prompt}]]
    outputs = llm.chat(messages=messages, sampling_params=translate_params)
    return normalize_zh(clean_llm_output(outputs[0].outputs[0].text))


def judge_safe_commit(
    llm: LLM,
    observed_source: str,
    committed: str,
    greedy_ext: str,
    future_translations: List[str],
    judge_params: SamplingParams,
) -> Tuple[int, str]:
    """
    Ask the LLM judge how many characters of greedy_ext are safe to commit.

    Returns (safe_chars, reasoning).
    """
    prompt = build_judge_prompt(
        observed_source, committed, greedy_ext, future_translations,
    )
    messages = [[{"role": "user", "content": prompt}]]
    outputs = llm.chat(messages=messages, sampling_params=judge_params)
    raw = outputs[0].outputs[0].text
    return parse_judge_response(raw)


def translate_final(
    llm: LLM,
    full_source: str,
    committed: str,
    translate_params: SamplingParams,
) -> str:
    if committed:
        prompt = build_complete_prompt(full_source, committed)
    else:
        prompt = build_translate_prompt(full_source)

    messages = [[{"role": "user", "content": prompt}]]
    outputs = llm.chat(messages=messages, sampling_params=translate_params)
    raw = outputs[0].outputs[0].text
    cleaned = clean_llm_output(raw)

    if committed:
        committed_norm = normalize_zh(committed)
        cleaned_norm = normalize_zh(cleaned)
        if cleaned_norm.startswith(committed_norm):
            return cleaned_norm
        return committed_norm + cleaned_norm
    else:
        return normalize_zh(cleaned)


# ===================================================================
# Core Processing
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
    judge_params: SamplingParams,
    verbose_log_file: Optional[Any] = None,
) -> Dict[str, Any]:
    full_source_text = " ".join(sentences)
    n_chunks = len(trajectory)

    _vlog(verbose_log_file, f"\n{'#'*60}")
    _vlog(verbose_log_file, f"# Utterance: {utt_id}")
    _vlog(verbose_log_file, f"# Full text (EN): {full_source_text}")
    _vlog(verbose_log_file, f"# Trajectory chunks: {n_chunks}")
    _vlog(verbose_log_file, f"# Commit method: LLM-as-judge (v3)")
    _vlog(verbose_log_file, f"# M={args.num_candidates}, "
          f"min_commit_chars={args.min_commit_chars}")
    _vlog(verbose_log_file, f"{'#'*60}")

    committed_norm = ""
    accumulated_source = ""
    decisions: List[Tuple[str, str]] = []
    all_details: List[Optional[Dict]] = []

    tokenizer = getattr(args, "_tokenizer", None)
    no_think = getattr(args, "no_think_prefix", "")

    for chunk_pos, chunk in enumerate(trajectory):
        stripped = chunk.strip()
        if stripped:
            accumulated_source = (accumulated_source + " " + stripped).strip()

        is_last = (chunk_pos == n_chunks - 1)

        _vlog(verbose_log_file, f"\n{'='*60}")
        _vlog(verbose_log_file, f"  Chunk {chunk_pos + 1}/{n_chunks}")
        _vlog(verbose_log_file,
              f"  accumulated source (EN): \"{accumulated_source}\"")
        _vlog(verbose_log_file,
              f"  committed so far (ZH): \"{committed_norm}\"")
        _vlog(verbose_log_file, f"{'='*60}")

        # ----- Case 1: Empty chunk -> READ -----
        if not stripped and not is_last:
            decisions.append(("READ", ""))
            _vlog(verbose_log_file, "  [Case 1] Empty chunk -> READ")
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
                _vlog(verbose_log_file, "  [Case 2] End of utterance -> force commit")
                _vlog(verbose_log_file, f"  full_translation: \"{full_translation}\"")
                _vlog(verbose_log_file, f"  committed so far: \"{committed_norm}\"")
                _vlog(verbose_log_file, f"  remaining written: \"{remaining}\"")
                committed_norm = full_translation
                decisions.append(("WRITE", remaining))
            else:
                _vlog(verbose_log_file, "  [Case 2] End of utterance -> nothing new")
                _vlog(verbose_log_file, f"  full_translation: \"{full_translation}\"")
                decisions.append(("READ", ""))
            if args.save_details:
                all_details.append({
                    "reason": "end_of_utterance",
                    "full_translation": full_translation,
                })
            continue

        # ----- Case 3: future sampling + LLM judge -----

        observed_words = len(accumulated_source.split())
        if observed_words < args.min_observed_words:
            decisions.append(("READ", ""))
            _vlog(verbose_log_file,
                  f"  [Early skip] observed_words={observed_words} "
                  f"< {args.min_observed_words} -> READ")
            if args.save_details:
                all_details.append({"reason": "early_skip",
                                    "observed_words": observed_words})
            continue

        # Step 1: Sample M future source continuations
        futures = sample_source_futures(
            llm, accumulated_source, args.num_candidates, sample_params
        )
        _vlog(verbose_log_file, "  [Step 1] Sampled M future source continuations:")
        for fi, f in enumerate(futures):
            _vlog(verbose_log_file, f"    future[{fi}]: \"{f}\"")

        # Step 2: Translate (observed + each future), forced committed prefix
        full_translations = translate_with_futures(
            llm, accumulated_source, futures, "", translate_params,
            committed=committed_norm, tokenizer=tokenizer,
            no_think_prefix=no_think,
        )
        all_translations = [normalize_zh(t) for t in full_translations]
        valid_translations = [t for t in all_translations if t.startswith(committed_norm)]
        _vlog(verbose_log_file,
              f"  [Step 2] Translations (valid prefix "
              f"{len(valid_translations)}/{len(all_translations)}):")
        for ti, t in enumerate(all_translations):
            prefix_ok = "OK" if t.startswith(committed_norm) else "BAD"
            _vlog(verbose_log_file, f"    trans[{ti}] [{prefix_ok}]: \"{t}\"")

        if len(valid_translations) < 2:
            decisions.append(("READ", ""))
            _vlog(verbose_log_file,
                  f"  Too few valid translations "
                  f"({len(valid_translations)}) -> READ")
            if args.save_details:
                all_details.append({"reason": "too_few_valid",
                                    "observed": accumulated_source})
            continue

        # Step 3: Greedy translate observed-only source
        greedy_full = greedy_translate_observed(
            llm, accumulated_source, committed_norm, translate_params,
            tokenizer=tokenizer, no_think_prefix=no_think,
        )
        greedy_ext = greedy_full[len(committed_norm):] if greedy_full.startswith(committed_norm) else ""
        _vlog(verbose_log_file,
              f"  [Step 3] Greedy translation (observed-only):")
        _vlog(verbose_log_file,
              f"    greedy_full: \"{greedy_full}\"")
        _vlog(verbose_log_file,
              f"    greedy_ext (beyond committed): \"{greedy_ext}\"")

        if not greedy_ext:
            decisions.append(("READ", ""))
            _vlog(verbose_log_file, "  [Step 3] Empty greedy extension -> READ")
            if args.save_details:
                all_details.append({
                    "reason": "empty_greedy_ext",
                    "observed": accumulated_source,
                    "greedy_full": greedy_full,
                })
            continue

        # Step 4: LLM judge — how many chars of greedy_ext are safe?
        candidate_extensions = [t[len(committed_norm):] for t in valid_translations]
        safe_chars, reasoning = judge_safe_commit(
            llm, accumulated_source, committed_norm,
            greedy_ext, candidate_extensions, judge_params,
        )
        safe_chars = min(safe_chars, len(greedy_ext))

        _vlog(verbose_log_file, f"  [Step 4] LLM judge result:")
        _vlog(verbose_log_file, f"    safe_chars: {safe_chars}/{len(greedy_ext)}")
        _vlog(verbose_log_file, f"    reasoning: \"{reasoning}\"")

        # Step 5: Commit or READ
        if safe_chars >= args.min_commit_chars:
            commit_ext = greedy_ext[:safe_chars]
            committed_norm += commit_ext
            decisions.append(("WRITE", commit_ext))
            _vlog(verbose_log_file,
                  f"  [Step 5] WRITE  extension=\"{commit_ext}\"  "
                  f"committed=\"{committed_norm}\"")
        else:
            decisions.append(("READ", ""))
            reason = "judge_zero" if safe_chars == 0 else "below_min_chars"
            _vlog(verbose_log_file,
                  f"  [Step 5] READ ({reason}, safe_chars={safe_chars})  "
                  f"committed=\"{committed_norm}\"")

        if args.save_details:
            all_details.append({
                "observed": accumulated_source,
                "futures": futures,
                "translations": valid_translations,
                "greedy_full": greedy_full,
                "greedy_ext": greedy_ext,
                "candidate_extensions": candidate_extensions,
                "safe_chars": safe_chars,
                "reasoning": reasoning,
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
            "version": "v3_llm_judge",
            "num_candidates": args.num_candidates,
            "future_tokens": args.future_tokens,
            "min_commit_chars": args.min_commit_chars,
            "min_observed_words": args.min_observed_words,
        },
    }

    for k in ["audio", "n_frames", "speaker", "src_lang", "tgt_lang"]:
        if k in row:
            result[k] = row[k]

    if args.save_details:
        result["details"] = all_details

    return result


# ===================================================================
# Data-Parallel I/O
# ===================================================================

def iter_assigned_rows(input_tsv: str, task_id: int, num_tasks: int):
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if row_idx % num_tasks != task_id:
                continue
            yield row_idx, row


def count_assigned_rows(input_tsv: str, task_id: int, num_tasks: int) -> int:
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        total = sum(1 for _ in f) - 1
    if total <= task_id:
        return 0
    return int(math.ceil((total - task_id) / num_tasks))


def get_one_row_by_id(input_tsv: str, utt_id: str, id_column: str = "id") -> Optional[Tuple[int, Dict[str, str]]]:
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

    sample_params = SamplingParams(
        temperature=args.sample_temperature,
        max_tokens=args.future_tokens,
    )
    translate_params = SamplingParams(
        temperature=0.0,
        max_tokens=500,
        repetition_penalty=1.1,
    )
    judge_params = SamplingParams(
        temperature=0.0,
        max_tokens=200,
    )

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
        f"  M={args.num_candidates}, "
        f"future_tokens={args.future_tokens}, "
        f"min_commit_chars={args.min_commit_chars}\n"
        f"  commit_method=llm_judge (v3)"
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
                    args, sample_params, translate_params, judge_params,
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
