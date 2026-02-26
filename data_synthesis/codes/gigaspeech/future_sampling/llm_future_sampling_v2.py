#!/usr/bin/env python3
"""
Future Source Sampling + Char-Level Trie Vote for Streaming MT.

v2: simplified from v1 -- removed token-level vote and logprob safety.
Uses embedding clustering + char-level trie vote as the uncertainty signal.

Algorithm (per 960ms chunk, flat utterance stream):
  1. Accumulate observed source words across the whole utterance.
  2. Sample M future source continuations (LLM, temp > 0).
  3. Translate (observed + each future) -> M Chinese translations (forced prefix = committed).
  4. Embed translation extensions via sentence-transformers, cluster with
     AgglomerativeClustering.  Select the largest semantic cluster.
  5. Char-level trie vote within the cluster -> consensus extension.
     - M_total adapts: cluster_size if cluster dominates (>= ratio*M), else M.
     - At each character position, count votes; denominator = M_total.
     - Shorter translations implicitly vote "stop extending".
     - Stop when best_char agreement < tau.
  6. Commit extension if len >= min_commit_chars, else READ.
  7. At utterance end, force-commit remainder.

Usage:
  python llm_future_sampling_v2.py \\
    --input-tsv MANIFEST.tsv \\
    --output-root OUTPUT_DIR \\
    --task-id $SLURM_ARRAY_TASK_ID \\
    --num-tasks 8

  Test / verbose (one utterance, print every step to stdout AND log file):
  python llm_future_sampling_v2.py --input-tsv MANIFEST.tsv --output-root OUT \\
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

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ===================================================================
# CLI Arguments
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Future source sampling + char trie vote for streaming MT data synthesis (v2)."
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
    p.add_argument("--tau", type=float, default=0.7,
                   help="Char trie vote threshold (fraction of M that must agree, 0.6-0.8).")
    p.add_argument("--min-observed-words", type=int, default=5,
                   help="Skip future sampling when observed source has fewer words than this.")
    p.add_argument("--embed-model",
                   default="paraphrase-multilingual-MiniLM-L12-v2",
                   help="Sentence-transformers model for embedding clustering.")
    p.add_argument("--cluster-threshold", type=float, default=0.25,
                   help="Cosine distance threshold for AgglomerativeClustering (lower = tighter clusters).")
    p.add_argument("--min-cluster-ratio", type=float, default=0.8,
                   help="If largest_cluster_size >= this * M, use cluster_size as M_total; else use M.")

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


# ===================================================================
# Char-Level Trie Vote
# ===================================================================

def char_trie_vote(
    translations: List[str],
    committed: str,
    tau: float,
    M_total: int,
    min_commit_chars: int = 2,
) -> Tuple[str, List[Dict]]:
    """
    Char-level trie vote for consensus extension.

    Each translation's extension (after stripping committed prefix) is a
    "path" in the trie.  At each character position we count votes;
    the denominator is always M_total (fixed), so shorter translations
    and invalid ones implicitly vote "stop extending".

    Returns (extension_text, vote_log).
    extension_text is "" if consensus length < min_commit_chars.
    """
    extensions = []
    for t in translations:
        if t.startswith(committed):
            extensions.append(t[len(committed):])

    if not extensions:
        return "", []

    result_chars: List[str] = []
    vote_log: List[Dict] = []
    alive = list(extensions)
    max_len = max(len(e) for e in extensions)

    for pos in range(max_len):
        char_votes: Counter = Counter()
        for e in alive:
            if pos < len(e):
                char_votes[e[pos]] += 1

        if not char_votes:
            break

        best_char, best_count = char_votes.most_common(1)[0]
        agreement = best_count / M_total

        vote_log.append({
            "pos": pos,
            "char": best_char,
            "count": best_count,
            "agreement": round(agreement, 3),
            "passed": agreement >= tau,
        })

        if agreement >= tau:
            result_chars.append(best_char)
            alive = [e for e in alive if pos < len(e) and e[pos] == best_char]
        else:
            break

    result = "".join(result_chars)
    if len(result) < min_commit_chars:
        return "", vote_log
    return result, vote_log


# ===================================================================
# Embedding Clustering
# ===================================================================

def embed_and_cluster(
    translations: List[str],
    committed: str,
    embed_model: SentenceTransformer,
    distance_threshold: float = 0.25,
) -> Tuple[List[int], Dict]:
    """
    Embed translation extensions and cluster semantically.

    Returns (selected_indices, info_dict) where selected_indices are
    indices into the original `translations` list belonging to the
    largest (or tightest, on tie) cluster.
    """
    extensions: List[str] = []
    valid_indices: List[int] = []
    for i, t in enumerate(translations):
        if t.startswith(committed):
            ext = t[len(committed):]
            if ext:
                extensions.append(ext)
                valid_indices.append(i)

    n = len(extensions)
    info: Dict[str, Any] = {}

    if n == 0:
        info.update(n_valid=0, n_clusters=0, cluster_sizes=[], selected_cluster=-1)
        return [], info

    if n == 1:
        info.update(n_valid=1, n_clusters=1, cluster_sizes=[1], selected_cluster=0)
        return valid_indices, info

    embeddings = embed_model.encode(extensions, normalize_embeddings=True)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    label_counts = Counter(labels)
    max_count = label_counts.most_common(1)[0][1]
    tied_labels = [lbl for lbl, cnt in label_counts.items() if cnt == max_count]

    if len(tied_labels) == 1:
        best_label = tied_labels[0]
    else:
        best_label = _pick_tightest_cluster(embeddings, labels, tied_labels)

    selected = [valid_indices[j] for j, lbl in enumerate(labels) if lbl == best_label]

    info.update(
        n_valid=n,
        n_clusters=len(set(labels)),
        cluster_sizes=sorted(label_counts.values(), reverse=True),
        selected_cluster=int(best_label),
        selected_size=len(selected),
    )
    return selected, info


def _pick_tightest_cluster(
    embeddings: np.ndarray, labels: np.ndarray, candidate_labels: List[int],
) -> int:
    """Among clusters of equal size, pick the one with highest avg pairwise cosine similarity."""
    best_label = candidate_labels[0]
    best_sim = -1.0
    for lbl in candidate_labels:
        mask = labels == lbl
        vecs = embeddings[mask]
        if len(vecs) < 2:
            avg_sim = 1.0
        else:
            sim_matrix = vecs @ vecs.T
            n = len(vecs)
            avg_sim = float((sim_matrix.sum() - n) / (n * (n - 1)))
        if avg_sim > best_sim:
            best_sim = avg_sim
            best_label = lbl
    return best_label


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
    verbose_log_file: Optional[Any] = None,
) -> Dict[str, Any]:
    full_source_text = " ".join(sentences)
    n_chunks = len(trajectory)

    _vlog(verbose_log_file, f"\n{'#'*60}")
    _vlog(verbose_log_file, f"# Utterance: {utt_id}")
    _vlog(verbose_log_file, f"# Full text (EN): {full_source_text}")
    _vlog(verbose_log_file, f"# Trajectory chunks: {n_chunks}")
    _vlog(verbose_log_file, f"# Vote method: embed_cluster + char trie vote (v2)")
    _vlog(verbose_log_file, f"# tau={args.tau}, M={args.num_candidates}, "
          f"min_commit_chars={args.min_commit_chars}, "
          f"cluster_threshold={args.cluster_threshold}, "
          f"min_cluster_ratio={args.min_cluster_ratio}")
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

        # ----- Case 3: future sampling + char trie vote -----

        # Early-skip: not enough observed words for reliable future sampling
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
        n_valid = sum(1 for t in all_translations if t.startswith(committed_norm))
        _vlog(verbose_log_file,
              f"  [Step 2] Translations (valid prefix "
              f"{n_valid}/{len(all_translations)}):")
        for ti, t in enumerate(all_translations):
            prefix_ok = "✓" if t.startswith(committed_norm) else "✗"
            _vlog(verbose_log_file, f"    trans[{ti}] [{prefix_ok}]: \"{t}\"")

        # Step 3: Embedding clustering
        embed_model = getattr(args, "_embed_model", None)
        cluster_indices, cluster_info = embed_and_cluster(
            all_translations,
            committed_norm,
            embed_model,
            distance_threshold=args.cluster_threshold,
        )
        cluster_translations = [all_translations[i] for i in cluster_indices]
        cluster_size = len(cluster_translations)

        _vlog(verbose_log_file, "  [Step 3] Embedding clustering:")
        _vlog(verbose_log_file,
              f"    n_clusters={cluster_info.get('n_clusters', 0)}  "
              f"sizes={cluster_info.get('cluster_sizes', [])}  "
              f"selected_size={cluster_size}")

        if cluster_size == 0:
            decisions.append(("READ", ""))
            _vlog(verbose_log_file, "  [Step 5] READ (no valid cluster)")
            if args.save_details:
                all_details.append({
                    "observed": accumulated_source,
                    "futures": futures,
                    "translations": all_translations,
                    "cluster_info": cluster_info,
                    "reason": "no_cluster",
                    "action": "READ",
                })
            continue

        # Adaptive M_total
        M = args.num_candidates
        if cluster_size >= args.min_cluster_ratio * M:
            effective_m = cluster_size
        else:
            effective_m = M
        _vlog(verbose_log_file,
              f"    cluster_size={cluster_size}, "
              f"min_cluster_ratio={args.min_cluster_ratio}, "
              f"effective_M_total={effective_m}")

        # Step 4: Char-level trie vote within cluster
        commit_ext, vote_log = char_trie_vote(
            cluster_translations,
            committed_norm,
            tau=args.tau,
            M_total=effective_m,
            min_commit_chars=args.min_commit_chars,
        )

        _vlog(verbose_log_file, "  [Step 4] Char trie vote (within cluster):")
        for v in vote_log:
            tag = "PASS" if v["passed"] else "STOP"
            _vlog(verbose_log_file,
                  f"    pos={v['pos']:>2d}  char='{v['char']}'  "
                  f"count={v['count']}/{effective_m}  "
                  f"agreement={v['agreement']:.3f}  [{tag}]")
        if not vote_log:
            _vlog(verbose_log_file, "    (no extensions to vote on)")

        # Step 5: Commit or READ
        if commit_ext:
            committed_norm += commit_ext
            decisions.append(("WRITE", commit_ext))
            _vlog(verbose_log_file,
                  f"  [Step 5] WRITE  extension=\"{commit_ext}\"  "
                  f"committed=\"{committed_norm}\"")
        else:
            decisions.append(("READ", ""))
            reason = "below_min_chars" if vote_log and vote_log[-1]["passed"] else "no_consensus"
            _vlog(verbose_log_file,
                  f"  [Step 5] READ ({reason})  committed=\"{committed_norm}\"")

        if args.save_details:
            all_details.append({
                "observed": accumulated_source,
                "futures": futures,
                "translations": all_translations,
                "cluster_info": cluster_info,
                "cluster_translations": cluster_translations,
                "effective_m": effective_m,
                "vote_log": vote_log,
                "commit_ext": commit_ext,
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
            "version": "v2_embed_cluster_trie_vote",
            "num_candidates": args.num_candidates,
            "future_tokens": args.future_tokens,
            "tau": args.tau,
            "min_commit_chars": args.min_commit_chars,
            "min_observed_words": args.min_observed_words,
            "cluster_threshold": args.cluster_threshold,
            "min_cluster_ratio": args.min_cluster_ratio,
            "embed_model": args.embed_model,
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

    print(f"[Task {args.task_id}] Loading embedding model: {args.embed_model}")
    args._embed_model = SentenceTransformer(
        args.embed_model, device="cpu",
    )

    sample_params = SamplingParams(
        temperature=args.sample_temperature,
        max_tokens=args.future_tokens,
    )
    translate_params = SamplingParams(
        temperature=0.0,
        max_tokens=500,
        repetition_penalty=1.1,
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
        f"  M={args.num_candidates}, tau={args.tau}, "
        f"future_tokens={args.future_tokens}, "
        f"min_commit_chars={args.min_commit_chars}\n"
        f"  vote_method=embed_cluster + char_trie_vote (v2)\n"
        f"  embed_model={args.embed_model}, "
        f"cluster_threshold={args.cluster_threshold}, "
        f"min_cluster_ratio={args.min_cluster_ratio}"
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
