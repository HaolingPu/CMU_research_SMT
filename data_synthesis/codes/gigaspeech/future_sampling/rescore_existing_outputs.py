#!/usr/bin/env python3
"""Rescore existing output JSONs in place using a frozen reference dataset."""

import argparse
import csv
import json
import math
import os
from collections import Counter
from typing import Dict, Iterable, List, Tuple


DEFAULT_REFERENCE_TSV = (
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/"
    "eval_datasets/train_xl_case_robust_asr_filtered_frozen_llm_reference.tsv"
)

DEFAULT_ROOTS = [
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_final",
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_future_sampling_thinking_gemini",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rescore existing output JSONs with a frozen reference.")
    p.add_argument(
        "--reference-tsv",
        default=DEFAULT_REFERENCE_TSV,
        help="TSV with `id` and `llm_reference_text` columns.",
    )
    p.add_argument(
        "--roots",
        nargs="*",
        default=DEFAULT_ROOTS,
        help="Root directories to scan recursively for per-utterance JSON outputs.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats without overwriting JSON files.",
    )
    return p.parse_args()


def load_reference_map(tsv_path: str) -> Dict[str, str]:
    refs: Dict[str, str] = {}
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            utt_id = str(row.get("id", "")).strip()
            ref = str(row.get("llm_reference_text", "")).strip()
            if utt_id and ref:
                refs[utt_id] = ref
    return refs


def iter_json_paths(roots: Iterable[str]) -> Iterable[str]:
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if not name.endswith(".json"):
                    continue
                yield os.path.join(dirpath, name)


def compute_laal(
    source_chunks: List[str],
    target_deltas: List[str],
    actions: List[str],
    reference: str,
) -> float:
    timeline: List[int] = []
    source_read = 0

    for chunk, delta, action in zip(source_chunks, target_deltas, actions):
        words_in_chunk = len(str(chunk).strip().split()) if str(chunk).strip() else 0
        source_read += words_in_chunk
        if action == "WRITE" and str(delta).strip():
            for _ in str(delta).strip():
                timeline.append(source_read)

    y = "".join(d for d in target_deltas if d)
    y_len = len(y)
    yref_len = len(str(reference).replace(" ", ""))
    x_len = sum(len(str(c).strip().split()) for c in source_chunks if str(c).strip())
    if y_len == 0 or x_len == 0 or yref_len == 0:
        return float("nan")

    denom = max(y_len, yref_len)
    if denom <= 0 or len(timeline) == 0:
        return float("nan")

    total_lagging = 0.0
    for i in range(1, denom + 1):
        d_i = timeline[i - 1] if i <= len(timeline) else x_len
        d_star_i = (i - 1) * x_len / denom
        total_lagging += (d_i - d_star_i)
    return total_lagging / denom


def _char_tokens_zh(text: str) -> List[str]:
    return [c for c in str(text) if not c.isspace()]


def compute_bleu_char(
    hypothesis: str,
    reference: str,
    max_order: int = 4,
    smooth: bool = True,
) -> float:
    hyp = _char_tokens_zh(hypothesis)
    ref = _char_tokens_zh(reference)
    hyp_len = len(hyp)
    ref_len = len(ref)
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
        clipped = 0
        for ng, cnt in hyp_ngrams.items():
            clipped += min(cnt, ref_ngrams.get(ng, 0))
        if smooth:
            p_n = (clipped + 1.0) / (total + 1.0)
        else:
            if clipped == 0:
                return 0.0
            p_n = clipped / total
        precisions.append(p_n)

    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_len / hyp_len))

    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / eff_order)
    return bleu * 100.0


def rescore_file(path: str, refs: Dict[str, str], dry_run: bool = False) -> Tuple[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    utt_id = str(data.get("utt_id", "")).strip()
    if not utt_id:
        return "skip", f"{path}: missing utt_id"
    if "error" in data and "source_future_sampling" not in data:
        return "skip", f"{path}: error-only json"

    ref = refs.get(utt_id, "")
    if not ref:
        return "missing_ref", f"{path}: missing frozen reference for {utt_id}"

    source_chunks = data.get("source_future_sampling")
    target_deltas = data.get("target_future_sampling")
    actions = data.get("actions")
    if not isinstance(source_chunks, list) or not isinstance(target_deltas, list) or not isinstance(actions, list):
        return "skip", f"{path}: missing trajectory fields"

    system_output = str(data.get("system_output_text", "")).strip()
    if not system_output:
        system_output = "".join(str(x) for x in target_deltas if x)
        data["system_output_text"] = system_output

    laal_error = None
    bleu_error = None
    try:
        laal_value = compute_laal(source_chunks, target_deltas, actions, ref)
        bleu_value = compute_bleu_char(system_output, ref)
    except Exception as e:
        laal_value = float("nan")
        bleu_value = float("nan")
        laal_error = str(e)
        bleu_error = str(e)

    metrics = dict(data.get("metrics", {}) or {})
    metrics.update({
        "laal_text": laal_value,
        "laal_reference_mode": "frozen_llm_reference",
        "bleu_char": bleu_value,
        "bleu_reference_mode": "frozen_llm_reference",
        "effective_source_chunks": sum(1 for c in source_chunks if str(c).strip()),
        "system_output_chars": len(system_output),
        "reference_chars": len(ref.replace(" ", "")) if ref else 0,
        "laal_error": laal_error,
        "bleu_char_error": bleu_error,
    })
    data["reference_text"] = ref
    data["laal_reference_text"] = ref
    data["metrics"] = metrics

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return "ok", path


def main() -> None:
    args = parse_args()
    refs = load_reference_map(args.reference_tsv)
    print(f"[Refs] Loaded {len(refs)} frozen references from {args.reference_tsv}")

    counts = Counter()
    by_root = Counter()
    for path in sorted(iter_json_paths(args.roots)):
        status, msg = rescore_file(path, refs, dry_run=args.dry_run)
        counts[status] += 1
        if status == "ok":
            matched_root = None
            for root in args.roots:
                if path.startswith(root):
                    matched_root = root
                    break
            by_root[matched_root or "<other>"] += 1
        elif status in {"missing_ref"}:
            print(msg)

    print("[Done] Rescore summary:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    if by_root:
        print("[Done] Overwritten by root:")
        for key in sorted(by_root):
            print(f"  {key}: {by_root[key]}")


if __name__ == "__main__":
    main()
