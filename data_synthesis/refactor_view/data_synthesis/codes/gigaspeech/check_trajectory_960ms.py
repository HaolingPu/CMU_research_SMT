#!/usr/bin/env python3
"""
Check that src_trajectory chunks align with 960ms windows from MFA TextGrid.

Uses word-level assignment: each MFA word is assigned to chunk k where
  word.end <= (k+1) * chunk_seconds  =>  k = ceil(end/chunk_seconds) - 1.
Expected trajectory = one chunk per 960ms window, words in order.

Note: If your trajectory was built by multi_trajectory_gigaspeech.py (LLM chunks
matched to MFA), chunks are by *LLM segment* end time, so word-based check here
may differ; this script checks strict word-based 960ms alignment.

Input: --mfa-dir (TextGrid root) + --input-tsv (id, src_trajectory) or --input-json.
Output: ok/fail per utterance; optional --report JSON, --verbose for diffs.
"""

import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import tgt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check src_trajectory vs 960ms windows from MFA TextGrid."
    )
    p.add_argument("--mfa-dir", required=True, help="MFA TextGrid directory (recursive).")
    p.add_argument(
        "--input-tsv",
        default=None,
        help="Manifest TSV with id column and src_trajectory column (or trajectory from JSON).",
    )
    p.add_argument(
        "--input-json",
        default=None,
        help="Single JSON file with src_trajectory / source_future_sampling / source_low_latency.",
    )
    p.add_argument(
        "--input-dir",
        default=None,
        help="Directory of JSON files (e.g. llm_output); check each up to --max-items.",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="With --input-dir: max number of JSON files to check (e.g. 100).",
    )
    p.add_argument(
        "--utt-id",
        default=None,
        help="With --input-tsv: check only this utterance id.",
    )
    p.add_argument(
        "--id-column",
        default="id",
        help="TSV column for utterance id.",
    )
    p.add_argument(
        "--trajectory-column",
        default="src_trajectory",
        help="TSV column for trajectory (list of chunk strings).",
    )
    p.add_argument(
        "--chunk-ms",
        type=int,
        default=960,
        help="Chunk duration in ms (default: 960).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Max rows to check when using --input-tsv (debug).",
    )
    p.add_argument(
        "--report",
        default=None,
        help="Write report JSON to this path.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-chunk diff for mismatches.",
    )
    return p.parse_args()


def normalize_word(w: str) -> str:
    s = (w or "").lower().strip()
    s = re.sub(r"[^a-z0-9']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def trajectory_to_word_lists(trajectory: List[str]) -> List[List[str]]:
    """Split each chunk string into normalized words."""
    out: List[List[str]] = []
    for chunk in trajectory:
        words = []
        for w in (chunk or "").split():
            n = normalize_word(w)
            if n:
                words.append(n)
        out.append(words)
    return out


def load_word_alignment(textgrid_path: str) -> Tuple[List[Dict[str, Any]], float]:
    tg = tgt.read_textgrid(textgrid_path)
    audio_duration = float(tg.end_time)
    words = tg.get_tier_by_name("words").intervals
    out = []
    for w in words:
        raw = (w.text or "").strip()
        if not raw:
            continue
        out.append({
            "word": raw,
            "start": float(w.start_time),
            "end": float(w.end_time),
        })
    return out, audio_duration


def assign_words_to_960ms_chunks(
    mfa_words: List[Dict[str, Any]],
    chunk_seconds: float,
) -> List[List[str]]:
    """
    Assign each MFA word to chunk k by same rule as multi_trajectory:
    word belongs to chunk k iff word.end <= (k+1)*chunk_seconds (first such k).
    So k = ceil(end / chunk_seconds) - 1, clamped to 0.
    Returns list of chunks; each chunk is list of word strings in order.
    """
    if not mfa_words:
        return []

    chunk_seconds = float(chunk_seconds)
    eps = 1e-9
    chunks: Dict[int, List[str]] = {}

    for w in mfa_words:
        end = w["end"]
        # k such that (k+1)*chunk_seconds >= end  =>  k = ceil(end/chunk_seconds) - 1
        k = max(0, int(math.ceil((end - eps) / chunk_seconds)) - 1)
        if k not in chunks:
            chunks[k] = []
        chunks[k].append(w["word"])

    max_k = max(chunks.keys()) if chunks else -1
    return [chunks.get(i, []) for i in range(max_k + 1)]


def parse_list_column(raw: Any) -> List[str]:
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    try:
        parsed = __import__("ast").literal_eval(raw)
    except Exception:
        return [raw] if raw else []
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed]
    return [str(parsed).strip()] if str(parsed).strip() else []


def list_textgrids(root: str) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".TextGrid"):
                path = os.path.join(dirpath, fn)
                utt_id = os.path.basename(path)[: -len(".TextGrid")]
                if utt_id not in idx:
                    idx[utt_id] = path
    return idx


def compare_chunks(
    traj_word_lists: List[List[str]],
    expected_word_lists: List[List[str]],
) -> Tuple[bool, List[str]]:
    """Compare trajectory chunks vs expected (from MFA 960ms). Return (ok, list of messages)."""
    issues: List[str] = []
    n_t = len(traj_word_lists)
    n_e = len(expected_word_lists)
    if n_t != n_e:
        issues.append(f"chunk_count_mismatch: trajectory={n_t}, expected_from_mfa={n_e}")

    n = min(n_t, n_e)
    for k in range(n):
        traj_w = traj_word_lists[k]
        exp_w = expected_word_lists[k]
        if traj_w != exp_w:
            issues.append(
                f"chunk_{k}: trajectory {len(traj_w)} words vs expected {len(exp_w)} words"
            )
            issues.append(f"  trajectory[{k}]: {traj_w[:15]}{'...' if len(traj_w) > 15 else ''}")
            issues.append(f"  expected[{k}]:  {exp_w[:15]}{'...' if len(exp_w) > 15 else ''}")

    return len(issues) == 0, issues


def check_one(
    utt_id: str,
    trajectory: List[str],
    textgrid_path: str,
    chunk_ms: int,
    verbose: bool,
) -> Dict[str, Any]:
    chunk_seconds = chunk_ms / 1000.0
    mfa_words, audio_duration = load_word_alignment(textgrid_path)
    expected_chunks = assign_words_to_960ms_chunks(mfa_words, chunk_seconds)
    traj_word_lists = trajectory_to_word_lists(trajectory)
    expected_word_lists = [[normalize_word(w) for w in ch] for ch in expected_chunks]

    ok, issues = compare_chunks(traj_word_lists, expected_word_lists)
    result = {
        "utt_id": utt_id,
        "ok": ok,
        "trajectory_chunks": len(trajectory),
        "expected_chunks": len(expected_chunks),
        "audio_duration_sec": round(audio_duration, 3),
        "textgrid_path": textgrid_path,
    }
    if not ok:
        result["issues"] = issues
        if verbose:
            result["traj_sample"] = traj_word_lists[:5]
            result["expected_sample"] = expected_word_lists[:5]
    return result


def main() -> None:
    args = parse_args()
    if args.chunk_ms <= 0:
        raise ValueError("--chunk-ms must be > 0")

    mfa_index = list_textgrids(args.mfa_dir)
    print(f"chunk_ms={args.chunk_ms}, mfa TextGrids={len(mfa_index)}")

    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        utt_id = data.get("utt_id", os.path.basename(args.input_json).replace(".json", ""))
        trajectory = (
            data.get("src_trajectory")
            or data.get("source_future_sampling")
            or data.get("source_low_latency")
            or data.get("source_medium_latency")
            or data.get("source_high_latency")
        )
        if trajectory is None:
            raise ValueError(f"No trajectory list in {args.input_json}")
        if isinstance(trajectory, str):
            trajectory = parse_list_column(trajectory)
        tg_path = mfa_index.get(utt_id)
        if not tg_path:
            print(f"ERROR: no TextGrid for utt_id={utt_id} in {args.mfa_dir}")
            return
        res = check_one(utt_id, trajectory, tg_path, args.chunk_ms, args.verbose)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump({"results": [res]}, f, ensure_ascii=False, indent=2)
        return

    if args.input_dir:
        # Batch check: scan dir for *.json, up to --max-items
        json_files: List[str] = []
        for dirpath, _, files in os.walk(args.input_dir):
            for fn in files:
                if fn.endswith(".json"):
                    json_files.append(os.path.join(dirpath, fn))
        json_files.sort()
        if args.max_items:
            json_files = json_files[: args.max_items]
        print(f"Checking {len(json_files)} JSON files from {args.input_dir}")

        results = []
        ok_count = 0
        miss_count = 0
        fail_count = 0
        for jpath in json_files:
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            utt_id = data.get("utt_id") or os.path.basename(jpath).replace(".json", "")
            trajectory = (
                data.get("src_trajectory")
                or data.get("source_future_sampling")
                or data.get("source_low_latency")
                or data.get("source_medium_latency")
                or data.get("source_high_latency")
            )
            if trajectory is None:
                miss_count += 1
                results.append({"utt_id": utt_id, "ok": False, "issues": ["no_trajectory_in_json"]})
                continue
            if isinstance(trajectory, str):
                trajectory = parse_list_column(trajectory)
            if not trajectory:
                miss_count += 1
                results.append({"utt_id": utt_id, "ok": False, "issues": ["empty_trajectory"]})
                continue
            tg_path = mfa_index.get(utt_id)
            if not tg_path:
                miss_count += 1
                results.append({"utt_id": utt_id, "ok": False, "issues": ["missing_textgrid"]})
                continue
            try:
                res = check_one(utt_id, trajectory, tg_path, args.chunk_ms, args.verbose)
                results.append(res)
                if res["ok"]:
                    ok_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                results.append({"utt_id": utt_id, "ok": False, "issues": [str(e)]})

        print(f"\n========== Summary ==========")
        print(f"  Total checked: {len(results)}")
        print(f"  OK (match 960ms): {ok_count}")
        print(f"  Fail (mismatch): {fail_count}")
        print(f"  Missing/skip:    {miss_count}")
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(
                    {"chunk_ms": args.chunk_ms, "summary": {"ok": ok_count, "fail": fail_count, "missing": miss_count}, "results": results},
                    f, ensure_ascii=False, indent=2,
                )
            print(f"  Report: {args.report}")
        return

    if not args.input_tsv:
        raise ValueError("Provide --input-tsv, --input-json, or --input-dir")

    rows: List[Tuple[int, Dict[str, str]]] = []
    with open(args.input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            uid = str(row.get(args.id_column, "")).strip()
            if args.utt_id and uid != args.utt_id:
                continue
            rows.append((row_idx, row))
            if args.utt_id:
                break
            if args.max_rows and len(rows) >= args.max_rows:
                break

    results: List[Dict[str, Any]] = []
    ok_count = 0
    miss_count = 0
    fail_count = 0

    for row_idx, row in rows:
        utt_id = str(row.get(args.id_column, "")).strip() or f"row_{row_idx}"
        trajectory = parse_list_column(row.get(args.trajectory_column))
        if not trajectory:
            print(f"[SKIP] {utt_id}: empty trajectory")
            miss_count += 1
            continue
        tg_path = mfa_index.get(utt_id)
        if not tg_path:
            print(f"[MISS] {utt_id}: no TextGrid in --mfa-dir")
            miss_count += 1
            results.append({"utt_id": utt_id, "ok": False, "issues": ["missing_textgrid"]})
            continue
        try:
            res = check_one(utt_id, trajectory, tg_path, args.chunk_ms, args.verbose)
            results.append(res)
            if res["ok"]:
                ok_count += 1
            else:
                fail_count += 1
                print(f"[FAIL] {utt_id}: {res['issues'][:3]}")
        except Exception as e:
            fail_count += 1
            results.append({"utt_id": utt_id, "ok": False, "issues": [str(e)]})
            print(f"[ERR]  {utt_id}: {e}")

    print(f"\n========== Summary ==========")
    print(f"  Total checked: {len(results)}")
    print(f"  OK (match 960ms): {ok_count}")
    print(f"  Fail (mismatch): {fail_count}")
    print(f"  Missing/skip:    {miss_count}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(
                {"chunk_ms": args.chunk_ms, "summary": {"ok": ok_count, "fail": fail_count, "missing": miss_count}, "results": results},
                f, ensure_ascii=False, indent=2,
            )
        print(f"  Report: {args.report}")


if __name__ == "__main__":
    main()
