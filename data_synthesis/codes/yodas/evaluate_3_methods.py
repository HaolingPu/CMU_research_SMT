#!/usr/bin/env python3
"""
Prepare a manual-review bundle for comparing two final datasets.

What it does:
- Load two final dataset roots (EAST vs BASELINE style final_jsonl dirs).
- Find common utt_id per latency.
- Sample N examples per latency (default: low=10, medium=10, high=10).
- Export one CSV that includes:
  - utt_id / latency
  - final outputs from both methods
  - paths (+ existence flags) for streaming json, llm raw/merged json,
    mfa TextGrid, mfa wav/lab
  - optional snippets from streaming/llm json for quick inspection
"""

import argparse
import csv
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple


LAT_ORDER = ["low", "medium", "high", "offline"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build manual review CSV for two methods.")
    p.add_argument("--method-a-name", default="EAST")
    p.add_argument("--method-a-final-dir", required=True)
    p.add_argument("--method-b-name", default="BASELINE")
    p.add_argument("--method-b-final-dir", required=True)

    p.add_argument("--method-a-stream-dir", default=None)
    p.add_argument("--method-a-llm-dir", default=None)
    p.add_argument("--method-a-mfa-textgrid-dir", default=None)
    p.add_argument("--method-a-mfa-corpus-dir", default=None)

    p.add_argument("--method-b-stream-dir", default=None)
    p.add_argument("--method-b-llm-dir", default=None)
    p.add_argument("--method-b-mfa-textgrid-dir", default=None)
    p.add_argument("--method-b-mfa-corpus-dir", default=None)

    p.add_argument(
        "--latency-plan",
        default="low:10,medium:10,high:10",
        help="Comma-separated plan, e.g. low:10,medium:10,high:10",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-json", default=None, help="Optional sampled rows as JSON.")
    return p.parse_args()


def parse_latency_plan(s: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad latency-plan item: {item}")
        lat, n = item.split(":", 1)
        lat = lat.strip().lower()
        n = int(n.strip())
        if lat not in LAT_ORDER:
            raise ValueError(f"Unsupported latency in plan: {lat}")
        if n <= 0:
            raise ValueError(f"Sample count must be >0 in plan: {item}")
        out.append((lat, n))
    if not out:
        raise ValueError("Empty --latency-plan")
    return out


def read_jsonl_records(path: str, fallback_latency: Optional[str] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            utt_id = obj.get("utt_id")
            if not isinstance(utt_id, str) or not utt_id.strip():
                continue
            lat = obj.get("latency")
            if not isinstance(lat, str) or not lat.strip():
                lat = fallback_latency
            if not isinstance(lat, str) or not lat.strip():
                continue
            src = obj.get("source")
            tgt = obj.get("target")
            if not isinstance(src, list) or not isinstance(tgt, list):
                continue
            records.append(
                {
                    "utt_id": utt_id.strip(),
                    "latency": lat.strip().lower(),
                    "source": [str(x) for x in src],
                    "target": [str(x) for x in tgt],
                    "file": path,
                    "line_no": i,
                }
            )
    return records


def load_final_index(final_root: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Returns: idx[latency][utt_id] -> record
    """
    idx: Dict[str, Dict[str, Dict[str, Any]]] = {}
    dup = 0
    files_seen = 0
    for dirpath, _, filenames in os.walk(final_root):
        for fn in filenames:
            if not fn.endswith(".jsonl"):
                continue
            files_seen += 1
            p = os.path.join(dirpath, fn)
            fallback_latency = None
            if fn.endswith("_latency.jsonl"):
                fallback_latency = fn[: -len("_latency.jsonl")].lower()
            for rec in read_jsonl_records(p, fallback_latency=fallback_latency):
                lat = rec["latency"]
                if lat not in idx:
                    idx[lat] = {}
                uid = rec["utt_id"]
                if uid in idx[lat]:
                    dup += 1
                    continue
                idx[lat][uid] = rec
    print(f"Loaded final index from {final_root}: files={files_seen}, dups_skipped={dup}")
    for lat in sorted(idx.keys(), key=lambda x: LAT_ORDER.index(x) if x in LAT_ORDER else 999):
        print(f"  latency={lat:<7} utts={len(idx[lat])}")
    return idx


def recording_id_from_utt(utt_id: str) -> str:
    if "_" not in utt_id:
        return utt_id
    return utt_id.rsplit("_", 1)[0]


def resolve_candidate_path(root: Optional[str], utt_id: str, ext: str) -> str:
    if not root:
        return ""
    rec = recording_id_from_utt(utt_id)
    p1 = os.path.join(root, f"{utt_id}{ext}")
    p2 = os.path.join(root, rec, f"{utt_id}{ext}")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return p2  # return expected path for easier debugging


def read_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not path or (not os.path.exists(path)):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def get_stream_keys(latency: str) -> Tuple[str, str]:
    if latency == "offline":
        return "source_offline", "target_offline"
    return f"source_{latency}_latency", f"target_{latency}_latency"


def stringify_segments(v: Any, sep: str = " || ") -> str:
    if isinstance(v, list):
        return sep.join(str(x) for x in v)
    return ""


def build_rows(
    plan: List[Tuple[str, int]],
    idx_a: Dict[str, Dict[str, Dict[str, Any]]],
    idx_b: Dict[str, Dict[str, Dict[str, Any]]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rng = random.Random(args.seed)
    rows: List[Dict[str, Any]] = []

    for lat, k in plan:
        a = idx_a.get(lat, {})
        b = idx_b.get(lat, {})
        common = sorted(set(a.keys()) & set(b.keys()))
        if not common:
            print(f"[WARN] latency={lat}: no common utt_id between two methods.")
            continue

        actual_k = min(k, len(common))
        sampled = rng.sample(common, actual_k)
        print(f"latency={lat}: common={len(common)}, sampled={actual_k}")

        for utt_id in sampled:
            rec_a = a[utt_id]
            rec_b = b[utt_id]

            a_stream = resolve_candidate_path(args.method_a_stream_dir, utt_id, ".json")
            b_stream = resolve_candidate_path(args.method_b_stream_dir, utt_id, ".json")
            a_llm = resolve_candidate_path(args.method_a_llm_dir, utt_id, ".json")
            b_llm = resolve_candidate_path(args.method_b_llm_dir, utt_id, ".json")
            a_tg = resolve_candidate_path(args.method_a_mfa_textgrid_dir, utt_id, ".TextGrid")
            b_tg = resolve_candidate_path(args.method_b_mfa_textgrid_dir, utt_id, ".TextGrid")
            a_wav = resolve_candidate_path(args.method_a_mfa_corpus_dir, utt_id, ".wav")
            b_wav = resolve_candidate_path(args.method_b_mfa_corpus_dir, utt_id, ".wav")
            a_lab = resolve_candidate_path(args.method_a_mfa_corpus_dir, utt_id, ".lab")
            b_lab = resolve_candidate_path(args.method_b_mfa_corpus_dir, utt_id, ".lab")

            a_stream_obj = read_json_if_exists(a_stream)
            b_stream_obj = read_json_if_exists(b_stream)
            a_llm_obj = read_json_if_exists(a_llm)
            b_llm_obj = read_json_if_exists(b_llm)

            src_key, tgt_key = get_stream_keys(lat)
            row = {
                "utt_id": utt_id,
                "latency": lat,

                f"{args.method_a_name}_final_source": stringify_segments(rec_a["source"], sep=" || "),
                f"{args.method_a_name}_final_target": stringify_segments(rec_a["target"], sep=" || "),
                f"{args.method_a_name}_final_file": rec_a["file"],
                f"{args.method_a_name}_final_line_no": rec_a["line_no"],

                f"{args.method_b_name}_final_source": stringify_segments(rec_b["source"], sep=" || "),
                f"{args.method_b_name}_final_target": stringify_segments(rec_b["target"], sep=" || "),
                f"{args.method_b_name}_final_file": rec_b["file"],
                f"{args.method_b_name}_final_line_no": rec_b["line_no"],

                f"{args.method_a_name}_stream_path": a_stream,
                f"{args.method_a_name}_stream_exists": int(os.path.exists(a_stream)),
                f"{args.method_a_name}_stream_original_text": (a_stream_obj or {}).get("original_text", ""),
                f"{args.method_a_name}_stream_source": stringify_segments((a_stream_obj or {}).get(src_key, [])),
                f"{args.method_a_name}_stream_target": stringify_segments((a_stream_obj or {}).get(tgt_key, [])),

                f"{args.method_b_name}_stream_path": b_stream,
                f"{args.method_b_name}_stream_exists": int(os.path.exists(b_stream)),
                f"{args.method_b_name}_stream_original_text": (b_stream_obj or {}).get("original_text", ""),
                f"{args.method_b_name}_stream_source": stringify_segments((b_stream_obj or {}).get(src_key, [])),
                f"{args.method_b_name}_stream_target": stringify_segments((b_stream_obj or {}).get(tgt_key, [])),

                f"{args.method_a_name}_llm_path": a_llm,
                f"{args.method_a_name}_llm_exists": int(os.path.exists(a_llm)),
                f"{args.method_a_name}_llm_input": (a_llm_obj or {}).get("input", ""),

                f"{args.method_b_name}_llm_path": b_llm,
                f"{args.method_b_name}_llm_exists": int(os.path.exists(b_llm)),
                f"{args.method_b_name}_llm_input": (b_llm_obj or {}).get("input", ""),

                f"{args.method_a_name}_textgrid_path": a_tg,
                f"{args.method_a_name}_textgrid_exists": int(os.path.exists(a_tg)),
                f"{args.method_a_name}_wav_path": a_wav,
                f"{args.method_a_name}_wav_exists": int(os.path.exists(a_wav)),
                f"{args.method_a_name}_lab_path": a_lab,
                f"{args.method_a_name}_lab_exists": int(os.path.exists(a_lab)),

                f"{args.method_b_name}_textgrid_path": b_tg,
                f"{args.method_b_name}_textgrid_exists": int(os.path.exists(b_tg)),
                f"{args.method_b_name}_wav_path": b_wav,
                f"{args.method_b_name}_wav_exists": int(os.path.exists(b_wav)),
                f"{args.method_b_name}_lab_path": b_lab,
                f"{args.method_b_name}_lab_exists": int(os.path.exists(b_lab)),

                "manual_alignment_ok": "",
                "manual_translation_ok": "",
                "manual_notes": "",
            }
            rows.append(row)

    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    plan = parse_latency_plan(args.latency_plan)

    idx_a = load_final_index(args.method_a_final_dir)
    idx_b = load_final_index(args.method_b_final_dir)
    rows = build_rows(plan, idx_a, idx_b, args)

    if not rows:
        raise RuntimeError("No sampled rows generated. Check paths/latency-plan/intersection.")

    write_csv(args.out_csv, rows)
    print(f"\nSaved CSV: {args.out_csv}")
    print(f"Rows: {len(rows)}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON: {args.out_json}")


if __name__ == "__main__":
    main()



# python /data/user_data/haolingp/data_synthesis/codes/evaluate_3_methods.py \
#   --method-a-name EAST \
#   --method-a-final-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_refined_east/final_jsonl_refined_east \
#   --method-a-stream-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_refined_east/streaming_dataset \
#   --method-a-llm-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_refined_east/llm_output_merged \
#   --method-a-mfa-textgrid-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_refined_east/mfa_textgrid \
#   --method-a-mfa-corpus-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_refined_east/mfa_corpus \
#   --method-b-name BASELINE \
#   --method-b-final-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/final_jsonl_dataset \
#   --method-b-stream-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/streaming_dataset \
#   --method-b-llm-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/llm_output_merged \
#   --method-b-mfa-textgrid-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/mfa_textgrid \
#   --method-b-mfa-corpus-dir /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/mfa_corpus \
#   --latency-plan low:10,medium:10,high:10 \
#   --seed 42 \
#   --out-csv /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/manual_eval/east_vs_baseline_manual_review_30.csv \
#   --out-json /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/manual_eval/east_vs_baseline_manual_review_30.json
