#!/usr/bin/env python3
"""SALAMI llm_output_merged -> consensus-format JSONs for SEGALE pipeline.

Reads per-utt merged JSONs (Source/Target chunks) plus manifest TSV
(for src_text_full sentence list), writes one consensus-format JSON per
utt under <out-root>/job_salami_<lang>/task_0/<utt>.json with fields:
  utt_id, prediction (joined target), src_text_full (list of sentences),
  reference_text, source_full_text, east_utt_id, east_latency, east_stream_json.

Only utts whose merged file has non-empty offline.Source/Target are emitted,
so this also acts as a downstream gate equivalent to "fix_llm_raw passed".
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm


SPACE_JOIN_LANGS = {"de", "en", "fr", "es"}


def join_target(segments: List[str], lang: str) -> str:
    sep = " " if (lang or "").lower() in SPACE_JOIN_LANGS else ""
    return sep.join(str(x) for x in segments).strip()


def parse_src_text_full(raw) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return None
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    except Exception:
        pass
    return [s] if s else None


def load_manifest_src_full(tsv_path: str) -> dict:
    out = {}
    csv.field_size_limit(sys.maxsize)
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, desc="load manifest src_text_full"):
            utt = row.get("id")
            if not utt:
                continue
            out[utt] = row.get("src_text_full", "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-dir", required=True,
                    help="Dir of llm_output_merged JSONs (post fix_llm_raw + adapt + merge)")
    ap.add_argument("--manifest-tsv", required=True)
    ap.add_argument("--stream-dir", required=True,
                    help="streaming_salami_dataset root (for east_stream_json field)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--sys-id", default="salami_ja")
    ap.add_argument("--latency", default="offline")
    ap.add_argument("--target-lang", default="ja")
    args = ap.parse_args()

    merged_dir = Path(args.merged_dir)
    out_dir = Path(args.out_root) / f"job_{args.sys_id}" / "task_0"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[salami2consensus] loading manifest src_text_full...")
    manifest_src_full = load_manifest_src_full(args.manifest_tsv)
    print(f"[salami2consensus] manifest entries: {len(manifest_src_full):,}")

    stream_dir = Path(args.stream_dir)
    files = sorted([p for p in merged_dir.glob("*.json")])
    print(f"[salami2consensus] merged files: {len(files):,}")

    ok = skipped_err = skipped_empty = skipped_no_src = 0
    for path in tqdm(files, desc="convert"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            skipped_err += 1
            continue
        if "error" in d:
            skipped_err += 1
            continue
        utt = d.get("utt_id") or path.stem
        offline = d.get("offline", {})
        src_chunks = offline.get("Source", []) or []
        tgt_chunks = offline.get("Target", []) or []
        if not src_chunks or not tgt_chunks:
            skipped_empty += 1
            continue
        raw_src_full = manifest_src_full.get(utt) or d.get("src_text_full")
        src_text_full = parse_src_text_full(raw_src_full)
        if not src_text_full:
            input_text = d.get("input", "").strip()
            if input_text:
                src_text_full = [input_text]
            else:
                skipped_no_src += 1
                continue

        prediction = join_target(tgt_chunks, args.target_lang)
        if not prediction:
            skipped_empty += 1
            continue
        source_full_text = " ".join(src_text_full)
        stream_json = stream_dir / utt[:11] / f"{utt}.json"

        obj = {
            "utt_id": utt,
            "prediction": prediction,
            "src_text_full": src_text_full,
            "source_full_text": source_full_text,
            "reference_text": "",
            "east_utt_id": utt,
            "east_latency": args.latency,
            "east_stream_json": str(stream_json),
            "target_lang": args.target_lang,
        }
        (out_dir / f"{utt}.json").write_text(
            json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        ok += 1

    print(f"\n[salami2consensus] done. ok={ok:,} "
          f"skipped_err={skipped_err:,} skipped_empty={skipped_empty:,} "
          f"skipped_no_src={skipped_no_src:,}")
    print(f"[salami2consensus] output: {out_dir}")


if __name__ == "__main__":
    main()
