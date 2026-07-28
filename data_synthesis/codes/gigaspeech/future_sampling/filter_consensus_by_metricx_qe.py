#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def iter_metricx_rows(path: Path) -> Iterable[Tuple[str, float, Path]]:
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {exc}") from exc

            if not isinstance(obj, dict):
                continue

            pred = obj.get("prediction")
            try:
                score = float(pred)
            except (TypeError, ValueError):
                continue
            if math.isnan(score):
                continue

            meta = obj.get("metadata") or {}
            utt_id = str(meta.get("utt_id", "")).strip()
            stream_json = str(meta.get("stream_json", "")).strip()
            if not utt_id or not stream_json:
                continue
            yield utt_id, score, Path(stream_json)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy consensus JSON outputs whose MetricX QE score is <= threshold."
    )
    parser.add_argument("--metricx-output", required=True, help="Path to metricx_output.jsonl")
    parser.add_argument("--output-dir", required=True, help="Destination directory for kept JSON files")
    parser.add_argument("--threshold", type=float, default=3.0, help="Keep if QE <= threshold")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing JSON files in output-dir before copying",
    )
    args = parser.parse_args()

    metricx_output = Path(args.metricx_output).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_output:
        for path in output_dir.glob("*.json"):
            path.unlink()

    total = 0
    kept = 0
    missing = 0
    duplicate = 0
    seen: Dict[str, Any] = {}

    for utt_id, score, src_json in iter_metricx_rows(metricx_output):
        total += 1
        if score > args.threshold:
            continue
        if utt_id in seen:
            duplicate += 1
            continue
        seen[utt_id] = True

        if not src_json.is_file():
            missing += 1
            continue

        dst_json = output_dir / src_json.name
        shutil.copy2(src_json, dst_json)
        kept += 1

    print("===== Filter Consensus by MetricX QE =====")
    print(f"MetricX output : {metricx_output}")
    print(f"Output dir     : {output_dir}")
    print(f"Threshold      : {args.threshold}")
    print(f"Total rows     : {total}")
    print(f"Kept JSONs     : {kept}")
    print(f"Missing JSONs  : {missing}")
    print(f"Duplicate utts : {duplicate}")


if __name__ == "__main__":
    main()
