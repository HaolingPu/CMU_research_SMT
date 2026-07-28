#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


def collect_json_files(root: str) -> List[Path]:
    out: List[Path] = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            out.append(Path(dirpath) / fn)
    return sorted(out)


def iter_consensus_examples(root: str) -> Iterable[Dict[str, object]]:
    for path in collect_json_files(root):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if "error" in data:
            continue

        utt_id = str(data.get("utt_id", "")).strip()
        source = str(data.get("source_full_text", "")).strip()
        hypothesis = str(data.get("prediction", "")).strip()
        reference = str(data.get("reference_text", "")).strip()
        if not utt_id or not source or not hypothesis:
            continue

        yield {
            "source": source,
            "hypothesis": hypothesis,
            "reference": reference,
            "metadata": {
                "utt_id": utt_id,
                "latency": "future_sampling",
                "stream_json": str(path),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert consensus decoding JSON outputs to MetricX QE input.jsonl."
    )
    parser.add_argument("--input-dir", required=True, help="Consensus experiment directory.")
    parser.add_argument("--output", required=True, help="Output MetricX input jsonl.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for item in iter_consensus_examples(args.input_dir):
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Input dir : {args.input_dir}")
    print(f"Output    : {output_path}")
    print(f"Examples  : {kept}")


if __name__ == "__main__":
    main()
