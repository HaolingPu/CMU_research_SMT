#!/usr/bin/env python3
"""Concatenate the per-shard SEGALE alignments (shard_NN/system/aligned_spacy_system.jsonl) into one file."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", required=True, help="Directory with shard_NN/system/aligned_spacy_system.jsonl.")
    parser.add_argument("--output", required=True, help="Merged jsonl.")
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    doc_ids = set()
    with output.open("w", encoding="utf-8") as merged:
        for shard_id in range(args.num_shards):
            # Opening a missing shard raises: every shard must be aligned before the merge.
            shard_path = Path(args.shards_root) / f"shard_{shard_id:02d}" / "system" / "aligned_spacy_system.jsonl"
            for line in shard_path.read_text(encoding="utf-8").splitlines():
                if line:
                    merged.write(line + "\n")
                    doc_ids.add(json.loads(line)["doc_id"])
                    rows += 1
    print(f"Merged {rows} rows ({len(doc_ids)} docs) from {args.num_shards} shards -> {output}")


if __name__ == "__main__":
    main()
