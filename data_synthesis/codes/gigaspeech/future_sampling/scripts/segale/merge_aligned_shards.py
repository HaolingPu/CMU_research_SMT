#!/usr/bin/env python3
"""Merge per-shard aligned_spacy_system.jsonl into one file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-root", required=True,
                    help="Dir with shard_NN/ subdirs (each having system/aligned_spacy_system.jsonl).")
    ap.add_argument("--output", required=True, help="Merged jsonl output path")
    ap.add_argument("--num-shards", type=int, default=8)
    args = ap.parse_args()

    shards_root = Path(args.shards_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    docs = set()
    missing = []
    with output.open("w", encoding="utf-8") as fout:
        for sid in range(args.num_shards):
            shard_aligned = shards_root / f"shard_{sid:02d}" / "system" / "aligned_spacy_system.jsonl"
            if not shard_aligned.is_file():
                missing.append(str(shard_aligned))
                continue
            with shard_aligned.open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    fout.write(line + "\n")
                    try:
                        d = json.loads(line)
                        docs.add(d.get("doc_id", ""))
                    except Exception:
                        pass
                    total += 1

    print(f"Merged: {total} rows from {args.num_shards - len(missing)}/{args.num_shards} shards")
    print(f"Unique docs: {len(docs)}")
    if missing:
        print(f"MISSING shards ({len(missing)}):")
        for m in missing:
            print(f"  - {m}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
