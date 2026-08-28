#!/usr/bin/env python3
"""Merge per-utterance LA JSON outputs into one JSONL for inspection."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List


def natural_key(p: Path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)]


def collect_jsons(root: Path) -> List[Path]:
    nested = sorted(root.glob("job_*/task_*/*.json"), key=natural_key)
    direct = sorted(root.glob("task_*/*.json"), key=natural_key)
    return nested if nested else direct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-docs", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.input_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    files = collect_jsons(root)
    if args.num_docs is not None:
        files = files[: args.num_docs]

    written = 0
    skipped = 0
    with output.open("w", encoding="utf-8") as fout:
        for path in files:
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                skipped += 1
                continue
            if not isinstance(obj, dict) or "error" in obj:
                skipped += 1
                continue
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"Input root : {root}")
    print(f"Found      : {len(files)}")
    print(f"Written    : {written}")
    print(f"Skipped    : {skipped}")
    print(f"Output     : {output}")


if __name__ == "__main__":
    main()
