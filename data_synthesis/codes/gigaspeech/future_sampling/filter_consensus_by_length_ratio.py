#!/usr/bin/env python3
"""Keep the decoded utterances whose length ratio to the reference lies in [--min-ratio-ref, --max-ratio-ref].

metrics.length_ratio_ref is prediction characters over reference characters, whitespace ignored; an
utterance without a finite ratio is kept. Kept JSONs are copied into --output-dir, which is first emptied
of JSONs from an earlier run, and one report line per utterance goes to --report.
"""
import argparse
import collections
import json
import math
import shutil
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Decoded JSONs that passed the QE filter.")
    parser.add_argument("--output-dir", required=True, help="Receives the kept decoded JSONs.")
    parser.add_argument("--min-ratio-ref", type=float, required=True)
    parser.add_argument("--max-ratio-ref", type=float, required=True)
    parser.add_argument("--report", required=True, help="Per-utterance report jsonl.")
    args = parser.parse_args()

    records = []
    for path in sorted(Path(args.input_dir).resolve().glob("*.json")):
        decoded = json.loads(path.read_text(encoding="utf-8"))
        records.append((path, decoded["utt_id"], decoded["metrics"]["length_ratio_ref"]))
    finite = sorted(ratio for _, _, ratio in records if math.isfinite(ratio))
    if finite:
        n = len(finite)
        print(f"length_ratio_ref over {len(records)} JSONs: n={n} min={finite[0]:.3f} p25={finite[n // 4]:.3f} "
              f"median={finite[n // 2]:.3f} p75={finite[n * 3 // 4]:.3f} p95={finite[int(n * 0.95)]:.3f} "
              f"max={finite[-1]:.3f} mean={mean(finite):.3f}")
    else:
        print(f"length_ratio_ref over {len(records)} JSONs: no finite values")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_json in output_dir.glob("*.json"):
        stale_json.unlink()
    dropped = collections.Counter()
    with open(args.report, "w", encoding="utf-8") as report:
        for path, utt_id, ratio in records:
            if not math.isfinite(ratio):
                reason = "pass"  # no reference-length evidence either way
            elif ratio > args.max_ratio_ref:
                reason = f"ratio_ref>{args.max_ratio_ref}"
            elif ratio < args.min_ratio_ref:
                reason = f"ratio_ref<{args.min_ratio_ref}"
            else:
                reason = "pass"
            report.write(json.dumps({"utt_id": utt_id, "path": str(path), "ratio_ref": ratio if math.isfinite(ratio) else None,
                                     "passed": reason == "pass", "reason": reason}, ensure_ascii=False) + "\n")
            if reason == "pass":
                shutil.copy2(path, output_dir / path.name)
            else:
                dropped[reason] += 1
    print(f"Kept {len(records) - sum(dropped.values())} / {len(records)} -> {output_dir}; dropped: {dict(dropped)}")


if __name__ == "__main__":
    main()
