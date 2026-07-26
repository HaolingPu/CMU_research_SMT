#!/usr/bin/env python3
"""Filter consensus JSONs by length_ratio metrics.

Reads per-utt consensus JSONs, examines `metrics.length_ratio_ref` and/or
`metrics.length_ratio_src`; keeps only those whose ratios are within the
specified bounds. `--dry-run` skips copying and just reports the distribution
plus what each threshold would keep.

Typical usage
-------------
Stack on top of the QE filter output:

  python filter_consensus_by_length_ratio.py \\
    --input-dir  <QE-filtered dir> \\
    --output-dir <final dir> \\
    --max-ratio-ref 1.5 \\
    --clean-output

Inspect distribution first:

  python filter_consensus_by_length_ratio.py \\
    --input-dir <QE-filtered dir> \\
    --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


def iter_json_files(root: str) -> Iterable[Path]:
    for p in Path(root).rglob("*.json"):
        yield p


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def load_ratios(path: Path) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None, {}
    if not isinstance(data, dict) or "error" in data:
        return None, None, {}
    metrics = data.get("metrics") or {}
    return (
        _finite(metrics.get("length_ratio_ref")),
        _finite(metrics.get("length_ratio_src")),
        {
            "utt_id": data.get("utt_id"),
            "pred_chars": metrics.get("pred_chars"),
            "ref_chars": metrics.get("ref_chars"),
            "src_words": metrics.get("src_words"),
        },
    )


def distribution(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    xs = sorted(values)
    n = len(xs)
    return {
        "n": n,
        "min": xs[0],
        "p25": xs[n // 4],
        "median": xs[n // 2],
        "p75": xs[n * 3 // 4],
        "p90": xs[int(n * 0.90)],
        "p95": xs[int(n * 0.95)],
        "p99": xs[int(n * 0.99)],
        "max": xs[-1],
        "mean": mean(xs),
    }


def fmt_dist(name: str, d: Dict[str, float]) -> str:
    if d.get("n", 0) == 0:
        return f"{name}: no values"
    return (
        f"{name}: n={d['n']}  min={d['min']:.3f}  p25={d['p25']:.3f}  "
        f"median={d['median']:.3f}  p75={d['p75']:.3f}  p90={d['p90']:.3f}  "
        f"p95={d['p95']:.3f}  p99={d['p99']:.3f}  max={d['max']:.3f}  "
        f"mean={d['mean']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter consensus JSONs by length_ratio.")
    parser.add_argument("--input-dir", required=True, help="Directory of per-utt consensus JSONs.")
    parser.add_argument("--output-dir", default=None, help="Destination; required unless --dry-run.")
    parser.add_argument("--max-ratio-ref", type=float, default=None,
                        help="Drop if length_ratio_ref > this.")
    parser.add_argument("--min-ratio-ref", type=float, default=None,
                        help="Drop if length_ratio_ref < this (catch under-translation).")
    parser.add_argument("--max-ratio-src", type=float, default=None,
                        help="Drop if length_ratio_src > this (useful when reference unavailable).")
    parser.add_argument("--min-ratio-src", type=float, default=None,
                        help="Drop if length_ratio_src < this.")
    parser.add_argument("--require-ratio-ref", action="store_true",
                        help="Drop if length_ratio_ref is missing/NaN (default: ignore missing).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't copy; only report distribution + per-threshold counts.")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--report", default=None, help="Optional per-utt jsonl report.")
    args = parser.parse_args()

    if not args.dry_run and not args.output_dir:
        parser.error("--output-dir required unless --dry-run")

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        parser.error(f"input-dir not a directory: {input_dir}")

    ratios_ref: List[float] = []
    ratios_src: List[float] = []
    records: List[Dict[str, Any]] = []

    for path in iter_json_files(str(input_dir)):
        rr, rs, info = load_ratios(path)
        if rr is not None:
            ratios_ref.append(rr)
        if rs is not None:
            ratios_src.append(rs)
        records.append({"path": path, "ratio_ref": rr, "ratio_src": rs, "info": info})

    total = len(records)

    print("===== Length-Ratio Filter =====")
    print(f"Input dir   : {input_dir}")
    print(f"Total JSONs : {total}")
    print(fmt_dist("length_ratio_ref", distribution(ratios_ref)))
    print(fmt_dist("length_ratio_src", distribution(ratios_src)))
    print()

    if args.dry_run:
        def count_above(xs: List[float], t: float) -> int:
            return sum(1 for x in xs if x > t)

        print("--- Threshold preview (length_ratio_ref) ---")
        for t in [1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]:
            dropped = count_above(ratios_ref, t)
            kept = len(ratios_ref) - dropped
            pct = 100 * kept / max(1, len(ratios_ref))
            print(f"  <= {t:4.1f}: keep {kept:4d} / {len(ratios_ref):4d} ({pct:5.1f}%)")
        print()
        print("--- Threshold preview (length_ratio_src) ---")
        for t in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            dropped = count_above(ratios_src, t)
            kept = len(ratios_src) - dropped
            pct = 100 * kept / max(1, len(ratios_src))
            print(f"  <= {t:4.1f}: keep {kept:4d} / {len(ratios_src):4d} ({pct:5.1f}%)")
        return

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        for p in output_dir.glob("*.json"):
            p.unlink()

    def check(rec: Dict[str, Any]) -> Tuple[bool, str]:
        rr = rec["ratio_ref"]
        rs = rec["ratio_src"]
        if args.require_ratio_ref and rr is None:
            return False, "ratio_ref_missing"
        if args.max_ratio_ref is not None and rr is not None and rr > args.max_ratio_ref:
            return False, f"ratio_ref>{args.max_ratio_ref}"
        if args.min_ratio_ref is not None and rr is not None and rr < args.min_ratio_ref:
            return False, f"ratio_ref<{args.min_ratio_ref}"
        if args.max_ratio_src is not None and rs is not None and rs > args.max_ratio_src:
            return False, f"ratio_src>{args.max_ratio_src}"
        if args.min_ratio_src is not None and rs is not None and rs < args.min_ratio_src:
            return False, f"ratio_src<{args.min_ratio_src}"
        return True, "pass"

    kept = 0
    dropped_reasons: Dict[str, int] = {}
    report_rows: List[Dict[str, Any]] = []

    for rec in records:
        ok, reason = check(rec)
        if args.report is not None:
            report_rows.append({
                "utt_id": (rec["info"] or {}).get("utt_id"),
                "path": str(rec["path"]),
                "ratio_ref": rec["ratio_ref"],
                "ratio_src": rec["ratio_src"],
                "passed": ok,
                "reason": reason,
            })
        if not ok:
            dropped_reasons[reason] = dropped_reasons.get(reason, 0) + 1
            continue
        dst = output_dir / rec["path"].name
        shutil.copy2(rec["path"], dst)
        kept += 1

    if args.report is not None:
        with Path(args.report).open("w", encoding="utf-8") as fh:
            for row in report_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Output dir  : {output_dir}")
    print(f"Kept        : {kept} / {total}")
    for reason, cnt in sorted(dropped_reasons.items()):
        print(f"  dropped[{reason}]: {cnt}")
    if args.report is not None:
        print(f"Report      : {args.report}")


if __name__ == "__main__":
    main()
