#!/usr/bin/env python3
"""Build a one-JSON-per-utterance symlink view of a decode root.

Nothing under --decode-root is modified or deleted. For every utt_id that
appears in more than one task_*/per_utt directory, exactly one copy is kept
according to --prefer (a comma-separated list of task directory names in
priority order; unlisted directories rank after listed ones, ties broken by
earliest mtime). The kept file is symlinked into
<out-root>/<same task dir>/per_utt/<utt>.json so downstream stages that glob
task_*/per_utt/*.json see a clean root.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--prefer", default="task_12,task_13,task_14,task_15",
                    help="task dirs that win duplicate conflicts, highest priority first")
    ap.add_argument("--report", default=None, help="optional JSON report path")
    args = ap.parse_args()

    root = os.path.abspath(args.decode_root)
    out = os.path.abspath(args.out_root)
    prefer = [p.strip() for p in args.prefer.split(",") if p.strip()]
    rank = {name: i for i, name in enumerate(prefer)}

    by_utt: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in glob.glob(os.path.join(root, "task_*", "per_utt", "*.json")):
        task = os.path.basename(os.path.dirname(os.path.dirname(path)))
        by_utt[os.path.basename(path)].append((task, path))

    kept: dict[str, tuple[str, str]] = {}
    dropped: list[dict[str, str]] = []
    for utt, copies in by_utt.items():
        copies.sort(key=lambda tp: (rank.get(tp[0], len(rank)), os.path.getmtime(tp[1])))
        kept[utt] = copies[0]
        for task, path in copies[1:]:
            dropped.append({"utt": utt, "dropped_task": task, "kept_task": copies[0][0]})

    os.makedirs(out, exist_ok=True)
    made = 0
    for utt, (task, path) in kept.items():
        d = os.path.join(out, task, "per_utt")
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, utt)
        if os.path.lexists(link):
            if os.path.realpath(link) == os.path.realpath(path):
                continue
            os.unlink(link)
        os.symlink(path, link)
        made += 1

    # verify every kept link resolves to readable JSON with matching utt_id
    bad = 0
    for utt, (task, path) in kept.items():
        try:
            with open(os.path.join(out, task, "per_utt", utt), encoding="utf-8") as h:
                if str(json.load(h)["utt_id"]) + ".json" != utt:
                    bad += 1
        except Exception:
            bad += 1

    per_task_dropped: dict[str, int] = defaultdict(int)
    for d in dropped:
        per_task_dropped[d["dropped_task"]] += 1
    summary = {
        "decode_root": root, "out_root": out, "prefer": prefer,
        "total_files": sum(len(v) for v in by_utt.values()),
        "unique_utts": len(by_utt), "links_created": made,
        "duplicates_dropped": len(dropped), "dropped_by_task": dict(per_task_dropped),
        "bad_links": bad,
    }
    print(json.dumps(summary, indent=1))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as h:
            json.dump({"summary": summary, "dropped": dropped}, h, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
