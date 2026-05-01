#!/usr/bin/env python3
"""Build SEGALE input (system.jsonl + ref.jsonl) from consensus json dir, sharded.

Reads job_*/task_*/<utt>.json under --consensus-root, sorts by utt_id (natural),
takes the first --num-docs, splits into --num-shards contiguous chunks, and
writes one (system.jsonl, ref.jsonl) pair per shard. SEGALE convention: each
src_text_full[seg_id] becomes a row; the full prediction goes in seg 0 of
system.jsonl, the full reference goes in seg 0 of ref.jsonl, all later segs
have empty tgt -- segale-align then redistributes.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List


def natural_key(p: Path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)]


def collect_consensus_jsons(root: Path) -> List[Path]:
    """Find consensus JSONs whether root is a parent (with job_* subdirs) or a job_* itself."""
    nested = sorted(root.glob("job_*/task_*/*.json"), key=natural_key)
    direct = sorted(root.glob("task_*/*.json"), key=natural_key)
    return nested if nested else direct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus-root", required=True,
                    help="Dir containing job_*/task_*/<utt>.json")
    ap.add_argument("--out-root", required=True,
                    help="Output base; writes <out-root>/shards/shard_NN/{system,ref}.jsonl")
    ap.add_argument("--num-docs", type=int, default=50000)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--sys-id", default="consensus_run",
                    help="Logical system identifier written into system.jsonl rows.")
    args = ap.parse_args()

    cons_root = Path(args.consensus_root).resolve()
    out_root = Path(args.out_root).resolve()
    shards_root = out_root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    files = collect_consensus_jsons(cons_root)
    print(f"Found {len(files)} consensus jsons under {cons_root}")
    if len(files) < args.num_docs:
        print(f"WARNING: only {len(files)} available; using all of them")
    files = files[: args.num_docs]
    total = len(files)
    print(f"Selected {total} for SEGALE")

    chunk = (total + args.num_shards - 1) // args.num_shards
    manifest = {
        "consensus_root": str(cons_root),
        "out_root": str(out_root),
        "num_docs": total,
        "num_shards": args.num_shards,
        "sys_id": args.sys_id,
        "shard_size": chunk,
        "shards": [],
    }

    rows_total = 0
    skipped_no_src = 0
    for sid in range(args.num_shards):
        shard_dir = shards_root / f"shard_{sid:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        sys_path = shard_dir / "system.jsonl"
        ref_path = shard_dir / "ref.jsonl"

        sub = files[sid * chunk : (sid + 1) * chunk]
        n_docs = 0
        n_rows = 0
        with sys_path.open("w", encoding="utf-8") as fsys, ref_path.open("w", encoding="utf-8") as fref:
            for path in sub:
                try:
                    d = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(d, dict) or "error" in d:
                    continue
                utt_id = str(d.get("utt_id", "")).strip()
                if not utt_id:
                    continue
                src_full = d.get("src_text_full")
                if not isinstance(src_full, list):
                    src_full = [str(d.get("source_full_text", "")).strip()]
                src_full = [str(x).strip() for x in src_full if str(x).strip()]
                if not src_full:
                    skipped_no_src += 1
                    continue
                pred = str(d.get("prediction", "")).strip()
                refer = str(d.get("reference_text", "")).strip()
                for seg_id, src in enumerate(src_full):
                    sys_row = {
                        "doc_id": utt_id,
                        "sys_id": args.sys_id,
                        "seg_id": seg_id,
                        "src": src,
                        "tgt": pred if seg_id == 0 else "",
                    }
                    ref_row = {
                        "doc_id": utt_id,
                        "seg_id": seg_id,
                        "src": src,
                        "tgt": refer if seg_id == 0 else "",
                    }
                    fsys.write(json.dumps(sys_row, ensure_ascii=False) + "\n")
                    fref.write(json.dumps(ref_row, ensure_ascii=False) + "\n")
                    n_rows += 1
                n_docs += 1

        rows_total += n_rows
        manifest["shards"].append({
            "shard_id": sid,
            "shard_dir": str(shard_dir),
            "system_file": str(sys_path),
            "ref_file": str(ref_path),
            "docs": n_docs,
            "rows": n_rows,
        })
        print(f"shard_{sid:02d}: {n_docs} docs, {n_rows} rows -> {shard_dir}")

    manifest["total_rows"] = rows_total
    manifest["skipped_no_src"] = skipped_no_src
    with (out_root / "shard_manifest.json").open("w", encoding="utf-8") as fmani:
        json.dump(manifest, fmani, ensure_ascii=False, indent=2)
    print(f"\nManifest: {out_root / 'shard_manifest.json'}")
    print(f"Total rows : {rows_total}")
    print(f"Skipped no-src docs: {skipped_no_src}")


if __name__ == "__main__":
    main()
