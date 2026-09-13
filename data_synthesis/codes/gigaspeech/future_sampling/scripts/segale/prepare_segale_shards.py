#!/usr/bin/env python3
"""Write SEGALE input shards (system.jsonl + ref.jsonl) from the decoded per-utterance JSONs.

Utterances under <consensus-root>/task_*/per_utt/ are sorted naturally by file name, the first --num-docs
are split into --num-shards contiguous shards, and every src_text_full unit becomes one row. The whole
prediction goes in segment 0 of system.jsonl and the whole reference in segment 0 of ref.jsonl; later
segments get an empty tgt, and segale-align redistributes the text across segments.
"""
import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus-root", required=True, help="Decode root containing task_*/per_utt/<utt_id>.json.")
    parser.add_argument("--out-root", required=True,
                        help="Receives shards/shard_NN/{system,ref}.jsonl and shard_manifest.json.")
    parser.add_argument("--num-docs", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--sys-id", required=True, help="System identifier written into system.jsonl rows.")
    args = parser.parse_args()

    consensus_root = Path(args.consensus_root).resolve()
    out_root = Path(args.out_root).resolve()
    paths = sorted(consensus_root.glob("task_*/per_utt/*.json"),
                   key=lambda path: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)])
    print(f"Found {len(paths)} consensus jsons under {consensus_root}; using the first {min(len(paths), args.num_docs)}")
    paths = paths[:args.num_docs]
    shard_size = (len(paths) + args.num_shards - 1) // args.num_shards
    manifest = {"consensus_root": str(consensus_root), "out_root": str(out_root), "num_docs": len(paths),
                "num_shards": args.num_shards, "sys_id": args.sys_id, "shard_size": shard_size, "shards": []}
    for shard_id in range(args.num_shards):
        shard_dir = out_root / "shards" / f"shard_{shard_id:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_paths = paths[shard_id * shard_size:(shard_id + 1) * shard_size]
        rows = 0
        with (shard_dir / "system.jsonl").open("w", encoding="utf-8") as system_file, \
                (shard_dir / "ref.jsonl").open("w", encoding="utf-8") as ref_file:
            for path in shard_paths:
                decoded = json.loads(path.read_text(encoding="utf-8"))
                prediction = decoded["prediction"].strip()
                reference = decoded["reference_text"].strip()
                for seg_id, unit in enumerate(unit.strip() for unit in decoded["src_text_full"]):
                    system_row = {"doc_id": decoded["utt_id"], "sys_id": args.sys_id, "seg_id": seg_id, "src": unit,
                                  "tgt": prediction if seg_id == 0 else ""}
                    ref_row = {"doc_id": decoded["utt_id"], "seg_id": seg_id, "src": unit,
                               "tgt": reference if seg_id == 0 else ""}
                    system_file.write(json.dumps(system_row, ensure_ascii=False) + "\n")
                    ref_file.write(json.dumps(ref_row, ensure_ascii=False) + "\n")
                    rows += 1
        manifest["shards"].append({"shard_id": shard_id, "shard_dir": str(shard_dir),
                                   "system_file": str(shard_dir / "system.jsonl"), "ref_file": str(shard_dir / "ref.jsonl"),
                                   "docs": len(shard_paths), "rows": rows})
        print(f"shard_{shard_id:02d}: {len(shard_paths)} docs, {rows} rows -> {shard_dir}")
    manifest["total_rows"] = sum(shard["rows"] for shard in manifest["shards"])
    with (out_root / "shard_manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
    print(f"Manifest: {out_root / 'shard_manifest.json'}; total rows {manifest['total_rows']}")


if __name__ == "__main__":
    main()
