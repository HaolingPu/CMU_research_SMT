#!/usr/bin/env python3
"""Convert vecalign-aligned consensus outputs to per-sentence MetricX QE input.

Reads `aligned_spacy_system.jsonl` (vecalign output), groups rows by doc_id, and
writes one MetricX input row per (doc_id, segment) using the aligned (src, tgt)
pair as (source, hypothesis). The consensus json path is resolved by searching
job_*/task_*/<doc_id>.json under --consensus-root so the downstream
filter_consensus_by_metricx_qe_per_sentence.py can copy the original case.

This bypasses chunk-midpoint assignment, which misaligns deltas when streaming
defers commits (see the horizon-filter alignment artifact we hit on top_5_k4).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_aligned_groups(path: Path) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("doc_id", "")).strip()
            if not doc_id:
                continue
            groups[doc_id].append(obj)
    for doc_id, rows in groups.items():
        rows.sort(key=lambda r: int(r.get("seg_id", 0)))
    return groups


def index_consensus_jsons(consensus_root: Path) -> Dict[str, Path]:
    """Find consensus JSONs. Supports three layouts:
      - <root>/job_*/task_*/<utt>.json           (early multi-job runs)
      - <root>/task_*/<utt>.json                 (single-job flat)
      - <root>/task_*/per_utt/<utt>.json         (J 40k production layout)
    """
    idx: Dict[str, Path] = {}
    root = consensus_root.resolve()
    for pattern in ("task_*/per_utt/*.json", "job_*/task_*/*.json", "task_*/*.json"):
        paths = list(root.glob(pattern))
        for path in paths:
            utt = path.stem
            idx.setdefault(utt, path.resolve())
        if idx:
            break
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert vecalign-aligned consensus outputs to MetricX per-sentence input.jsonl."
    )
    parser.add_argument("--aligned", required=True, help="Path to aligned_spacy_system.jsonl")
    parser.add_argument("--consensus-root", required=True,
                        help="Root containing job_*/task_*/<utt_id>.json")
    parser.add_argument("--output", required=True, help="Output MetricX input jsonl")
    parser.add_argument("--require-stream-json", action="store_true",
                        help="Skip docs whose consensus json is not found.")
    args = parser.parse_args()

    aligned = load_aligned_groups(Path(args.aligned))
    consensus_idx = index_consensus_jsons(Path(args.consensus_root))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    docs = 0
    docs_missing_stream = 0
    skipped_empty_pair = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for doc_id, segs in sorted(aligned.items()):
            stream_json = consensus_idx.get(doc_id)
            if stream_json is None:
                docs_missing_stream += 1
                if args.require_stream_json:
                    continue
            total = len(segs)
            for local_idx, seg in enumerate(segs):
                src = str(seg.get("src", "")).strip()
                tgt = str(seg.get("tgt", "")).strip()
                if not src and not tgt:
                    skipped_empty_pair += 1
                    continue
                row = {
                    "source": src,
                    "hypothesis": tgt,
                    "reference": str(seg.get("ref", "")).strip(),
                    "metadata": {
                        "utt_id": doc_id,
                        "sentence_idx": local_idx,
                        "total_sentences": total,
                        "latency": "future_sampling",
                        "stream_json": str(stream_json) if stream_json else "",
                        "aligner": "vecalign_spacy",
                        "aligned_seg_id": int(seg.get("seg_id", local_idx)),
                    },
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
            docs += 1

    print("===== Convert vecalign-aligned -> MetricX per-sentence input =====")
    print(f"Aligned input        : {args.aligned}")
    print(f"Consensus root       : {args.consensus_root}")
    print(f"Output               : {output_path}")
    print(f"Docs (utts)          : {docs}")
    print(f"Sentence rows        : {rows}")
    print(f"Empty (src+tgt) pairs skipped : {skipped_empty_pair}")
    print(f"Docs without consensus json   : {docs_missing_stream}"
          + (" (skipped)" if args.require_stream_json else " (kept w/ empty stream_json)"))


if __name__ == "__main__":
    main()
