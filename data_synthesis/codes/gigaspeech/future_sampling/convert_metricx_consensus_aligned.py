#!/usr/bin/env python3
"""Turn the merged SEGALE alignment into per-sentence MetricX QE input.

Rows of aligned_all.jsonl are grouped by doc_id and ordered by seg_id. Every aligned (src, tgt) pair
becomes one MetricX row whose metadata names the decoded JSON (stream_json) that the QE filter copies.
Aligned pairs avoid the misalignment of assigning streaming deltas to sentences by chunk midpoint.
"""
import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", required=True, help="Merged aligned_all.jsonl.")
    parser.add_argument("--consensus-root", required=True, help="Decode root containing task_*/per_utt/<utt_id>.json.")
    parser.add_argument("--output", required=True, help="MetricX input jsonl.")
    args = parser.parse_args()

    segments_by_doc = collections.defaultdict(list)
    for line in Path(args.aligned).read_text(encoding="utf-8").splitlines():
        if line.strip():
            segment = json.loads(line)
            segments_by_doc[segment["doc_id"]].append(segment)
    decoded_paths = {path.stem: path.resolve() for path in Path(args.consensus_root).resolve().glob("task_*/per_utt/*.json")}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = skipped_empty_pairs = 0
    with output.open("w", encoding="utf-8") as metricx_input:
        for doc_id, segments in sorted(segments_by_doc.items()):
            segments.sort(key=lambda segment: int(segment["seg_id"]))
            for sentence_idx, segment in enumerate(segments):
                source, hypothesis = segment["src"].strip(), segment["tgt"].strip()
                if not source and not hypothesis:
                    # The utterance keeps this index in total_sentences, so the QE filter will drop it as incomplete.
                    skipped_empty_pairs += 1
                else:
                    metricx_input.write(json.dumps({
                        "source": source,
                        "hypothesis": hypothesis,
                        "reference": segment["ref"].strip(),
                        "metadata": {"utt_id": doc_id, "sentence_idx": sentence_idx, "total_sentences": len(segments),
                                     "stream_json": str(decoded_paths[doc_id]), "aligned_seg_id": int(segment["seg_id"])},
                    }, ensure_ascii=False) + "\n")
                    rows += 1
    print(f"{len(segments_by_doc)} docs, {rows} sentence rows, {skipped_empty_pairs} empty (src+tgt) pairs skipped -> {output}")


if __name__ == "__main__":
    main()
