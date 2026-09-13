#!/usr/bin/env python3
"""Keep the decoded utterances whose every aligned sentence has MetricX QE <= --threshold.

An utterance passes only when its rows cover sentence indices 0..total_sentences-1 and no score exceeds
the threshold (lower MetricX is better). Passing decoded JSONs are copied into --output-dir, which is
first emptied of JSONs from an earlier run, and one report line per utterance goes to --report.
"""
import argparse
import collections
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metricx-output", required=True, help="Merged per-sentence MetricX output jsonl.")
    parser.add_argument("--output-dir", required=True, help="Receives the passing decoded JSONs.")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--report", required=True, help="Per-utterance report jsonl.")
    args = parser.parse_args()

    scores_by_utt = collections.defaultdict(list)  # utt_id -> [(sentence_idx, score)] in file order
    total_sentences, stream_json = {}, {}
    rows = 0
    with open(args.metricx_output, encoding="utf-8") as metricx_output:
        for line in metricx_output:
            if line.strip():
                row = json.loads(line)
                utt_id = row["metadata"]["utt_id"]
                scores_by_utt[utt_id].append((int(row["metadata"]["sentence_idx"]), float(row["prediction"])))
                total_sentences.setdefault(utt_id, int(row["metadata"]["total_sentences"]))
                stream_json[utt_id] = Path(row["metadata"]["stream_json"])
                rows += 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_json in output_dir.glob("*.json"):
        stale_json.unlink()
    kept = dropped_incomplete = dropped_over_threshold = 0
    with open(args.report, "w", encoding="utf-8") as report:
        for utt_id, scores in scores_by_utt.items():
            is_complete = set(range(total_sentences[utt_id])) <= {sentence_idx for sentence_idx, _ in scores}
            worst_score = max(score for _, score in scores) if is_complete else None
            passed = is_complete and worst_score <= args.threshold
            report.write(json.dumps({
                "utt_id": utt_id, "total_expected": total_sentences[utt_id], "num_rows_seen": len(scores),
                "is_complete": is_complete, "all_scored": is_complete, "worst_score": worst_score, "passed": passed,
                "per_sentence": [{"sentence_idx": sentence_idx, "score": score}
                                 for sentence_idx, score in sorted(scores, key=lambda item: item[0])],
            }, ensure_ascii=False) + "\n")
            if not is_complete:
                dropped_incomplete += 1
            elif not passed:
                dropped_over_threshold += 1
            else:
                shutil.copy2(stream_json[utt_id], output_dir / stream_json[utt_id].name)
                kept += 1
    print(f"{rows} MetricX rows, {len(scores_by_utt)} utterances: kept {kept}; dropped {dropped_over_threshold} "
          f"with a sentence QE > {args.threshold} and {dropped_incomplete} with unscored sentences -> {output_dir}")


if __name__ == "__main__":
    main()
