#!/usr/bin/env python3
"""
Build final trajectory jsonl dataset for GigaSpeech from:
  - MetricX filtered jsonl
  - streaming trajectory JSON files

Output layout:
  <output_dir>/<recording_id>/<latency>_latency.jsonl
where recording_id is utt_id without the last "_<idx>" suffix.
"""

import argparse
import json
import os
from typing import Dict, Optional, Tuple

from tqdm import tqdm


def recording_id_from_utt(utt_id: str) -> str:
    if "_" not in utt_id:
        return utt_id
    return utt_id.rsplit("_", 1)[0]

# 新增：扁平输出文件名规则
def flat_output_path(output_root: str, utt_id: str, latency: str) -> str:
    # Keep offline file names as <utt_id>.json for Salami workflow.
    # For non-offline latencies, add suffix to avoid collisions per utt_id.
    if latency == "offline":
        fn = f"{utt_id}.json"
    else:
        fn = f"{utt_id}_{latency}.json"
    return os.path.join(output_root, fn)


def build_stream_path_index(stream_root: str, only_recording: Optional[str] = None) -> Dict[str, str]:
    """
    Build utt_id -> stream json path index.
    """
    print(f"\nIndexing streaming dataset paths from: {stream_root}")
    if only_recording:
        print(f"  only_recording = {only_recording}")

    index: Dict[str, str] = {}
    bad = 0

    for dirpath, _, files in os.walk(stream_root):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                utt_id = data.get("utt_id")
                if not isinstance(utt_id, str) or not utt_id.strip():
                    bad += 1
                    continue
                utt_id = utt_id.strip()

                if only_recording and recording_id_from_utt(utt_id) != only_recording:
                    continue
                index[utt_id] = p
            except Exception:
                bad += 1

    print(f"Indexed {len(index)} utt paths. Bad/failed: {bad}\n")
    return index


def open_writers(output_root: str, append: bool = False):
    """
    No longer caches file handles to avoid hitting the OS open-file-descriptor
    limit (ulimit -n) when there are many unique (recording_id, latency) pairs.
    Each write opens in append mode and closes immediately.
    On the first visit to a (recording_id, latency) pair, the file is truncated
    unless append=True.
    """
    writers: Dict[Tuple[str, str], any] = {}  # kept only to track seen pairs
    os.makedirs(output_root, exist_ok=True)

    def get_writer(recording_id: str, latency: str):
        key = (recording_id, latency)
        out_dir = os.path.join(output_root, recording_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{latency}_latency.jsonl")
        # Truncate on first visit (unless appending); afterwards always append.
        if not append and key not in writers:
            open(out_path, "w", encoding="utf-8").close()
        writers[key] = True
        return open(out_path, "a", encoding="utf-8")

    return writers, get_writer


def build_jsonl(
    metricx_file: str,
    stream_root: str,
    output_root: str,
    only_recording: Optional[str] = None,
    max_lines: Optional[int] = None,
    append: bool = False,
    flat_output: bool = False,
) -> None:
    print("\n============================================================")
    print("Building final GigaSpeech trajectory dataset")
    print("============================================================\n")

    if not os.path.exists(metricx_file):
        raise FileNotFoundError(f"MetricX file not found: {metricx_file}")

    stream_path_index = build_stream_path_index(stream_root, only_recording=only_recording)
    writers, get_writer = open_writers(output_root, append=append)

    processed = 0
    skipped = 0
    read_lines = 0

    try:
        with open(metricx_file, "r", encoding="utf-8") as f:
            it = tqdm(f, desc="Processing MetricX entries", total=max_lines if max_lines else None)
            for line in it:
                if max_lines is not None and read_lines >= max_lines:
                    break
                read_lines += 1

                line = line.strip()
                if not line:
                    skipped += 1
                    continue

                try:
                    item = json.loads(line)
                    meta = item.get("metadata", {})
                    utt_id = meta.get("utt_id")
                    latency = meta.get("latency")
                    if not isinstance(utt_id, str) or not utt_id.strip():
                        skipped += 1
                        continue
                    utt_id = utt_id.strip()

                    if latency not in {"low", "medium", "high", "offline"}:
                        skipped += 1
                        continue

                    recording_id = recording_id_from_utt(utt_id)
                    if only_recording and recording_id != only_recording:
                        skipped += 1
                        continue

                    p = stream_path_index.get(utt_id)
                    if not p:
                        skipped += 1
                        continue

                    with open(p, "r", encoding="utf-8") as sf:
                        seg = json.load(sf)

                    if latency == "offline":
                        src_key = "source_offline"
                        tgt_key = "target_offline"
                    else:
                        src_key = f"source_{latency}_latency"
                        tgt_key = f"target_{latency}_latency"

                    if src_key not in seg or tgt_key not in seg:
                        skipped += 1
                        continue

                    out_item = {
                        "utt_id": utt_id,
                        "latency": latency,
                        "source": seg[src_key],
                        "target": seg[tgt_key],
                    }

                    if flat_output:
                        os.makedirs(output_root, exist_ok=True)
                        out_path = flat_output_path(output_root, utt_id, latency)
                        with open(out_path, "w", encoding="utf-8") as wf:
                            json.dump(out_item, wf, ensure_ascii=False, indent=2)
                    else:
                        with get_writer(recording_id, latency) as wf:
                            wf.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                    processed += 1

                except Exception:
                    skipped += 1
                    continue
    finally:
        pass  # files are closed after each write; nothing to clean up

    print("\n============================================================")
    print("DONE")
    print("============================================================")
    print(f"Read lines       : {read_lines}")
    print(f"Processed entries: {processed}")
    print(f"Skipped entries  : {skipped}")
    print(f"Output directory : {output_root}")
    print("============================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metricx_jsonl", required=True, help="MetricX filtered jsonl.")
    parser.add_argument("--stream_dir", required=True, help="Streaming dataset directory.")
    parser.add_argument("--output_dir", required=True, help="Output directory for final jsonl dataset.")
    parser.add_argument("--only-recording", default=None, help="Debug filter, e.g. POD1000000010")
    parser.add_argument("--max-lines", type=int, default=None, help="Only read first N MetricX lines.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output files instead of overwrite per run.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help=(
            "Write per-utt JSON directly under output_dir. "
            "offline -> <utt_id>.json; others -> <utt_id>_<latency>.json"
        ),
    )
    args = parser.parse_args()

    build_jsonl(
        metricx_file=args.metricx_jsonl,
        stream_root=args.stream_dir,
        output_root=args.output_dir,
        only_recording=args.only_recording,
        max_lines=args.max_lines,
        append=args.append,
        flat_output=args.flat_output,

    )


if __name__ == "__main__":
    main()