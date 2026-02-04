import json
import os
import re
from tqdm import tqdm


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def collect_json_files(root):
    """Recursively collect all .json files in streaming dataset."""
    all_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".json"):
                all_files.append(os.path.join(dirpath, f))
    return sorted(all_files)


def join_zh_segments(seg_list):
    """Join Chinese segments into a full sentence."""
    # your chunks likely don't need spaces
    s = "".join(seg_list).strip()
    # compress accidental whitespace if any
    s = re.sub(r"\s+", " ", s).strip()
    return s


def convert_dataset(stream_root, output_file):
    """
    Convert streaming dataset into sentence-level MetricX QE input.jsonl
    For each latency, combine target segments into a single hypothesis.
    """
    json_files = collect_json_files(stream_root)
    print(f"Found {len(json_files)} streaming JSON files.\n")

    kept = 0
    skipped = 0

    with open(output_file, "w", encoding="utf8") as out:
        for path in tqdm(json_files, desc="Converting"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                skipped += 1
                continue

            # sanity check: must be our streaming json
            if not isinstance(data, dict) or "utt_id" not in data or "original_text" not in data:
                skipped += 1
                continue

            utt_id = data["utt_id"]
            full_source_raw = data["original_text"]
            full_source = normalize_text(full_source_raw)  # optional but recommended

            # latency levels + keys
            if "source_offline" in data and "target_offline" in data:
                levels = [("offline", "source_offline", "target_offline")]
            else:
                levels = [
                    ("low", "source_low_latency", "target_low_latency"),
                    ("medium", "source_medium_latency", "target_medium_latency"),
                    ("high", "source_high_latency", "target_high_latency"),
                ]

            for latency, src_key, tgt_key in levels:
                if src_key not in data or tgt_key not in data:
                    continue

                target_segments = data[tgt_key]
                if not isinstance(target_segments, list) or len(target_segments) == 0:
                    continue

                full_hypothesis = join_zh_segments(target_segments)
                if not full_hypothesis:
                    continue

                example = {
                    "source": full_source,
                    "hypothesis": full_hypothesis,
                    "reference": "",
                    "metadata": {
                        "utt_id": utt_id,
                        "latency": latency,
                        "num_segments": len(target_segments),
                        "source_raw": full_source_raw,   # keep for traceability
                        "stream_json": path,
                    }
                }

                out.write(json.dumps(example, ensure_ascii=False) + "\n")
                kept += 1

    print(f"\n{'='*60}")
    print(f"🎉 Done!")
    print(f"Output: {output_file}")
    print(f"Kept examples: {kept}")
    print(f"Skipped files : {skipped}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--stream_dir", required=True, help="Root of streaming dataset")
    parser.add_argument("--output", required=True, help="Output JSONL file")

    args = parser.parse_args()
    convert_dataset(args.stream_dir, args.output)
