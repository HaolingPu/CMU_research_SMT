import json
import os
from tqdm import tqdm


def collect_json_files(root):
    """Recursively collect all .json files in streaming dataset."""
    all_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".json"):
                all_files.append(os.path.join(dirpath, f))
    return sorted(all_files)


def join_segments(seg_list):
    """Join segments into a full sentence."""
    return "".join(seg_list).strip()


def convert_dataset(stream_root, output_file):
    """
    Convert streaming dataset into *Sentence-level* MetricX QE input.jsonl
    For each latency (low/medium/high), we combine segments into a single sentence.
    """
    out = open(output_file, "w", encoding="utf8")

    json_files = collect_json_files(stream_root)
    print(f"Found {len(json_files)} streaming JSON files.\n")

    for path in tqdm(json_files, desc="Converting"):
        data = json.load(open(path, "r"))

        utt_id = data["utt_id"]
        full_source = data["original_text"]

        for latency in ["low", "medium", "high"]:
            src_key = f"source_{latency}_latency"
            tgt_key = f"target_{latency}_latency"

            if src_key not in data or tgt_key not in data:
                continue

            target_segments = data[tgt_key]

            # ⚠️ If segmentation failed → no valid target
            if not isinstance(target_segments, list) or len(target_segments) == 0:
                continue

            # === Step 1: join segments into full-sentence translation ===
            full_hypothesis = join_segments(target_segments)

            if not full_hypothesis:
                continue

            # === Step 2: build MetricX QE entry ===
            example = {
                "source": full_source,
                "hypothesis": full_hypothesis,
                "reference": "",
                "metadata": {
                    "utt_id": utt_id,
                    "latency": latency,
                    "num_segments": len(target_segments)
                }
            }

            out.write(json.dumps(example, ensure_ascii=False) + "\n")

    out.close()
    print(f"\n{'='*60}")
    print(f"🎉 Done!")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--stream_dir", required=True,
                       help="Root of streaming dataset")
    parser.add_argument("--output", required=True,
                       help="Output JSONL file")

    args = parser.parse_args()
    convert_dataset(args.stream_dir, args.output)
