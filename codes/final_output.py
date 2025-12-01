import os
import json
from tqdm import tqdm
from collections import defaultdict


def parse_utt_id(utt):
    """utt_en000_00000000_0017 → (en000, 00000000)"""
    parts = utt.split("_")
    return parts[1], parts[2]


def load_streaming_dataset(stream_root):
    """
    Load all streaming dataset JSON into a dict:
    utt_id → { ... streaming json ... }
    """
    index = {}

    for dirpath, _, files in os.walk(stream_root):
        for f in files:
            if f.endswith(".json"):
                p = os.path.join(dirpath, f)
                data = json.load(open(p))
                index[data["utt_id"]] = data

    print(f"Loaded {len(index)} streaming items.")
    return index


def build_jsonl(metricx_file, stream_root, output_root):
    """
    Build structured trajectory-style dataset:
    en000/00000000/low_latency.jsonl
    ...
    """

    os.makedirs(output_root, exist_ok=True)

    # Load streaming dataset mapping: utt_id → segmentation
    stream_index = load_streaming_dataset(stream_root)

    # Accumulator
    dataset = defaultdict(lambda: {
        "low": [],
        "medium": [],
        "high": []
    })

    print(f"\n📥 Loading MetricX QE file: {metricx_file}")
    lines = open(metricx_file).readlines()
    print(f"Found {len(lines)} lines.\n")

    for line in tqdm(lines, desc="Parsing QE output"):
        item = json.loads(line)

        meta = item["metadata"]
        utt = meta["utt_id"]
        latency = meta["latency"]

        lang, pqid = parse_utt_id(utt)

        # ----- 关键：从 streaming dataset 取 segmentation -----
        if utt not in stream_index:
            continue

        seg = stream_index[utt]  # segmentation dict

        src_key = f"source_{latency}_latency"
        tgt_key = f"target_{latency}_latency"

        if src_key not in seg or tgt_key not in seg:
            continue

        traj_entry = {
            "utt_id": utt,
            "latency": latency,
            "source": seg[src_key],      # LIST
            "target": seg[tgt_key]       # LIST
        }

        dataset[(lang, pqid)][latency].append(traj_entry)

    # ----- write output -----
    print("\n💾 Writing JSONL output...\n")

    for (lang, pqid), groups in dataset.items():
        base = os.path.join(output_root, lang, pqid)
        os.makedirs(base, exist_ok=True)

        for latency in ["low", "medium", "high"]:
            out_path = os.path.join(base, f"{latency}_latency.jsonl")

            with open(out_path, "w", encoding="utf-8") as f:
                for entry in groups[latency]:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"Saved {out_path}  ({len(groups[latency])} entries)")

    print("\n🎉 DONE! Trajectory-style dataset built.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--metricx_jsonl", required=True,
                        help="Sentence-level MetricX QE output")

    parser.add_argument("--stream_dir", required=True,
                        help="Streaming dataset directory")

    parser.add_argument("--output_dir", required=True,
                        help="Where to store structured dataset")

    args = parser.parse_args()

    build_jsonl(
        metricx_file=args.metricx_jsonl,
        stream_root=args.stream_dir,
        output_root=args.output_dir
    )
