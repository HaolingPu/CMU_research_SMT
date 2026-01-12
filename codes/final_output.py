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
    print(f"\n📥 Loading streaming dataset from: {stream_root}")
    index = {}

    for dirpath, _, files in os.walk(stream_root):
        for f in files:
            if f.endswith(".json"):
                p = os.path.join(dirpath, f)
                try:
                    data = json.load(open(p))
                    index[data["utt_id"]] = data
                except Exception as e:
                    print(f"  ⚠️ Error loading {f}: {e}")
                    continue

    print(f"✓ Loaded {len(index)} streaming items.\n")
    return index


def build_jsonl(metricx_file, stream_root, output_root):
    """
    Build structured trajectory-style dataset:
    en000/00000000/low_latency.json
    en000/00000000/medium_latency.json
    en000/00000000/high_latency.json
    """

    print(f"\n{'='*60}")
    print(f"Building final dataset")
    print(f"{'='*60}\n")

    os.makedirs(output_root, exist_ok=True)

    # Load streaming dataset mapping: utt_id → segmentation
    stream_index = load_streaming_dataset(stream_root)

    # Accumulator: (lang, pqid) → {latency → [entries]}
    dataset = defaultdict(lambda: {
        "low": [],
        "medium": [],
        "high": []
    })

    print(f"📥 Loading MetricX filtered file: {metricx_file}")
    
    if not os.path.exists(metricx_file):
        print(f"❌ Error: MetricX file not found!")
        return
    
    lines = open(metricx_file).readlines()
    print(f"✓ Found {len(lines)} filtered entries.\n")

    processed = 0
    skipped = 0

    for line in tqdm(lines, desc="Processing entries"):
        try:
            item = json.loads(line)

            meta = item["metadata"]
            utt = meta["utt_id"]
            latency = meta["latency"]

            lang, pqid = parse_utt_id(utt)

            # ----- 关键：从 streaming dataset 取 segmentation -----
            if utt not in stream_index:
                skipped += 1
                continue

            seg = stream_index[utt]  # segmentation dict

            src_key = f"source_{latency}_latency"
            tgt_key = f"target_{latency}_latency"

            if src_key not in seg or tgt_key not in seg:
                skipped += 1
                continue

            traj_entry = {
                "utt_id": utt,
                "latency": latency,
                "source": seg[src_key],      # LIST of segments
                "target": seg[tgt_key]       # LIST of segments
            }

            dataset[(lang, pqid)][latency].append(traj_entry)
            processed += 1

        except Exception as e:
            print(f"\n  ⚠️ Error processing line: {e}")
            skipped += 1
            continue

    print(f"\n✓ Processed: {processed}")
    print(f"✓ Skipped: {skipped}\n")

    # ----- write output -----
    print("💾 Writing output files...\n")

    total_files = 0
    total_entries = 0

    for (lang, pqid), groups in sorted(dataset.items()):
        base = os.path.join(output_root, lang, pqid)
        os.makedirs(base, exist_ok=True)

        for latency in ["low", "medium", "high"]:
            # 修改：使用 .json 扩展名（匹配你的截图）
            out_path = os.path.join(base, f"{latency}_latency.jsonl")

            entries = groups[latency]
            
            if len(entries) > 0:
                with open(out_path, "w", encoding="utf-8") as f:
                    for entry in entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                total_files += 1
                total_entries += len(entries)
                
                # 只显示前几个和最后几个，避免输出太多
                if total_files <= 10 or (lang, pqid) == list(sorted(dataset.items()))[-1][0]:
                    print(f"  ✓ {lang}/{pqid}/{latency}_latency.json ({len(entries)} entries)")

    print(f"\n{'='*60}")
    print(f"🎉 DONE! Trajectory-style dataset built.")
    print(f"{'='*60}")
    print(f"Total output files: {total_files}")
    print(f"Total entries: {total_entries}")
    print(f"Output directory: {output_root}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--metricx_jsonl", required=True,
                        help="Sentence-level MetricX QE output (filtered)")

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