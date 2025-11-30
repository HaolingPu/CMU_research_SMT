import json
import os
from tqdm import tqdm

def convert_dataset(stream_dir, output_file):
    """
    Convert streaming segmentation JSON files into MetricX QE input.jsonl.
    """
    out = open(output_file, "w", encoding="utf8")

    json_files = [f for f in os.listdir(stream_dir) if f.endswith(".json")]

    for jf in tqdm(json_files, desc="Converting"):
        path = os.path.join(stream_dir, jf)

        with open(path, "r") as f:
            data = json.load(f)

        utt_id = data["utt_id"]
        full_source = data["original_text"]  # full ASR sentence

        # Process ALL latency levels
        for latency in ["low", "medium", "high"]:
            src_key = f"source_{latency}_latency"
            tgt_key = f"target_{latency}_latency"

            source_list = data[src_key]
            target_list = data[tgt_key]

            # Every pair (src_segment, tgt_segment) is one QE example
            for i, (src_seg, tgt_seg) in enumerate(zip(source_list, target_list)):
                if tgt_seg.strip() == "":
                    continue  # skip empty target segments

                example = {
                    "utt_id": utt_id,
                    "latency": latency,
                    "segment_id": i,
                    "source": full_source,      # Full source
                    "hypothesis": tgt_seg,      # Only target segment is evaluated
                    "reference": ""             # QE mode → empty reference
                }

                out.write(json.dumps(example, ensure_ascii=False) + "\n")

    out.close()
    print(f"\nDone! Written to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    convert_dataset(args.stream_dir, args.output)
