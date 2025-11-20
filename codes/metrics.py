#!/usr/bin/env python3
import os
import json
import sacrebleu
import csv

STREAM_DIR = "/data/user_data/haolingp/outputs/streaming_dataset_v2"
REF_DIR    = "/data/user_data/haolingp/outputs/llm_reference_json"

OUT_CSV = "/data/user_data/haolingp/outputs/trajectory_results.csv"


############################################################
#  Utils
############################################################

def join_segments(segs):
    """Combine non-empty segments into one sentence."""
    return "".join([s for s in segs if s.strip()])


def compute_bleu(hyp, ref):
    return sacrebleu.corpus_bleu(
        [hyp],
        [[ref]],
        tokenize='zh'    # 🔥关键
    ).score


def compute_laal(timeline, source_length, streaming_len, reference_len):
    """
    timeline: [t1, t2, ...]  # seconds when each chunk is produced
    source_length: total duration L (seconds)
    streaming_len: |Y|
    reference_len: |Y*|
    """

    M = max(streaming_len, reference_len)

    # pad timeline if needed
    if len(timeline) < M:
        # if streaming shorter, last timestamp repeats (interpreted as 'finished reading')
        timeline = timeline + [timeline[-1]] * (M - len(timeline))

    laal_sum = 0.0
    for i in range(1, M + 1):
        d_i = timeline[i - 1]
        d_i_star = (i - 1) * (source_length / reference_len)
        laal_sum += (d_i - d_i_star)

    return laal_sum / M



############################################################
# Load timeline (source timestamps for segments)
############################################################

def load_timeline(data, level):
    """
    Loads timestamps: timeline = [t1, t2, ...]
    Your streaming JSON should contain:
        source_low_latency
        target_low_latency
        timeline_low_latency (optional depending on your structure)
    """
    key = f"timeline_{level}"
    if key not in data:
        # fallback: if you structured timeline differently adjust here
        return []
    return data[key]


############################################################
# Main evaluation loop
############################################################

def main():
    results = []

    stream_files = sorted(os.listdir(STREAM_DIR))

    for fn in stream_files:
        if not fn.endswith(".json"):
            continue

        utt_id = fn[:-5]
        stream_path = os.path.join(STREAM_DIR, fn)
        ref_path = os.path.join(REF_DIR, f"{utt_id}.json")

        # Skip if reference missing
        if not os.path.exists(ref_path):
            print(f"[WARN] missing offline reference for {utt_id}")
            continue

        stream_data = json.load(open(stream_path))
        # print(join_segments(stream_data["target_low_latency"]))

        ref_data    = json.load(open(ref_path))
        ref_text    = ref_data["offline_reference"]
        # print(ref_text)

        row = {"utt_id": utt_id}
        print(row)

        for level in ["low_latency", "medium_latency", "high_latency"]:
            hyp_segments = stream_data[f"target_{level}"]
            # print(hyp_segments)
            hyp_text = join_segments(hyp_segments)
            print(hyp_text)

            # --- BLEU ---
            bleu = compute_bleu(hyp_text, ref_text)
            row[f"bleu_{level}"] = round(bleu,2)

            # --- LAAL ---
            # timeline = stream_data.get(f"timeline_{level}", [])
            # laal = compute_laal(timeline)
            # row[f"laal_{level}"] = laal

        results.append(row)

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utt_id",
                "bleu_low_latency","laal_low_latency",
                "bleu_medium_latency","laal_medium_latency",
                "bleu_high_latency","laal_high_latency",
            ]
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("Done. Saved →", OUT_CSV)


if __name__ == "__main__":
    main()

