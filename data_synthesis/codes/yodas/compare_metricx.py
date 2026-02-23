#!/usr/bin/env python3
import argparse, json, random
import numpy as np

def get_key(obj, fallback_idx):
    md = obj.get("metadata") or {}
    for k in ["utt_id", "file", "id", "key", "name", "path"]:
        if k in md:
            return str(md[k])
    # fallback to line index
    return f"line_{fallback_idx}"

def load_scores(path):
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            key = get_key(obj, i)
            pred = obj.get("prediction")
            if pred is None:
                continue
            scores[key] = float(pred)
    return scores

def summarize(name, arr):
    arr = np.array(arr, dtype=float)
    if len(arr) == 0:
        print(f"\n== {name} ==\ncount: 0 (empty)")
        return
    print(f"\n== {name} ==")
    print(f"count: {len(arr)}")
    print(f"mean : {arr.mean():.4f}")
    print(f"p10  : {np.percentile(arr,10):.4f}")
    print(f"p25  : {np.percentile(arr,25):.4f}")
    print(f"p50  : {np.percentile(arr,50):.4f}")
    print(f"p75  : {np.percentile(arr,75):.4f}")
    print(f"p90  : {np.percentile(arr,90):.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--sample", type=int, default=50)
    args = ap.parse_args()

    ref = load_scores(args.refined)
    base = load_scores(args.baseline)

    common = sorted(set(ref.keys()) & set(base.keys()))
    print(f"refined only: {len(ref) - len(common)}")
    print(f"baseline only: {len(base) - len(common)}")
    print(f"common: {len(common)}")

    ref_scores = [ref[k] for k in common]
    base_scores = [base[k] for k in common]

    summarize("refined (common)", ref_scores)
    summarize("baseline (common)", base_scores)

    if len(common) == 0:
        print("\nNo common keys. Try line-index alignment by rerunning with identical ordering.")
        return

    # filter ratio
    ref_bad = sum(s < args.threshold for s in ref_scores)
    base_bad = sum(s < args.threshold for s in base_scores)
    print(f"\n<threshold={args.threshold}> refined bad: {ref_bad}/{len(common)} ({ref_bad/len(common):.2%})")
    print(f"<threshold={args.threshold}> baseline bad: {base_bad}/{len(common)} ({base_bad/len(common):.2%})")

    # sample comparisons
    print("\n== Sample differences ==")
    random.seed(0)
    for k in random.sample(common, min(args.sample, len(common))):
        print(f"{k}\tref={ref[k]:.3f}\tbase={base[k]:.3f}\tdelta={ref[k]-base[k]:+.3f}")

if __name__ == "__main__":
    main()


# python /data/user_data/haolingp/data_synthesis/codes/compare_metricx.py \
#   --refined /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_refined_east/metricx_output.jsonl \
#   --baseline /data/user_data/haolingp/data_synthesis/outputs/gigaspeech/train_xl_east/metricx_output.jsonl \
#   --threshold 3.0 \
#   --sample 50
