#!/usr/bin/env python3
"""Build the best-of-5 reference-selected manifest (wiki: 2026-07-refsel-bestof5).

For each utt_id present in ANY pool's QE-filtered dir, pick the QE-surviving
candidate with max metrics.bleu_char vs the frozen reference, guarded by
length_ratio_ref in [0.7, 1.5], and copy its per-utt json into --output-dir
(a MANIFEST_ROOT consumable by convert2swift_consensus.py). Decoding stayed
ref-free; the reference is used only here, at selection time.
"""
import argparse
import json
import os
import shutil
from collections import Counter
from glob import glob

PROD = "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod"
# pool -> QE-filtered (threshold 3.0) per-utt dir
POOL_QE_DIRS = {
    "J_40k": f"{PROD}/J_40k-segale-p24/qe3-aligned-max-full",
    "J_40k_fut100": f"{PROD}/J_40k_fut100-segale-p24/qe3-aligned-max",
    "J_40k_n100": f"{PROD}/J_40k_n100-segale-p24/qe3-aligned-max",
    "J_40k_softvote": f"{PROD}/J_40k_softvote-segale-p24/qe3-aligned-max",
    "anchor_40k": f"{PROD}/anchor_40k-segale-p24/qe3-aligned-max",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pools", nargs="*", default=list(POOL_QE_DIRS))
    ap.add_argument("--min-lr", type=float, default=0.7)
    ap.add_argument("--max-lr", type=float, default=1.5)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pool_dirs = {p: POOL_QE_DIRS[p] for p in args.pools}
    candidates = {}  # utt_id -> list of (pool, path, bleu, lr)
    for pool, d in pool_dirs.items():
        files = glob(os.path.join(d, "*.json"))
        kept = 0
        for path in files:
            try:
                with open(path) as f:
                    rec = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            m = rec.get("metrics") or {}
            bleu, lr = m.get("bleu_char"), m.get("length_ratio_ref")
            if bleu is None or lr is None or not (args.min_lr <= lr <= args.max_lr):
                continue
            candidates.setdefault(rec["utt_id"], []).append((pool, path, bleu, lr))
            kept += 1
        print(f"[pool] {pool}: {len(files)} QE files, {kept} pass LR guard")

    wins = Counter()
    bleu_sum = 0.0
    for utt_id, cands in candidates.items():
        pool, path, bleu, _ = max(cands, key=lambda c: c[2])
        shutil.copyfile(path, os.path.join(args.output_dir, f"{utt_id}.json"))
        wins[pool] += 1
        bleu_sum += bleu

    n = len(candidates)
    report = {
        "n_selected": n,
        "mean_selected_bleu_char": bleu_sum / n if n else 0.0,
        "win_shares": {p: wins[p] / n for p in pool_dirs} if n else {},
        "win_counts": dict(wins),
        "lr_guard": [args.min_lr, args.max_lr],
        "output_dir": args.output_dir,
    }
    print(json.dumps(report, indent=2))
    report_path = args.report or os.path.join(args.output_dir + "_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()
