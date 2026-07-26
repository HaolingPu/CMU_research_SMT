#!/usr/bin/env python3
"""Step 0 of the best-of-5 reference-selection design (wiki: 2026-07-refsel-bestof5).

Aggregates stored per-utt metrics across the five ref-free 40k pools and reports
the oracle best-of-5 char-BLEU headroom vs the flagship pool alone. CPU-only.
"""
import argparse
import json
import os
from collections import Counter
from glob import glob

PROD = "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/consensus_decoding_prod"
POOLS = ["J_40k", "J_40k_fut100", "J_40k_n100", "J_40k_softvote", "anchor_40k"]
FLAGSHIP = "J_40k"


def load_pool(pool: str):
    out = {}
    for path in glob(os.path.join(PROD, pool, "task_*", "per_utt", "*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        m = d.get("metrics") or {}
        if "bleu_char" not in m:
            continue
        out[d["utt_id"]] = {
            "bleu": m["bleu_char"],
            "lr": m.get("length_ratio_ref"),
            "laal": m.get("laal_text"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="*", default=POOLS)
    ap.add_argument("--min-lr", type=float, default=0.7)
    ap.add_argument("--max-lr", type=float, default=1.5)
    ap.add_argument("--out-json", default=os.path.join(PROD, "bestof5_headroom.json"))
    args = ap.parse_args()

    pool_names = args.pools
    pools = {}
    for p in pool_names:
        pools[p] = load_pool(p)
        print(f"[load] {p}: {len(pools[p])} utts")

    common = set(pools[FLAGSHIP])
    for p in pool_names:
        common &= set(pools[p])
    print(f"[common] {len(common)} utts in all {len(pool_names)} pools")

    def eligible(rec):
        return rec["lr"] is not None and args.min_lr <= rec["lr"] <= args.max_lr

    flag_sum = oracle_sum = oracle_lr_sum = 0.0
    wins = Counter()
    n = 0
    for u in common:
        flag = pools[FLAGSHIP][u]
        cands = [(p, pools[p][u]) for p in pool_names]
        best_p, best = max(cands, key=lambda kv: kv[1]["bleu"])
        guarded = [(p, r) for p, r in cands if eligible(r)] or [(FLAGSHIP, flag)]
        gbest_p, gbest = max(guarded, key=lambda kv: kv[1]["bleu"])
        flag_sum += flag["bleu"]
        oracle_sum += best["bleu"]
        oracle_lr_sum += gbest["bleu"]
        wins[gbest_p] += 1
        n += 1

    report = {
        "n_common": n,
        "flagship_mean_bleu": flag_sum / n,
        "oracle_mean_bleu_unguarded": oracle_sum / n,
        "oracle_mean_bleu_lr_guarded": oracle_lr_sum / n,
        "oracle_gain_lr_guarded": oracle_lr_sum / n - flag_sum / n,
        "win_shares_lr_guarded": {p: wins[p] / n for p in pool_names},
        "pool_mean_bleu": {p: sum(r["bleu"] for r in pools[p].values()) / len(pools[p])
                           for p in pool_names},
        "pool_mean_laal": {p: sum(r["laal"] for r in pools[p].values() if r["laal"] is not None)
                           / max(1, sum(1 for r in pools[p].values() if r["laal"] is not None))
                           for p in pool_names},
    }
    print(json.dumps(report, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
