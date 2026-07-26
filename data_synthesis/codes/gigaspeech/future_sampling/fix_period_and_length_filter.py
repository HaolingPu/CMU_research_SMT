#!/usr/bin/env python3
"""Fix the deferred-period (。X) artifact + apply length-ratio filter.

Input : a dir of QE-passed consensus JSONs (qe3-aligned-max).
Step 1: period-fix — move any leading 。！？ of a target_trajectory delta to the
        END of the previous non-empty delta (restores the LLM's true sentence
        boundary, which `prediction` already has correct).
Step 2: length-ratio filter — keep doc iff MIN <= len(prediction)/len(reference_text) <= MAX.
Output: a flat dir of fixed+kept JSONs, ready for convert2swift_consensus.py.
"""
import argparse, json, os, glob

SENTP = set("。！？")

def fix_traj(traj):
    out = list(traj)
    moved = 0
    for i in range(len(out)):
        d = out[i] or ""
        if not d:
            continue
        j = 0
        while j < len(d) and d[j] in SENTP:
            j += 1
        if j == 0:
            continue
        lead, rest = d[:j], d[j:]
        k = i - 1
        while k >= 0 and not (out[k] or "").strip():
            k -= 1
        if k < 0:
            continue  # sentence-initial, nothing to attach to
        out[k] = (out[k] or "") + lead
        out[i] = rest
        moved += 1
    return out, moved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-ratio", type=float, default=0.7)
    ap.add_argument("--max-ratio", type=float, default=1.5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.in_dir, "*.json")))
    n_in = len(files)
    n_lead_before = n_lead_after = n_deltas = 0
    n_drop_ratio = n_drop_noref = kept = 0
    total_moved = 0
    for f in files:
        d = json.load(open(f))
        traj = d.get("target_trajectory") or []
        for c in traj:
            c = (c or "").strip()
            if c:
                n_deltas += 1
                if c[0] in SENTP:
                    n_lead_before += 1
        fixed, moved = fix_traj(traj)
        total_moved += moved
        for c in fixed:
            c = (c or "").strip()
            if c and c[0] in SENTP:
                n_lead_after += 1
        # length-ratio filter
        pred = d.get("prediction") or ""
        ref = d.get("reference_text") or ""
        if not ref:
            n_drop_noref += 1
            continue
        ratio = len(pred) / max(1, len(ref))
        if not (args.min_ratio <= ratio <= args.max_ratio):
            n_drop_ratio += 1
            continue
        d["target_trajectory"] = fixed
        json.dump(d, open(os.path.join(args.out_dir, os.path.basename(f)), "w"), ensure_ascii=False)
        kept += 1

    print(f"input JSONs            : {n_in}")
    print(f"period-led deltas BEFORE: {n_lead_before}/{n_deltas} = {100*n_lead_before/n_deltas:.2f}%")
    print(f"period-led deltas AFTER : {n_lead_after}/{n_deltas} = {100*n_lead_after/n_deltas:.2f}%  (moved {total_moved} periods)")
    print(f"length-ratio filter     : dropped {n_drop_ratio} (ratio outside [{args.min_ratio},{args.max_ratio}]), {n_drop_noref} no-ref")
    print(f"KEPT (fixed+filtered)   : {kept}  -> {args.out_dir}")

if __name__ == "__main__":
    main()
