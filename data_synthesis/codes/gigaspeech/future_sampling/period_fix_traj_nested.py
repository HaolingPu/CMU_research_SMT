#!/usr/bin/env python3
"""Step-1 period-fix for the NEW dASR data, applied right after decode.

Walks a consensus decode root (task_*/per_utt/*.json), moves any leading
。！？ of a target_trajectory delta to the END of the previous non-empty delta
(the `。X` chunk-start artifact that the new sub-sentence-split dASR introduces),
and writes a MIRRORED tree (task_NN/per_utt/<same>.json) with the fixed
target_trajectory. NO filtering, NO downsampling — every doc is carried through,
so the canonical SEGALE+QE pipeline can run on the fixed mirror unchanged.

`prediction` is left untouched (it already has the period in the right place);
only the streaming `target_trajectory` deltas are repaired, which is exactly what
convert2swift consumes for the multi-turn training trajectory.
"""
import argparse, glob, json, os

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
    ap.add_argument("--in-root", required=True, help="decode root with task_*/per_utt/*.json")
    ap.add_argument("--out-root", required=True, help="mirrored output root")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_root, "task_*", "per_utt", "*.json")))
    n_in = len(files)
    n_deltas = n_lead_before = n_lead_after = total_moved = written = 0

    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
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
        d["target_trajectory"] = fixed

        # mirror task_NN/per_utt/<name>.json
        rel = os.path.relpath(f, args.in_root)
        dst = os.path.join(args.out_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(d, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
        written += 1

    pct_b = 100 * n_lead_before / n_deltas if n_deltas else 0.0
    pct_a = 100 * n_lead_after / n_deltas if n_deltas else 0.0
    print(f"input JSONs             : {n_in}")
    print(f"period-led deltas BEFORE: {n_lead_before}/{n_deltas} = {pct_b:.2f}%")
    print(f"period-led deltas AFTER : {n_lead_after}/{n_deltas} = {pct_a:.2f}%  (moved {total_moved})")
    print(f"written (mirror)        : {written}  -> {args.out_root}")


if __name__ == "__main__":
    main()
