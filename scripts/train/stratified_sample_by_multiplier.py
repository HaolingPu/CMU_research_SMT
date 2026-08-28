"""Stratified-sample EAST-latency2mult manifest by multiplier (latency proxy).

multiplier:
  1     -> low_latency
  2     -> medium_latency
  3-12  -> high_latency

Default: 12,500 total = 4167/4167/4166 across (low, medium, high).
"""
import argparse, json, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--total", type=int, default=12500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    groups = {"low": [], "medium": [], "high": []}
    with open(args.input) as f:
        for L in f:
            d = json.loads(L)
            m = d.get("multiplier", 0)
            if m == 1:        groups["low"].append(L)
            elif m == 2:      groups["medium"].append(L)
            elif 3 <= m <= 12: groups["high"].append(L)

    n_each = args.total // 3
    remainder = args.total - n_each * 3
    sizes = {"low": n_each, "medium": n_each, "high": n_each + remainder}

    out = []
    for k in ("low", "medium", "high"):
        avail = groups[k]
        n = sizes[k]
        if len(avail) < n:
            print(f"WARN: {k} only has {len(avail)} (want {n}); taking all")
            out.extend(avail)
        else:
            out.extend(random.sample(avail, n))
    random.shuffle(out)
    with open(args.output, "w") as f:
        f.writelines(out)
    print(f"Wrote {len(out)} rows to {args.output}")
    print(f"  low    : {sizes['low']:>5}  (pool {len(groups['low'])})")
    print(f"  medium : {sizes['medium']:>5}  (pool {len(groups['medium'])})")
    print(f"  high   : {sizes['high']:>5}  (pool {len(groups['high'])})")

if __name__ == "__main__":
    main()
