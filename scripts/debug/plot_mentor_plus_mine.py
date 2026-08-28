#!/usr/bin/env python3
"""Best-effort overlay: mentor's earlier baselines + my fresh runs on ONE figure.

CAVEAT (annotated on the plot): the two groups use DIFFERENT latency metrics.
  - mentor's baselines: StreamLAAL (old/InfiniSST pipeline), hardcoded from the CSV
  - my fresh runs:       LongYAAL (CU), current Qwen3-Omni pipeline, read live from scores.tsv
So horizontal positions are NOT directly comparable; only BLEU levels/trends are indicative.
The honest same-axis comparison is plot_latency_quality_compare.py once the mentor
policies finish re-running on the LongYAAL pipeline.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HAOLINGP_ROOT = Path("/data/user_data/haolingp/ckpts/infinisst-omni")
SEGS = [960, 1920, 2880, 3840]

# ---- mentor's earlier baselines: (StreamLAAL ms, BLEU) ----
MENTOR = {
    "EAST-latency2mult (mentor)":      [(1264, 39.74), (1978, 45.74), (2767, 47.52), (3599, 49.94)],
    "Refined-EAST-l2m (mentor)":       [(1247, 40.20), (1962, 45.31), (2864, 48.17), (3541, 49.59)],
    "Simul-MuST-C-v2 (mentor)":        [(1125, 38.32), (2006, 46.34), (2607, 48.62), (3057, 48.26)],
    "Word-Alignment (mentor)":         [(1180, 42.33), (1808, 45.60), (2251, 48.19), (2615, 48.48)],
}


def _latest_hf(exp_name: str) -> Path:
    cands = sorted(HAOLINGP_ROOT.glob(f"{exp_name}/v*-hf"),
                   key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return cands[-1] if cands else HAOLINGP_ROOT / f"{exp_name}/<no-hf-yet>"


# ---- my fresh runs (LongYAAL, read live): (label, exp_name, skip_segs) ----
MINE = [
    ("consensus-top5-axis5 (mine)", "gigaspeech-zh-consensus-top5-axis5-s-bsz4", ()),
    ("consensus-fut100_n100 (mine)", "gigaspeech-zh-consensus-top5-axis5-fut100_n100-s-bsz4", ()),
    ("PA-40k (mine)", "gigaspeech-zh-PA-40k-s-bsz4", ()),
    ("LA-40k-seg14 (mine)", "gigaspeech-zh-LA-40k-seg14-LA2-s-bsz4", (960,)),
]

OUT_PATH = Path(__file__).resolve().parent / "mentor_plus_mine.png"


def parse_scores(path: Path):
    scores = {}
    with path.open() as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                try:
                    scores[row[0]] = float(row[1])
                except ValueError:
                    pass
    return scores


def collect_mine(exp_name, skip):
    ckpt = _latest_hf(exp_name)
    yaal, bleu = [], []
    for seg in SEGS:
        if seg in skip:
            continue
        tsv = ckpt / "evaluation/acl_6060/en-zh" / f"seg{seg}" / "segmentation_output/scores.tsv"
        if not tsv.exists():
            continue
        s = parse_scores(tsv)
        if "LongYAAL (CU)" in s:
            yaal.append(s["LongYAAL (CU)"])
            bleu.append(s.get("BLEU", float("nan")))
    return yaal, bleu


def main():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    cmap = plt.cm.tab10.colors
    ci = 0

    # mentor: dashed, hollow markers
    for label, pts in MENTOR.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, linestyle="--", marker="s", markersize=8, markerfacecolor="none",
                markeredgewidth=1.8, linewidth=1.6, color=cmap[ci % 10], label=label)
        ci += 1

    # mine: solid, filled markers
    for label, exp, skip in MINE:
        yaal, bleu = collect_mine(exp, skip)
        if not yaal:
            print(f"skip {label}: no scores yet")
            continue
        order = np.argsort(yaal)
        xs = np.array(yaal)[order]
        ys = np.array(bleu)[order]
        ax.plot(xs, ys, linestyle="-", marker="o", markersize=8, linewidth=2.2,
                color=cmap[ci % 10], label=label)
        ci += 1
        print(f"plotted {label}: {len(xs)} pts")

    ax.set_xlabel("latency [ms]  —  mentor: StreamLAAL  |  mine: LongYAAL (CU)")
    ax.set_ylabel("BLEU (char zh)")
    ax.set_title("Mentor baselines (dashed, StreamLAAL) + my runs (solid, LongYAAL)\n"
                 "⚠ different latency metrics — x positions NOT directly comparable")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
