#!/usr/bin/env python3
"""Consensus: my methods vs futures=200 baseline. BLEU + COMET vs LongYAAL latency.
ACL6060 dev, en->zh. Numbers from each ckpt's seg{960,1920,2880,3840} scores.tsv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# lat (LongYAAL CU ms), BLEU, COMET  @ seg 960/1920/2880/3840
M = {
    "futures200 baseline (topk5)": (
        [1318.9, 1882.7, 2378.0, 2855.1],
        [32.44, 35.21, 34.73, 37.42],
        [0.7773, 0.7965, 0.7986, 0.8059], "o", "#888888"),
    "5-axis (top5-axis5)": (
        [1461.4, 2175.8, 2745.0, 3106.9],
        [34.91, 39.61, 40.12, 40.14],
        [0.7867, 0.8083, 0.8117, 0.8174], "o", "#1f77b4"),
    "soft-vote (top5-axis5-sv)": (
        [1289.3, 1963.9, 2464.1, 2894.2],
        [30.87, 37.64, 38.52, 38.65],
        [0.7626, 0.8077, 0.8127, 0.8116], "s", "#2ca02c"),
    "fut100_n100": (
        [1209.2, 1889.8, 2371.1, 2756.5],
        [32.05, 36.94, 38.37, 39.30],
        [0.7674, 0.8049, 0.8071, 0.8158], "^", "#ff7f0e"),
}

fig, (axb, axc) = plt.subplots(1, 2, figsize=(13, 5.2))
for name, (lat, bleu, comet, mk, col) in M.items():
    base = dict(marker=mk, color=col, lw=2.2, ms=7)
    if "baseline" in name:
        base.update(ls="--", lw=2.4)
    axb.plot(lat, bleu, label=name, **base)
    axc.plot(lat, comet, label=name, **base)

axb.set_title("BLEU vs latency")
axb.set_ylabel("BLEU"); axb.set_xlabel("LongYAAL (CU), ms  ← faster")
axc.set_title("COMET vs latency  (ranking metric)")
axc.set_ylabel("COMET (XCOMET-XL)"); axc.set_xlabel("LongYAAL (CU), ms  ← faster")
for ax in (axb, axc):
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")

# annotate the efficiency win on the COMET panel
axc.annotate("5-axis @ seg1920 (0.808)\nbeats baseline's best (0.806)\n~680ms sooner",
             xy=(2175.8, 0.8083), xytext=(2150, 0.772), fontsize=8,
             arrowprops=dict(arrowstyle="->", color="#1f77b4"))

fig.suptitle("Consensus on ACL6060 en→zh: my methods vs futures=200 baseline", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "/home/haolingp/CMU_research_SMT/scripts/debug/consensus_vs_futures200.png"
fig.savefig(out, dpi=150)
print("saved:", out)
