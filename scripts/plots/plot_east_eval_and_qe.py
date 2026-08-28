#!/usr/bin/env python3
"""
Two figure groups:

Fig 1 (eval): EAST ja v1 + EAST de v2 — BLEU/COMET vs LongAL latency (4 seg points).
Fig 2 (qe):  ja sub-sentence MetricX distributions: EAST ja vs SALAMI ja
             (raw-sub-sentence histogram + per-doc-MAX CDF).
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "/home/haolingp/CMU_research_SMT/scripts/plots/out"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- Fig 1: BLEU/COMET vs Latency, per-language, 3 policies ----------
# seg -> (BLEU, chrF, COMET, LongAL_CU_ms)
# All policies trained on 12.5k examples (fair-comparison).
# EAST     = EAST with latency-stratified 12.5k (low/med/high 4166/4166/4168)
# EAST-low = EAST with low-latency-only 12.5k, multiplier 1..12 uniform
# SALAMI   = Simul-MuST-C-style chunked Qwen translation, 12.5k SEGALE+MetricX-filtered

ja_policies = {
    "EAST": {
        960:  (16.87, 32.48, 0.519,  386),
        1920: (24.04, 37.13, 0.654, 1580),
        2880: (27.92, 39.87, 0.708, 2566),
        3840: (28.43, 40.33, 0.743, 3530),
    },
    "EAST-low": {
        960:  (17.82, 33.03, 0.539,  257),
        1920: (22.14, 35.69, 0.630, 1572),
        2880: (22.75, 36.55, 0.639, 2130),
        3840: (23.17, 36.91, 0.645, 2466),
    },
    "SALAMI": {
        960:  (16.86, 30.27, 0.4510, 1063),
        1920: (21.55, 33.68, 0.5389, 2044),
        2880: (23.40, 35.13, 0.5943, 2637),
        3840: (23.84, 35.24, 0.5911, 3149),
    },
}

de_policies = {
    "EAST": {
        960:  (22.20, 60.44, 0.835, 1060),
        1920: (30.51, 64.64, 0.898, 3532),
        2880: (33.96, 66.36, 0.909, 5908),
        3840: (34.07, 65.55, 0.903, 6879),
    },
    "EAST-low": {
        960:  (21.30, 59.75, 0.8245,  979),
        1920: (25.27, 63.21, 0.8813, 1980),
        2880: (25.50, 63.62, 0.8804, 2454),
        3840: (25.61, 63.57, 0.8877, 2893),
    },
    "SALAMI": {
        960:  (20.04, 59.64, 0.7735,  891),
        1920: (27.18, 63.84, 0.8741, 2780),
        2880: (28.39, 64.53, 0.8849, 2676),
        3840: (27.97, 64.87, 0.8851, 3209),
    },
}

POLICY_STYLE = {
    "EAST":     ("#1f77b4", "o"),
    "EAST-low": ("#2ca02c", "s"),
    "SALAMI":   ("#d62728", "^"),
}


def _plot_quality_panel(ax, policies, metric_idx, ylabel, title):
    for name, d in policies.items():
        color, marker = POLICY_STYLE[name]
        items = sorted(d.items())
        xs, ys = [], []
        for seg, vals in items:
            y = vals[metric_idx]
            if y is None:
                continue
            xs.append(vals[3] / 1000.0)
            ys.append(y)
        if not xs:
            continue
        ax.plot(xs, ys, color=color, marker=marker, label=name, lw=1.8, ms=7)
        for seg, vals in items:
            y = vals[metric_idx]
            if y is None:
                continue
            ax.annotate(f"seg{seg}", xy=(vals[3]/1000.0, y),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=color, alpha=0.8)
    ax.set_xlabel("LongAL (CU)  [seconds]  — lower = lower latency")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")


def plot_lat_quality(ja_policies, de_policies):
    # Combined BLEU panel: ja + de side-by-side, 3 policies each.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    _plot_quality_panel(axes[0], ja_policies, 0,
                        "BLEU  (mecab-ja)", "en→ja  BLEU vs Latency")
    _plot_quality_panel(axes[1], de_policies, 0,
                        "BLEU  (13a-de)",   "en→de  BLEU vs Latency")
    fig.suptitle("Gigaspeech offline-synthesis (12.5k fair-comparison) — BLEU vs LongAL",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    p = f"{OUT_DIR}/fig1_bleu_vs_latency.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"saved {p}")

    # COMET panel — only EAST has full COMET numbers right now; ja+de.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    _plot_quality_panel(axes[0], ja_policies, 2,
                        "COMET  (XCOMET-XL)", "en→ja  COMET vs Latency")
    _plot_quality_panel(axes[1], de_policies, 2,
                        "COMET  (XCOMET-XL)", "en→de  COMET vs Latency")
    fig.suptitle("Gigaspeech offline-synthesis (12.5k fair-comparison) — COMET vs LongAL",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    p = f"{OUT_DIR}/fig1b_comet_vs_latency.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"saved {p}")


# ---------- Fig 2: QE distribution comparison ----------
EAST_JA = "/data/user_data/haolingp/data_synthesis/outputs/EAST/gigaspeech_ja/segale_qe/metricx/metricx_output.jsonl"
SAL_JA  = "/data/user_data/haolingp/data_synthesis/outputs/SALAMI/gigaspeech_ja/segale_pipeline/metricx_aligned/metricx_output.jsonl"


def load_scores(path):
    raw = []
    per_doc = defaultdict(lambda: -1e9)
    for L in open(path):
        try:
            r = json.loads(L)
        except Exception:
            continue
        s = r.get("prediction")
        if s is None:
            continue
        s = float(s)
        raw.append(s)
        md = r.get("metadata", {})
        utt = md.get("utt_id") or md.get("doc_id") or md.get("source_id") or ""
        lat = md.get("latency", "offline")
        doc = f"{utt}::{lat}"
        if s > per_doc[doc]:
            per_doc[doc] = s
    return np.array(raw), np.array(list(per_doc.values()))


def plot_qe(east_raw, east_max, sal_raw, sal_max):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- raw sub-sentence histogram ----
    ax = axes[0]
    bins = np.linspace(0, 20, 60)
    ax.hist(np.clip(east_raw, 0, 20), bins=bins, density=True,
            alpha=0.55, label=f"EAST ja  (n={len(east_raw):,})", color="#1f77b4")
    ax.hist(np.clip(sal_raw, 0, 20), bins=bins, density=True,
            alpha=0.55, label=f"SALAMI ja (n={len(sal_raw):,})", color="#d62728")
    ax.axvline(4.0, color="grey", ls="--", lw=1, alpha=0.7)
    ax.text(4.05, ax.get_ylim()[1]*0.92, "t=4", fontsize=8, color="grey")
    ax.set_xlabel("MetricX sub-sentence score (lower=better)")
    ax.set_ylabel("density")
    ax.set_title("ja sub-sentence QE distribution (raw)")
    ax.set_xlim(0, 20)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- per-doc MAX CDF (gating-equivalent) ----
    ax = axes[1]
    for name, arr, color in [
        (f"EAST ja  (n={len(east_max):,})", east_max, "#1f77b4"),
        (f"SALAMI ja (n={len(sal_max):,})", sal_max, "#d62728"),
    ]:
        arr_s = np.sort(np.clip(arr, 0, 20))
        cdf = np.arange(1, len(arr_s)+1) / len(arr_s)
        ax.plot(arr_s, cdf, label=name, lw=2, color=color)
    # mark used thresholds
    for t, lbl, c in [(4.26, "EAST high (t=4.26)", "#1f77b4"),
                      (6.60, "EAST low (t=6.60)",  "#1f77b4"),
                      (7.01, "SALAMI 12.5k (t=7.01)", "#d62728")]:
        ax.axvline(t, color=c, ls=":", lw=1.2, alpha=0.8)
        ax.text(t+0.05, 0.05, lbl, fontsize=7, color=c, rotation=90, va="bottom")
    ax.set_xlabel("per-doc MAX MetricX score")
    ax.set_ylabel("CDF (fraction of docs ≤ x)")
    ax.set_title("ja per-doc MAX CDF (gating-equivalent)")
    ax.set_xlim(0, 20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    p = f"{OUT_DIR}/fig2_ja_qe_distribution.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"saved {p}")


# ---------- Fig 3: latency stratum count for EAST ja v1 ----------
def plot_strata_count():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    cats = ["low\n(mult=1)", "medium\n(mult=2)", "high\n(mult=3-12)"]
    ja = [4166, 4166, 4168]
    ax.bar(cats, ja, color=["#7fb3d5", "#5dade2", "#2e86c1"])
    for i, v in enumerate(ja):
        ax.text(i, v+50, f"{v:,}", ha="center", fontsize=10)
    ax.axhline(12500/3, color="grey", ls="--", lw=1, alpha=0.7)
    ax.text(2.4, 12500/3+50, "1/3 of 12.5k", fontsize=8, color="grey")
    ax.set_ylabel("training examples")
    ax.set_title("EAST ja v1 train manifest — stratified by latency (total=12,500)")
    ax.set_ylim(0, 5200)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = f"{OUT_DIR}/fig3_east_ja_strata.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"saved {p}")


if __name__ == "__main__":
    plot_lat_quality(ja_policies, de_policies)
    plot_strata_count()

    print("loading EAST ja sub-sentence scores...")
    east_raw, east_max = load_scores(EAST_JA)
    print(f"  raw={len(east_raw):,}  per-doc-max={len(east_max):,}")
    print("loading SALAMI ja sub-sentence scores...")
    sal_raw, sal_max = load_scores(SAL_JA)
    print(f"  raw={len(sal_raw):,}  per-doc-max={len(sal_max):,}")
    plot_qe(east_raw, east_max, sal_raw, sal_max)
