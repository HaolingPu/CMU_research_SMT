#!/usr/bin/env python3
"""Honest latency-quality comparison on ACL 6060 dev en-zh, all on ONE pipeline.

LongYAAL (CU) vs COMET (primary panel) and BLEU (with bootstrap 95% CI error bars).
Pulls every model from the SAME Qwen3-Omni + omnisteval pipeline so the lines are
actually comparable. Models whose scores.tsv don't exist yet are skipped, so this
can be re-run as evals land.

BLEU CI: bootstrap over the 468 resegmented (pred, ref) pairs, recomputing corpus
char-level zh BLEU each resample (sacrebleu tokenize='zh' reproduces the eval exactly).
COMET CI is NOT shown: per-segment COMET scores are not dumped by the eval, and
recomputing XCOMET-XL per resample is infeasible.
"""
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import sacrebleu
    _HAVE_SACREBLEU = True
except ImportError:
    _HAVE_SACREBLEU = False

HAOLINGP_ROOT = Path("/data/user_data/haolingp/ckpts/infinisst-omni")
SEGS = [960, 1920, 2880, 3840]
N_BOOT = 1000
BOOT_SEED = 0


def _latest_hf(exp_name: str) -> Path:
    cands = sorted(
        HAOLINGP_ROOT.glob(f"{exp_name}/v*-hf"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
    )
    return cands[-1] if cands else HAOLINGP_ROOT / f"{exp_name}/<no-hf-yet>"


# (label, ckpt_path, eval_dir, linestyle, skip_segs)
# ref-free (our method) drawn solid; ref-based baselines dashed.
MODELS = [
    # original-figure consensus lines that still have scores on disk
    ("consensus-topk5 (orig)", _latest_hf("gigaspeech-zh-consensus-topk5-s-bsz4"), "en-zh", "-", ()),
    ("consensus-topk5_v2 (orig)", _latest_hf("gigaspeech-zh-consensus-topk5_v2-s-bsz4"), "en-zh", "-", ()),
    # newer consensus variants
    ("consensus-top5-axis5 (ours, ref-free)", _latest_hf("gigaspeech-zh-consensus-top5-axis5-s-bsz4"), "en-zh", "-", ()),
    ("consensus-fut100_n100 (ours, ref-free)", _latest_hf("gigaspeech-zh-consensus-top5-axis5-fut100_n100-s-bsz4"), "en-zh", "-", ()),
    ("PA-40k", _latest_hf("gigaspeech-zh-PA-40k-s-bsz4"), "en-zh", "-", ()),
    # LA seg960 dropped: degenerates into burst mode at small chunks (~7.2s YAAL outlier).
    ("LA-40k-seg14 (best LA)", _latest_hf("gigaspeech-zh-LA-40k-seg14-LA2-s-bsz4"), "en-zh", "-", (960,)),
    ("hibiki (ref-based)", _latest_hf("gigaspeech-zh-hibiki-s-bsz4"), "en-zh", "--", ()),
    ("EAST-lowonly", _latest_hf("gigaspeech-zh-EAST-lowonly-s-bsz4"), "en-zh", "--", ()),
    ("EAST-even", _latest_hf("gigaspeech-zh-EAST-even-s-bsz4"), "en-zh", "--", ()),
    ("Simul-MuST-C (ref-based)", _latest_hf("gigaspeech-zh-Simul-MuST-C-fixed-v2-s_origin-bsz4"), "en-zh", "--", ()),
]

# Mentor's lines copied (digitized) from the original LongYAAL figure
# (latency_quality_3models.png); their ckpts/scores are deleted, so these are
# eyeballed off the figure -> APPROXIMATE, but on the same LongYAAL axis so
# positionally comparable. Drawn dashed. (s_origin == word-alignment.)
# fields: yaal, bleu, comet  (4 pts: seg 960/1920/2880/3840)
MENTOR_REF = [
    ("s_origin = word-align (mentor)", dict(
        yaal=[1150, 1850, 2400, 2820],
        bleu=[37.7, 41.25, 43.35, 43.5],
        comet=[0.767, 0.8025, 0.8142, 0.8145])),
    ("hibiki (mentor)", dict(
        yaal=[1560, 2230, 2800, 3270],
        bleu=[41.85, 44.8, 46.45, 46.7],
        comet=[0.7845, 0.803, 0.819, 0.818])),
]

OUT_PATH = Path(__file__).resolve().parent / "latency_quality_compare.png"


def parse_scores(path: Path) -> dict:
    scores = {}
    with path.open() as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            try:
                scores[row[0]] = float(row[1])
            except ValueError:
                pass
    return scores


def collect(ckpt_path: Path, eval_dir: str, skip_segs=()):
    yaal, bleu, comet = [], [], []
    for seg in SEGS:
        if seg in skip_segs:
            continue
        base = ckpt_path / "evaluation/acl_6060" / eval_dir / f"seg{seg}" / "segmentation_output"
        tsv = base / "scores.tsv"
        if not tsv.exists():
            continue
        s = parse_scores(tsv)
        if "LongYAAL (CU)" not in s:
            continue
        yaal.append(s["LongYAAL (CU)"])
        comet.append(s.get("COMET", math.nan))
        bleu.append(s.get("BLEU", math.nan))
    return yaal, bleu, comet


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = plt.cm.tab10.colors

    for i, (label, path, eval_dir, ls, skip) in enumerate(MODELS):
        try:
            yaal, bleu, comet = collect(path, eval_dir, skip)
        except Exception as e:
            print(f"skip {label}: {type(e).__name__}: {e}")
            continue
        if not yaal:
            print(f"skip {label}: no scores yet")
            continue
        c = colors[i % len(colors)]
        order = np.argsort(yaal)
        yaal = np.array(yaal)[order]
        comet = np.array(comet)[order]
        bleu = np.array(bleu)[order]
        # ACL 6060 dev is a fixed benchmark (all 5 talks, every eval) -> deterministic
        # points, no resampling CI. Decoding randomness (temp 0.6) is the only residual
        # noise; measure it by re-running inference if a close call needs defending.
        axes[0].plot(yaal, comet, marker="o", label=label, linewidth=2, markersize=7, linestyle=ls, color=c)
        axes[1].plot(yaal, bleu, marker="o", label=label, linewidth=2, markersize=7, linestyle=ls, color=c)
        print(f"plotted {label}: {len(yaal)} pts")

    # mentor's lines digitized from the original figure: dashed, hollow markers, grey-ish
    mentor_colors = ["#555555", "#aa6600"]
    for j, (label, d) in enumerate(MENTOR_REF):
        c = mentor_colors[j % len(mentor_colors)]
        axes[0].plot(d["yaal"], d["comet"], marker="s", markersize=7, markerfacecolor="none",
                     markeredgewidth=1.6, linewidth=1.6, linestyle="--", color=c, label=label)
        axes[1].plot(d["yaal"], d["bleu"], marker="s", markersize=7, markerfacecolor="none",
                     markeredgewidth=1.6, linewidth=1.6, linestyle="--", color=c, label=label)
        print(f"plotted (mentor, digitized) {label}")

    for ax, ylabel in zip(axes, ["COMET (XCOMET-XL)", "BLEU (char zh)"]):
        ax.set_xlabel("LongYAAL (CU) [ms]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].set_title("Latency vs COMET  (primary)")
    axes[1].set_title("Latency vs BLEU")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("ACL 6060 dev en-zh: one pipeline, ref-free (solid) vs ref-based (dashed)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
