#!/usr/bin/env python3
"""Plot LongYAAL (CU) vs BLEU and COMET for three checkpoints across 4 seg multipliers."""
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

CKPT_ROOT = Path("/data/user_data/siqiouya/ckpts/infinisst-omni")
SEGS = [960, 1920, 2880, 3840]

MODELS = [
    ("s_origin", "gigaspeech-zh-s_origin-bsz4/v1-20260122-055820-hf"),
    ("hibiki", "gigaspeech-zh-hibiki-s-bsz4/v0-20260326-141050-hf"),
    ("consensus-topk5", "gigaspeech-zh-consensus-topk5-s-bsz4/v1-20260418-002956-hf"),
    ("consensus-topk5_v2", "gigaspeech-zh-consensus-topk5_v2-s-bsz4/v0-20260425-121058-hf"),
    ("consensus-topk5_f200", "gigaspeech-zh-consensus-topk5_f200-s-bsz4/v0-20260502-125501-hf"),
]

OUT_PATH = Path(__file__).resolve().parent / "latency_quality_3models.png"


def parse_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open() as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            try:
                scores[row[0]] = float(row[1])
            except ValueError:
                pass
    return scores


def collect(model_path: str):
    yaal, bleu, comet, bleurt, taser = [], [], [], [], []
    for seg in SEGS:
        tsv = CKPT_ROOT / model_path / "evaluation/acl_6060/en-zh" / f"seg{seg}" / "segmentation_output/scores.tsv"
        s = parse_scores(tsv)
        yaal.append(s["LongYAAL (CU)"])
        bleu.append(s["BLEU"])
        comet.append(s["COMET"])
        bleurt.append(s["BLEURT"])
        taser.append(s.get("TASER", math.nan))
    return yaal, bleu, comet, bleurt, taser


def main():
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    for label, path in MODELS:
        yaal, bleu, comet, bleurt, taser = collect(path)
        axes[0].plot(yaal, bleu, marker="o", label=label, linewidth=2, markersize=7)
        axes[1].plot(yaal, comet, marker="o", label=label, linewidth=2, markersize=7)
        axes[2].plot(yaal, bleurt, marker="o", label=label, linewidth=2, markersize=7)
        axes[3].plot(yaal, taser, marker="o", label=label, linewidth=2, markersize=7)

    for ax, ylabel in zip(axes, ["BLEU", "COMET (XCOMET-XL)", "BLEURT", "TASER"]):
        ax.set_xlabel("LongYAAL (CU) [ms]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("ACL 6060 dev en-zh: latency vs quality (seg 960/1920/2880/3840 ms)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
