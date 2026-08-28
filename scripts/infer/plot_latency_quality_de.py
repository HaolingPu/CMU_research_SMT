#!/usr/bin/env python3
"""Plot LongYAAL (CU) vs BLEU and COMET for EAST de checkpoint(s)."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HAOLINGP_ROOT = Path("/data/user_data/haolingp/ckpts/infinisst-omni")
SEGS = [960, 1920, 2880, 3840]

MODELS = [
    ("EAST-de (v0, unbalanced 87k)", HAOLINGP_ROOT / "gigaspeech-de-EAST-latency2mult-s_origin-bsz4_de/v0-20260509-183510-hf"),
    ("EAST-de (v2, 12.5k stratified)", HAOLINGP_ROOT / "gigaspeech-de-EAST-latency2mult-s_origin-bsz4_de/v2-20260513-010501-hf"),
]

OUT_PATH = Path(__file__).resolve().parent / "latency_quality_de.png"


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


def collect(ckpt_path: Path):
    yaal, bleu, comet = [], [], []
    for seg in SEGS:
        tsv = ckpt_path / "evaluation/acl_6060/en-de" / f"seg{seg}" / "segmentation_output/scores.tsv"
        s = parse_scores(tsv)
        yaal.append(s["LongYAAL (CU)"])
        bleu.append(s["BLEU"])
        comet.append(s["COMET"])
    return yaal, bleu, comet


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for label, path in MODELS:
        try:
            yaal, bleu, comet = collect(path)
        except (PermissionError, FileNotFoundError, KeyError) as e:
            print(f"skip {label}: {type(e).__name__}: {e}")
            continue
        axes[0].plot(yaal, bleu, marker="o", label=label, linewidth=2, markersize=7)
        axes[1].plot(yaal, comet, marker="o", label=label, linewidth=2, markersize=7)
        for x, y, seg in zip(yaal, bleu, SEGS):
            axes[0].annotate(f"seg{seg}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
        for x, y, seg in zip(yaal, comet, SEGS):
            axes[1].annotate(f"seg{seg}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    for ax, ylabel in zip(axes, ["BLEU (sacrebleu 13a)", "COMET (XCOMET-XL)"]):
        ax.set_xlabel("LongYAAL (CU) [ms]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("ACL 6060 dev en-de: latency vs quality (seg 960/1920/2880/3840 ms)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
