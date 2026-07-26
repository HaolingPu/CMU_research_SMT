#!/usr/bin/env python3
"""Plot LongYAAL (CU) vs BLEU and COMET for selected checkpoints across 4 seg multipliers."""
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

HAOLINGP_ROOT = Path("/data/user_data/haolingp/ckpts/infinisst-omni")
SEGS = [960, 1920, 2880, 3840]

def _latest_hf(exp_name: str):
    cands = sorted(HAOLINGP_ROOT.glob(f"{exp_name}/v*-hf"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return cands[-1] if cands else HAOLINGP_ROOT / f"{exp_name}/<no-hf-yet>"


MODELS = [
    ("consensus-topk5", HAOLINGP_ROOT / "gigaspeech-zh-consensus-topk5-s-bsz4/v0-20260427-000433-hf", "en-zh", "-", ()),
    ("consensus-topk5_v2", HAOLINGP_ROOT / "gigaspeech-zh-consensus-topk5_v2-s-bsz4/v2-20260426-101916-hf", "en-zh", "-", ()),
    ("consensus-topk5_k4", HAOLINGP_ROOT / "gigaspeech-zh-consensus-topk5_k4-s-bsz4/v1-20260429-023458-hf", "en-zh", "-", ()),
    ("consensus-top5-axis5", HAOLINGP_ROOT / "gigaspeech-zh-consensus-top5-axis5-s-bsz4/v0-20260516-121120-hf", "en-zh", "-", ()),
    ("consensus-top5-axis5-sv", HAOLINGP_ROOT / "gigaspeech-zh-consensus-top5-axis5-sv-s-bsz4/v0-20260518-141037-hf", "en-zh", "-", ()),
    ("PA-40k", HAOLINGP_ROOT / "gigaspeech-zh-PA-40k-s-bsz4/v0-20260504-150226-hf", "en-zh", "-", ()),
    # LA seg960 dropped: LA-trained model degenerates into burst mode at small chunks,
    # producing a ~7.2s LongYAAL outlier that distorts the latency axis.
    ("LA-40k-seg14 (best LA)", _latest_hf("gigaspeech-zh-LA-40k-seg14-LA2-s-bsz4"), "en-zh", "-", (960,)),
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


def collect(ckpt_path: Path, eval_dir: str, skip_segs=()):
    yaal, bleu, comet, chrf = [], [], [], []
    for seg in SEGS:
        if seg in skip_segs:
            continue
        tsv = ckpt_path / "evaluation/acl_6060" / eval_dir / f"seg{seg}" / "segmentation_output/scores.tsv"
        s = parse_scores(tsv)
        yaal.append(s["LongYAAL (CU)"])
        bleu.append(s["BLEU"])
        comet.append(s["COMET"])
        chrf.append(s.get("chrF", math.nan))
    return yaal, bleu, comet, chrf


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for label, path, eval_dir, linestyle, skip_segs in MODELS:
        try:
            yaal, bleu, comet, chrf = collect(path, eval_dir, skip_segs)
        except (PermissionError, FileNotFoundError, KeyError) as e:
            print(f"skip {label}: {type(e).__name__}: {e}")
            continue
        axes[0].plot(yaal, bleu, marker="o", label=label, linewidth=2, markersize=7, linestyle=linestyle)
        axes[1].plot(yaal, comet, marker="o", label=label, linewidth=2, markersize=7, linestyle=linestyle)
        axes[2].plot(yaal, chrf, marker="o", label=label, linewidth=2, markersize=7, linestyle=linestyle)

    for ax, ylabel in zip(axes, ["BLEU", "COMET (XCOMET-XL)", "chrF"]):
        ax.set_xlabel("LongYAAL (CU) [ms]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("ACL 6060 dev en-zh: latency vs quality (seg 960/1920/2880/3840 ms)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
