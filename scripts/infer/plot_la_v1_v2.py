#!/usr/bin/env python3
"""LA-40k & LA-40k-seg13 v1 (default sampling) vs v2 (rep_pen=1.05) side-by-side."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HAOLINGP_ROOT = Path("/data/user_data/haolingp/ckpts/infinisst-omni")
SEGS = [960, 1920, 2880, 3840]

def _latest_hf(exp_name: str):
    cands = sorted(HAOLINGP_ROOT.glob(f"{exp_name}/v*-hf"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return cands[-1] if cands else HAOLINGP_ROOT / f"{exp_name}/<no-hf-yet>"


CKPTS = [
    ("LA-40k",            HAOLINGP_ROOT / "gigaspeech-zh-LA-40k-s-bsz4/v0-20260506-221535-hf"),
    ("LA-40k-seg13",      HAOLINGP_ROOT / "gigaspeech-zh-LA-40k-seg13-s-bsz4/v0-20260507-214653-hf"),
    ("LA-2-40k-seg14",    _latest_hf("gigaspeech-zh-LA-40k-seg14-LA2-s-bsz4")),
]

OUT_PATH = Path(__file__).resolve().parent / "la_v1_v2_compare.png"


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


def collect(ckpt_path: Path, eval_dir: str):
    yaal, bleu, comet = [], [], []
    for seg in SEGS:
        tsv = ckpt_path / "evaluation/acl_6060" / eval_dir / f"seg{seg}" / "segmentation_output/scores.tsv"
        s = parse_scores(tsv)
        yaal.append(s["LongYAAL (CU)"])
        bleu.append(s["BLEU"])
        comet.append(s["COMET"])
    return yaal, bleu, comet


def main():
    fig, axes = plt.subplots(len(CKPTS), 2, figsize=(11, 4.5 * len(CKPTS)))
    if len(CKPTS) == 1:
        axes = [axes]

    for row_idx, (name, path) in enumerate(CKPTS):
        for variant, eval_dir, linestyle in [("v1 default", "en-zh", "-"),
                                             ("v2 rep_pen=1.05", "en-zh-v2", "--")]:
            try:
                yaal, bleu, comet = collect(path, eval_dir)
            except (PermissionError, FileNotFoundError, KeyError) as e:
                print(f"skip {name} {variant}: {type(e).__name__}: {e}")
                continue
            axes[row_idx][0].plot(yaal, bleu, marker="o", label=variant, linewidth=2, markersize=7, linestyle=linestyle)
            axes[row_idx][1].plot(yaal, comet, marker="o", label=variant, linewidth=2, markersize=7, linestyle=linestyle)

        for ax, ylabel in zip(axes[row_idx], ["BLEU", "COMET (XCOMET-XL)"]):
            ax.set_xlabel("LongYAAL (CU) [ms]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{name}: {ylabel}")
            ax.grid(alpha=0.3)
            ax.legend()

    fig.suptitle("ACL 6060 dev en-zh: LA v1 vs v2 (rep_pen=1.05)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
