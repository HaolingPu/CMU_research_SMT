#!/usr/bin/env python3
"""Wait-k and local agreement quality/latency figures.

Three panels share the LAAL x-axis:
  (a) BLEU (higher is better)
  (b) Avg MetricX QE (lower is better)
  (c) QE <= 3.0 pass rate (higher is better)

Saves PDF and PNG: fig_waitk_la_50k.{pdf,png}

Run:
    python3 graph.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ===========================================================================
# SECTION 1: Paths  (edit input/output file locations here)
# ===========================================================================
HERE = Path(__file__).resolve().parent

# Input: wait-k summary JSON produced by aggregation script.
# Expected schema per record: {k, n, bleu, laal, metricx_avg, qe_pass_pct, ...}
WAITK_SUMMARY = Path(
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/"
    "rule_based_SMT/wait-k/waitk_50k_summary.json"
)
LA_SUMMARY = Path(
    "/data/user_data/haolingp/data_synthesis/outputs/gigaspeech/"
    "rule_based_SMT/local_agreement/la_50k_summary.json"
)

# Output PDF (same directory as this script).
# Change to .png / .svg if you need raster / web formats.
OUT_FIG = HERE / "fig_waitk_la_50k.pdf"


# ===========================================================================
# SECTION 2: Data loading  (no need to edit unless summary schema changes)
# ===========================================================================
# Load and sort so lines connect in ascending policy order.
wk = sorted(json.loads(WAITK_SUMMARY.read_text()), key=lambda r: r["k"])
la = sorted(json.loads(LA_SUMMARY.read_text()), key=lambda r: r["seg"])

def series(rows, label_key):
    return {
        "x": [r["laal"] for r in rows],
        "bleu": [r["bleu"] for r in rows],
        "qe": [r["metricx_avg"] for r in rows],
        "pass_rate": [r["qe_pass_pct"] for r in rows],
        "labels": [r[label_key] for r in rows],
    }


waitk = series(wk, "k")
local_agreement = series(la, "seg")


# ===========================================================================
# SECTION 3: Global style  (font, grid, line widths — applies to all panels)
# ===========================================================================
# These are matplotlib rcParams. To tune for a specific journal/venue,
# change font.size (e.g. 8 for IEEE two-column, 10 for ACL/EMNLP one-column).
# pdf.fonttype=42 forces TrueType — required by ACL for PDF submissions.
plt.rcParams.update({
    "font.family": "serif",      # use "sans-serif" for Neurips / ICML style
    "font.size": 10,             # base font size; everything else relative
    "axes.labelsize": 10,        # axis label (xlabel/ylabel)
    "axes.titlesize": 11,        # subplot title (we don't use titles here)
    "legend.fontsize": 9,        # legend text
    "xtick.labelsize": 9,        # tick numbers on x-axis
    "ytick.labelsize": 9,        # tick numbers on y-axis
    "axes.grid": True,           # show grid on every axis
    "grid.alpha": 0.25,          # 0=invisible, 1=solid; 0.25 = very subtle
    "grid.linestyle": "--",      # dashed; use "-" for solid, ":" for dotted
    "grid.linewidth": 0.6,       # thin grid lines so they don't dominate
    "axes.linewidth": 0.8,       # border thickness of the plot box
    "lines.linewidth": 1.6,      # the connecting line between markers
    "lines.markersize": 7,       # marker diameter in points
    "pdf.fonttype": 42,          # required by ACL; leave as-is for paper PDFs
    "ps.fonttype": 42,           # same for EPS outputs
})

# ---- Curve appearance (edit to match your color scheme) ------------------
# Common paper-friendly colors (Tableau 10 palette):
#   red  #d62728   blue #1f77b4   green #2ca02c
#   orange #ff7f0e   purple #9467bd   brown #8c564b
WAITK_COLOR = "#ff7f0e"
LA_COLOR = "#1f77b4"

# Marker shapes: 'o' circle, 's' square, 'v' triangle-down, '^' triangle-up,
#                'D' diamond, 'P' plus-filled, 'X' x-filled
WAITK_MARKER = "s"
LA_MARKER = "o"


# ===========================================================================
# SECTION 4: Figure layout  (adjust size, panel count, aspect)
# ===========================================================================
# figsize is in INCHES. For ACL/EMNLP papers:
#   - single-column:  ~(3.3, H)
#   - double-column / full-page: ~(7.0-9.0, H)
# Height depends on how tall you want panels vs. label overhead.
# constrained_layout handles spacing between panels automatically.
FIG_WIDTH = 9.0
FIG_HEIGHT = 2.7
NCOLS = 3                         # number of panels side by side
NROWS = 1

fig, axes = plt.subplots(
    NROWS, NCOLS,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    constrained_layout=True,      # auto-spacing; disable if you want manual subplots_adjust
)

# ---- Panel definitions ---------------------------------------------------
# Each tuple: (ax, y-values list, y-axis label, is_percent).
# To add a 4th panel, change NCOLS=4 and append another tuple here.
panels = [
    (axes[0], "bleu",      "BLEU",                    False),
    (axes[1], "qe",        "Avg MetricX QE",          False),
    (axes[2], "pass_rate", "QE \u2264 3.0 pass rate", True),
]


# ===========================================================================
# SECTION 5: Draw each panel  (loop; edit inner body to change appearance)
# ===========================================================================
LABEL_BOX = {
    "boxstyle": "round,pad=0.12",
    "fc": "white",
    "ec": "none",
    "alpha": 0.72,
}

LA_OFFSETS = {
    "bleu": {
        1: (8, 0, "left", "center"),
        2: (-6, 16, "right", "bottom"),
        3: (-8, -12, "right", "top"),
        4: (0, -12, "center", "top"),
        5: (-10, -4, "right", "center"),
    },
    "qe": {
        1: (8, 8, "left", "bottom"),
        2: (-8, 12, "right", "bottom"),
        3: (0, 14, "center", "bottom"),
        4: (6, 10, "left", "bottom"),
        5: (-8, -6, "right", "top"),
    },
    "pass_rate": {
        1: (8, 0, "left", "center"),
        2: (-6, 16, "right", "bottom"),
        3: (-8, 24, "right", "bottom"),
        4: (0, -12, "center", "top"),
        5: (-10, -4, "right", "center"),
    },
}


def annotate_points(ax, xs, ys, labels, prefix, metric_key):
    n_pts = len(xs)
    for i, (x, y, label) in enumerate(zip(xs, ys, labels)):
        if prefix == "seg" and label in LA_OFFSETS.get(metric_key, {}):
            dx, dy, ha, va = LA_OFFSETS[metric_key][label]
            xytext = (dx, dy)
        elif i == 0:
            xytext, ha, va = (8, 0), "left", "center"
        elif i == n_pts - 1:
            xytext, ha, va = (-8, 0), "right", "center"
        else:
            xytext, ha, va = (0, 9), "center", "bottom"
        ax.annotate(
            f"{prefix}={label}",
            xy=(x, y),
            xytext=xytext,
            textcoords="offset points",
            fontsize=6.8,
            ha=ha,
            va=va,
            color="#555",
            bbox=LABEL_BOX,
            clip_on=False,
        )


for ax, metric_key, ylabel, as_percent in panels:
    # ---- Plot the main curve ----
    ax.plot(
        waitk["x"], waitk[metric_key],
        marker=WAITK_MARKER,
        color=WAITK_COLOR,
        label="Wait-k (stride=1)",
        zorder=3,                      # draw on top of grid
        markeredgecolor="white",       # white halo around marker (pops off grid)
        markeredgewidth=1,
    )
    ax.plot(
        local_agreement["x"], local_agreement[metric_key],
        marker=LA_MARKER,
        color=LA_COLOR,
        label="Local agreement (n=2)",
        zorder=3,
        markeredgecolor="white",
        markeredgewidth=1,
    )

    # ---- Axis labels ----
    ax.set_xlabel("LAAL")
    ax.set_ylabel(ylabel)

    # ---- Format y-axis as percent for pass-rate panel ----
    if as_percent:
        # decimals=0 → "10%"; decimals=1 → "10.1%"
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    annotate_points(ax, waitk["x"], waitk[metric_key], waitk["labels"], "k", metric_key)
    annotate_points(
        ax,
        local_agreement["x"],
        local_agreement[metric_key],
        local_agreement["labels"],
        "seg",
        metric_key,
    )

    # ---- Reserve label room above/below markers ----
    ymin, ymax = ax.get_ylim()
    pad = 0.18 * (ymax - ymin)
    ax.set_ylim(ymin - 0.08 * (ymax - ymin), ymax + pad)


# ===========================================================================
# SECTION 6: Shared legend at top  (hide/move as needed)
# ===========================================================================
# Pull handle from first panel; since all panels use the same label,
# the legend has just one entry.  When you add more methods (LA, min-p, ...),
# increase ncol and they will flow horizontally.
handles, labels = axes[0].get_legend_handles_labels()
# bbox_to_anchor y=1.12 pushes the legend well above the panels' top edge so
# it never collides with interior "k=..." annotations that sit above markers.
# If you add more methods (ncol=2/3), you may need to bump y up to 1.15.
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.12),           # (x, y) in figure coords; y>1 → above plot
    frameon=False,                         # True to show box around legend
)


# ===========================================================================
# SECTION 7: Save  (PDF is lossless; add dpi= for PNG)
# ===========================================================================
# bbox_inches="tight" crops whitespace. For exact size in paper, use
# pad_inches=0 to remove all padding.
fig.savefig(OUT_FIG, bbox_inches="tight")
fig.savefig(OUT_FIG.with_suffix(".png"), dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_FIG}")
print(f"Saved: {OUT_FIG.with_suffix('.png')}")
