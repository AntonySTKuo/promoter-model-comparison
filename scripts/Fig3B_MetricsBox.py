"""
Plot per-model metric distributions as box + strip panels.

Companion to Fig3_MetricsHeatmap.py: the heatmap shows every (model, dataset)
cell individually, this one collapses the dataset axis so the four model
variants can be compared as distributions.

Two figures are produced, one per regulation type, because the first model
column differs between them (RM for repressors, AM for activators) and mixing
the two would put unlike models in the same box. Each figure carries one panel
per metric:

    x = the four model variants,  y = the metric (panels run R2, RMSE, MAE)
    box   = quartiles and median over that variant's datasets
    strip = one jittered point per dataset, so n and the outliers stay visible

Points from peaked-trade-off (PT) datasets are red, the rest grey, mirroring
the bold red dataset IDs in Fig3A_MetricsHeatmap.py.

Metrics are recomputed from the fitted checkpoints in results/models/.

Usage:
    python scripts/Fig3B_MetricsBox.py

Outputs (results/figures/):
    metrics_box_repressors.png / .svg
    metrics_box_activators.png / .svg
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.loader import DATASET_SUMMARY
from common.models import metrics_tables
from common.paths import FIGURE_DIR, display_path
from common.style import PT_FRAME_COLOR


OUT_DIR = FIGURE_DIR

MM_PER_INCH = 25.4
FIG_WIDTH_MM = 86
FIG_HEIGHT_MM = 38
DPI = 300
# Margins are fractions of a 90 x 40 mm canvas, so they are wider in relative
# terms than a large figure would need: the y label plus tick labels take a
# fixed few mm whatever the figure size.
GRID_LEFT = 0.085
GRID_RIGHT = 0.99
GRID_BOTTOM = 0.17
GRID_TOP = 0.96
GRID_WSPACE = 0.45

# Greyscale throughout: the four columns are already named on the x axis, so
# colour would only repeat that. Keeping box and points neutral also stops this
# figure from implying a model-colour mapping that differs from Fig2/Fig5.
BOX_EDGE_GREY = "0.35"
BOX_FACE_GREY = "0.90"
MEDIAN_GREY = "0.15"
STRIP_FACE_GREY = "0.45"
STRIP_EDGE_GREY = "white"

# Points from peaked-trade-off (PT) datasets are drawn red instead of grey,
# matching the bold red dataset IDs in Fig3A. Boxes stay grey: they summarise
# all datasets in the column, PT and IS together.
PT_POINT_COLOR = PT_FRAME_COLOR

# matplotlib draws the whisker caps at 0.5 * the box width unless capwidths is
# given, so the two are set to the same value here to make them equal width.
BOX_WIDTH = 0.42
CAP_WIDTH = BOX_WIDTH
# Centre-to-centre spacing of the four boxes, in the same data units as
# BOX_WIDTH. Below 1.0 the columns sit closer than one category apart, so the
# gap between boxes narrows while the boxes themselves keep their width.
BOX_SPACING = 0.82
# Blank space kept outside the first and last box.
X_MARGIN = 0.5
BOX_LINEWIDTH = 0.5
MEDIAN_LINEWIDTH = 0.9
WHISKER_CAP_LINEWIDTH = 0.5

STRIP_SIZE = 3.5
STRIP_ALPHA = 0.85
STRIP_EDGE_LINEWIDTH = 0.25
# Half-width of the horizontal jitter, in x-axis units (boxes sit at 0, 1, 2...).
STRIP_JITTER = 0.13
# Jitter is cosmetic, but a fixed draw keeps the figure reproducible.
STRIP_SEED = 0

# Type sized for a 90 mm-wide, three-panel figure; matches the 5 pt ticks used
# by the other small multi-panel figures in this project.
AXIS_LINEWIDTH = 0.4
TICK_LABEL_SIZE = 5
AXIS_LABEL_SIZE = 6.5

# (metric column, y-axis label)
METRICS = [
    ("r2",       "R²"),
    ("geo_rmse", "RMSE"),
    ("mae",      "MAE"),
    ]

# (x tick label, model tag per regulation code)
MODEL_COLUMNS = [
    ("RM/AM", {"R": "RM",     "A": "AM"}),
    ("GM",    {"R": "GM",     "A": "GM"}),
    ("GM*",   {"R": "cGM",    "A": "cGM"}),
    ("GM**",  {"R": "cGM_a1", "A": "cGM_a1"}),
    ]

# (Regulation value in dataset_summary, regulation code, output basename)
FIGURES = [
    ("Repressor", "R", "metrics_box_repressors"),
    ("Activator", "A", "metrics_box_activators"),
    ]


def _load_metrics():
    """Recompute metrics from the fitted checkpoints under results/models/."""
    metrics = metrics_tables()
    empty = [tag for tag, table in metrics.items() if table.empty]
    if empty:
        raise FileNotFoundError(
            f"No fitted checkpoints for: {', '.join(empty)}. "
            f"Run scripts/ModelFit.py first.")
    return metrics


def _dataset_keys(regulation):
    rows = DATASET_SUMMARY[DATASET_SUMMARY["Regulation"] == regulation]
    return list(rows["Dataset Key"]), list(rows["Trend"])


def _column_values(metrics, tag, dataset_keys, trends, metric_col):
    """
    That model's metric over the datasets it was fitted against.

    Returns (values, is_pt) with NaNs dropped from both together, so a dataset
    missing from this model's table cannot shift the point colours out of step
    with the values.
    """
    table = metrics[tag]
    paired = [(tf, trend) for tf, trend in zip(dataset_keys, trends)
              if tf in table.index]
    values = table.loc[[tf for tf, _ in paired], metric_col].to_numpy(dtype=float)
    is_pt = np.array([trend == "PT" for _, trend in paired], dtype=bool)
    finite = np.isfinite(values)
    return values[finite], is_pt[finite]


def _x_label(template, regulation_code):
    """'RM/AM' resolves to whichever half applies; the rest pass through."""
    if template == "RM/AM":
        return "RM" if regulation_code == "R" else "AM"
    return template


def _style_box(bp):
    """Grey box with a pale fill, so the darker strip points read on top."""
    for patch in bp["boxes"]:
        patch.set_facecolor(BOX_FACE_GREY)
        patch.set_edgecolor(BOX_EDGE_GREY)
        patch.set_linewidth(BOX_LINEWIDTH)
    for median in bp["medians"]:
        median.set_color(MEDIAN_GREY)
        median.set_linewidth(MEDIAN_LINEWIDTH)
    for element in list(bp["whiskers"]) + list(bp["caps"]):
        element.set_color(BOX_EDGE_GREY)
        element.set_linewidth(WHISKER_CAP_LINEWIDTH)


def _draw_panel(ax, columns, labels, rng):
    positions = np.arange(len(columns)) * BOX_SPACING

    bp = ax.boxplot([values for values, _ in columns],
                    positions=positions, widths=BOX_WIDTH,
                    capwidths=CAP_WIDTH, patch_artist=True, showfliers=False,
                    zorder=2)
    _style_box(bp)

    # Every observation is drawn by the strip, so the box's own fliers are off
    # above -- otherwise outliers would appear twice.
    for pos, (values, is_pt) in zip(positions, columns):
        jitter = rng.uniform(-STRIP_JITTER, STRIP_JITTER, size=len(values))
        # PT drawn last so a red point is never hidden under a grey one.
        for mask, color, zorder in ((~is_pt, STRIP_FACE_GREY, 3),
                                    (is_pt, PT_POINT_COLOR, 4)):
            if not mask.any():
                continue
            ax.scatter((pos + jitter)[mask], values[mask], s=STRIP_SIZE,
                       color=color, alpha=STRIP_ALPHA, edgecolors=STRIP_EDGE_GREY,
                       linewidth=STRIP_EDGE_LINEWIDTH, zorder=zorder)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_SIZE)
    ax.set_xlim(-X_MARGIN, (len(columns) - 1) * BOX_SPACING + X_MARGIN)
    ax.tick_params(axis="both", which="major", direction="in",
                   labelsize=TICK_LABEL_SIZE, width=AXIS_LINEWIDTH,
                   length=1.8, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_LINEWIDTH)


def _save_figure(metrics, regulation, regulation_code, basename):
    dataset_keys, trends = _dataset_keys(regulation)
    labels = [_x_label(template, regulation_code) for template, _ in MODEL_COLUMNS]

    fig, axes = plt.subplots(
        1, len(METRICS),
        figsize=(FIG_WIDTH_MM / MM_PER_INCH, FIG_HEIGHT_MM / MM_PER_INCH),
        dpi=DPI,
        )
    fig.subplots_adjust(left=GRID_LEFT, right=GRID_RIGHT, bottom=GRID_BOTTOM,
                        top=GRID_TOP, wspace=GRID_WSPACE)

    for ax, (metric_col, _metric_label) in zip(axes, METRICS):
        columns = [_column_values(metrics, tag_by_type[regulation_code],
                                  dataset_keys, trends, metric_col)
                   for _, tag_by_type in MODEL_COLUMNS]
        # One RNG per panel, seeded the same way, so the jitter is identical
        # across metrics and a dataset keeps its horizontal offset panel to panel.
        _draw_panel(ax, columns, labels, np.random.default_rng(STRIP_SEED))

    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT_DIR, f"{basename}.{ext}"))
    plt.close(fig)
    print(f"Saved -> {display_path(os.path.join(OUT_DIR, f'{basename}.png'))}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics = _load_metrics()
    for regulation, regulation_code, basename in FIGURES:
        _save_figure(metrics, regulation, regulation_code, basename)


if __name__ == "__main__":
    main()
