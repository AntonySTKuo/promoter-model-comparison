"""
Plot metric heatmaps for the model comparison.

Each output figure shows repressors on the left and activators on the right.
Rows are model classes; columns are datasets. One shared colorbar is used per metric.

Dataset IDs of peaked-trade-off (PT) datasets are drawn bold and red, so the
PT/IS split from the registry's Trend column can be read straight off the axis.

Metrics are recomputed from the fitted checkpoints in results/models/.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.loader import DATASET_SUMMARY
from common.models import metrics_tables
from common.paths import FIGURE_DIR, display_path
from common.style import PT_FRAME_COLOR


OUT_DIR = FIGURE_DIR
MM_PER_INCH = 25.4
FIG_WIDTH_MM = 180
FIG_HEIGHT_MM = 30
DPI = 300
GRID_LEFT = 0.08
GRID_RIGHT = 0.96
GRID_BOTTOM = 0.08
GRID_TOP = 0.88
HEATMAP_GAP_WIDTH = 2.0
COLORBAR_GAP_WIDTH = 0.6
COLORBAR_WIDTH = 0.4
BORDER_COLOR = "black"
BORDER_LINEWIDTH = 0.4
GRID_LINE_COLOR = "0.35"
GRID_LINEWIDTH = 0.35
COLORBAR_TICK_WIDTH = 0.4
XTICK_FONT_SIZE = 6
YTICK_FONT_SIZE = 6.5
YTICK_PAD = 2
YTICK_ALIGNMENT = "right"

# Datasets whose Trend column reads PT get a bold red column label.
PT_LABEL_COLOR = PT_FRAME_COLOR
PT_LABEL_WEIGHT = "bold"
IS_LABEL_COLOR = "black"
IS_LABEL_WEIGHT = "normal"

METRICS = [
    ("r2",       "R²",   "metrics_heatmap_R2",    True,  False),
    ("geo_rmse", "RMSE", "metrics_heatmap_RMSE",  False, False),
    ("mae",      "MAE",  "metrics_heatmap_MAE",   False, False),
    ]

MODEL_ROWS = [
    ("RM/AM", {"R": "RM",     "A": "AM"}),
    ("GM",    {"R": "GM",     "A": "GM"}),
    ("GM*",   {"R": "cGM",    "A": "cGM"}),
    ("GM**",  {"R": "cGM_a1", "A": "cGM_a1"}),
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


def _dataset_order(regulation):
    rows = DATASET_SUMMARY[DATASET_SUMMARY["Regulation"] == regulation]
    dataset_ids = rows["Dataset ID"].str.replace("DS", "", regex=False)
    return list(rows["Dataset Key"]), list(dataset_ids), list(rows["Trend"])


def _matrix(metrics, regulation_code, dataset_keys, metric_col):
    data = np.full((len(MODEL_ROWS), len(dataset_keys)), np.nan)
    for row_idx, (_, model_by_type) in enumerate(MODEL_ROWS):
        model = model_by_type[regulation_code]
        table = metrics[model]
        for col_idx, tf in enumerate(dataset_keys):
            if tf in table.index:
                data[row_idx, col_idx] = table.loc[tf, metric_col]
    return data


def _make_norm_cmap_and_extend(all_values, higher_is_better, log_scale):
    """Build colour norm and choose the right YlOrRd variant.

    Both branches render good fits pale and poor fits dark red, so the two
    metric families stay visually comparable.

    higher_is_better (R²):
        YlOrRd_r – large value → position 1 → pale yellow → best
        colorbar: bottom = the smallest value present, top = 1, neither end
                  extended. Spanning the full range rather than clipping at a
                  percentile is what lets both ends stay unextended without
                  hiding any cell outside the bar.

    not higher_is_better (RMSE/MAE):
        YlOrRd   – small value → position 0 → pale yellow → best
        colorbar: bottom = 0 with no lower-end extension;
                  top = 98th percentile with upper-end extension
    """
    finite = all_values[np.isfinite(all_values)]
    if log_scale:
        finite = finite[finite > 0]

    if higher_is_better:
        vmin = float(np.nanmin(finite))
        vmax = 1.0
        cmap_name = "YlOrRd_r"
        norm = Normalize(vmin=vmin, vmax=vmax)
        extend = "neither"
    else:
        vmin = 0.0
        vmax = float(np.percentile(finite, 98))
        cmap_name = "YlOrRd"
        if log_scale:
            positive_min = float(np.nanmin(finite[finite > 0]))
            norm = LogNorm(vmin=positive_min, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        extend = "max" if np.nanmax(finite) > vmax else "neither"

    return norm, cmap_name, extend


def _text_color(cmap_obj, norm_val):
    """White or black text based on perceived luminance at a colormap position."""
    r, g, b, _ = cmap_obj(norm_val)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.45 else "black"


def _draw_values(ax, data, norm, cmap_obj):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if not np.isfinite(value):
                continue
            clipped = float(np.clip(value, norm.vmin, norm.vmax))
            mapped = float(norm(clipped))
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=4.5, color=_text_color(cmap_obj, mapped))


def _mark_pt_labels(ax, trends):
    """
    Colour the PT column labels red and bold.

    Applied to both label1 and label2 of each tick: the heatmap moves its
    labels to the top, which draws label2, while set_xticklabels only styles
    label1 -- setting just one of them silently leaves the labels black.
    """
    for tick, trend in zip(ax.xaxis.get_major_ticks(), trends):
        is_pt = trend == "PT"
        color = PT_LABEL_COLOR if is_pt else IS_LABEL_COLOR
        weight = PT_LABEL_WEIGHT if is_pt else IS_LABEL_WEIGHT
        for label in (tick.label1, tick.label2):
            label.set_color(color)
            label.set_fontweight(weight)


def _style_axis(ax, dataset_ids, trends, row_labels, show_y_labels=True):
    ax.set_xticks(np.arange(len(dataset_ids)))
    ax.set_xticklabels(dataset_ids, rotation=0, fontsize=XTICK_FONT_SIZE)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, pad=1)
    ax.set_yticks(np.arange(len(row_labels)))
    if show_y_labels:
        ytick_labels = ax.set_yticklabels(row_labels, fontsize=YTICK_FONT_SIZE)
        for label in ytick_labels:
            label.set_horizontalalignment(YTICK_ALIGNMENT)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both", length=0)
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", labelleft=show_y_labels, labelright=False, pad=YTICK_PAD)
    _mark_pt_labels(ax, trends)
    ax.set_xticks(np.arange(-0.5, len(dataset_ids), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color=GRID_LINE_COLOR, linestyle="-",
            linewidth=GRID_LINEWIDTH)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BORDER_COLOR)
        spine.set_linewidth(BORDER_LINEWIDTH)


def _save_metric_heatmap(metrics, metric_col, metric_label, basename, higher_is_better, log_scale):
    rep_keys, rep_ids, rep_trends = _dataset_order("Repressor")
    act_keys, act_ids, act_trends = _dataset_order("Activator")
    rep_data = _matrix(metrics, "R", rep_keys, metric_col)
    act_data = _matrix(metrics, "A", act_keys, metric_col)

    all_values = np.concatenate([rep_data[np.isfinite(rep_data)], act_data[np.isfinite(act_data)]])

    norm, cmap_name, extend = _make_norm_cmap_and_extend(all_values, higher_is_better, log_scale)
    cmap_obj = matplotlib.colormaps[cmap_name].with_extremes(bad="lightgrey")

    fig = plt.figure(figsize=(FIG_WIDTH_MM / MM_PER_INCH, FIG_HEIGHT_MM / MM_PER_INCH), dpi=DPI)
    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[
            len(rep_ids),
            HEATMAP_GAP_WIDTH,
            len(act_ids),
            COLORBAR_GAP_WIDTH,
            COLORBAR_WIDTH,
            ],
        left=GRID_LEFT,
        right=GRID_RIGHT,
        bottom=GRID_BOTTOM,
        top=GRID_TOP,
        wspace=0,
        )
    ax_rep = fig.add_subplot(gs[0, 0])
    ax_act = fig.add_subplot(gs[0, 2])
    cax    = fig.add_subplot(gs[0, 4])

    rep_masked = np.ma.masked_invalid(rep_data)
    act_masked = np.ma.masked_invalid(act_data)

    image_rep = ax_rep.imshow(rep_masked, aspect="auto", cmap=cmap_obj, norm=norm)
    ax_act.imshow(act_masked, aspect="auto", cmap=cmap_obj, norm=norm)

    _style_axis(ax_rep, rep_ids, rep_trends, ["RM", "GM", "GM*", "GM**"])
    _style_axis(ax_act, act_ids, act_trends, ["AM", "GM", "GM*", "GM**"])
    _draw_values(ax_rep, rep_data, norm, cmap_obj)
    _draw_values(ax_act, act_data, norm, cmap_obj)

    colorbar = fig.colorbar(image_rep, cax=cax, extend=extend)
    colorbar.outline.set_edgecolor(BORDER_COLOR)
    colorbar.outline.set_linewidth(BORDER_LINEWIDTH)
    colorbar.ax.tick_params(labelsize=6, length=2, width=COLORBAR_TICK_WIDTH)

    tick_vals = list(np.linspace(norm.vmin, norm.vmax, 3))
    colorbar.set_ticks(tick_vals)
    colorbar.set_ticklabels([f"{v:.2f}" for v in tick_vals])
    colorbar.set_label(f"{metric_label}", fontsize=7)

    for ext in ("png", "svg"):
        path = os.path.join(OUT_DIR, f"{basename}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"Saved -> {display_path(os.path.join(OUT_DIR, f'{basename}.png'))}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics = _load_metrics()
    for metric_col, metric_label, basename, higher_is_better, log_scale in METRICS:
        _save_metric_heatmap(metrics, metric_col, metric_label, basename, higher_is_better, log_scale)


if __name__ == "__main__":
    main()
