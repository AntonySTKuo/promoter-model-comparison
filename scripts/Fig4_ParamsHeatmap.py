"""
Plot alpha, beta and P_T parameter heatmaps for model comparisons.

Parameters are read from the fitted checkpoints in results/models/, which store
log10 values. Heatmaps use a diverging scale centered at 0, corresponding to the
original parameter value of 1.

Cell labels show the original (delogged) value. alpha and beta span many decades
in both directions, so they fall back to scientific notation outside 1e-2..1e2;
P_T is mostly larger than 100, so it stays in plain decimals up to the point
where the cell can no longer hold the digits.
"""

import os
import sys
import colorsys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.loader import DATASET_SUMMARY
from common.models import params_tables
from common.paths import FIGURE_DIR, display_path
from common.style import MODEL_COLORS, PT_FRAME_COLOR


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
VALUE_FONT_SIZE = 4.5
COLORBAR_FONT_SIZE = 6
COLORBAR_LABEL_SIZE = 7

# Datasets whose Trend column reads PT get a bold, coloured column label.
PT_LABEL_COLOR = PT_FRAME_COLOR
PT_LABEL_WEIGHT = "bold"
IS_LABEL_COLOR = "black"
IS_LABEL_WEIGHT = "normal"

# (column, colorbar label, output basename, cell-label style)
PARAMETERS = [
    ("log_alpha", "alpha",  "params_heatmap_alpha", "scientific"),
    ("log_beta",  "beta",   "params_heatmap_beta",  "scientific"),
    ("log_PT",    "$P_T$",  "params_heatmap_PT",    "decimal"),
    ]

MODEL_ROWS = [
    ("RM/AM", {"R": "RM",     "A": "AM"}),
    ("GM",    {"R": "GM",     "A": "GM"}),
    ("GM*",   {"R": "cGM",    "A": "cGM"}),
    ("GM**",  {"R": "cGM_a1", "A": "cGM_a1"}),
    ]

def _saturate_hex(hex_color, saturation_factor=1.45, lightness_factor=0.95):
    hex_color = hex_color.lstrip("#")
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in range(0, 6, 2)]
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    saturation = min(1.0, saturation * saturation_factor)
    lightness = max(0.0, min(1.0, lightness * lightness_factor))
    saturated = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#" + "".join(f"{round(channel * 255):02X}" for channel in saturated)


PARAM_CMAP = LinearSegmentedColormap.from_list(
    "parameter_effect",
    [
        (0.00, _saturate_hex(MODEL_COLORS["RM"])),
        (0.50, "#ffffff"),
        (1.00, MODEL_COLORS["AM"]),
        ],
    )


def _load_params():
    """Read fitted parameters from the checkpoints under results/models/."""
    params = params_tables()
    empty = [tag for tag, table in params.items() if table.empty]
    if empty:
        raise FileNotFoundError(
            f"No fitted checkpoints for: {', '.join(empty)}. "
            f"Run scripts/ModelFit.py first.")
    return params


def _dataset_order(regulation):
    rows = DATASET_SUMMARY[DATASET_SUMMARY["Regulation"] == regulation]
    dataset_ids = rows["Dataset ID"].str.replace("DS", "", regex=False)
    return list(rows["Dataset Key"]), list(dataset_ids), list(rows["Trend"])


def _matrix(params, regulation_code, dataset_keys, param_col):
    data = np.full((len(MODEL_ROWS), len(dataset_keys)), np.nan)
    for row_idx, (_, model_by_type) in enumerate(MODEL_ROWS):
        model = model_by_type[regulation_code]
        table = params[model]
        if param_col not in table.columns:
            continue
        values = pd.to_numeric(table[param_col], errors="coerce")
        for col_idx, tf in enumerate(dataset_keys):
            if tf in values.index:
                data[row_idx, col_idx] = values.loc[tf]
    return data


def _make_norm_and_extend(all_values):
    finite = all_values[np.isfinite(all_values)]
    max_abs = float(np.nanmax(np.abs(finite)))
    max_abs = max(max_abs, 1.0)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    return norm, "neither"


def _text_color(cmap_obj, norm_val):
    r, g, b, _ = cmap_obj(norm_val)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.45 else "black"


def _plain_decimal(value):
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_cell_value(log_value, style):
    """Delogged cell label; 'decimal' keeps large values out of exponent form."""
    value = 10 ** log_value
    upper = float("inf") if style == "decimal" else 100.0
    if 0.01 <= value <= upper:
        return _plain_decimal(value)
    mantissa, exponent = f"{value:.1E}".split("E")
    return f"{mantissa}E\n{exponent}"


def _draw_values(ax, data, norm, cmap_obj, style):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if not np.isfinite(value):
                continue
            mapped = float(norm(value))
            ax.text(j, i, _format_cell_value(value, style), ha="center", va="center",
                    fontsize=VALUE_FONT_SIZE, color=_text_color(cmap_obj, mapped))


def _mark_pt_labels(ax, trends):
    """
    Colour the PT column labels and make them bold.

    Applied to both label1 and label2 of each tick: the heatmap moves its
    labels to the top, which draws label2, while set_xticklabels only styles
    label1 -- setting just one of them silently leaves the labels unstyled.
    """
    for tick, trend in zip(ax.xaxis.get_major_ticks(), trends):
        is_pt = trend == "PT"
        color = PT_LABEL_COLOR if is_pt else IS_LABEL_COLOR
        weight = PT_LABEL_WEIGHT if is_pt else IS_LABEL_WEIGHT
        for label in (tick.label1, tick.label2):
            label.set_color(color)
            label.set_fontweight(weight)


def _style_axis(ax, dataset_ids, trends, row_labels):
    ax.set_xticks(np.arange(len(dataset_ids)))
    ax.set_xticklabels(dataset_ids, rotation=0, fontsize=XTICK_FONT_SIZE)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, pad=1)
    _mark_pt_labels(ax, trends)
    ax.set_yticks(np.arange(len(row_labels)))
    ytick_labels = ax.set_yticklabels(row_labels, fontsize=YTICK_FONT_SIZE)
    for label in ytick_labels:
        label.set_horizontalalignment(YTICK_ALIGNMENT)
    ax.tick_params(axis="both", which="both", length=0)
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", labelleft=True, labelright=False, pad=YTICK_PAD)
    ax.set_xticks(np.arange(-0.5, len(dataset_ids), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color=GRID_LINE_COLOR, linestyle="-",
            linewidth=GRID_LINEWIDTH)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BORDER_COLOR)
        spine.set_linewidth(BORDER_LINEWIDTH)


def _colorbar_ticks(norm):
    low = int(np.ceil(norm.vmin))
    high = int(np.floor(norm.vmax))
    ticks = [tick for tick in range(low, high + 1) if tick % 2 == 0]
    if 0 not in ticks:
        ticks.append(0)
    ticks = sorted(ticks)
    labels = ["1" if tick == 0 else rf"$10^{{{tick}}}$" for tick in ticks]
    return ticks, labels


def _save_param_heatmap(params, param_col, param_label, basename, style):
    rep_keys, rep_ids, rep_trends = _dataset_order("Repressor")
    act_keys, act_ids, act_trends = _dataset_order("Activator")
    rep_data = _matrix(params, "R", rep_keys, param_col)
    act_data = _matrix(params, "A", act_keys, param_col)

    all_values = np.concatenate([rep_data[np.isfinite(rep_data)],
                                 act_data[np.isfinite(act_data)]])
    norm, extend = _make_norm_and_extend(all_values)
    cmap_obj = PARAM_CMAP.with_extremes(bad="lightgrey")

    fig = plt.figure(figsize=(FIG_WIDTH_MM / MM_PER_INCH,
                              FIG_HEIGHT_MM / MM_PER_INCH),
                     dpi=DPI)
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
    cax = fig.add_subplot(gs[0, 4])

    rep_masked = np.ma.masked_invalid(rep_data)
    act_masked = np.ma.masked_invalid(act_data)

    image_rep = ax_rep.imshow(rep_masked, aspect="auto", cmap=cmap_obj, norm=norm)
    ax_act.imshow(act_masked, aspect="auto", cmap=cmap_obj, norm=norm)

    _style_axis(ax_rep, rep_ids, rep_trends, ["RM", "GM", "GM*", "GM**"])
    _style_axis(ax_act, act_ids, act_trends, ["AM", "GM", "GM*", "GM**"])
    _draw_values(ax_rep, rep_data, norm, cmap_obj, style)
    _draw_values(ax_act, act_data, norm, cmap_obj, style)

    colorbar = fig.colorbar(image_rep, cax=cax, extend=extend)
    colorbar.outline.set_edgecolor(BORDER_COLOR)
    colorbar.outline.set_linewidth(BORDER_LINEWIDTH)
    colorbar.ax.tick_params(labelsize=COLORBAR_FONT_SIZE, length=2,
                            width=COLORBAR_TICK_WIDTH)

    tick_vals, tick_labels = _colorbar_ticks(norm)
    colorbar.set_ticks(tick_vals)
    colorbar.set_ticklabels(tick_labels)
    colorbar.set_label(param_label, fontsize=COLORBAR_LABEL_SIZE)

    for ext in ("png", "svg"):
        path = os.path.join(OUT_DIR, f"{basename}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"Saved -> {display_path(os.path.join(OUT_DIR, f'{basename}.png'))}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    params = _load_params()
    for param_col, param_label, basename, style in PARAMETERS:
        _save_param_heatmap(params, param_col, param_label, basename, style)


if __name__ == "__main__":
    main()
