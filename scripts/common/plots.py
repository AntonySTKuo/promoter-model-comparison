"""
Shared plotting helpers for the model-curve panels.

The drawing logic lives here; the per-dataset tables it reads (scatter marker
sizes, CRP tick placement) live in common.style.

showPlot/showScatter/showCurve touch only the shared con_model/reg_model
interface, so every model variant is drawn by the same code path.

Fig5_PP-Expression.py plots against log10(P_P) rather than log10(ConExp) and
draws its own curves and scatter, so it takes only dataset_scatter_setup from
this module.
"""

import numpy as np
import torch
from matplotlib.patches import Rectangle

from common.style import SCATTER_SETUP, SCATTER_SETUP_DEFAULT, SELECTIVE_TICKS
from common.utils import sci_label, sci_ticks


def draw_pt_id_box(ax, text_obj, pad_x, pad_top, pad_bottom, color, linewidth):
    """
    Draw a rectangle around a dataset-ID label (Fig2/Fig5's PT highlight).

    Built manually rather than through the text's own bbox= so top/bottom
    padding can differ -- a text's bbox pad (boxstyle="square,pad=...") is
    always symmetric on every side.
    """
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bbox_ax = text_obj.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
    x0, x1 = bbox_ax.x0 - pad_x, bbox_ax.x1 + pad_x
    y0, y1 = bbox_ax.y0 - pad_bottom, bbox_ax.y1 + pad_top
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=ax.transAxes,
                           fill=False, edgecolor=color, linewidth=linewidth,
                           clip_on=False))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific panel setup
# ─────────────────────────────────────────────────────────────────────────────

def dataset_scatter_setup(dataset_keys):
    """{dataset key: (marker size, edge linewidth)} for every requested dataset."""
    setup = dict(SCATTER_SETUP)
    setup.update({tf: SCATTER_SETUP_DEFAULT
                  for tf in dataset_keys if tf not in setup})
    return setup


def apply_selective_ticks(ax, TF):
    """
    Apply explicit tick placement for datasets that define one (the CRP panels).

    Datasets absent from style.SELECTIVE_TICKS are left with whatever ticks
    showPlot already set.
    """
    spec = SELECTIVE_TICKS.get(TF)
    if spec is None:
        return

    xticks, x_labelled, yticks, y_labelled = spec
    ax.set_xticks(xticks)
    ax.set_xticklabels([sci_label(v, x_labelled) for v in xticks])
    ax.set_yticks(yticks)
    ax.set_yticklabels([sci_label(v, y_labelled) for v in yticks])


# ─────────────────────────────────────────────────────────────────────────────
# Model-curve drawing
# ─────────────────────────────────────────────────────────────────────────────

def showPlot(ax, limits):
    """
    Apply axis limits, the FC = 1 baseline, and sci-notation tick formatting.

    The baseline is dashed so it reads as a reference rather than as data:
    Fig2_ModelCurves puts log10(FC) on y, so it marks "no regulation".
    """
    xMin, xMax, yMin, yMax = limits
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', zorder=1)
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    sci_ticks(ax, 'x', ticksize=4.5, pad=1.5)
    sci_ticks(ax, 'y', ticksize=4.5, pad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


def showScatterPoints(ax, x, y, limits, s=8, lw=0.4):
    """
    Scatter with the shared observation style, for pre-computed x/y.

    showScatter derives its own axes from (ConExp, RegExp); use this directly
    when the x axis is something else.
    """
    ax.scatter(x, y, color='0.92', alpha=0.7, edgecolors='0.6', linewidth=lw, s=s, zorder=2)
    xMin, xMax, yMin, yMax = limits
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


def showScatter(ax, X_obs, Y_obs, limits, s=8, lw=0.4):
    """
    Scatter: log10(FC) = log10(RegExp) − log10(ConExp)  vs  log10(ConExp).
    """
    showScatterPoints(ax, X_obs, Y_obs - X_obs, limits, s=s, lw=lw)


def _curve_points(model):
    """
    Trace the fitted manifold: (log10 ConExp, log10 RegExp).

    Sweeps P_P over [10⁻¹⁵, 10¹⁰], which covers the full thermodynamic range.
    """
    Ps = torch.logspace(-15, 10, 200, dtype=torch.float32)
    with torch.no_grad():
        X_fit = model.con_model(Ps).numpy()
        Y_fit = model.reg_model(Ps).numpy()
    return X_fit, Y_fit


def showCurve(ax, model, col='#e0bb00', lw=1.2, a=0.8, ls='-', zorder=3):
    """
    Plot the fitted model curve: log10(FC) vs log10(ConExp).
    Sweeps P_P over [10⁻¹⁵, 10¹⁰] to trace the full thermodynamic manifold.
    """
    X_fit, Y_fit = _curve_points(model)
    # Only plot finite values
    valid = np.isfinite(X_fit) & np.isfinite(Y_fit)
    ax.plot(X_fit[valid], (Y_fit - X_fit)[valid], color=col,
            linewidth=lw, alpha=a, linestyle=ls, zorder=zorder)
