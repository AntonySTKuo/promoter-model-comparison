"""
Draw the regulatory fold change against RNAP occupancy: E_TF+/E_TF- vs P_P.

One panel per dataset:

    x = log10(P_P)     -- the per-dataset latent (model.P_latent), cropped for
                          RM/AM since the curve sits flat at FC = 1 at either
                          end of the sweep (see PP_AUTOCROP_TAGS)
    y = log10(E_TF+/E_TF-) = log10(RegExp) - log10(ConExp)

This is the fold-change half of what Fig5_PP-Expression.py stacks above a
basal sub-panel, now drawn on its own at the full panel height of
Fig5_PP-Expression.py, which is the companion figure showing E_TF- and E_TF+
over the same x axis. Everything the two share -- the x crop, the tick
thinning, the DS label and its PT box, the grid geometry -- lives in
common.pp_panels.

Each panel marks P_P,opt (where the fold change reaches its extremum) with a
vertical dashed line, and reports IPR90, the 5th-95th interpercentile range of
log10(P_P) the dataset's variants span.

Covers RM (repressors), AM (activators), and GM -- GM is fitted against both
dataset types, so it gets its own repressor and activator figures too, each
using the GM curve color instead of the RM/AM palette. cGM/cGM_a1 are not
covered by this figure.

Axis titles and panel letters are added downstream when the repressor and
activator figures are composited; this script draws the panel grids only.

Usage:
    python scripts/Fig6_PP-FC.py

Outputs (results/figures/, alongside the other figure scripts):
    PP-FC_RM_repressors.png / .svg
    PP-FC_AM_activators.png / .svg
    PP-FC_GM_repressors.png / .svg
    PP-FC_GM_activators.png / .svg
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.loader import REPRESSOR_IDs, ACTIVATOR_IDs, ALL_IDs, loadData
from common.models import applies_to, load_model, pp_opt
from common.paths import FIGURE_DIR, display_path
from common.plots import dataset_scatter_setup
from common.pp_panels import (FC_Y_TICK_STEPS, TICK_LABEL_SIZE, apply_axes,
                              draw_ds_label, draw_ipr90, draw_pp_opt, fc_ylim,
                              model_curve, panel_grid, panel_xlim, plot_series,
                              thin_x_labels, thin_y_labels)
from common.style import EXPRESSION_PP_COLORS


EXPRESSION_PP_TAGS = ('RM', 'AM', 'GM')

SETUP = dataset_scatter_setup(ALL_IDs)

OUT_DIR = FIGURE_DIR

# Which two corners the DS ID and IPR90 go in, as (x, y, ha, va). Repression
# drives the curve below FC = 1 and activation above it, and FC_YLIMS pins that
# line high for repressors and low for activators, so the free corners are the
# bottom pair in a repressor panel and the top pair in an activator one.
# The ID takes the left corner and IPR90 the right, as in Fig5_PP-Expression.
LABEL_POS_REPRESSOR = ((0.03, 0.04, "left", "bottom"),
                       (0.97, 0.04, "right", "bottom"))
LABEL_POS_ACTIVATOR = ((0.03, 0.96, "left", "top"),
                       (0.97, 0.96, "right", "top"))


def _draw_panel(ax, TF, ds_id, tag):
    """One dataset's fold-change panel. False if nothing was drawn."""
    model = load_model(tag, TF)
    if model is None:
        return False

    _, df_sample = loadData(TF)
    n_obs = len(df_sample)
    n_fit = model.P_latent.shape[0]
    if n_fit != n_obs:
        # The checkpoint was fitted against a different sample, so P_latent no
        # longer lines up row-for-row with the observations.
        raise ValueError(f"checkpoint has {n_fit} latents for {n_obs} observations; refit {tag}")

    p_log = model.P_latent.detach().numpy()
    con_log = np.log10(df_sample["ConExp"].values)
    reg_log = np.log10(df_sample["RegExp"].values)
    fc_log = reg_log - con_log

    is_repressor = TF in REPRESSOR_IDs
    xlim = panel_xlim(model, tag, p_log, TF)
    s, lw = SETUP[TF]
    colors = EXPRESSION_PP_COLORS[tag]
    p_curve, con_fit, reg_fit = model_curve(model, xlim)

    apply_axes(ax, xlim, fc_ylim(TF, fc_log, is_repressor))
    plot_series(ax, p_log, fc_log, p_curve, reg_fit - con_fit,
                colors['fc_face'], colors['fc_edge'], colors['fc_curve'],
                s, lw, colors['scatter_alpha'])
    draw_pp_opt(ax, pp_opt(model, tag))

    thin_x_labels(ax)
    y_step = FC_Y_TICK_STEPS.get(TF)
    if y_step is not None:
        thin_y_labels(ax, *y_step)
    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_SIZE)

    id_pos, ipr_pos = LABEL_POS_REPRESSOR if is_repressor else LABEL_POS_ACTIVATOR
    draw_ds_label(ax, TF, ds_id, *id_pos)
    draw_ipr90(ax, p_log, *ipr_pos)
    return True


def _make_overview(tf_dict, tag):
    tf_list = [(TF, ds_id) for TF, ds_id in tf_dict.items() if applies_to(tag, TF)]
    if not tf_list:
        return None

    fig, axes = panel_grid(len(tf_list))

    drawn = 0
    for ax, (TF, ds_id) in zip(axes, tf_list):
        try:
            has_curve = _draw_panel(ax, TF, ds_id, tag)
        except Exception as exc:
            ax.set_visible(False)
            print(f"  [SKIP] {tag} / {TF}: {exc}")
            continue
        if has_curve:
            drawn += 1
        else:
            ax.set_visible(False)

    if drawn == 0:
        plt.close(fig)
        return None

    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    saved = 0
    for tag in EXPRESSION_PP_TAGS:
        for label, tf_dict in (("repressors", REPRESSOR_IDs), ("activators", ACTIVATOR_IDs)):
            fig = _make_overview(tf_dict, tag)
            if fig is None:
                continue
            basename = f"PP-FC_{tag}_{label}"
            for ext in ("png", "svg"):
                fig.savefig(os.path.join(OUT_DIR, f"{basename}.{ext}"))
            plt.close(fig)
            saved += 1
            print(f"Saved -> {display_path(os.path.join(OUT_DIR, f'{basename}.png'))}")

    if saved == 0:
        raise SystemExit("No fitted checkpoints found. Run scripts/ModelFit.py first.")


if __name__ == "__main__":
    main()
