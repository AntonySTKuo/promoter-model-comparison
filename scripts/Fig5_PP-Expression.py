"""
Draw basal and regulated expression against RNAP occupancy: E_TF-/E_TF+ vs P_P.

One panel per dataset, carrying both series on a single y axis:

    x = log10(P_P)   -- the per-dataset latent (model.P_latent), cropped for
                        RM/AM since both curves sit flat at either end of the
                        sweep (see PP_AUTOCROP_TAGS)
    y = log10(E_TF-) = log10(ConExp)   -- basal points and con_model curve
        log10(E_TF+) = log10(RegExp)   -- regulated points and reg_model curve

Both curves share the same r0 floor and r_max ceiling, so the single y axis is
pinned to those asymptotes and both series fit inside it. Each panel also marks
P_P,opt, the RNAP occupancy at which the fold change is most extreme.

Fig6_PP-FC.py is the companion figure: the same datasets over the same x axis,
with the fold change E_TF+/E_TF- on y instead. Everything the two share -- the
x crop, the tick thinning, the DS label and its PT box, the grid geometry --
lives in common.pp_panels.

Covers RM (repressors), AM (activators), and GM -- GM is fitted against both
dataset types, so it gets its own repressor and activator figures too, each
using the GM curve color instead of the RM/AM palette. cGM/cGM_a1 are not
covered by this figure.

Axis titles, panel letters, and the E_TF-/E_TF+ legend are added downstream when
the repressor and activator figures are composited; this script draws the panel
grids only.

Usage:
    python scripts/Fig5_PP-Expression.py

Outputs (results/figures/, alongside the other figure scripts):
    PP-Expression_RM_repressors.png / .svg
    PP-Expression_AM_activators.png / .svg
    PP-Expression_GM_repressors.png / .svg
    PP-Expression_GM_activators.png / .svg
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
from common.pp_panels import (EXPRESSION_Y_TICK_STEPS, TICK_LABEL_SIZE,
                              apply_axes, draw_ds_label, draw_pp_opt,
                              expression_ylim, model_curve, panel_grid,
                              panel_xlim, plot_series, thin_x_labels,
                              thin_y_labels)
from common.style import EXPRESSION_PP_COLORS


EXPRESSION_PP_TAGS = ('RM', 'AM', 'GM')

SETUP = dataset_scatter_setup(ALL_IDs)

OUT_DIR = FIGURE_DIR

# Both curves run low-left to high-right, so the top-left corner is the one
# they never reach; the DS ID goes there, clear of the r_max plateau.
DS_LABEL_POS = (0.03, 0.96, "left", "top")


def _draw_panel(ax, TF, ds_id, tag):
    """One dataset: both expression series on a single axis. False if empty."""
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

    xlim = panel_xlim(model, tag, p_log, TF)
    s, lw = SETUP[TF]
    colors = EXPRESSION_PP_COLORS[tag]
    p_curve, con_fit, reg_fit = model_curve(model, xlim)

    apply_axes(ax, xlim,
               expression_ylim(model, tag, np.concatenate([con_log, reg_log]), TF))

    plot_series(ax, p_log, con_log, p_curve, con_fit,
                colors['basal_face'], colors['basal_edge'], colors['basal_curve'],
                s, lw, colors['scatter_alpha'])
    plot_series(ax, p_log, reg_log, p_curve, reg_fit,
                colors['reg_face'], colors['reg_edge'], colors['reg_curve'],
                s, lw, colors['scatter_alpha'])
    draw_pp_opt(ax, pp_opt(model, tag))

    thin_x_labels(ax)
    y_step = EXPRESSION_Y_TICK_STEPS.get(TF)
    if y_step is not None:
        thin_y_labels(ax, *y_step)
    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_SIZE)

    draw_ds_label(ax, TF, ds_id, *DS_LABEL_POS)
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
            basename = f"PP-Expression_{tag}_{label}"
            for ext in ("png", "svg"):
                fig.savefig(os.path.join(OUT_DIR, f"{basename}.{ext}"))
            plt.close(fig)
            saved += 1
            print(f"Saved -> {display_path(os.path.join(OUT_DIR, f'{basename}.png'))}")

    if saved == 0:
        raise SystemExit("No fitted checkpoints found. Run scripts/ModelFit.py first.")


if __name__ == "__main__":
    main()
