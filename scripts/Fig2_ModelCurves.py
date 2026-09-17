"""
Draw model curves together for visual comparison.

Outputs are the two multi-panel overviews in results/figures/, built from the
fitted caches under results/models/:
    - GM plus RM for repressors or AM for activators
    - cGM (GM*) and cGM_a1 (GM**)
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.cache import load_cache
from common.loader import (REPRESSOR_IDs, ACTIVATOR_IDs, ALL_IDs,
                           DATASET_INFO, loadData)
from common.paths import FIGURE_DIR, display_path, model_cache_dir
from common.plots import (apply_selective_ticks, dataset_scatter_setup,
                          draw_pt_id_box, showCurve, showPlot, showScatter)
from common.style import (DS_LABEL_BOLD_REFERENCES, DS_LABEL_HIGHLIGHT_COLOR,
                          LIMITS, MODEL_CURVE_STYLES, PT_FRAME_COLOR,
                          PT_ID_BOX_LINEWIDTH, PT_ID_BOX_PAD_BOTTOM,
                          PT_ID_BOX_PAD_TOP, PT_ID_BOX_PAD_X)


SETUP = dataset_scatter_setup(ALL_IDs)

OUT_DIR = FIGURE_DIR

CACHE_GM = model_cache_dir("GM")
CACHE_RM = model_cache_dir("RM")
CACHE_AM = model_cache_dir("AM")
CACHE_CGM = model_cache_dir("cGM")
CACHE_CGM_A1 = model_cache_dir("cGM_a1")

CURVE_STYLES = MODEL_CURVE_STYLES

MM_PER_INCH = 25.4
OVERVIEW_WIDTH_MM = 170
OVERVIEW_HEIGHT_MM = 105
OVERVIEW_LEFT_MM = 10
OVERVIEW_RIGHT_MM = 4
OVERVIEW_BOTTOM_MM = 7
OVERVIEW_TOP_MM = 4
OVERVIEW_WSPACE_MM = 7
OVERVIEW_HSPACE_MM = 5
TICK_LABEL_SIZE = 5
DS_LABEL_SIZE = 7


def _load_models(TF):
    regulated_tag = "RM" if TF in REPRESSOR_IDs else "AM"
    regulated_cache = CACHE_RM if TF in REPRESSOR_IDs else CACHE_AM
    regulated_model = "Model_RM" if TF in REPRESSOR_IDs else "Model_AM"

    return [
        (regulated_tag, load_cache(TF, regulated_model, regulated_cache)),
        ("GM", load_cache(TF, "Model_GM", CACHE_GM)),
        ("cGM", load_cache(TF, "Model_GM", CACHE_CGM)),
        ("cGM_a1", load_cache(TF, "Model_GM", CACHE_CGM_A1)),
        ]


def _draw_panel(ax, TF, ds_id):
    _, df_sample = loadData(TF)
    x_obs = np.log10(df_sample["ConExp"].values)
    y_obs = np.log10(df_sample["RegExp"].values)

    limits = LIMITS[TF]
    s, lw = SETUP[TF]

    showPlot(ax, limits)
    showScatter(ax, x_obs, y_obs, limits, s=s, lw=lw)

    has_curves = False
    for tag, model in _load_models(TF):
        if model is None:
            continue
        style = CURVE_STYLES[tag]
        showCurve(ax, model, **style)
        has_curves = True

    apply_selective_ticks(ax, TF)
    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_SIZE)
    highlight = DATASET_INFO[TF]["Reference"] in DS_LABEL_BOLD_REFERENCES
    is_pt = DATASET_INFO[TF]["Trend"] == "PT"
    text_obj = ax.text(0.97, 0.96, ds_id, transform=ax.transAxes,
                       fontsize=DS_LABEL_SIZE, ha="right", va="top",
                       fontweight="bold" if highlight else "normal",
                       color=DS_LABEL_HIGHLIGHT_COLOR if highlight else "black")
    if is_pt:
        draw_pt_id_box(ax, text_obj, PT_ID_BOX_PAD_X, PT_ID_BOX_PAD_TOP,
                       PT_ID_BOX_PAD_BOTTOM, PT_FRAME_COLOR, PT_ID_BOX_LINEWIDTH)

    return has_curves


def _make_overview(tf_dict, ncols=4):
    tf_list = list(tf_dict.items())
    nrows = (len(tf_list) + ncols - 1) // ncols
    fig = plt.figure(figsize=(OVERVIEW_WIDTH_MM / MM_PER_INCH, OVERVIEW_HEIGHT_MM / MM_PER_INCH), dpi=600)

    panel_w_mm = (
        OVERVIEW_WIDTH_MM - OVERVIEW_LEFT_MM - OVERVIEW_RIGHT_MM
        - (ncols - 1) * OVERVIEW_WSPACE_MM
        ) / ncols
    panel_h_mm = (
        OVERVIEW_HEIGHT_MM - OVERVIEW_TOP_MM - OVERVIEW_BOTTOM_MM
        - (nrows - 1) * OVERVIEW_HSPACE_MM
        ) / nrows

    for idx, (TF, ds_id) in enumerate(tf_list):
        row = idx // ncols
        col = idx % ncols
        left_mm = OVERVIEW_LEFT_MM + col * (panel_w_mm + OVERVIEW_WSPACE_MM)
        bottom_mm = (
            OVERVIEW_BOTTOM_MM
            + (nrows - 1 - row) * (panel_h_mm + OVERVIEW_HSPACE_MM)
            )
        ax = fig.add_axes([
            left_mm / OVERVIEW_WIDTH_MM,
            bottom_mm / OVERVIEW_HEIGHT_MM,
            panel_w_mm / OVERVIEW_WIDTH_MM,
            panel_h_mm / OVERVIEW_HEIGHT_MM,
            ])
        try:
            has_curves = _draw_panel(ax, TF, ds_id)
        except Exception as exc:
            ax.set_visible(False)
            print(f"[SKIP] {TF}: {exc}")
            continue
        if not has_curves:
            ax.set_visible(False)

    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for label, tf_dict in (("repressors", REPRESSOR_IDs), ("activators", ACTIVATOR_IDs)):
        fig = _make_overview(tf_dict)
        basename = f"curves_{label}"
        for ext in ("png", "svg"):
            fig.savefig(os.path.join(OUT_DIR, f"{basename}.{ext}"))
        plt.close(fig)
        print(f"Saved -> {display_path(os.path.join(OUT_DIR, f'{basename}.png'))}")


if __name__ == "__main__":
    main()
