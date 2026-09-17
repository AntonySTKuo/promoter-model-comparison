"""
Shared panel machinery for the figures drawn against log10(P_P).

Fig5_PP-Expression.py (E_TF-/E_TF+ on one axis) and Fig6_PP-FC.py (the fold
change on its own) show the same datasets over the same x axis and differ only
in what they put on y, so the per-dataset tuning that has to agree between them
-- the x crop and its per-dataset padding overrides, the tick thinning, the
r0/rmax y anchoring, the DS label and its PT box, the grid geometry -- lives
here rather than being copied into both.

common.plots holds the equivalent helpers for the Fig2 panels, which are drawn
against log10(ConExp) instead and share none of this.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.loader import DATASET_INFO
from common.plots import draw_pt_id_box
from common.style import (BASAL_YLIMS, CRP_ACTIVATOR_TFS, CRP_REPRESSOR_TFS,
                          DS_LABEL_BOLD_REFERENCES, DS_LABEL_HIGHLIGHT_COLOR,
                          FC_YLIMS, PP_AUTOCROP_TAGS, PP_AUTOCROP_THRESHOLD,
                          PP_XLIM, PT_FRAME_COLOR, PT_ID_BOX_LINEWIDTH,
                          PT_ID_BOX_PAD_BOTTOM, PT_ID_BOX_PAD_TOP,
                          PT_ID_BOX_PAD_X)
from common.utils import sci_label, sci_ticks


# ─────────────────────────────────────────────────────────────────────────────
# Grid geometry
# ─────────────────────────────────────────────────────────────────────────────

MM_PER_INCH = 25.4
OVERVIEW_WIDTH_MM = 170
OVERVIEW_HEIGHT_MM = 105
OVERVIEW_LEFT_MM = 10
OVERVIEW_RIGHT_MM = 4
OVERVIEW_BOTTOM_MM = 7
OVERVIEW_TOP_MM = 4
OVERVIEW_WSPACE_MM = 7
OVERVIEW_HSPACE_MM = 5


def panel_grid(n_panels, ncols=4):
    """
    A figure plus one axes per dataset, laid out in mm on a fixed-size canvas.

    Sized in mm rather than through subplots/tight_layout so a panel occupies
    the same area in every figure this module draws, and so the repressor and
    activator figures composite together without rescaling.
    """
    nrows = (n_panels + ncols - 1) // ncols
    fig = plt.figure(
        figsize=(OVERVIEW_WIDTH_MM / MM_PER_INCH, OVERVIEW_HEIGHT_MM / MM_PER_INCH),
        dpi=600)

    panel_w_mm = (
        OVERVIEW_WIDTH_MM - OVERVIEW_LEFT_MM - OVERVIEW_RIGHT_MM
        - (ncols - 1) * OVERVIEW_WSPACE_MM
        ) / ncols
    panel_h_mm = (
        OVERVIEW_HEIGHT_MM - OVERVIEW_TOP_MM - OVERVIEW_BOTTOM_MM
        - (nrows - 1) * OVERVIEW_HSPACE_MM
        ) / nrows

    axes = []
    for idx in range(n_panels):
        row, col = divmod(idx, ncols)
        left_mm = OVERVIEW_LEFT_MM + col * (panel_w_mm + OVERVIEW_WSPACE_MM)
        bottom_mm = (OVERVIEW_BOTTOM_MM
                     + (nrows - 1 - row) * (panel_h_mm + OVERVIEW_HSPACE_MM))
        axes.append(fig.add_axes([
            left_mm / OVERVIEW_WIDTH_MM,
            bottom_mm / OVERVIEW_HEIGHT_MM,
            panel_w_mm / OVERVIEW_WIDTH_MM,
            panel_h_mm / OVERVIEW_HEIGHT_MM,
            ]))
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Axis ranges
# ─────────────────────────────────────────────────────────────────────────────

# RM/AM x crop: padding (in decades of log10 P_P) added on each side of the
# peak-centered departure window (see panel_xlim).
XLIM_PAD = 0.3

# Per-dataset override of XLIM_PAD, keyed by TF. The shared crop is
# peak-centered on where |FC| is largest, which for some datasets sits well
# away from where the fitted P_latent values actually are -- e.g. AgaR's peak
# is at log10(P_P)=0.28 but its 42 points top out at -0.63, so the
# (mathematically correct) shared window still leaves the point cloud crammed
# against one edge with near-zero margin. These datasets are padded further so
# their points sit comfortably inside the panel rather than flush against it
# (the two CpxR entries each had a single point landing just past the right
# edge); every other RM/AM panel keeps the shared default.
XLIM_PAD_OVERRIDES = {
    "TetR":                  1.0,
    "AgaR":                  0.6,
    "UlaR":                  1.1,
    "CpxR(DL5p)_Galactose":  0.7,
    "CpxR(DL5p)_Glycerol":   0.7,
    }

# GM x/y limits are set from the observed point positions instead: GM's
# con_model/reg_model curves don't share RM/AM's FC=1-at-both-ends or
# shared-rmax structure (alpha is trained independently), so the FC-departure
# crop and the r0/rmax anchoring below don't carry over to it.
GM_XLIM_PAD_DECADES = 0.3
GM_YLIM_PAD_FRAC_BOTTOM = 0.12
# Extra headroom above the highest point, well past GM_YLIM_PAD_FRAC_BOTTOM, so
# the point cloud clears the DS ID label in a top corner.
GM_YLIM_PAD_FRAC_TOP = 0.30

# RM/AM y axis is pinned to the fitted r0/rmax asymptotes (shared by con_model
# and reg_model -- both curves flatten to the same r0 at low P_P and the same
# rmax at high P_P) rather than auto-scaled to the observed data, so r0 and
# rmax sit at the same axes-fraction height in every panel. RMAX_FRAC is well
# below 1.0 so the flat rmax tail (which runs under a top corner, where the DS
# ID label may sit) never touches the label.
R0_FRAC = 0.12
RMAX_FRAC = 0.88

# Per-dataset override of (R0_FRAC, RMAX_FRAC), keyed by TF. TetR (DS01) is the
# only dataset that needs its floor/ceiling pushed higher.
YFRAC_OVERRIDES = {"TetR": (0.16, 0.80)}

# Height of FC = 1 within a fold-change panel, as an axes fraction. Repression
# drives FC below 1 and activation above it, so pinning the line high for
# repressors and low for activators gives each the room its curve actually
# uses, and puts the line at the same height in every panel of a figure.
FC_ONE_FRAC_REPRESSOR = 0.85
FC_ONE_FRAC_ACTIVATOR = 0.15


def panel_xlim(model, tag, p_log, TF):
    """
    x limits for one panel.

    RM/AM: a peak-centered crop. RM and AM sit at FC = 1 (con_model ==
    reg_model) at both ends of the P_P sweep, so panels are cropped to the
    stretch where the two curves actually separate. The window is built around
    log10(P_P) at max|FC| (the point of greatest separation between the
    E_TF-/E_TF+ curves), so that point always lands at the panel's horizontal
    center, and it is NOT widened to keep every observation in frame: a few
    baseline-noise points sitting well outside the departure window would
    otherwise drag a wide flat FC~1 stretch back into view on one side, which
    is exactly what this crop is meant to cut away. Padded by XLIM_PAD, or an
    XLIM_PAD_OVERRIDES entry for the handful of datasets that need more room.

    GM: FC isn't a meaningful crop signal here, so the range is set directly
    from the fitted P_latent positions instead, padded by GM_XLIM_PAD_DECADES.
    """
    if tag not in PP_AUTOCROP_TAGS:
        if len(p_log):
            return (float(p_log.min()) - GM_XLIM_PAD_DECADES,
                    float(p_log.max()) + GM_XLIM_PAD_DECADES)
        return PP_XLIM

    Ps = torch.logspace(-25, 20, 40000, dtype=torch.float32)
    with torch.no_grad():
        x = np.log10(Ps.numpy())
        fc = model.reg_model(Ps).numpy() - model.con_model(Ps).numpy()

    finite = np.isfinite(fc)
    x, fc = x[finite], fc[finite]
    departed = np.abs(fc) > PP_AUTOCROP_THRESHOLD
    if not departed.any():
        return PP_XLIM

    peak = float(x[np.argmax(np.abs(fc))])
    lo, hi = float(x[departed][0]), float(x[departed][-1])
    pad = XLIM_PAD_OVERRIDES.get(TF, XLIM_PAD)
    half_width = max(peak - lo, hi - peak) + pad
    return (peak - half_width, peak + half_width)


def expression_ylim(model, tag, y_obs, TF):
    """
    y limits for an expression panel, covering the E_TF- and E_TF+ series.

    RM/AM read style.BASAL_YLIMS, which is hand-tunable. Those limits are
    pinned to the fitted r0/rmax asymptotes, and both curves flatten to the
    same r0 at low P_P and the same rmax at high P_P, so the range that frames
    the basal series frames the regulated one too. The anchoring that generated
    that table stays as the fallback, so a dataset added without an entry still
    gets sensible limits: the floor sits at R0_FRAC and the ceiling at
    RMAX_FRAC of the panel height (or a YFRAC_OVERRIDES entry), so both line up
    across every panel regardless of the dataset's own dynamic range. The
    ceiling is evaluated as the actual P_P -> infinity value of both curves (at
    P_P = 1e15) rather than read off log_rmax directly, since that is what
    con_model/reg_model actually converge to for these two models. The floor
    uses log10(r0) directly (RM/AM fits always include a finite r0).

    GM: r0/rmax aren't a meaningful anchor here, so the range is set directly
    from the observed values in y_obs instead, padded by
    GM_YLIM_PAD_FRAC_BOTTOM/TOP of their span -- asymmetric, since the DS ID
    label sits in a top corner and needs more clearance above the highest point
    than the floor needs below the lowest.
    """
    limits = BASAL_YLIMS.get(TF)
    if limits is not None and tag in PP_AUTOCROP_TAGS:
        return limits

    if tag not in PP_AUTOCROP_TAGS:
        y = y_obs[np.isfinite(y_obs)]
        span = y.max() - y.min()
        return (y.min() - span * GM_YLIM_PAD_FRAC_BOTTOM,
                y.max() + span * GM_YLIM_PAD_FRAC_TOP)

    P_inf = torch.tensor(1e15, dtype=torch.float32)
    with torch.no_grad():
        top = model.con_model(P_inf).item()
    bottom = model.log_r0.item()

    r0_frac, rmax_frac = YFRAC_OVERRIDES.get(TF, (R0_FRAC, RMAX_FRAC))
    height = (top - bottom) / (rmax_frac - r0_frac)
    y_lo = bottom - r0_frac * height
    return (y_lo, y_lo + height)


def fc_ylim(TF, fc_log, is_repressor):
    """
    y limits for a fold-change panel.

    Read from the hand-tunable style.FC_YLIMS table. Datasets absent from it
    fall back to their own padded range, positioned so FC = 1 keeps the height
    the table was built with.
    """
    limits = FC_YLIMS.get(TF)
    if limits is not None:
        return limits

    frac = FC_ONE_FRAC_REPRESSOR if is_repressor else FC_ONE_FRAC_ACTIVATOR
    y = fc_log[np.isfinite(fc_log)]
    height = (y.max() - y.min()) * (1 + GM_YLIM_PAD_FRAC_BOTTOM + GM_YLIM_PAD_FRAC_TOP)
    return (-frac * height, (1.0 - frac) * height)


# ─────────────────────────────────────────────────────────────────────────────
# Ticks and frame
# ─────────────────────────────────────────────────────────────────────────────

TICK_LABEL_SIZE = 5

# A panel on the uncropped axis spans many decades, where sci_ticks' one label
# per decade collides; thin them to at most this many.
X_MAX_LABELS = 5

# The CRP datasets (DS27-DS30) span ~10 decades of expression, and their fold
# change spans up to 8; a label at every decade crowds together, so label only
# the even exponents (10^0, 10^2, ...). Reuses the CRP TF sets from
# common.style rather than listing the four dataset keys again.
CRP_TFS = CRP_REPRESSOR_TFS | CRP_ACTIVATOR_TFS

# Y tick-label thinning for an expression panel, as {dataset: (step, offset)}:
# a decade is labelled when its exponent % step == offset. These datasets span
# far more decades between r0 and r_max than a panel can label one by one.
# Datasets absent from this table keep a label on every decade.
EXPRESSION_Y_TICK_STEPS = {
    **{tf: (2, 0) for tf in CRP_TFS},           # CRP, ~10 decades: 10^(2n)
    'Fig2_AraC': (2, 1),                        # AraC / LasR, ~5: 10^(2n-1)
    'Fig2_LasR': (2, 1),
    }

# Same, for a fold-change panel. Only the CRP datasets need it there.
FC_Y_TICK_STEPS = {tf: (2, 0) for tf in CRP_TFS}


def thin_x_labels(ax, max_labels=X_MAX_LABELS):
    """Blank out all but every k-th major x label; ticks themselves are kept."""
    xl, xh = ax.get_xlim()
    majors = np.arange((xl // 1) + 1, (xh // 1) + 1)
    if len(majors) <= max_labels:
        return
    step = int(np.ceil(len(majors) / max_labels))
    keep = {float(v) for v in majors if int(v) % step == 0}
    ax.set_xticks(majors, [sci_label(v, keep) for v in majors])


def thin_y_labels(ax, step, offset=0):
    """
    Label only the decades whose exponent % step == offset; the ticks
    themselves are kept. step=2/offset=0 gives 10^(2n), step=2/offset=1 gives
    10^(2n-1), step=3/offset=0 gives 10^(3n). Python's modulo returns a
    non-negative result here, so negative exponents follow the same rule.
    """
    yl, yh = ax.get_ylim()
    majors = np.arange((yl // 1) + 1, (yh // 1) + 1)
    keep = {float(v) for v in majors if int(v) % step == offset}
    ax.set_yticks(majors, [sci_label(v, keep) for v in majors])


def apply_axes(ax, xlim, ylim):
    """Axis limits, sci-notation ticks, and a frame of uniform weight."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    sci_ticks(ax, 'x', ticksize=4.5, pad=1.5)
    sci_ticks(ax, 'y', ticksize=4.5, pad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


# ─────────────────────────────────────────────────────────────────────────────
# Series and annotations
# ─────────────────────────────────────────────────────────────────────────────

CURVE_LINEWIDTH = 1.0
DS_LABEL_SIZE = 7
IPR90_LABEL_SIZE = 6

# The P_P,opt marker is drawn on top of the data: it marks a position on the x
# axis, so it has to stay readable where it crosses the point clouds and curves.
PP_OPT_COLOR = 'black'
PP_OPT_LINEWIDTH = 0.5
PP_OPT_LINESTYLE = (0, (3, 2))
PP_OPT_ZORDER = 4


def model_curve(model, xlim):
    """con_model and reg_model swept over the panel's P_P window."""
    Ps = torch.logspace(xlim[0], xlim[1], 200, dtype=torch.float32)
    with torch.no_grad():
        con_fit = model.con_model(Ps).numpy()
        reg_fit = model.reg_model(Ps).numpy()
    return np.log10(Ps.numpy()), con_fit, reg_fit


def plot_series(ax, p_log, y_obs, p_curve, y_curve, face, edge, curve, s, lw, alpha):
    """
    One scatter plus its fitted curve.

    alpha=1.0 on the stroke so it renders as the exact MODEL_COLORS hex Fig2
    uses; at 0.8 the same colour came out visibly lighter than its counterpart.
    """
    ax.scatter(p_log, y_obs, color=face, alpha=alpha, edgecolors=edge,
               linewidth=lw, s=s, zorder=2)
    valid = np.isfinite(y_curve)
    ax.plot(p_curve[valid], y_curve[valid], color=curve,
            linewidth=CURVE_LINEWIDTH, alpha=1.0, zorder=3)


def draw_pp_opt(ax, p_opt):
    """The P_P,opt marker; a no-op when the model has no finite optimum."""
    if p_opt is None:
        return
    ax.axvline(p_opt, color=PP_OPT_COLOR, linewidth=PP_OPT_LINEWIDTH,
               linestyle=PP_OPT_LINESTYLE, zorder=PP_OPT_ZORDER)


def draw_ds_label(ax, TF, ds_id, x, y, ha, va):
    """
    The dataset ID, in the corner given by (x, y, ha, va).

    Bold and coloured for the sources in DS_LABEL_BOLD_REFERENCES, and boxed
    for datasets whose Trend column reads PT, so the same datasets stand out
    the same way in Fig2, Fig3, Fig4, Fig5, and Fig6.
    """
    highlight = DATASET_INFO[TF]["Reference"] in DS_LABEL_BOLD_REFERENCES
    text_obj = ax.text(x, y, ds_id, transform=ax.transAxes,
                       fontsize=DS_LABEL_SIZE, ha=ha, va=va,
                       fontweight="bold" if highlight else "normal",
                       color=DS_LABEL_HIGHLIGHT_COLOR if highlight else "black")
    if DATASET_INFO[TF]["Trend"] == "PT":
        draw_pt_id_box(ax, text_obj, PT_ID_BOX_PAD_X, PT_ID_BOX_PAD_TOP,
                       PT_ID_BOX_PAD_BOTTOM, PT_FRAME_COLOR, PT_ID_BOX_LINEWIDTH)


def draw_ipr90(ax, p_log, x, y, ha, va):
    """
    IPR90 -- the 5th-95th interpercentile range of the fitted P_latent -- in
    the corner given by (x, y, ha, va).

    In place of max-min, which is set by a single most extreme point and grows
    with n (Kuo's 8000-row resample vs. 21-91 rows elsewhere), so it isn't
    comparable across datasets; IPR90 is robust to both.
    """
    p5, p95 = np.percentile(p_log, [5, 95])
    ax.text(x, y, rf"$\mathrm{{IPR}}_{{90}}$ = {float(p95 - p5):.1f}",
            transform=ax.transAxes, fontsize=IPR90_LABEL_SIZE, ha=ha, va=va)
