"""
Draw the analytical E_TF- -- FC relationship predicted by RM and AM.

Two panels, one per (r0, rmax) pair, each showing FC against E_TF- for a series
of regulatory factors F_reg. Every curve is traced by sweeping P_P: both axes
carry model quantities in their own units, so the panel spans exactly
r0 <= E_TF- <= rmax on x and r0/rmax <= FC <= rmax/r0 on y.

Marked on top of the curves is the analytical optimum of S1 Text section II:
the dot on each curve sits at (E_TF-.opt, FC_opt) from Eq 5, and the gray line
is the locus those dots fall on, Eq 6,

    FC_opt = r0 * rmax / E_TF-.opt^2

which is a slope -2 line in log-log coordinates, drawn only across the span
the plotted F_reg series actually reaches.

Axis titles, the F_reg legend, and the A/B panel letters are added downstream
when the figure is composited, as for Fig 5 and Fig 6.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.paths import FIGURE_DIR, display_path
from common.style import MODEL_COLORS, _tint
from common.utils import sci_label


# ─────────────────────────────────────────────────────────────────────────────
# What the panels show
# ─────────────────────────────────────────────────────────────────────────────

# (r0, rmax) per panel. The ratio rmax/r0 sets both the x span and the FC
# bounds, so these two panels differ by 2 vs 4 decades of either.
PANELS = ((1.0, 1e2), (1.0, 1e3))

# F_reg > 1 activates and F_reg < 1 represses. F_reg = 1 is left out: it would
# be a flat line on FC = 1, which the baseline below already marks.
F_REG_VALUES = (1e5, 1e2, 1e1, 1e-1, 1e-2, 1e-5)

# Repression keeps the RM color and activation the AM color, as in Fig 2/5/6;
# within each, the weaker regulator is the lighter tint.
CURVE_COLORS = {
    1e5:  MODEL_COLORS['AM'],
    1e2:  _tint(MODEL_COLORS['AM'], 0.35),
    1e1:  _tint(MODEL_COLORS['AM'], 0.65),
    1e-1: _tint(MODEL_COLORS['RM'], 0.65),
    1e-2: _tint(MODEL_COLORS['RM'], 0.35),
    1e-5: MODEL_COLORS['RM'],
    }

# P_P sweep. Wide enough that the weakest F_reg still reaches F_reg*P_P >> 1,
# so every curve closes back onto FC = 1 at the rmax end of the x axis.
P_LOG_RANGE = (-12.0, 12.0)
P_POINTS = 6000


# ─────────────────────────────────────────────────────────────────────────────
# Geometry and styling
# ─────────────────────────────────────────────────────────────────────────────

MM_PER_INCH = 25.4
FIG_WIDTH_MM = 72
FIG_HEIGHT_MM = 51
LEFT_MM = 10
RIGHT_MM = 4
BOTTOM_MM = 7
TOP_MM = 4
WSPACE_MM = 10

CURVE_LINEWIDTH = 1.2
CURVE_ZORDER = 3

# The optimum locus and its dots read as a single annotation layer over the
# curves, so they share a z order above them.
LOCUS_COLOR = '#808080'
LOCUS_LINEWIDTH = 0.7
LOCUS_ZORDER = 5
DOT_COLOR = 'black'
DOT_SIZE = 9

# The FC = 1 baseline, styled as in common.plots.showPlot: dashed and behind
# the data, so it reads as "no regulation" rather than as another curve.
FC_ONE_COLOR = 'black'
FC_ONE_LINEWIDTH = 0.5
FC_ONE_LINESTYLE = '--'
FC_ONE_ZORDER = 1

TICK_LABEL_SIZE = 5
SPINE_LINEWIDTH = 0.4

# Both axes are padded past the data by this fraction of their span, so the
# curves and the dots at the corners are not clipped by the frame.
AXIS_PAD_FRAC = 0.08


def fc_curve(r0, rmax, F_reg):
    """log10 of (E_TF-, FC) along the P_P sweep, from Eq 3 of the main text."""
    P = 10.0 ** np.linspace(*P_LOG_RANGE, P_POINTS)
    con = (r0 + rmax * P) / (1.0 + P)
    reg = (r0 + rmax * F_reg * P) / (1.0 + F_reg * P)
    return np.log10(con), np.log10(reg / con)


def optimum(r0, rmax, F_reg):
    """
    log10 of (E_TF-.opt, FC_opt), the intermediate optimum of S1 Text Eq S8/S10.

    Written as the closed form in sqrt(r0), sqrt(rmax) rather than by
    substituting P_P.opt back into Eq 3, so the pair is exact rather than
    limited by the resolution of the P_P sweep.
    """
    a = np.sqrt(rmax) + np.sqrt(r0 * F_reg)
    b = np.sqrt(r0) + np.sqrt(rmax * F_reg)
    con_opt = np.sqrt(r0 * rmax) * a / b
    return np.log10(con_opt), 2.0 * np.log10(b / a)


def decade_ticks(ax, axis, lim, pad):
    """
    A labelled tick on every decade, boundaries included.

    common.utils.sci_ticks drops the tick sitting on the lower limit, which is
    where r0 (x) and r0/rmax (y) land here -- both are values the panel is
    meant to be read off against, so they get labelled like any other decade.
    """
    lo, hi = int(round(lim[0])), int(round(lim[1]))
    majors = np.arange(lo, hi + 1)
    minors = np.concatenate([np.log10(np.arange(1, 11)) + d
                             for d in np.arange(lo, hi)])

    setter = ax.set_xticks if axis == 'x' else ax.set_yticks
    setter(minors, minor=True)
    setter(majors, [sci_label(d) for d in majors])
    ax.tick_params(axis=axis, which='major', direction='in', width=0.3,
                   length=1.8, labelsize=TICK_LABEL_SIZE, pad=pad)
    ax.tick_params(axis=axis, which='minor', direction='in', width=0.3,
                   length=1)


def padded(lo, hi):
    pad = (hi - lo) * AXIS_PAD_FRAC
    return lo - pad, hi + pad


def draw_panel(ax, r0, rmax):
    # E_TF- runs from r0 to rmax, and FC_opt is bounded by r0/rmax and
    # rmax/r0, so the data span of each axis follows from the pair alone.
    xlim = (np.log10(r0), np.log10(rmax))
    span = xlim[1] - xlim[0]
    ax.set_xlim(*padded(*xlim))
    ax.set_ylim(*padded(-span, span))

    # The locus is only meaningful where the plotted F_reg series puts optima
    # on it: it is cut at the dots of the strongest activator and repressor
    # rather than extrapolated out to the F_reg -> 0 and -> infinity corners.
    ends = [optimum(r0, rmax, F) for F in (max(F_REG_VALUES), min(F_REG_VALUES))]
    ax.plot(*zip(*ends), color=LOCUS_COLOR, linewidth=LOCUS_LINEWIDTH,
            zorder=LOCUS_ZORDER)

    ax.axhline(0, color=FC_ONE_COLOR, linewidth=FC_ONE_LINEWIDTH,
               linestyle=FC_ONE_LINESTYLE, zorder=FC_ONE_ZORDER)

    for F_reg in F_REG_VALUES:
        con_log, fc_log = fc_curve(r0, rmax, F_reg)
        ax.plot(con_log, fc_log, color=CURVE_COLORS[F_reg],
                linewidth=CURVE_LINEWIDTH, solid_capstyle='round',
                zorder=CURVE_ZORDER)
        ax.scatter(*optimum(r0, rmax, F_reg), s=DOT_SIZE, color=DOT_COLOR,
                   linewidths=0, zorder=LOCUS_ZORDER + 1)

    decade_ticks(ax, 'x', xlim, pad=1.5)
    decade_ticks(ax, 'y', (-span, span), pad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)


def make_figure():
    fig = plt.figure(
        figsize=(FIG_WIDTH_MM / MM_PER_INCH, FIG_HEIGHT_MM / MM_PER_INCH),
        dpi=600)

    panel_w_mm = (FIG_WIDTH_MM - LEFT_MM - RIGHT_MM - WSPACE_MM) / len(PANELS)
    panel_h_mm = FIG_HEIGHT_MM - TOP_MM - BOTTOM_MM

    for idx, (r0, rmax) in enumerate(PANELS):
        left_mm = LEFT_MM + idx * (panel_w_mm + WSPACE_MM)
        ax = fig.add_axes([
            left_mm / FIG_WIDTH_MM,
            BOTTOM_MM / FIG_HEIGHT_MM,
            panel_w_mm / FIG_WIDTH_MM,
            panel_h_mm / FIG_HEIGHT_MM,
            ])
        draw_panel(ax, r0, rmax)

    return fig


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig = make_figure()
    for ext in ("png", "svg"):
        path = os.path.join(FIGURE_DIR, f"analytical_solution.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"Saved -> {display_path(os.path.join(FIGURE_DIR, 'analytical_solution.png'))}")


if __name__ == "__main__":
    main()
