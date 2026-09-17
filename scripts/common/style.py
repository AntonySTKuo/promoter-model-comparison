"""Shared style settings for model plots."""

import math

## original
# MODEL_COLORS = {
#     'GM': '#00abcc',
#     'RM': '#ff0000',
#     'AM': '#438a0f',
#     'cGM': '#3333ff',
#     'cGM_a1': '#c61aff',
#     }

## Okabe-Ito derived palette
MODEL_COLORS = {
    'GM': '#005381',  #0072B2
    'RM': '#CB7692',
    'AM': '#009E73',
    'cGM': '#D5A400',
    'cGM_a1': '#5F3D9A',
    }

MODEL_CLASS_COLORS = {
    'Model_GM': MODEL_COLORS['GM'],
    'Model_RM': MODEL_COLORS['RM'],
    'Model_AM': MODEL_COLORS['AM'],
    }

MODEL_CURVE_STYLES = {
    'GM':     {'col': MODEL_COLORS['GM'],     'lw': 1.0, 'a': 0.8,  'ls': '-',           'zorder': 7},
    'RM':     {'col': MODEL_COLORS['RM'],     'lw': 2.5, 'a': 1.0,  'ls': (2.5, (1, 3)), 'zorder': 4},
    'AM':     {'col': MODEL_COLORS['AM'],     'lw': 2.5, 'a': 1.0,  'ls': (2.5, (1, 3)), 'zorder': 4},
    'cGM':    {'col': MODEL_COLORS['cGM'],    'lw': 2.5, 'a': 0.9,  'ls': (1.5, (1, 3)), 'zorder': 5},
    'cGM_a1': {'col': MODEL_COLORS['cGM_a1'], 'lw': 2.5, 'a': 0.8,  'ls': (0.0, (1, 3)), 'zorder': 6},
    }

MODEL_CLASS_CURVE_STYLES = {
    'Model_GM': MODEL_CURVE_STYLES['GM'],
    'Model_RM': MODEL_CURVE_STYLES['RM'],
    'Model_AM': MODEL_CURVE_STYLES['AM'],
    }

MODEL_SCATTER_STYLES = {
    'GM':     {'color': MODEL_COLORS['GM'],     'alpha': 0.7, 'marker': 'o'},
    'cGM':    {'color': MODEL_COLORS['cGM'],    'alpha': 0.7, 'marker': 's'},
    'cGM_a1': {'color': MODEL_COLORS['cGM_a1'], 'alpha': 0.5, 'marker': '^'},
    }

# ── Dataset-specific model-curve panel details ───────────────────────────────
# Consumed by common.plots. Kept here so the per-dataset drawing details sit
# with the rest of the style tables; common.plots holds the drawing logic.

# Scatter marker size and edge linewidth per dataset. The Kuo promoter
# libraries carry thousands of points, so they use smaller, thinner markers
# than SCATTER_SETUP_DEFAULT -- but not so small that they disappear next to
# the other panels, and Fig2 and Fig5 share the value so DS01-03 look the same
# in both. Every other dataset falls back to SCATTER_SETUP_DEFAULT.
SCATTER_SETUP = {
    'TetR': (8, 0.4),
    'LuxR': (8, 0.4),
    'CueR': (8, 0.4),
    }

SCATTER_SETUP_DEFAULT = (12, 0.6)

# The CRP panels span far more decades than the default sci_ticks layout reads
# well at, so they place ticks explicitly. Each entry is
#     (xticks, x_labelled, yticks, y_labelled)
# where the *_labelled sets pick which ticks actually get a 10^n label; the
# remaining ticks are drawn unlabelled. Datasets absent from SELECTIVE_TICKS
# keep the default sci_ticks layout.
CRP_REPRESSOR_TFS = {'Fig2_CRP_repressor'}
CRP_ACTIVATOR_TFS = {'Fig5_CRP_-61.5', 'Fig5_CRP_-71.5', 'Fig9_CRP_-41.5'}

_CRP_ACTIVATOR_TICKS = (tuple(range(-6, 4)), {-6, -4, -2, 0, 2},
                        tuple(range(0, 7)),  {0, 2, 4, 6})

_CRP_REPRESSOR_TICKS = (tuple(range(-6, 3)), {-6, -4, -2, 0, 2},
                        tuple(range(-4, 2)), {-4, -2, 0})

SELECTIVE_TICKS = {
    **{tf: _CRP_ACTIVATOR_TICKS for tf in CRP_ACTIVATOR_TFS},
    **{tf: _CRP_REPRESSOR_TICKS for tf in CRP_REPRESSOR_TFS},
    }


PARAM_MANTISSA_DECIMALS = 1
PARAM_SCI_THRESHOLD = 0.01


def format_param(value, decimals=PARAM_MANTISSA_DECIMALS):
    value = float(value)
    if value >= PARAM_SCI_THRESHOLD:
        if value >= 100:
            return f"{value:.0f}"
        if value >= 10:
            return f"{value:.1f}"
        if value >= 1:
            return f"{value:.2f}"
        return f"{value:.{-math.floor(math.log10(value)) + 1}f}"

    log_value = math.log10(value)
    exponent = int(log_value // 1)
    mantissa = 10 ** (log_value - exponent)

    if round(mantissa, decimals) >= 10:
        mantissa /= 10
        exponent += 1

    return rf"${mantissa:.{decimals}f}$·$10^{{{exponent}}}$"


# Axis limits for model-curve BPM plots: (xmin, xmax, ymin, ymax)
REPRESSOR_LIMITS = {
    'TetR'                  : (0.4,  3.2, -2.3, 0.6),  # original = (0.4,  3.2, -2.3, 1.0)
    'AcrR'                  : (1.1,  4.5, -2.7, 0.6),  # original = (1.1,  4.5, -2.7, 1.2)
    'AgaR'                  : (1.1,  2.5, -0.8, 0.2),  # original = (1.1,  4.5, -2.7, 1.2)
    'AscG'                  : (1.1,  3.7, -1.9, 0.5),  # original = (1.1,  4.5, -2.7, 1.2)
    'GntR'                  : (1.1,  3.7, -1.9, 0.5),  # original = (1.1,  4.5, -2.7, 1.2)
    'LacI(DL5p)_Acetate'    : (1.1,  4.5, -2.7, 0.7),  # original = (1.1,  4.5, -2.7, 1.2)
    'LacI(DL5p)_Arabinose'  : (1.1,  4.5, -2.7, 0.6),  # original = (1.1,  4.5, -2.7, 1.2)
    'LacI(DL5p)_Galactose'  : (1.1,  4.5, -2.7, 0.6),  # original = (1.1,  4.5, -2.7, 1.2)
    'LacI(DL5p)_Glucose'    : (1.1,  4.5, -2.7, 0.6),  # original = (1.1,  4.5, -2.7, 1.2)
    'LacI(DL5p)_Glycerol'   : (1.1,  4.5, -2.7, 0.6),  # original = (1.1,  4.5, -2.7, 1.2)
    'LacI(DL5p)_Pyrruvate'  : (1.1,  4.5, -2.7, 0.6),  # original = (1.1,  4.5, -2.7, 1.2)
    'MngR(DL5p)'            : (1.1,  4.5, -2.7, 0.6),  # original = (1.1,  4.5, -2.7, 1.2)
    'MngR(MngRp)'           : (1.1,  2.8, -1.1, 0.3),  # original = (1.1,  4.5, -2.7, 1.2)
    'PdhR'                  : (1.1,  4.5, -2.7, 0.6),  # original = (1.1,  4.5, -2.7, 1.2)
    'UlaR'                  : (1.1,  2.5, -0.8, 0.2),  # original = (1.1,  4.5, -2.7, 1.2)
    'Fig2_CRP_repressor'    : (-6.8, 2.6, -4.3, 0.9),  # original = (-6.8, 2.6, -4.3, 1.9)
    }

ACTIVATOR_LIMITS = {
    'LuxR'                  : (0.3,  2.8, -0.3, 2.3),
    'CueR'                  : (0.1,  2.8, -0.3, 2.3),
    'CpxR(DL5p)_Arabinose'  : (1.01, 4.8, -0.3, 2.3),  # original = (1.01, 4.8, -0.3, 2.8)
    'CpxR(DL5p)_Galactose'  : (1.01, 4.8, -0.3, 2.3),  # original = (1.01, 4.8, -0.3, 2.8)
    'CpxR(DL5p)_Glucose'    : (1.01, 4.8, -0.3, 2.3),  # original = (1.01, 4.8, -0.3, 2.8)
    'CpxR(DL5p)_Glycerol'   : (1.01, 4.8, -0.3, 2.3),  # original = (1.01, 4.8, -0.3, 2.8)
    'MetR'                  : (1.01, 4.8, -0.3, 2.3),  # original = (1.01, 4.8, -0.3, 2.8)
    'SoxS(DL5p)'            : (1.01, 4.8, -0.3, 2.3),  # original = (1.01, 4.8, -0.3, 2.8)
    'SoxS(FldAp)'           : (1.01, 4.8, -0.3, 2.3),  # original = (1.01, 4.8, -0.3, 2.8)
    'Fig2_AraC'             : (0.5,  5.3, -0.4, 3.2),
    'Fig2_LasR'             : (0.2,  5.4, -0.3, 3.2),
    'Fig5_CRP_-61.5'        : (-6.9, 3.4, -0.5, 7.5),
    'Fig5_CRP_-71.5'        : (-6.9, 3.4, -0.5, 6.5),
    'Fig9_CRP_-41.5'        : (-6.9, 3.4, -0.5, 6.5),
    }

LIMITS = {**REPRESSOR_LIMITS, **ACTIVATOR_LIMITS}


# Fallback x axis for a Fig5 panel drawn without the autocrop: x = log10(P_P).
#
# One global range rather than a per-dataset one, so any two such panels can be
# read against each other directly. It spans the fitted P_latent values of every
# dataset (-10.97 to 6.01), which is ~18 decades, so narrow datasets occupy only
# part of the axis.
PP_XLIM = (-11.5, 6.5)

# Datasets from these sources get a bold, coloured DS label in the model-curve
# panels; every other dataset keeps a plain black one. Matched against the
# 'Reference' column of tables/dataset_summary.csv. Fig2 and Fig5 share both
# settings, so the same datasets stand out the same way in either figure.
DS_LABEL_BOLD_REFERENCES = {'Parisutham'}
DS_LABEL_HIGHLIGHT_COLOR = 'dodgerblue'

# Label/box colour for datasets whose Trend column reads PT (peaked
# trade-off), shared by Fig2, Fig3A, Fig3B, Fig4, and Fig5 so the same colour
# identifies the same datasets everywhere. Change it here to re-colour all of
# them at once.
PT_FRAME_COLOR = '#D7263D'
# Padding (axes fraction) around a PT dataset's ID label, for the box Fig2/Fig5
# draw around it. Independent top/bottom values, since draw_pt_id_box builds
# the box manually rather than through a text's (always-symmetric) bbox pad --
# the bottom pad is 1% less than the top so the bottom edge sits 1% higher.
PT_ID_BOX_PAD_X = 0.02
PT_ID_BOX_PAD_TOP = 0.02
PT_ID_BOX_PAD_BOTTOM = 0.01
PT_ID_BOX_LINEWIDTH = 0.8

# RM and AM return to FC = 1 at both ends of the P_P sweep -- con_model and
# reg_model meet there -- so on the shared axis their panels are mostly flat
# tail. Fig5 crops these models' panels to the stretch where the two curves
# actually separate instead of using PP_XLIM. Empty this set to put every model
# back on the shared axis.
PP_AUTOCROP_TAGS = {'RM', 'AM'}

# |log10 FC| above which the curve counts as having left FC = 1.
PP_AUTOCROP_THRESHOLD = 0.01

# Explicit y limits for the two Fig5 sub-panels, one entry per dataset, in the
# same spirit as LIMITS above: tabulated so they can be hand-tuned, rather than
# derived at draw time. Used by the RM and AM figures; the GM figures still
# scale to their own data, since GM fixes r0 = 0 and has no finite floor.
#
# The values were generated from the rules they replace, so the figures are
# unchanged on first run:
#   BASAL_YLIMS  places log10(r0) at 0.12 and log10(rmax) at 0.88 of the panel
#                (TetR at 0.16/0.80), so both asymptotes line up across panels.
#   FC_YLIMS     keeps each dataset's previously tuned height but places
#                FC = 1 at 0.85 of the panel for repressors and 0.15 for
#                activators. Editing a pair freely will move that line.
#
# Two datasets have a basal point outside these limits by design, since the
# limits follow the fitted asymptotes rather than the data: DS07 (an outlier
# above the plateau) and DS26 (one point below the floor).
BASAL_YLIMS = {
    'TetR'                 : (  0.18,  3.61),   # DS01
    'AcrR'                 : (  1.10,  4.74),   # DS04
    'AgaR'                 : (  1.34,  2.50),   # DS05
    'AscG'                 : (  1.13,  3.57),   # DS06
    'GntR'                 : (  1.25,  2.95),   # DS07
    'LacI(DL5p)_Acetate'   : (  0.80,  4.49),   # DS08
    'LacI(DL5p)_Arabinose' : (  1.00,  4.21),   # DS09
    'LacI(DL5p)_Galactose' : (  0.80,  4.26),   # DS10
    'LacI(DL5p)_Glucose'   : (  1.00,  3.89),   # DS11
    'LacI(DL5p)_Glycerol'  : (  1.00,  4.05),   # DS12
    'LacI(DL5p)_Pyrruvate' : (  1.00,  4.06),   # DS13
    'MngR(DL5p)'           : (  1.10,  4.71),   # DS14
    'MngR(MngRp)'          : (  1.35,  2.72),   # DS15
    'PdhR'                 : (  1.03,  4.72),   # DS16
    'UlaR'                 : (  1.39,  2.38),   # DS17
    'Fig2_CRP_repressor'   : ( -7.12,  2.55),   # DS27
    'LuxR'                 : (  0.30,  3.11),   # DS02
    'CueR'                 : ( -0.00,  3.24),   # DS03
    'CpxR(DL5p)_Arabinose' : (  1.10,  4.51),   # DS18
    'CpxR(DL5p)_Galactose' : (  1.12,  4.46),   # DS19
    'CpxR(DL5p)_Glucose'   : (  1.45,  4.31),   # DS20
    'CpxR(DL5p)_Glycerol'  : (  1.00,  4.50),   # DS21
    'MetR'                 : (  1.00,  4.99),   # DS22
    'SoxS(DL5p)'           : (  1.12,  4.90),   # DS23
    'SoxS(FldAp)'          : (  1.00,  3.81),   # DS24
    'Fig2_AraC'            : (  0.55,  5.57),   # DS25
    'Fig2_LasR'            : (  0.55,  5.72),   # DS26
    'Fig5_CRP_-61.5'       : ( -6.75,  3.96),   # DS28
    'Fig5_CRP_-71.5'       : ( -6.79,  3.79),   # DS29
    'Fig9_CRP_-41.5'       : ( -7.01,  3.98),   # DS30
    }

FC_YLIMS = {
    'TetR'                 : ( -2.38,  0.42),   # DS01
    'AcrR'                 : ( -2.81,  0.50),   # DS04
    'AgaR'                 : ( -0.90,  0.18),   # DS05
    'AscG'                 : ( -2.12,  0.38),   # DS06
    'GntR'                 : ( -2.12,  0.50),   # DS07
    'LacI(DL5p)_Acetate'   : ( -2.81,  0.50),   # DS08
    'LacI(DL5p)_Arabinose' : ( -2.81,  0.50),   # DS09
    'LacI(DL5p)_Galactose' : ( -2.81,  0.50),   # DS10
    'LacI(DL5p)_Glucose'   : ( -2.81,  0.50),   # DS11
    'LacI(DL5p)_Glycerol'  : ( -2.81,  0.50),   # DS12
    'LacI(DL5p)_Pyrruvate' : ( -2.81,  0.50),   # DS13
    'MngR(DL5p)'           : ( -2.81,  0.50),   # DS14
    'MngR(MngRp)'          : ( -1.20,  0.26),   # DS15
    'PdhR'                 : ( -2.81,  0.50),   # DS16
    'UlaR'                 : ( -0.80,  0.21),   # DS17
    'Fig2_CRP_repressor'   : ( -4.42,  0.99),   # DS27
    'LuxR'                 : ( -0.39,  2.35),   # DS02
    'CueR'                 : ( -0.39,  2.21),   # DS03
    'CpxR(DL5p)_Arabinose' : ( -0.39,  2.35),   # DS18
    'CpxR(DL5p)_Galactose' : ( -0.39,  2.21),   # DS19
    'CpxR(DL5p)_Glucose'   : ( -0.39,  2.35),   # DS20
    'CpxR(DL5p)_Glycerol'  : ( -0.39,  2.21),   # DS21
    'MetR'                 : ( -0.39,  1.99),   # DS22
    'SoxS(DL5p)'           : ( -0.39,  2.21),   # DS23
    'SoxS(FldAp)'          : ( -0.39,  2.21),   # DS24
    'Fig2_AraC'            : ( -0.54,  2.99),   # DS25
    'Fig2_LasR'            : ( -0.53,  2.98),   # DS26
    'Fig5_CRP_-61.5'       : ( -1.20,  6.80),   # DS28
    'Fig5_CRP_-71.5'       : ( -1.05,  5.95),   # DS29
    'Fig9_CRP_-41.5'       : ( -1.05,  5.95),   # DS30
    }


def _tint(color, t):
    """Lighten a hex color toward white by fraction t (0 = unchanged, 1 = white)."""
    c = color.lstrip('#')
    rgb = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return '#%02x%02x%02x' % tuple(round(v + (255 - v) * t) for v in rgb)


# Scatter fill and edge are the curve color lightened toward white, so the point
# cloud stays legible under heavy overplotting while still reading as the same
# series as its fitted curve.
_FACE_TINT = 0.88
_EDGE_TINT = 0.5


def _series(curve):
    return _tint(curve, _FACE_TINT), _tint(curve, _EDGE_TINT), curve


# Basal ETF- is drawn gray in every panel; the regulated series takes the color
# of the model that produced it, so Fig 5 and Fig 6 stay keyed to Fig 2.
_BASAL_FACE, _BASAL_EDGE, _BASAL_CURVE = '#f5f5f5', '#aaaaaa', '#555555'


def _palette(curve):
    face, edge, _ = _series(curve)
    return {
        'basal_face':  _BASAL_FACE,
        'basal_edge':  _BASAL_EDGE,
        'basal_curve': _BASAL_CURVE,
        'reg_face':    face,
        'reg_edge':    edge,
        'reg_curve':   curve,
        # Fold change is a single series, drawn in the regulated colors.
        'fc_face':     face,
        'fc_edge':     edge,
        'fc_curve':    curve,
        'scatter_alpha': 0.8,
        }


_REPRESSOR_COLORS = _palette(MODEL_COLORS['RM'])
_ACTIVATOR_COLORS = _palette(MODEL_COLORS['AM'])
_GM_COLORS = _palette(MODEL_COLORS['GM'])


# Basal ('con'/ETF-)/regulated ('reg'/ETF+) color scheme keyed by model tag
# rather than by TF: used by Fig5_PP-Expression.py. Every RM dataset is a
# repressor and every AM dataset is an activator, so those two tags alone pick the right palette; GM is fitted
# against both dataset types and keeps one palette (_GM_COLORS) for either.
EXPRESSION_PP_COLORS = {
    'RM': _REPRESSOR_COLORS,
    'AM': _ACTIVATOR_COLORS,
    'GM': _GM_COLORS,
    }
