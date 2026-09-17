"""
Model variant registry and summary tables derived from cached checkpoints.

The fitted checkpoints under results/models/ are the single source of truth.
Parameters are read straight out of the state_dict; metrics are recomputed from
the model plus its observations, since they are not stored in the checkpoint.
"""

import numpy as np
import pandas as pd

from Model import Model_GM, Model_RM, Model_AM, calculate_metrics
from common.cache import load_cache
from common.loader import DATASET_SUMMARY, REPRESSOR_IDs, ALL_IDs, loadData
from common.paths import MODEL_TAGS, model_cache_dir


# cls   : model class used for fitting
# name  : class name embedded in the checkpoint filename (cGM/cGM_a1 are Model_GM)
# types : 'R' repressor / 'A' activator datasets this variant applies to
MODEL_SPECS = {
    'RM':     {'cls': Model_RM, 'name': 'Model_RM', 'types': ('R',)},
    'AM':     {'cls': Model_AM, 'name': 'Model_AM', 'types': ('A',)},
    'GM':     {'cls': Model_GM, 'name': 'Model_GM', 'types': ('R', 'A')},
    'cGM':    {'cls': Model_GM, 'name': 'Model_GM', 'types': ('R', 'A')},
    'cGM_a1': {'cls': Model_GM, 'name': 'Model_GM', 'types': ('R', 'A')},
    }

PARAM_COLUMNS = ['DS', 'n', 'n_fit', 'Type',
                 'log_rmax', 'log_r0', 'log_PT', 'log_beta', 'log_alpha']
METRIC_COLUMNS = ['DS', 'n', 'n_fit', 'Type',
                  'geo_rmse', 'tls', 'r2', 'pearson_r', 'mae']

# 'Sample size' is stored with thousands separators, e.g. "16,380".
_DATASET_SIZE = (DATASET_SUMMARY.set_index('Dataset Key')['Sample size']
                 .astype(str).str.replace(',', '', regex=False).astype(int))


def tf_type(TF):
    return 'R' if TF in REPRESSOR_IDs else 'A'


def dataset_size(TF):
    """Total points in the dataset, before the Kuo libraries are subsampled."""
    return int(_DATASET_SIZE[TF])


def applies_to(tag, TF):
    return tf_type(TF) in MODEL_SPECS[tag]['types']


def datasets_for(tag):
    """Dataset keys this variant was fitted against, in canonical order."""
    return [TF for TF in ALL_IDs if applies_to(tag, TF)]


def load_model(tag, TF):
    """Load one fitted checkpoint, or None when it has not been fitted yet."""
    return load_cache(TF, MODEL_SPECS[tag]['name'], model_cache_dir(tag))


def pp_opt(model, tag):
    """
    log10 of P_P,opt, the RNAP occupancy at which |log FC| peaks:

        P_P,opt = sqrt( r0 / (r_max * F_reg) )

    with the regulated-state factor F_reg reading

        RM:  F_reg = 1 / (1 + P_T)
        AM:  F_reg = (1 + beta * P_T) / (1 + P_T)

    Setting d(log FC)/dP_P = 0 for either model gives
    r0*r_max*(1+P_P)*(...) = (...)*(r0 + r_max*P_P), which in the regime the
    peak actually sits in (P_P << 1 and r0 << r_max*P_P) reduces to the
    expression above.

    Returns None where there is no finite optimum: GM, which is fitted with
    r0 = 0, and any checkpoint fitted without r0.
    """
    if tag not in ('RM', 'AM') or not model.include_r0:
        return None

    P_T = 10.0 ** model.log_PT.item()
    if tag == 'RM':
        F_reg = 1.0 / (1.0 + P_T)
    else:
        beta = 10.0 ** model.log_beta.item()
        F_reg = (1.0 + beta * P_T) / (1.0 + P_T)
    return 0.5 * (model.log_r0.item() - model.log_rmax.item() - np.log10(F_reg))


def _params_row(model, TF):
    return {
        'TF': TF,
        'DS': ALL_IDs[TF],
        'n': dataset_size(TF),
        'n_fit': model.P_latent.shape[0],
        'Type': tf_type(TF),
        'log_rmax': round(model.log_rmax.item(), 4),
        'log_r0': round(model.log_r0.item(), 4) if model.include_r0 else np.nan,
        'log_PT': round(model.log_PT.item(), 4) if hasattr(model, 'log_PT') else np.nan,
        'log_beta': round(model.log_beta.item(), 4) if hasattr(model, 'log_beta') else np.nan,
        'log_alpha': round(model.log_alpha.item(), 4) if hasattr(model, 'log_alpha') else np.nan,
        }


def _metrics_row(model, TF, X_obs, Y_obs):
    metrics = calculate_metrics(model, TF, X_obs, Y_obs)
    row = {'TF': TF, 'DS': ALL_IDs[TF], 'n': dataset_size(TF),
           'n_fit': model.P_latent.shape[0], 'Type': tf_type(TF)}
    for key in ('geo_rmse', 'tls', 'r2', 'pearson_r', 'mae'):
        row[key] = round(metrics[key], 4)
    return row


def _empty(columns):
    return pd.DataFrame(columns=columns, index=pd.Index([], name='TF'))


def params_tables(tags=MODEL_TAGS):
    """
    {tag: DataFrame indexed by TF} of fitted parameters, read from the caches.

    Datasets without a checkpoint are omitted. No observation data is loaded --
    every value comes from the checkpoint itself.
    """
    tables = {}
    for tag in tags:
        rows = []
        for TF in datasets_for(tag):
            model = load_model(tag, TF)
            if model is not None:
                rows.append(_params_row(model, TF))
        tables[tag] = (pd.DataFrame(rows).set_index('TF')[PARAM_COLUMNS]
                       if rows else _empty(PARAM_COLUMNS))
    return tables


def metrics_tables(tags=MODEL_TAGS):
    """
    {tag: DataFrame indexed by TF} of goodness-of-fit metrics.

    Metrics are not stored in the checkpoints, so each dataset is reloaded and
    the metrics recomputed. Observations are loaded once per dataset and shared
    across model variants.
    """
    observations = {}

    def _observations(TF):
        if TF not in observations:
            _, df_sample = loadData(TF)
            observations[TF] = (np.log10(df_sample['ConExp'].values),
                                np.log10(df_sample['RegExp'].values))
        return observations[TF]

    tables = {}
    for tag in tags:
        rows = []
        for TF in datasets_for(tag):
            model = load_model(tag, TF)
            if model is None:
                continue
            X_obs, Y_obs = _observations(TF)
            rows.append(_metrics_row(model, TF, X_obs, Y_obs))
        tables[tag] = (pd.DataFrame(rows).set_index('TF')[METRIC_COLUMNS]
                       if rows else _empty(METRIC_COLUMNS))
    return tables
