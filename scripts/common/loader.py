import os

import numpy as np
import pandas as pd

from common.paths import repo_path
from common.utils import Motif2Seqs

TABLES_DIR = repo_path("tables")
DATASET_SUMMARY_PATH = repo_path("tables", "dataset_summary.csv")
DATASET_SUMMARY_COLUMNS = [
    "TF",
    "Regulation",
    "Trend",
    "Medium",
    "Technology",
    "Sample size",
    "Dataset ID",
    "Reference",
    "Dataset Key",
    "Directory",
    ]


# Trend classification carried in the registry's 'Trend' column, mirroring the
# published Table 1: PT = peaked trade-off, IS = inverse scaling.
TRENDS = {"PT": "peaked trade-off", "IS": "inverse scaling"}


def _table_path(*parts):
    return os.path.join(TABLES_DIR, *parts)


def stratified_sample(df, col, counts, seed=77777):
    """
    Sample `counts[i]` rows (with replacement) from each of len(counts) equal-width
    bins of log10(df[col]) -- used to flatten the Kuo expression distributions.

    Bins on an expression column.
    separate routine that bins on summed -35/-10 energy instead.
    """
    d = df.copy()
    d["grp"] = pd.cut(np.log10(d[col]), bins=len(counts))
    parts = []
    for grp, cnt in zip(d["grp"].unique(), counts):
        block = d[d["grp"] == grp]
        if block.empty:
            continue
        parts.append(block.sample(cnt, replace=True, random_state=seed))
    out = pd.concat(parts).drop(columns="grp")
    return out


def _load_dataset_summary():
    summary = pd.read_csv(DATASET_SUMMARY_PATH, dtype=str).fillna("")
    missing = [col for col in DATASET_SUMMARY_COLUMNS if col not in summary.columns]
    if missing:
        raise ValueError(f"dataset_summary.csv missing required columns: {missing}")
    if summary["Dataset Key"].duplicated().any():
        dupes = summary.loc[summary["Dataset Key"].duplicated(), "Dataset Key"].tolist()
        raise ValueError(f"dataset_summary.csv has duplicate Dataset Key values: {dupes}")
    if not summary["Regulation"].isin(["Repressor", "Activator"]).all():
        raise ValueError("dataset_summary.csv Regulation must be 'Repressor' or 'Activator'.")
    if not summary["Trend"].isin(list(TRENDS)).all():
        raise ValueError(f"dataset_summary.csv Trend must be one of {sorted(TRENDS)}.")
    return summary


def _dataset_ids_for(summary, regulation):
    rows = summary[summary["Regulation"] == regulation]
    return dict(zip(rows["Dataset Key"], rows["Dataset ID"]))


DATASET_SUMMARY = _load_dataset_summary()
DATASET_INFO = DATASET_SUMMARY.set_index("Dataset Key").to_dict("index")
REPRESSOR_IDs = _dataset_ids_for(DATASET_SUMMARY, "Repressor")
ACTIVATOR_IDs = _dataset_ids_for(DATASET_SUMMARY, "Activator")
ALL_IDs = {**REPRESSOR_IDs, **ACTIVATOR_IDs}


def loadOurData(lib):
    dfB = pd.read_csv(_table_path("source_Kuo", f"PL_{lib}b.tsv"), sep='\t', index_col=0)
    dfI = pd.read_csv(_table_path("source_Kuo", f"PL_{lib}i.tsv"), sep='\t', index_col=0)

    if lib == 'TetR':
        df = pd.DataFrame({'ConExp': 10**dfI['LogGFP'], 'RegExp': 10**dfB['LogGFP']})
    elif lib in ['LuxR', 'CueR']:
        df = pd.DataFrame({'ConExp': 10**dfB['LogGFP'], 'RegExp': 10**dfI['LogGFP']})
    
    df['FC'] = df['RegExp'] / df['ConExp']
    return df


def loadSciData(file, dir_path=None):
    if dir_path is None:
        dir_path = _table_path("source_Parisutham")
    df = pd.read_csv(os.path.join(dir_path, f"{file}.csv"), index_col=0).dropna()
    
    df['ConExp'] = df['Yfinal']
    df['FC'] = df['FCfinal']
    df['RegExp'] = df['ConExp'] * df['FC']
    
    df['ConExp_SE'] = df['WeightY']
    df['FC_SE'] = df['WeightF']
    
    return df[['ConExp', 'ConExp_SE', 'RegExp', 'FC', 'FC_SE']]


def loadChenData(file):
    df = pd.read_csv(_table_path("source_Chen", f"{file}.csv"))

    TF = file.split('.')[0].split('_')[1]
    
    if TF == 'LacI':
        df = pd.DataFrame({'Minus35': df['Minus35'], 'Minus10': df['Minus10'], 'ConExp': 10**df['Induced'], 'RegExp': 10**df['Basal']})
    elif TF in ['AraC', 'LasR']:
        df = pd.DataFrame({'Minus35': df['Minus35'], 'Minus10': df['Minus10'], 'ConExp': 10**df['Basal'], 'RegExp': 10**df['Induced']})
    
    df['FC'] = df['RegExp'] / df['ConExp']
    return df


def loadForcierData(file, dir_path=None):
    if dir_path is None:
        dir_path = _table_path("source_Forcier")
    df = pd.read_csv(os.path.join(dir_path, f"{file}.csv"), index_col=0).dropna()
    df.columns = ['ConExp', 'ConExp_SE', 'RegExp', 'RegExp_SE']
    df = df[['ConExp', 'RegExp']]
    df = 10**df
    return df


def loadData(TF):
    """
    Load and (for the Kuo promoter libraries) stratify-sample a dataset.

    Returns:
        n         : total number of data points before sampling
        df_sample : DataFrame with columns 'ConExp' and 'RegExp' (linear scale)
    """
    if TF not in DATASET_INFO:
        raise ValueError(f"Unknown TF dataset: '{TF}'")

    reference = DATASET_INFO[TF]["Reference"]

    if reference == "Kuo":
        df = loadOurData(TF).dropna()
        if TF == "LuxR":
            drop = [s for s in Motif2Seqs("tTGacNGaTANt") if s in df.index]
            df = df.drop(drop, errors="ignore")
        elif TF == "CueR":
            drop = [s for s in Motif2Seqs("TTGaccAaNNTt") if s in df.index]
            df = df.drop(drop, errors="ignore")
        df["MeanExp"] = (df["ConExp"] + df["RegExp"]) / 2
        df_sample = stratified_sample(df, "MeanExp", [1000] * 8)

    elif reference == "Parisutham":
        df = loadSciData(TF).dropna()
        df_sample = df

    elif reference == "Chen":
        df = loadChenData(TF).dropna()
        df_sample = df

    elif reference == "Forcier":
        df = loadForcierData(TF).dropna()
        df_sample = df

    else:
        raise ValueError(f"Unknown data reference for '{TF}': {reference}")

    return len(df), df_sample
