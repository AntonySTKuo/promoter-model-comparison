# promoter-model-comparison

Code and curated datasets for:

> Kuo S-TA, Hsu C-P, Chou H-HD. **Data coverage and model formulation reshape
> quantitative interpretations of transcriptional regulation.**

The study compiles 30 transcription-factor-regulated promoter datasets from four
prior studies, fits thermodynamic models to each, and asks whether the
relationship between basal promoter strength (E<sub>TF−</sub>) and regulatory
fold change (FC) is universally inverse, as previously reported, or peaked.

This repository redraws every figure in the paper from the datasets in
`tables/`.

## Models

Each model predicts basal and regulated promoter strength from the Boltzmann
weights of the promoter states it defines. P<sub>P</sub>, the Boltzmann weight
of the RNAP-bound state, is fitted separately for every promoter variant; the
remaining parameters are single values shared across variants within a dataset.

| Tag | Name in paper | Free parameters | Notes |
|---|---|---|---|
| `RM` | RM | r<sub>max</sub>, r<sub>0</sub>, P<sub>T</sub> | Repression model; RNAP and repressor binding mutually exclusive |
| `AM` | AM | r<sub>max</sub>, r<sub>0</sub>, P<sub>T</sub>, β | Activation model |
| `GM` | GM | r<sub>max</sub>, P<sub>T</sub>, β, α | Generalized model, r<sub>0</sub> fixed at 0 |
| `cGM` | GM\* | r<sub>max</sub>, r<sub>0</sub>, P<sub>T</sub>, β, α | GM with r<sub>0</sub> freed |
| `cGM_a1` | GM\*\* | r<sub>max</sub>, r<sub>0</sub>, P<sub>T</sub>, β | GM with r<sub>0</sub> freed and α fixed at 1 |

The asterisked names are spelled out as tags here so that they work as directory
names and shell globs.

## Layout

```text
scripts/
  ModelFit.py                  Fit any model variant to any dataset -> results/models/
  Fig2_ModelCurves.py          Fig 2  ETF- vs FC, data and fitted curves
  Fig3A_MetricsHeatmap.py      Fig 3A R2 / RMSE / MAE per dataset
  Fig3B_MetricsBox.py          Fig 3B distribution of each metric across datasets
  Fig4_ParamsHeatmap.py        Fig 4  fitted alpha and beta
  Fig5_PP-Expression.py        Fig 5  PP vs ETF- and ETF+
  Fig6_PP-FC.py                Fig 6  PP vs FC
  Fig7_AnalyticalSolution.py   Fig 7  analytical ETF--FC relationship (uses no data)
  Model.py                     Model definitions, fitting loop, metrics
  common/
    loader.py                  Dataset registry and per-source loaders
    models.py                  Model variant registry; parameter and metric tables
    cache.py                   Read and write fitted checkpoints
    paths.py                   Repository paths
    plots.py                   Shared plotting for the Fig 2 panels
    pp_panels.py               Shared plotting for the Fig 5 and Fig 6 panels
    style.py                   Colors, axis limits, per-dataset plot overrides
    utils.py                   Tick and label helpers

tables/
  dataset_summary.csv          Registry: DS ID, TF, regulation, source, file paths
  SOURCES.md                   Provenance, citation, and license for every dataset
  source_Kuo/                  DS01-DS03  sort-seq libraries
  source_Parisutham/           DS04-DS24  fluorescent reporter assays
  source_Chen/                 DS25-DS26  fluorescent reporter assays
  source_Forcier/              DS27-DS30  beta-galactosidase assays

results/
  models/                      Fitted checkpoints, one directory per model variant
  figures/                     Generated figures (not tracked; regenerate them)
```

## Requirements

Python 3.12 with:

```
torch  numpy  pandas  scipy  matplotlib
```

Developed against torch 2.4.1, numpy 1.26.4, pandas 2.2.2, scipy 1.13.1,
matplotlib 3.9.2. No GPU is required; each fit runs on CPU in seconds to
minutes.

## Redrawing the figures

The fitted checkpoints are committed, so the figures can be redrawn without
refitting. Run from the repository root:

```bash
python scripts/Fig2_ModelCurves.py
python scripts/Fig3A_MetricsHeatmap.py
python scripts/Fig3B_MetricsBox.py
python scripts/Fig4_ParamsHeatmap.py
python scripts/Fig5_PP-Expression.py
python scripts/Fig6_PP-FC.py
python scripts/Fig7_AnalyticalSolution.py
```

Each writes PNG and SVG into `results/figures/`. Panel letters, axis titles, and
legends are added afterwards when the figures are composited for the manuscript,
so what these scripts produce is the plotted content only.

## Refitting from scratch

```bash
python scripts/ModelFit.py                        # every variant, every dataset
python scripts/ModelFit.py --models RM --tf UlaR  # one variant, one dataset
python scripts/ModelFit.py --jobs 8               # fits are independent
```

P<sub>P</sub> is initialized by an independent random draw for every promoter
variant, so a fixed seed is what makes a refit reproduce a committed checkpoint.
`--seed` defaults to 0, the value the committed checkpoints were fitted with.
The seed is combined with the dataset and model name, so each fit draws from its
own stream and `--jobs` changes only wall time, not results.

Refitting overwrites `results/models/`.

## Data

The datasets under `tables/` were published by other studies and are
redistributed here so that this analysis can be reproduced end to end. See
[`tables/SOURCES.md`](tables/SOURCES.md) for the citation and license of each
one. Please cite the original study, not this repository, when using them.

## License

Code and fitted checkpoints: MIT, see [`LICENSE`](LICENSE).
Datasets under `tables/`: the terms set by their original publishers.
