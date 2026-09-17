# Data sources

Every dataset analyzed in this study was published by another study. Nothing in
`tables/` is original data: the files here are the published measurements,
re-extracted and reformatted into a common layout so that one loader can read
all 30 datasets. All credit belongs to the original authors, and anyone using
these data should cite the source paper listed below, not this repository.

Dataset IDs (DS01–DS30) follow Table 1 of the accompanying manuscript. The
machine-readable registry mapping each ID to its TF, regulation type, and file
path is `dataset_summary.csv`, which is what the code actually reads.

| Directory | Datasets | n | Source | License |
|---|---|---|---|---|
| `source_Kuo/` | DS01–DS03 | 3 | Kuo et al. 2026, *Nat Commun* | CC BY 4.0 |
| `source_Parisutham/` | DS04–DS24 | 21 | Parisutham et al. 2025, *Science* | see note below |
| `source_Chen/` | DS25–DS26 | 2 | Chen et al. 2018, *Nat Commun* | CC BY 4.0 |
| `source_Forcier/` | DS27–DS30 | 4 | Forcier et al. 2018, *eLife* | CC BY 4.0 |

## References

**DS01–DS03** — Kuo S-TA, Shen W-Y, Lai S-W, Chang JK, Ni C-W, Chang C-C, et al.
Core elements play distinct roles in promoter birth and transcriptional
regulation. *Nat Commun*. 2026;17: 8756.
[doi:10.1038/s41467-026-75686-2](https://doi.org/10.1038/s41467-026-75686-2)

Sort-seq libraries of >16,000 promoter variants each, for TetR, LuxR, and CueR.
The loader subsamples these to 8,000 variants per dataset before fitting; see
the Model fitting section of the manuscript.

**DS04–DS24** — Parisutham V, Guharajan S, Lian M, Ali MZ, Rogers H, Joyce S, et
al. *E. coli* transcription factors regulate promoter activity by a universal,
homeostatic mechanism. *Science*. 2025;389: eadv2064.
[doi:10.1126/science.adv2064](https://doi.org/10.1126/science.adv2064)

**DS25–DS26** — Chen Y, Ho JML, Shis DL, Gupta C, Long J, Wagner DS, et al.
Tuning the dynamic range of bacterial promoters regulated by ligand-inducible
transcription factors. *Nat Commun*. 2018;9: 64.
[doi:10.1038/s41467-017-02473-5](https://doi.org/10.1038/s41467-017-02473-5)

**DS27–DS30** — Forcier TL, Ayaz A, Gill MS, Jones D, Phillips R, Kinney JB.
Measuring cis-regulatory energetics in living cells using allelic manifolds.
*eLife*. 2018;7: e40618.
[doi:10.7554/eLife.40618](https://doi.org/10.7554/eLife.40618)

## Note on DS04–DS24

Parisutham et al. state only that "all data is available in the supplementary
data file"; the data are not deposited in a public repository and carry no
explicit open license. The files in `source_Parisutham/` are the reporter
measurements extracted from that supplementary material and rewritten as one CSV
per dataset.

They are redistributed here so that the analysis in this study can be reproduced
end to end. They are measurements — facts, not creative expression — and are
reproduced in full attribution to the original authors. If you use them, cite
Parisutham et al. 2025 and obtain the data from the publisher where possible.

## File format

Column names differ between sources, which is why `common/loader.py` normalizes
them. After loading, every dataset exposes the same two columns:

- `ConExp` — basal promoter strength, E<sub>TF−</sub> (linear scale)
- `RegExp` — regulated promoter strength, E<sub>TF+</sub> (linear scale)

Units are arbitrary and differ between datasets, which is why each dataset is
fitted independently.
