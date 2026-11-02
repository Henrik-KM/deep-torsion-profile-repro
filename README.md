# Few-Shot Active Correction of Atomistic Foundation Models for Torsional Energy Profiles

Reproduction package for the article and Supporting Information "Few-Shot Active
Correction of Atomistic Foundation Models for Torsional Energy Profiles" by Henrik
Klein Moberg and Anders Hellman (Chalmers University of Technology).

The package regenerates every reported table and figure from the derived result
tables of the study. The raw THEMol HDF5 files and the UMA embedding caches are
not redistributed; the derived tables are sufficient for all reported results.

## Contents

- `reproduce.py` regenerates the LaTeX tables in `paper/tables/` and the figures
  in `results/figures/`.
- `results/tables/` holds the derived result tables: profile-level metrics,
  paired effects and summaries from the development, external, confirmation and
  fresh-shard evaluations.
- `src/deep_torsion_profile_selection/` contains the analysis and plotting code,
  including the periodic Gaussian-process update and the extrema-weighted
  acquisition rule used for the synthetic example in Figure 1.
- `configs/experiments/e006b_confirmation.yaml` is the frozen confirmation
  configuration; Figure 1a reads the model and acquisition settings from it.
- `o001/` contains the profile-level metrics and the summary script for the
  separate screen with inexpensive input geometries.

## Reproducing the results

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

Use Python 3.11. With the pinned package versions, the regenerated tables are
identical to those in the manuscript and Supporting Information, and the figure
images are identical to the published figures.

## Licence

The code and our results are licensed under CC BY 4.0. Three tables contain
values derived from THEMol, which is licensed under CC BY-NC 4.0, and are
distributed under that licence. See `LICENSE.md` for the list of files and the
attribution.

## Data source

THEMol, ByteDance-Seed, available on Hugging Face at
<https://huggingface.co/datasets/ByteDance-Seed/THEMol>, used at revision
`e029fa677b2638b29efa5930a1173d4e41abcab8`.

## Citation

Please cite the article and this archive: <https://doi.org/10.5281/zenodo.22917254>.
