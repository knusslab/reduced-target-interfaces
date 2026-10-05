# External datasets and source pins

No external scientific dataset is redistributed in this repository. The exact author-produced diffusion-sorption K = 8 checkpoints are included under `experiments/diffusion_sorption/main/checkpoints/`, and the complete author-produced CFDBench trained checkpoint set is included under `experiments/cfdbench/checkpoints/`. No third-party or externally sourced trained checkpoints are bundled. Download the datasets from their original publishers and verify the recorded identifiers before running the corresponding experiments.

## NS2D — PDEArena NavierStokes-2D

Hugging Face dataset: `pdearena/NavierStokes-2D`

Pinned revision used by the recovered experiment source:

```text
cd99556a883a20acb9102c1f4bfdfae66ae33495
```

The dataset metadata and PDEArena source code are MIT licensed.

Example pinned acquisition:

```bash
hf download pdearena/NavierStokes-2D \
  --repo-type dataset \
  --revision cd99556a883a20acb9102c1f4bfdfae66ae33495 \
  --local-dir ./data/pdearena/NavierStokes-2D
```

The approximately 43 GB dataset is not bundled here.

## Diffusion-sorption — PDEBench

Pinned PDEBench source commit used by the recovered experiment contract:

```text
4ff3e3a4aa1561721b5571fa3a048a0a463e0568
```

Required dataset file:

- file: `1D_diff-sorp_NA_NA.h5`
- publisher: DaRUS PDEBench Datasets
- DOI: `10.18419/darus-2986`
- data-file ID: `133020`
- publisher MD5: `9d466d1213065619d087319e16d9a938`
- recovered SHA-256: `8e48ab3efd39ab63524d92e85a4e0db46347b72e7c5b05edf81e1c9637a3ab2d`
- byte size: `4217044280`

Publisher endpoint:

```text
https://darus.uni-stuttgart.de/api/access/datafile/133020
```

Example download:

```bash
curl -L 'https://darus.uni-stuttgart.de/api/access/datafile/133020' \
  -o 1D_diff-sorp_NA_NA.h5
```

DaRUS publishes the PDEBench dataset under CC BY 4.0. The PDEBench source repository is MIT licensed except where otherwise stated. The dataset itself is not redistributed here.

## CFDBench

The frozen public protocol records:

- source commit: `6c30c62649780645eb961114775e1d76b5ec8f0f`
- dataset-tree commit: `09d3f237f53a3da8704236b43d7bb89cb1edd7d3`

The `chen-yingfa/CFDBench` Hugging Face dataset metadata reports Apache-2.0 for the dataset. That dataset license does not establish a license for the separate upstream source repository.

The recovered private experiment source used U-Net and FNO definitions described as copies of pinned upstream CFDBench semantics. Those executable model definitions are therefore excluded from this repository. See `experiments/cfdbench/SOURCE_AVAILABILITY.md`.

## Broadband 3D

Broadband 3D uses synthetic data tied to the included spectral-solver/protocol lineage. There is no external dataset bundle for this study.
