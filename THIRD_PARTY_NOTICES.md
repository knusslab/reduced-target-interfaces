# Third-party notices

This repository references external scientific resources but does not redistribute the large external datasets or trained checkpoints used by the experiments.

The repository-level MIT license covers author-written material distributed under that license. Third-party software, datasets, and other resources remain governed by their own terms.

## PDEArena

- PDEArena source code: MIT license.
- `pdearena/NavierStokes-2D` dataset metadata: MIT.
- The project records the pinned dataset revision `cd99556a883a20acb9102c1f4bfdfae66ae33495`.
- The dataset itself is not bundled here.

## PDEBench

- PDEBench source repository: MIT licensed except where otherwise stated.
- PDEBench dataset DOI `10.18419/darus-2986`: CC BY 4.0.
- The diffusion-sorption file used by the project is data-file ID `133020`, publisher MD5 `9d466d1213065619d087319e16d9a938`.
- The dataset and pretrained checkpoints are not bundled here.

## CFDBench

- The `chen-yingfa/CFDBench` Hugging Face dataset metadata reports Apache-2.0 for the dataset.
- Dataset licensing does not establish a license for the separate upstream source repository.
- The recovered private experiment implementation used model definitions described as copies of pinned upstream CFDBench semantics.
- Those executable definitions are excluded from this repository because their redistribution terms were not established.
- This repository therefore includes only the frozen protocol and author-generated reference-result artifacts for the CFDBench study.

See `experiments/cfdbench/SOURCE_AVAILABILITY.md` for the exact boundary.

## Broadband 3D

The Broadband 3D study uses synthetic data tied to the included numerical-solver/protocol lineage. No external dataset bundle is required for the public source-recovery boundary described here.
