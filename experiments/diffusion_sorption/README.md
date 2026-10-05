# Diffusion-sorption / PDEBench

This directory contains the paper-facing diffusion-sorption experiment source, cross-learner reuse study, matched-rank control, tests, and reference results.

## Layout

- `main/` — recovered primary experiment package and production corrections.
- `reuse/` — cross-learner reuse runner, protocol summary, tests, and result bindings.
- `matched_control/` — matched-rank gain-selected versus leading-POD control.
- `reference_results/` — paper-facing result records.

The matched-control runner imports the canonical core from `main/package/code/`; this repository does not carry the byte-identical duplicate that existed in earlier private trees.

## Reported results

The original K = 8 study has seedwise reduced/dense full-field MSE ratios of approximately 0.6181, 1.3141, and 0.7076. Their median is 0.7076.

The reuse study reports 10 seedwise ratios across MLP and FNO families. The larger family median is 0.9239, while 3 of the 10 individual runs exceed 1.05. The paper uses this contrast to show why the aggregation rule must be fixed before held-out evaluation.

The matched-rank gain-selected and leading-POD controls have medians near 1.0315 and 1.0344, respectively.

## Data

The external PDEBench diffusion-sorption dataset is not bundled. See `../../DATASETS.md` for the exact DaRUS file identifier, checksums, and acquisition instructions.

Several derived basis/mean/index arrays from the reuse lineage are also not redistributed in this public candidate. The corresponding public tests make that packaging boundary explicit.
