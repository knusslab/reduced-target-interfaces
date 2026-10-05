# Diffusion-sorption / PDEBench

This directory contains the paper-facing diffusion-sorption experiment source, cross-learner reuse study, matched-rank control, checkpoints, tests, and reference results.

## Layout

- `main/` — primary K = 8 experiment package and production runner.
- `main/checkpoints/` — exact dense-target and reduced-target checkpoints plus decode artifacts for the reported K = 8 study.
- `reuse/` — cross-learner reuse runner, protocol summary, tests, and result bindings.
- `matched_control/` — matched-rank gain-selected versus leading-POD control.
- `reference_results/` — paper-facing result records.

## Reported results

The original K = 8 study has seedwise reduced/dense full-field MSE ratios of approximately **0.6181**, **1.3141**, and **0.7076**. Their median is **0.7076**.

The reuse study reports 10 seedwise ratios across MLP and FNO families. The larger family median is **0.9239**, while 3 of the 10 individual runs exceed 1.05. This is the aggregation-sensitivity case discussed in the paper.

The matched-rank gain-selected and leading-POD controls have medians near **1.0315** and **1.0344**, respectively.

## Data

Dataset: [PDEBench on DaRUS](https://doi.org/10.18419/darus-2986)

See `../../DATASETS.md` for the exact diffusion-sorption file identifier, checksums, and download command.
