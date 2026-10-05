# NS2D / PDEArena Navier-Stokes 2D

This directory contains the paper-facing NS2D experiment source, frozen protocol, focused tests, portable control runner, and reference results.

## Main question

The projection-error screen at the selected width is below the 1.05 limit, so the screen is inconclusive. A fresh reduced-target model is therefore trained and compared with the paired dense-target baseline in full-field MSE.

## Key files

- `code/pod_basis.py` — canonical POD-basis construction used by the public runner.
- `code/predictive_gain_selector.py` — predictive-gain selector used to choose the reduced width.
- `code/run_quality_convergence_confirm_v1*.py` — staged experiment implementation.
- `protocols/QUALITY_CONVERGENCE_CONFIRM_V1.md` — frozen experiment protocol.
- `tests/` — focused tests.
- `controls/v033/run_portable.py` — portable reserve/control runner.
- `reference_results/REFERENCE_RESULT_SUMMARY.json` — paper-facing summary.
- `reference_results/STAGE_E.json` and `PHASE1_COMPUTE.json` — preserved result records.

The frozen protocol retains a few historical identifiers because its exact digest is part of the executed experiment record; the publication-facing Python module names are neutral.

## Reported result

For K = 32, the three reduced/dense held-out full-field MSE ratios are approximately **0.9747**, **0.9182**, and **0.9911**; their median is **0.9747**.

The normalized projection-error lower bound is approximately **0.6997**. Because this is below 1.05, it is an inconclusive screen rather than the final decision.

## Data

Dataset: [pdearena/NavierStokes-2D](https://huggingface.co/datasets/pdearena/NavierStokes-2D)

See `../../DATASETS.md` for the pinned revision and acquisition command.
