# NS2D / PDEArena Navier–Stokes 2D

This directory contains the paper-facing NS2D experiment source, its frozen protocol, focused tests, a portable reserve/control runner, and the reference results used by the manuscript.

## Main question

The projection-error screen at the selected width is below the 1.05 limit, so the screen cannot decide the case by itself. A fresh reduced-target model must therefore be trained and compared with the paired dense-target baseline in full-field MSE.

## Key files

- `code/` — recovered experiment implementation used by the NS2D lineage.
- `protocols/QUALITY_CONVERGENCE_CONFIRM_V1.md` — frozen experiment protocol.
- `tests/` — focused tests for the public package.
- `controls/v033/run_portable.py` — path-neutral portable control runner.
- `reference_results/REFERENCE_RESULT_SUMMARY.json` — paper-facing summary.
- `reference_results/STAGE_E.json` and `PHASE1_COMPUTE.json` — preserved result records used by the summary.

## Reported result

For K = 32, the three reduced/dense held-out full-field MSE ratios are approximately 0.9747, 0.9182, and 0.9911; their median is 0.9747.

The normalized projection-error lower bound is approximately 0.6997. Because this is below 1.05, it is only an inconclusive screen, not the final decision.

## Data

The external PDEArena NavierStokes-2D dataset is not bundled. See `../../DATASETS.md` for the pinned revision and acquisition instructions.

## Historical-path boundary

The historical V033 producing runner itself is not redistributed because the preserved file contained a server-local absolute path. The public package includes the previously verified path-neutral derivative instead.
