# Broadband 3D

This directory contains the synthetic Broadband 3D study used for the projection-error rejection example.

## Key files

- `solver/ns3d_spectral.py` — exact recovered numerical solver source.
- `reference_implementation/` — executable reference implementation of the frozen Broadband protocol.
- `reference_results/REFERENCE_RESULT_SUMMARY.json` — paper-facing result summary.

The exact historical training/evaluation executor for the reported run was not recovered. The reference implementation therefore makes the preserved protocol mechanics executable without presenting itself as the byte-identical historical trainer.

## Reported result

The normalized projection-error lower bound is already above the 1.05 limit:

- K = 80: **10.69345**
- complete training span K = 95: **10.5994**

The K = 95 screen therefore rejects the fixed representation without another reduced-target training run.

## Data

Broadband 3D is synthetic. Its data-generation mechanics are tied to the included solver and frozen protocol.
