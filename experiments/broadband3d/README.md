# Broadband 3D

This directory contains the recoverable public pieces of the synthetic broadband 3D study.

## What is historical

- `solver/ns3d_spectral.py` is recovered historical numerical-solver source.

## What is a later reimplementation

- `reimplementation/` is a later protocol-faithful implementation with focused tests and a structural/algebraic verifier.

The reimplementation is included to make the public package useful and testable, but it is not presented as recovered historical orchestration, training, or TEST-evaluator source.

## Reported result

The normalized projection-error lower bound is already far above the 1.05 limit:

- K = 80: 10.69345
- complete training span K = 95: 10.5994

The K = 95 screen therefore rejects the fixed representation without requiring another reduced-target training run.

## Recovery boundary

The historical orchestration/training/TEST-evaluator bytes were not found in the durable stores inspected during release engineering. The recovery record is in `provenance/ORIGINAL_EXECUTOR_RECOVERY_STATUS.md`.

There is no external dataset bundle for this study; the data are synthetic and tied to the included solver/protocol lineage.
