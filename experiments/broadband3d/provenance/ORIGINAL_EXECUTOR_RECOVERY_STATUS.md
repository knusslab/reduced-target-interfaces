# Broadband 3D source-recovery boundary

Status: `PARTIAL_EXACT_SOURCE_RECOVERY`

## Recovered exactly

- numerical solver source `solver/ns3d_spectral.py`, with the preserved SHA-256 identity;
- the frozen data roles, optimizer/model settings, candidate-width ladder, decision rule, and reported endpoint values from durable protocol and result records;
- paper-facing Broadband result values and figure inputs included in this public package.

## Not recovered as historical source

The following original executor files were not found as exact historical bytes:

- the orchestration that generated the frozen TRAIN/VALIDATION/SELECTION/TEST trajectories from the solver;
- the historical segmented 3D U-Net training launcher/orchestrator;
- the original TEST evaluator and verification-replay script.

The release audit searched archived project sources, durable research storage, prior uploaded-file records, and preserved protocol/result records. No exact copy of those executor files was found.

This package therefore does **not** reconstruct code from prose and present it as historical source.

## Later reimplementation

The `reimplementation/` directory contains a later protocol-faithful implementation with focused tests and a structural/algebraic verifier.

It is provided for transparency and usability, but it remains explicitly separate from the recovered historical source. The absence of the original orchestration/training/evaluator bytes is unchanged by that reimplementation.
