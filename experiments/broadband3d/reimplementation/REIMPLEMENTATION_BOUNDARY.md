# Broadband 3D clean reimplementation boundary

**Status: REIMPLEMENTATION — NOT HISTORICAL SOURCE**

This directory is a clean implementation written from the frozen Broadband 3D protocol and preserved endpoint records after the original orchestration/training/evaluator Python bytes could not be recovered.

It MUST NOT be described as the historical executor, byte-recovered source, or an exact source snapshot of the 2026-08-17 run.

What this implementation is intended to preserve:
- the exact recovered `../solver/ns3d_spectral.py` numerical solver bytes;
- frozen trajectory identity ranges and one-pair `t=0 -> 0.25` construction;
- the frozen broadband PCG64 initial-condition construction;
- the one-level 3D U-Net architecture and exact 46,707 trainable parameters;
- model seeds, optimizer constants, epoch-order seed rule, dense segmentation and compact microsegmentation semantics;
- the gain-tail selector rule, K ladder, fixed K=80 endpoint binding, two-bottleneck squared-error decomposition, and shared 10,000-draw trajectory bootstrap definition;
- V1.5-style JSON-safe conversion as a serialization utility.

What is deliberately NOT done by the offline verifier:
- no TRAIN/VALIDATION/SELECTION/TEST scientific trajectory generation;
- no model fitting;
- no historical checkpoint regeneration;
- no protected TEST reopening;
- no claim that synthetic unit-test outputs reproduce the scientific endpoint.

The included tests validate protocol structure, deterministic identities, parameter count, algebraic decomposition, bootstrap determinism, segment schedules, and JSON-safe conversion. Scientific endpoint values remain bound to the already-verified frozen artifacts rather than regenerated here.
