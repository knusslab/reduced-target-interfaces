# Broadband 3D protocol reference implementation

This directory provides an executable reference implementation of the preserved Broadband 3D experiment contract.

The exact historical numerical solver is kept separately at `../solver/ns3d_spectral.py`. The historical orchestration/training/TEST-evaluator Python bytes for the reported run were not recovered, so this reference implementation reconstructs those protocol mechanics from the frozen contract and endpoint records rather than claiming byte identity with the original executor.

It preserves the declared trajectory roles, one-pair `t=0 -> 0.25` construction, broadband PCG64 initial conditions, one-level 3D U-Net with 46,707 trainable parameters, model seeds, optimizer and epoch-order rules, segmentation semantics, gain-tail selector, K ladder, K = 80 endpoint binding, squared-error decomposition, shared 10,000-draw trajectory bootstrap, and JSON-safe serialization.

The included tests and verifier check the protocol structure, deterministic identities, parameter count, decomposition, bootstrap, and segmentation logic. The reported scientific endpoint values remain bound to the preserved result records.
