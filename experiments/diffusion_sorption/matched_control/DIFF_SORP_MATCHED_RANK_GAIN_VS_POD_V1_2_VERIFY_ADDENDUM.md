# DIFF_SORP_MATCHED_RANK_GAIN_VS_POD_V1.2 — pre-FINAL verifier-only repair

**STATUS:** FROZEN AFTER V1.1 VERIFIER FAILURE, BEFORE ANY TEST ACCESS — 2026-08-17

This addendum changes no scientific experiment. It repairs only the V1.1 verifier's impossible aggregate-digest equality when paired arms stop at different epochs.

V1.1's nine trained checkpoints, traces, TRAIN basis, K=8, gain set `[0,1,2,3,4,5,6,8]`, and POD set `[0,1,2,3,4,5,6,7]` are immutable. No training, selection, hyperparameter change, checkpoint reselection, or data-role access is authorized.

The batch-order contract is clarified as: for the same optimization seed, the deterministic permutation **at every epoch shared by both arm executions** is identical. Each arm may have additional later epochs if its own frozen early-stopping rule has not yet fired.

V1.2 must independently replay each arm's complete recorded permutation digest from the frozen formula and stop epoch, and then prove byte equality of each shared-epoch permutation. Any mismatch closes the lineage with TEST access 0.

Only after all V1.2 pre-FINAL gates pass may a new create-only pre-TEST seal authorize the originally frozen one-shot TEST cohort 70000..70023.
