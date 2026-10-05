# DIFF-SORP SECOND CONTRACT V1 — infrastructure correction R2 addendum

Status: `FROZEN_BEFORE_R2_TEST`

Successor lineage: `DIFF_SORP_SECOND_CONTRACT_V1_INFRA_CORRECTION_R2`

Predecessor: `DIFF_SORP_SECOND_CONTRACT_V1_PRODUCTION_R1`, closed as
`R1_CLOSED_NO_TEST` after a post-SELECTION, pre-TEST child-process auditor
launch failure.

## Correction class

`POST_SELECTION_PRETEST_AUDITOR_LAUNCH_INFRASTRUCTURE_CORRECTION`

The R1 producer launched the independent auditor as a child process. That
child terminated with `SIGABRT (-6)` and OpenMP error 179 while opening shared
memory. The failure occurred after the R1 PRETEST seal was created and before
any TEST or UNUSED access. A subsequent standalone read-only invocation of the
same byte-sealed auditor passed. R1 is not resumed or rewritten.

## Only permitted R2 changes

1. Run the byte-identical R1 independent auditor as a standalone top-level
   process before TEST, rather than as a producer child process.
2. Bind the launch environment to CPU-only execution with
   `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
   `OPENBLAS_NUM_THREADS=1`.

These are launch-infrastructure changes. They do not change the frozen
scientific numerical path: CPU float32, deterministic PyTorch algorithms, and
one Torch thread.

## Byte-identical inheritance

R2 MUST inherit the following R1 quantities and bytes without regeneration:

- raw HDF5 bytes and released-data schema;
- all role split digests and the exact TEST subset;
- TRAIN mean, float64 selector basis, float32 deployment basis, Gram matrix,
  and eigenvalues;
- dense checkpoints for seeds 0, 1, and 2;
- SELECTION targets, predictions, gain/order/q/e vectors, selected indices,
  and selector seal;
- `K_prop = 8` and selected indices `[0,1,2,3,4,5,6,8]`;
- compressed checkpoints for seeds 0, 1, and 2;
- architecture, parameter counts, optimizer, stopping rules, seeds, and all
  training traces;
- byte-identical evaluator and bootstrap sources;
- `10,000` paired-trajectory bootstrap replicates and its frozen seed;
- `Q_TEST = median_s R_TEST(compressed,s)/R_TEST(dense,s)`;
- success threshold `Q_TEST <= 1.05`;
- UNUSED remains unopened.

R2 performs no training, no checkpoint selection, no selector recomputation,
no K change, and no model or evaluator replacement.

## Fail-closed gates

Before TEST, a standalone auditor MUST verify the exact R1 RUN seal, PRETEST
seal, bound scientific files, training-stop replay, selection replay, role
ledger, and absence of TEST/UNUSED access. R2 then re-verifies the standalone
receipt and the inherited-byte manifest.

Any mismatch closes R2 as `R2_CLOSED_NO_TEST`. TEST MUST NOT open.

If all gates pass, R2 may open TEST once. The evaluator MUST exclusive-create
`TEST_ACCESS_CONSUMED.json` before reading TEST. Any failure after this marker
is terminal and MUST NOT be retried.

## One-shot TEST and terminal review

The one-shot TEST evaluates the six inherited checkpoints on the frozen 200
TEST trajectories (4,000 rows). TEST access is exactly 200 trajectories;
UNUSED access remains zero. After TEST, the byte-identical independent auditor
runs again as a standalone top-level process. Only after its PASS may the
workflow proceed through `ANALYZE -> VERIFY -> REVIEW`.

Permitted claim scope is limited to this preregistered diffusion-sorption
dataset, model family, training contract, seeds, and selector. The result does
not establish a universal K, intrinsic rank, cross-backbone transfer, or
architecture-level causality.

## Preservation and publication policy

R1 failure artifacts and namespace are immutable and remain part of the
provenance history. R2 uses a fresh create-only namespace. Intermediate source,
seals, results, and reviews are persisted to the active branch Google Drive
working state. No GitHub commit or push is authorized by this addendum.
