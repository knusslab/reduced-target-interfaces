# CFDBench

This directory contains the frozen CFDBench protocol, the complete author-produced trained checkpoint set, and the author-generated reference-result artifacts used by the paper.

## Reported result

For the fixed K = 16 representation:

- FNO family median reduced/dense full-field MSE ratio: 0.3627
- U-Net family median: 0.5335
- larger family median used for the cross-family decision: 0.5335

All six reported seedwise ratios are below 1.05.

The machine-readable values are in `reference_results/REFERENCE_RESULT_SUMMARY.json`.

## Trained checkpoints

The author-produced trained checkpoints are in `checkpoints/`. The public set contains all 24 recovered trained states used by the authoritative run lineage: dense/reduced targets, FNO/U-Net learner families, seeds 0--2, and both `FINAL_MODEL.pt` and `RESUME.pt`. Exact SHA-256 values and byte sizes are in `checkpoints/CHECKPOINT_MANIFEST.tsv`.

## Why executable source is not included

The recovered private CFDBench experiment source states that its U-Net and FNO definitions copy pinned upstream CFDBench model semantics. The separate upstream source repository did not expose a root license in the release audit.

For that reason, this repository does **not** redistribute those copied executable model definitions. This is a licensing boundary, not a change to the reported scientific result.

See `SOURCE_AVAILABILITY.md` and `../../THIRD_PARTY_NOTICES.md` for the exact boundary.

## Data

The external CFDBench dataset is not bundled. See `../../DATASETS.md` for the pinned dataset/source revisions and license notes.
