# Diffusion-sorption K = 8 checkpoints

These are the exact dense-target and reduced-target model checkpoints sealed before the reported held-out TEST evaluation for `DIFF_SORP_SECOND_CONTRACT_V1`.

- `dense/`: three dense-target MLP checkpoints.
- `reduced/`: three K = 8 reduced-target MLP checkpoints.
- `artifacts/BASIS_FLOAT32.npy`: training POD basis.
- `artifacts/TRAIN_MEAN_FLOAT32.npy`: training-field mean.
- `artifacts/SELECTED_INDICES_INT64.npy`: selected coefficient indices `[0,1,2,3,4,5,6,8]`.
- `CHECKPOINT_MANIFEST.json`: exact SHA-256 and byte size for every public artifact.

The paper-facing held-out reduced/dense MSE ratios for seeds 0, 1, and 2 are approximately `0.6181`, `1.3141`, and `0.7076`, with median `0.7076`.

The dataset itself is obtained from the PDEBench DaRUS record linked in the repository-level `DATASETS.md`.
