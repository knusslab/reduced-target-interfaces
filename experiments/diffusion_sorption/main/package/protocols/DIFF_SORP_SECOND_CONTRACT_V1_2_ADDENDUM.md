# DIFF_SORP_SECOND_CONTRACT_V1.2 ADDENDUM — analytical selector vs deployed basis

**STATUS:** FROZEN BEFORE RELEASED DATA DOWNLOAD COMPLETION OR HDF5 OPEN — 2026-08-15

**Relation to V1/V1.1:** additive only. This addendum fixes the numerical representation used by the proposal selector before any released scientific tensor value is read.

## Analytical selector coordinates

The canonical float64 TRAIN mean `mu64` and canonical float64 eigenvector rows `B64` produced by V1.1 are retained and sealed in addition to the deployed float32 objects.

- `mu64` and `B64` are used only for the SELECTION directional accounting (`q_j`, `e_j`, `g_j`) and energy comparator.
- The seed-0 dense prediction and target remain raw float32 model/data outputs, but are promoted to float64 before centring/projecting with `mu64` and `B64`.
- The primary threshold remains `tau * D * R_SELECTION(dense seed 0)` where the risk is raw full-field per-element MSE promoted to float64.
- Because all D=1024 canonical float64 directions are retained, the directional masking identity is evaluated in the full orthonormal TRAIN basis up to the numerical residual of the float64 eigensolver.

## Deployed compressed supervision

After indices are frozen, the same direction indices are applied to the corresponding deployed float32 rows `B32`.

- TRAIN coefficient targets use `(y_float32 - mu32) @ B32_S.T` in Torch/NumPy float32-compatible arithmetic.
- Compressed training uses the realised float32 Gram exactly as fixed in V1.1.
- Full-field validation and TEST decode with `B32_S` and `mu32`.

The selector representation and deployment representation therefore share direction identities but not an assumed bit-exact orthogonality.

## Additional seals

The basis artifact and PRETEST seal additionally record:

- canonical `mu64` SHA256 over contiguous float64 bytes;
- canonical `B64` SHA256 over contiguous float64 bytes;
- float64 orthonormality residual;
- explicit labels `selector_basis=float64_canonical` and `deployment_basis=float32_realised_gram`.

No outcome from the released data may change this representation split.
