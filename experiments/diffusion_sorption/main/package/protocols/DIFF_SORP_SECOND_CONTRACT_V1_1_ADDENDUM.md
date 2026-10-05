# DIFF_SORP_SECOND_CONTRACT_V1.1 ADDENDUM — numerical and byte-level disambiguation

**STATUS:** FROZEN BEFORE DATA DOWNLOAD OR HDF5 OPEN — 2026-08-15

**Relation to V1:** additive only. `DIFF_SORP_SECOND_CONTRACT_V1.md` is immutable and remains the scientific contract. This addendum resolves implementation-level ambiguities before any released tensor value is read.

## 1. Key-list digest encoding

Every trajectory-key digest is

`SHA256("\\n".join(keys).encode("utf-8"))`

with keys in the frozen permutation order and with no trailing newline.

## 2. Schema dtype

The released generator writes `data`, `grid/x`, and `grid/t` as float32. V1 therefore admits exactly float32 for these three datasets; a different numeric dtype is `SCHEMA_NO_GO` under this version rather than an implicit conversion choice.

## 3. Basis arithmetic

Let `Y` be the TRAIN target matrix with shape `[N,D]`, `N=9600`, `D=1024`.

1. Compute `mu64 = mean(Y, axis=0)` in float64.
2. Centre `X = Y - mu64` in float64.
3. Use the **population covariance** `C = X.T @ X / N` (not `N-1`).
4. Compute `numpy.linalg.eigh(C)` in float64.
5. Sort eigenpairs by descending eigenvalue.
6. For each eigenvector, find the first index attaining the largest absolute component and multiply the vector by `-1` iff that pivot is negative.
7. Store basis directions as rows: `B64[j,:]` is eigenvector `j`; deployed `B32 = B64.astype(float32)`.
8. Store `mu32 = mu64.astype(float32)` and the descending eigenvalue vector in float64.
9. Energy fractions use `max(lambda_j,0)` before cumulative summation. `K_energy` is the smallest prefix reaching 0.999 of the clipped total.
10. Realised Gram is computed by the same training backend and dtype used by the compressed loss: CPU Torch float32 `G32 = torch.from_numpy(B32) @ torch.from_numpy(B32).T`, then sealed as contiguous float32 bytes.

The producing NumPy/Torch versions are recorded in the basis seal. No exact cross-version eigenvector-byte equivalence is assumed before the seal; once produced, the sealed bytes are authoritative for this protocol.

## 4. Loss and validation units

Dense validation and TEST risk are per-element MSE:

`mean_rows mean_D (prediction - target)^2`.

For selected basis rows `B_S` and coefficient error `e`, compressed training uses

`mean_batch [e G_S e^T] / D`, where `G_S = B_S B_S^T` is realised in Torch float32.

This is the exact lifted-field MSE under the stored float32 basis rows. Decoding is

`y_hat = c_hat @ B_S + mu32`.

## 5. Stopping semantics

Epoch 1 is the initial retained checkpoint. At later epoch `t`, an improvement is significant iff

`(best - val_t) / best >= 1e-3`.

Only significant improvement updates both the retained checkpoint and the running `best`, and resets patience to zero. Every other finite epoch increments patience. The first epoch reaching patience 20 terminates as `PATIENCE`; reaching epoch 300 first remains `CEILING_REACHED` and fails the production gate.

## 6. Selector deterministic ordering

Sort by descending `g_j^+`; ties preserve ascending basis index. Tail mass at rung K is the sum of clipped positive gain outside the first K ordered directions. At K=D the tail is exactly zero by construction.

## 7. Pre-TEST binding

`PRETEST_SEAL.json` records the SHA256 of both V1 and this V1.1 addendum. A source implementing only V1 without this addendum may not open TEST.
