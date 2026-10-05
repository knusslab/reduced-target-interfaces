# DIFF_SORP_SECOND_CONTRACT_V1 — prospective cross-PDE fresh-retraining contract

**STATUS:** FROZEN BEFORE DATA DOWNLOAD OR HDF5 OPEN — 2026-08-15

**Purpose:** Add one genuinely prospective fresh-retraining contract on a PDE not previously used by the active `closed-loop-compressibility` evidence. This protocol tests cross-PDE replication while deliberately keeping the model family and one-step direct-supervision setup close to the NS2D contract. It is not a cross-backbone experiment and it does not replace `K32_UNUSED_CONFOUND_CONTROL_V2`.

## 1. Single question

For PDEBench 1D diffusion-sorption, can one dense reference model select a reduced output supervision budget from a disjoint SELECTION split such that fresh models trained only on that reduced supervision meet the frozen dense-quality tolerance on a previously unopened TEST split?

Primary claim if supported:

> A second prospective PDE contract, using the same MLP family but a different equation and output dimension, also admits a predictive-gain-selected reduced supervision budget under fresh retraining.

It may **not** be described as cross-backbone transfer, an intrinsic rank, a universal K, or a removal of the architecture/parameter-count confound. The latter is a separate question owned by `K32_UNUSED_CONFOUND_CONTROL_V2`.

## 2. Public source and byte gate

Pinned PDEBench source commit:

`4ff3e3a4aa1561721b5571fa3a048a0a463e0568`

Released file:

- `1D_diff-sorp_NA_NA.h5`
- DaRUS datafile id `133020`
- official direct URL `https://darus.uni-stuttgart.de/api/access/datafile/133020`
- publisher MD5 `9d466d1213065619d087319e16d9a938`

The file is not opened semantically before this protocol freeze. Download may begin only after freeze. Before any HDF5 open, local byte count, MD5 and SHA256 are written to a create-only `RAW_BYTE_SEAL.json`; MD5 mismatch is terminal `DATA_BYTE_NO_GO`.

## 3. Schema-only admission

The expected structure is frozen from the official generator/config, not from the released tensor values:

- exactly 10,000 seed groups named `0000` through `9999`;
- each group contains `data`, `grid/x`, `grid/t`;
- `data` shape exactly `[501, 1024, 1]`;
- `grid/x` length 1024;
- `grid/t` length 501;
- numeric tensors are float-compatible;
- official generator configuration: `tdim=501`, `xdim=1024`, one channel;
- official `FNODatasetMult` split convention is sorted keys with the last 10% held out.

The schema gate may inspect group names, dataset names, shapes and dtypes only. It may not read scientific tensor values. Any mismatch closes this version as `SCHEMA_NO_GO`; no loader repair is allowed under V1.

## 4. Frozen trajectory roles

Official development pool: keys `0000`–`8999`.
Official test pool: keys `9000`–`9999`.

Development permutation:

- RNG: NumPy `Generator(PCG64)`
- seed uint64: `2221047658708269114`
- derivation SHA256: `1ed2bf13f684f83ae85f5d82803669e1893318dc0e85b655b3e69b333f128898`
- literal: `DIFF_SORP_SECOND_CONTRACT_V1|development-split|PCG64`

Take, in permutation order:

- TRAIN: first 480 trajectories; key-list SHA256 `bdd5fe2abc0dc9706ddec253f702b9ebdd6751f69bbd5d9f5005d663cc60fcd2`
- VALIDATION: next 160; SHA256 `823b6f6a0ce08224ef14153babd0a52c54c1d2212492629fd6551d7c923e3ee6`
- UNUSED: next 160; SHA256 `060451f6180354ffe574c8095da4bf0349a5032ea5cd8003102d77c9ac0242da`
- SELECTION: next 200; SHA256 `d4ab1c03f1d2eb2a96baaeb1fea61ca42e73272a9f4ca5816df5198901290f44`
- all remaining official-development trajectories are OUT-OF-PROTOCOL and may not be used to rescue a result.

TEST subset permutation over keys `9000`–`9999`:

- RNG: NumPy `Generator(PCG64)`
- seed uint64: `3220641587463752856`
- derivation SHA256: `2cb20475c9a08c98b347f7e3692518745fd312dfac835afb0bbf5d12100c0914`
- literal: `DIFF_SORP_SECOND_CONTRACT_V1|official-test-subset|PCG64`
- TEST: first 200 permuted official-test keys; key-list SHA256 `38c307667103362555436ba5a02203df0dc6f60c69add785bb44e563d4db73a3`
- the other 800 official-test trajectories remain unread.

Role access is fail closed. TRAIN and VALIDATION are available to training; SELECTION becomes available only after all dense checkpoints are frozen; TEST remains inaccessible until the complete pre-TEST seal exists; UNUSED is never used by this V1.

## 5. One-step rows and dimensions

Use only the first 101 released time frames, matching the published forward-model horizon limit `t_train=101`.

For every admitted trajectory, create exactly 20 one-step rows at

`t = 0, 5, 10, ..., 95`,

with

- input `x = field[t]`, flattened to `D=1024`;
- target `y = field[t+1]`, flattened to `D=1024`.

No other time offset, phase or stride is allowed. Splitting is by trajectory, never by row.

Rows by role:

- TRAIN: 9,600
- VALIDATION: 3,200
- UNUSED: 3,200 (unopened)
- SELECTION: 4,000
- TEST: 4,000 (one-shot only)

## 6. TRAIN basis

Compute the TRAIN target mean in float64. Form the centred 9,600×1,024 target matrix and its float64 covariance. Use `numpy.linalg.eigh`, sort eigenpairs by descending eigenvalue, and apply this sign convention to each eigenvector: the component of largest absolute magnitude is positive; ties use the first index.

The deployed basis contains all 1,024 directions and is stored in float32 after canonicalisation. Seal:

- TRAIN mean float32 SHA256;
- basis float32 SHA256;
- eigenvalues float64 SHA256;
- realised float32 orthonormality residual;
- realised Gram `B B^T` float32 SHA256.

No basis direction is fitted on VALIDATION, SELECTION, UNUSED or TEST.

Energy comparator: smallest K carrying 99.9% of TRAIN target energy.

## 7. Dense reference model

Architecture, same family as the NS2D direct-supervision contract:

`MLP(1024 -> 256 -> 256 -> 1024)` with GELU between linear layers.

Seeds: `0,1,2`.

Training contract:

- CPU float32;
- `torch.use_deterministic_algorithms(True)`;
- `torch.set_num_threads(1)`;
- Adam, learning rate `1e-3`;
- batch size 64;
- raw full-field MSE on VALIDATION after every epoch;
- patience 20;
- relative `min_delta=1e-3`, governing both checkpoint retention and patience reset;
- max epochs 300;
- per-epoch batch order `torch.Generator().manual_seed(seed*1000 + epoch - 1)` then `torch.randperm`.

`CEILING_REACHED`, any nonfinite value, or stopping-rule replay mismatch is a hard failure. No retry under V1 except a documented infrastructure-only failure before SELECTION with byte-identical source/protocol/data.

All three dense checkpoints are sealed before SELECTION opens.

## 8. Proposal reference and selector

**Seed 0 is frozen as the sole proposal reference.** Seeds 1 and 2 are stability diagnostics and TEST baselines; they do not change K or the selected direction set.

On SELECTION, evaluate dense seed 0 exactly once. In the exact frozen TRAIN basis, compute per direction

- `q_j = mean(c_j^2)`;
- `e_j = mean((c_j - c_hat_j)^2)`;
- `g_j = q_j - e_j`;
- selector mass `g_j^+ = max(g_j,0)`.

Order directions by descending `g_j^+`, stable tie by basis index.

Frozen ladder:

`{2,4,8,16,32,48,64,96,128,256,512,1024}`.

Primary tolerance `tau=0.05`; robustness `tau=0.10` is descriptive only.

For each rung K, tail mass is the sum of positive predictive gain outside the first K selected directions. Select the smallest rung whose tail mass is at most

`tau * D * R_SELECTION(dense seed 0)`.

The resulting `K_prop`, selected indices, gain vector, basis/mean/checkpoint digests, SELECTION split digest, tolerance and ladder are written to a create-only selection seal. No outcome from seeds 1/2 changes the proposal. Their independently computed K values are reported only as stability diagnostics.

If no rung below 1024 passes, `K_prop=1024` and the contract remains valid but demonstrates no reduced supervision.

## 9. Fresh compressed retraining

After the selection seal, train from scratch for seeds `0,1,2`:

`MLP(1024 -> 256 -> 256 -> K_prop)`.

Targets are only the selected TRAIN coefficients. The physical prediction is decoded with the frozen selected basis rows plus TRAIN mean. Training loss is the exact lifted-field MSE under the realised float32 Gram of the selected rows; no dense target batch is required by the coefficient loss. VALIDATION scoring is raw full-field MSE after decoding.

Optimizer, seeds, batch order, stopping and device contract are identical to the dense arm.

This arm is intentionally **not parameter matched** to the dense model. Parameter counts are reported. A pass is cross-PDE fresh-retraining evidence under the native narrow-head realization, not a causal statement about why it passes. Architecture/parameterisation causality belongs to the separate V2 confound-control protocol.

## 10. Pre-TEST seal

TEST remains unopened until a create-only `PRETEST_SEAL.json` contains and independently verifies:

- this protocol byte SHA256;
- pinned PDEBench source commit and official metadata/config source hashes;
- released-data MD5, local SHA256 and byte count;
- schema record and every split key digest;
- TRAIN mean/basis/eigenvalue/Gram digests;
- three dense checkpoint file/state digests and selected epochs;
- seed-0 SELECTION gain/order/K/index digests;
- seed1/2 selector stability records;
- three compressed checkpoint file/state digests and selected epochs;
- exact architecture/parameter counts;
- all batch-order digests and stopping-rule replay results;
- access counters proving `TEST=0` and `UNUSED=0`;
- evaluator and bootstrap source SHA256;
- the fixed interpretation map below.

Any production gate failure closes V1 without TEST.

## 11. One-shot TEST

The evaluator creates `TEST_ACCESS_CONSUMED.json` with exclusive-create semantics **before** reading TEST. Existing marker refuses every retry; a crash after marker creation cannot be retried under V1.

Evaluate all dense and compressed seeds in one transaction on the same 200 TEST trajectories / 4,000 rows.

Per seed:

`r_s = R_TEST(compressed,s) / R_TEST(dense,s)`.

Primary statistic:

`Q_TEST = median_s r_s`.

Frozen pass threshold:

`Q_TEST <= 1.05`.

Trajectory-cluster bootstrap:

- 10,000 paired replicates;
- shared trajectory draw across all arms/seeds;
- NumPy `Generator(PCG64)`;
- seed uint64 `4072430940531852417`;
- derivation SHA256 `38842e5d1eb604810820e5651f0641426ad59377af419b8ba4adcc1a162fe872`;
- empirical 2.5%/97.5% quantiles.

The interval is TEST-trajectory uncertainty only, not uncertainty over optimization seeds/PDEs/datasets.

## 12. Preregistered interpretation

- `K_prop < 1024` and `Q_TEST <= 1.05`: a second PDE contract prospectively supports reduced-supervision sufficiency under fresh retraining in the same MLP family.
- `K_prop < 1024` and `Q_TEST > 1.05`: the selector did not transfer to fresh retraining on this PDE; report failure, no rescue.
- `K_prop = 1024`: no compression was admitted; report that the method did not find a reduced budget on this contract.
- Seed1/2 proposal disagreement is stability evidence only and never changes K after seed0 proposal.
- A ratio below one is reported without mechanism attribution.
- No result establishes cross-backbone transfer, a universal K, an exact minimum K, or architecture causality.

## 13. Existing evidence kept separate

- NS2D main confirmation and `K32_UNUSED_CONFOUND_CONTROL_V2` remain untouched.
- Existing PDEBench diff-react/radial-dam-break final-test artifacts are consumed and diagnostic; they are not reused as prospective evidence.
- The `computational-compressibility` Advection beta holdouts belong to a separate project and are not opened or repurposed here.

## 14. Failure-first implementation order

1. protocol freeze and SHA;
2. downloader/hash verifier tests;
3. schema-only gate tests;
4. role firewall/split-digest tests;
5. basis canonicalisation and Gram-equivalence tests;
6. deterministic training/stopping replay tests;
7. selector tests;
8. PRETEST seal mutation/fail-closed tests;
9. one-shot TEST marker/bootstrap tests;
10. only then download/byte-seal/schema-open scientific production.

No scientific result may be used before independent verification of its producing artifact.
