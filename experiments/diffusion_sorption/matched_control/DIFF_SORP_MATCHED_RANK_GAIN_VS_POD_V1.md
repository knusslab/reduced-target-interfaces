# DIFF_SORP_MATCHED_RANK_GAIN_VS_POD_V1

**STATUS:** FROZEN BEFORE NEW MLP TRAINING AND BEFORE FRESH TEST GENERATION — 2026-08-17
**MODE:** REVIEW -> PLAN -> TEST -> RUN -> ANALYZE -> VERIFY -> WRITE -> REVIEW
**CLASS:** post-confirmatory matched-rank falsification control; separate successor lineage.

## 1. Question
At the same selected output width K, does prediction-aware gain ordering provide fresh-retraining value beyond simply taking the leading K POD directions?

This control is deliberately allowed to falsify the selector story. It is not a rescue experiment. A result in which leading-POD K performs as well as gain-selected K weakens any claim that the gain ordering itself is needed.

## 2. Existing evidence kept immutable
- Parent BigData V18 scientific results are unchanged.
- Original NS2D TEST is closed and is not accessed.
- `K32_UNUSED_CONFOUND_CONTROL_V2` remains unopened/BLOCKED and is not reused.
- Diffusion-sorption R2 TEST is closed; R2 UNUSED remains unopened.
- FNO V1.2 TEST and all FNO V2/V4 endpoints remain separate.

## 3. Development data inheritance
Use only these already-generated, already-exposed rows from the FNO V1 lineage:
- TRAIN_ROWS.npz SHA256 `1218ec0651dcb8f4b3e4b10e978492d87425ec61a65734ad3675fd4c6d29f4be`, X/Y `[1280,1024]`, 64 trajectories x 20 pairs.
- VALIDATION_ROWS.npz SHA256 `deffc3daca734cc9794dcbc1195221784ef21d800f2d6f865bae474811f7441c`, X/Y `[320,1024]`, 16 trajectories x 20 pairs.
- SELECTION_ROWS.npz SHA256 `aaaf420f3e92bebd17619299f67d6c4097e1da47b83021aca0c4c12b5fe5e688`, X/Y `[480,1024]`, 24 trajectories x 20 pairs.

These rows were generated from pinned PDEBench diffusion-sorption generator semantics with disjoint scientific generator seeds:
- TRAIN 10000..10063
- VALIDATION 20000..20015
- SELECTION 30000..30023

This control is therefore not an independent development-data replication. Its independence is the new, unopened TEST cohort below.

## 4. Basis and reference
Re-fit the TRAIN target mean/POD basis using the same float64 covariance/eigendecomposition and sign convention as `DIFF_SORP_SECOND_CONTRACT_V1`.

Train dense MLP references for seeds `{0,1,2}`:
- architecture `MLP(1024 -> 256 -> 256 -> 1024)` with GELU;
- Adam, lr `1e-3`;
- batch size `64`;
- max epochs `300`;
- validation raw full-field MSE each epoch;
- patience `20`, relative `min_delta=1e-3`;
- per-epoch batch permutation `torch.Generator().manual_seed(seed*1000 + epoch - 1)` then `torch.randperm`;
- CPU float32, deterministic algorithms, torch threads=1.

After all three dense checkpoints are sealed, open inherited SELECTION rows. Seed 0 alone proposes K and the gain-selected set. Seeds 1/2 are diagnostics only.

Selector:
- basis = new TRAIN basis;
- `q_j = mean c_j^2`;
- `e_j = mean(c_j-c_hat_j)^2`;
- `g_j = q_j-e_j`, order by descending `max(g_j,0)`, stable index tie-break;
- ladder `{2,4,8,16,32,48,64,96,128,256,512,1024}`;
- `tau=0.05`;
- select smallest K whose omitted positive-gain tail is <= `tau * 1024 * dense_seed0_selection_MSE`.

If K=1024, close `NO_REDUCED_BUDGET` without TEST.

## 5. Matched-rank arms
After K is frozen:
- `GAIN_K`: selected K gain-ordered basis directions.
- `POD_K`: leading basis directions `{0,...,K-1}`.

Both use `MLP(1024 -> 256 -> 256 -> K)` and the identical optimization/stopping contract above. For each seed, model initialization and batch permutations MUST be identical across GAIN_K and POD_K; an initialization state digest is recorded before the first optimizer step.

Training targets are only the K coefficients for that arm. Training loss is lifted-field MSE under that arm's realized float32 Gram. Validation is raw full-field MSE after fixed-basis decoding.

If the two sets are exactly identical, close `SETS_IDENTICAL_NO_ORDERING_TEST_NEEDED` without TEST.

## 6. Fresh TEST cohort
Qualification-only generator seed: `69999` (never scientific).
Fresh TEST generator seeds are fixed now as `70000..70023` (24 trajectories), disjoint from all inherited development generator seeds and prior FNO test blocks.

Generator is exact pinned PDEBench commit `4ff3e3a4aa1561721b5571fa3a048a0a463e0568`, files:
- `pdebench/data_gen/src/sim_diff_sorp.py`
- `pdebench/data_gen/configs/diff-sorp.yaml`
with `D=5e-4`, `por=0.29`, `rho_s=2880`, `k_f=3.5e-4`, `n_f=0.874`, `sol=1.0`, `t=500`, `tdim=501`, `xdim=1024`, NumPy `default_rng(seed)`, SciPy `solve_ivp` default RK45.

For each trajectory use exactly 20 pairs with input time indices `0,5,...,95` and target index `input+1`, matching the inherited row semantics.

TEST generation is forbidden until `PRETEST_SEAL.json` is independently verified and `TEST_ACCESS_ONCE.json` is exclusive-created. After the TEST marker, no retry, seed replacement, solver change, exclusion, or rescue is allowed.

## 7. Primary endpoints
For seed s, on the same fresh TEST rows:
- `r_gain,s = MSE_GAIN_K,s / MSE_DENSE,s`
- `r_pod,s = MSE_POD_K,s / MSE_DENSE,s`
- `d_s = MSE_GAIN_K,s / MSE_POD_K,s`

Primary budget-quality statistics:
- `Q_gain = median_s r_gain,s`
- `Q_pod = median_s r_pod,s`
- frozen quality tolerance = `1.05`.

Predeclared interpretation:
- `GAIN_ONLY_PASS`: `Q_gain <= 1.05` and `Q_pod > 1.05`. Supports matched-rank evidence that the gain-selected set matters on this control.
- `POD_ONLY_PASS`: `Q_gain > 1.05` and `Q_pod <= 1.05`. Refutes any ordering advantage here; manuscript selector claims must be weakened.
- `BOTH_PASS`: both <=1.05. Shows the small supervision width transfers, but this control does not establish that gain ordering is necessary; matched-rank POD is sufficient too.
- `BOTH_FAIL`: both >1.05. The proposed K does not transfer under this successor; report failure and do not rescue.

`d_s` is reported descriptively for all seeds. No post-hoc superiority margin is invented.

Uncertainty: 10,000 paired trajectory-cluster bootstrap draws over the 24 TEST trajectories, PCG64 seed `20260817`, resampling the same trajectory indices across dense/gain/POD and all model seeds. Intervals condition on the trained seed set; they are not optimization-seed population uncertainty.

## 8. Failure-first and provenance gates
Before scientific training:
1. protocol/source/cache hashes fixed;
2. cache shape/dtype and 20-rows-per-trajectory checks;
3. TEST seed disjointness;
4. synthetic basis/selector and matched-set construction tests;
5. same-initialization digest test for GAIN/POD arms;
6. TEST generator refuses to run without marker;
7. qualification seed 69999 sequential/worker deterministic equivalence and finite shape `[501,1024,1]`;
8. no existing TEST marker/result in the new namespace.

Pre-TEST seal binds protocol/code hashes, cache hashes, basis, all nine checkpoints (3 dense + 3 gain + 3 pod), training traces, K and both sets, access ledger, environment, and tests.

## 9. Claim boundary
This is a secondary/post-confirmatory control on Diffusion-Sorption with reused development rows and a fresh generated TEST cohort.

It can establish only whether gain-selected versus leading-POD directions differ at the same prospectively selected K under this exact MLP/data/training contract.

It does NOT establish universal selector superiority, cross-PDE ordering superiority, intrinsic rank, architecture invariance, or a minimum K. A BOTH_PASS result specifically requires the paper to stop implying that predictive-gain ordering is essential.
