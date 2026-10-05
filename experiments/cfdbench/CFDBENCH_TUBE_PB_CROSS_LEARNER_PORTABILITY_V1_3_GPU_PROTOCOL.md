# CFDBENCH_TUBE_PB_CROSS_LEARNER_PORTABILITY_V1.3 GPU

**STATUS: FROZEN AS A NEW SUCCESSOR AFTER V1.2 TERMINAL SELECTION EXECUTION FAILURE AND BEFORE ANY V1.3 SCIENTIFIC VALUE READ**  
Date: 2026-08-20  
Mode at seal: `AUDIT -> PLAN -> PREREGISTER`

## 1. Why V1.3 exists

V1.2 is preserved as a terminal execution failure and is not retried or reinterpreted. V1.2 completed schema admission, TRAIN basis construction, all six dense 100-epoch arms, and the dense VALIDATION qualification. It then opened the prespecified 20-case SELECTION role once. Before any proposal statistic, K, index set, or artifact was produced, selection inference failed because a field tensor and mask tensor were on different devices inside `flatten_masked`. V1.2 therefore closed with SELECTION already consumed, TEST untouched, and no artifact.

V1.3 is a create-only successor. It keeps the scientific contract unchanged and fixes only the selection execution device placement. No result-dependent threshold, seed, learner, candidate K, role assignment, temporal pair, optimizer, training budget, or acceptance criterion changes.

## 2. Inherited scientific contract

V1.3 inherits the full V1.2 scientific contract unchanged:

- exact `tube.zip` SHA256 `d2b8a65a7198a7af31556a4c47c688e4841a5e4143347a40fd29394b3b8861f6`;
- source commit `6c30c62649780645eb961114775e1d76b5ec8f0f` and dataset tree commit `09d3f237f53a3da8704236b43d7bb89cb1edd7d3`;
- admitted identities `bc/case0000..case0049` and `prop/case0050..case0149`, excluding `geo`;
- role-manifest digest `c5dc52f07f207bf1d77b130f031fbb67d818956b50a503c9d2364b623c466c31` with TRAIN 100 / SELECTION 20 / VALIDATION 15 / TEST 15;
- temporal starts `[0,4,8]`, exactly three pairs per case and rows 300/60/45/45;
- source-derived FNO and U-Net, seeds `{0,1,2}`, paired sealed INIT_STATEs;
- Adam lr `1e-4`, weight decay 0, batch 16, exactly 100 epochs, no scheduler and no early stopping;
- TRAIN float64 SVD basis and candidate-ladder rule with relative singular-value floor `1e-8`;
- proposal reference dense FNO seed 0 only, `tau=0.05`, unchanged gain-tail rule and deterministic tie-breaking;
- FNO seeds 1/2 are diagnostics only; U-Net never participates in selection; no reselection;
- reduced training uses the same sealed artifact for all six arms;
- TEST family statistic is the median of three seedwise reduced/dense ratios, `Q_port=max(Q_FNO,Q_UNet)`, PASS iff `Q_port<=1.05`;
- 10,000 paired case-cluster bootstrap draws with NumPy PCG64 seed `20260818`;
- exact RTX 4090 CUDA fp32 deterministic execution contract from V1.1/V1.2.

Known V1.2 dense outcomes do not authorize any V1.3 scientific-contract change. V1.3 reruns the frozen pipeline from a fresh scientific-state namespace.

## 3. Sole execution correction

The only admitted V1.3 code correction is in selection inference. For each FNO selection batch, model, inputs, case parameters, and mask are placed on the frozen CUDA device before the forward pass. The same device-resident mask is passed to `flatten_masked`; the resulting flattened prediction is then copied to CPU for the unchanged NumPy proposal calculation.

Semantically, the corrected operation is:

`model -> cuda`, `inputs -> cuda`, `params -> cuda`, `mask -> cuda`, `raw = model(inputs, params, mask)`, `flat = flatten_masked(raw, mask).cpu()`.

This changes no mathematical objective, weights, data values, selected rows, proposal rule, or candidate ladder. It only makes tensor device placement internally consistent.

## 4. Execution order

V1.3 must execute failure-first and create-only:

1. schema/header gate with zero scientific-value reads;
2. TRAIN basis;
3. all six dense arms to 100 epochs using TRAIN only;
4. one VALIDATION materialization after all six dense checkpoints exist; dense family medians must beat persistence or V1.3 closes pre-SELECTION;
5. one SELECTION materialization and one artifact proposal/seal using corrected device placement;
6. six reduced arms from paired INIT_STATEs using TRAIN coefficient loss; reduced VALIDATION endpoints are reporting only and cannot alter the artifact;
7. structural PRETEST with TEST counters still zero;
8. one-shot TEST after writing `TEST_ACCESS_ONCE.json`; any crash after the marker is terminal and cannot be retried under V1.3.

## 5. No-go actions

V1.3 forbids changing roles, temporal pairs, seeds, architectures, optimizer, epoch budget, candidate ladder, proposal tolerance, proposal reference, selected representation for U-Net, TEST threshold, bootstrap seed, or reporting rule based on V1.2 or V1.3 outcomes. It also forbids importing any V1.2 artifact because none exists, reusing V1.2 SELECTION access as if it were V1.3 access, reopening V1.2 TEST, or editing V1/V1.1/V1.2 terminal records.

## 6. Required reporting

Always report schema/frame summary, role digest, TRAIN rank and ladder, dense endpoints and qualification, selected K and exact indices, all reduced endpoints, all six TEST ratios, `Q_FNO`, `Q_UNet`, `Q_port`, bootstrap intervals, per-case TEST dense/reduced MSEs, access ledger, TEST marker, source/data/checkpoint/artifact hashes, and the terminal PASS/FAIL/blocked outcome without rescue.
