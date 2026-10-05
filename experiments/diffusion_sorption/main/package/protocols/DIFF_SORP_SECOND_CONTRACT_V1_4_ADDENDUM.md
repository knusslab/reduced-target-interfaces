# DIFF_SORP_SECOND_CONTRACT_V1.4 ADDENDUM — released-view schema reconciliation

**STATUS:** FROZEN BEFORE DATA DOWNLOAD OR HDF5 OPEN — 2026-08-15

**Relation:** additive/corrective to V1–V1.3. No earlier protocol file is edited. This addendum resolves a source-static contradiction discovered before any released scientific tensor was read.

## 1. Source contradiction discovered pre-data

V1 §3 inherited `tdim=501` from the official generation configuration and therefore froze `data=[501,1024,1]` and `grid/t=501` as the expected released schema.

A later independent source audit, still before download/HDF5 open, found that the **pinned official PDEBench release-consumer code** opens the exact file `1D_diff-sorp_NA_NA.h5` and documents the loaded trajectory as:

`data shape = [101, 1024, 1]`.

The pinned released-model configuration independently fixes `t_train=101`. The benchmark release therefore distinguishes the 501-step simulation-generation grid from the 101-frame benchmark/model view. The experimental object in this contract is the released benchmark file, not a regenerated 501-frame simulation.

Bindings:

- PDEBench commit: `4ff3e3a4aa1561721b5571fa3a048a0a463e0568`
- official release visualizer: `pdebench/data_download/visualize_pdes.py`
- visualizer Git blob: `1c75218ea977fc408434c770acfb7b218417fa25`
- released-model config Git blob: `7ed4371bafc4922d89e3f9245d1c4e7f82a2bf92`
- generation config Git blob: `2ad7e1f3725250b75ce876c16290ffa450984af9`

No released HDF5 value or metadata was inspected to make this correction.

## 2. Corrected schema admission

V1 §3 is superseded only for the time-axis schema clauses. The admitted released benchmark view is now:

- exactly 10,000 groups `0000`…`9999`;
- group datasets `data`, `grid/x`, `grid/t` present;
- `data` shape exactly `[101,1024,1]`;
- `grid/x` shape exactly `[1024]`;
- `grid/t` is one-dimensional, float-compatible and has length **at least 101**.

`grid/t` values and any elements beyond the first 101 are not used by this experiment. Its length is not used to select, tune or rescue any scientific result. Pair times remain the frozen **index positions** `0,5,…,95` with target index `t+1`.

All scientific field values remain unread during schema admission.

## 3. No fallback to regenerated 501-frame data

A locally regenerated 501-frame file is not an admissible substitute for the released DaRUS byte object. The exact publisher MD5 gate from V1 remains binding. If the byte-identical released file does not expose `data=[101,1024,1]`, this version closes as `RELEASED_SCHEMA_CONTRADICTION_NO_GO`; no result-dependent reshaping/downsampling is allowed.

## 4. Seal propagation

Every schema, basis, selection, pre-TEST and TEST marker artifact must bind this addendum byte SHA256 and the official visualizer blob SHA above. A missing/mutated V1.4 binding fails before scientific field access.
