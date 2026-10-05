# DIFF_SORP_SECOND_CONTRACT_V1.3 ADDENDUM — official source registry and selector-order seal

**STATUS:** FROZEN BEFORE RELEASED DATA DOWNLOAD COMPLETION OR HDF5 OPEN — 2026-08-15

**Relation to V1/V1.1/V1.2:** additive only. No scientific choice is changed.

## Official source registry

All source-static expectations are bound to PDEBench commit
`4ff3e3a4aa1561721b5571fa3a048a0a463e0568` and the following Git blob IDs:

- data URL/checksum registry `pdebench/data_download/pdebench_data_urls.csv`:
  `3332b679127a6db3059243983a237332273483e7`
- generator `pdebench/data_gen/gen_diff_sorp.py`:
  `2f1462b81996d71df10ec8570420e951331fbf07`
- generator config `pdebench/data_gen/configs/diff-sorp.yaml`:
  `2ad7e1f3725250b75ce876c16290ffa450984af9`
- released-model config `pdebench/models/config/args/config_diff-sorp.yaml`:
  `7ed4371bafc4922d89e3f9245d1c4e7f82a2bf92`
- official FNO dataset loader `pdebench/models/fno/utils.py`:
  `d5bc69db79639ab2dc5e2c7fa3ca286eade8ea55`

These identifiers are source provenance only; the experiment does not run the released FNO.

## Selector order digest

The seed-0 selection seal records the complete direction order as contiguous little-endian/native NumPy int64 bytes using the producing environment and seals
`SHA256(np.ascontiguousarray(order, dtype=np.int64).tobytes())`.
The actual selected index prefix is stored separately and retains its own digest.

## Transport boundary

The canonical transport URL remains the V1 DaRUS Data Access API URL. A local inability to resolve or broker that URL is `TRANSPORT_INFRASTRUCTURE_BLOCKED`, not `DATA_BYTE_NO_GO` and not scientific evidence. No mirror or regenerated substitute may be used under V1.3 unless it is separately preregistered before semantic data access and shown byte-identical to the publisher MD5.
