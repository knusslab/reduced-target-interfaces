# V033 path-neutralization note

The preserved production V033 runner is restricted provenance because one top-level constant named a server-local workspace directory. The review derivative changes only the source of that root path: it resolves to the package root via `Path(__file__)`. No literal historical private path is reproduced in this review archive.

Historical production identity is still independently bound by `PREFLIGHT.json`, `TRAIN_SEAL.json`, and `UNUSED_EVAL.json`, all of which record the exact historical runner SHA-256. `SOURCE_BINDING.json` records that historical digest and the digest of the portable derivative.

A full V033 rerun from the derivative additionally requires the omitted training shards, Stage A parent basis, Stage B dense checkpoints, and six newly trained control checkpoints in the expected package-local layout. Packaging did not recreate or reopen those scientific objects.
