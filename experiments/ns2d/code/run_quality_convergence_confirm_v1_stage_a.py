"""QUALITY_CONVERGENCE_CONFIRM_V1 Stage A -- TRAIN-only preparation.

Builds the parent POD basis and the channel mean from the sixty TRAIN trajectories of each of
the eight confirmation families, and seals them. Nothing else is read.

Two firewalls, and the second is the one that is easy to get wrong.

  role level        selection and test files are refused before they are opened
  trajectory level  TRAIN, VALIDATION and UNUSED live in the *same* physical train HDF5, so
                    "only train files were opened" says nothing about which trajectories were
                    read. Every dataset access goes through `request_trajectories`, which
                    refuses any index outside the sealed TRAIN set for the stage. Reading all
                    100 and slicing afterwards is therefore impossible, not merely discouraged.

`basis_cap` is an argument of the canonical decomposition, not a policy here. It is passed at
the algebraic upper bound `train_rows - 1` purely for API compatibility, and the postcondition
`selector_rank == r_gram_resolved` is asserted so that no external truncation can re-enter. That
parameter, set to 256, is behind two realised defects and one vulnerability in this line.

Protocol: `protocols/QUALITY_CONVERGENCE_CONFIRM_V1.md`
Upstream: `results/quality_convergence_confirm_v1/stage0/DATA_SEAL_V1.json`
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import time
from datetime import datetime, timezone

import numpy as np

import sys
HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pod_basis import (  # noqa: E402
    canonical_parent_basis, orthonormality_residual)
from run_quality_convergence_confirm_v1 import (  # noqa: E402
    FAMILIES, PAIRS_PER_TRAJECTORY, PROTOCOL_ID, PROTOCOL_SHA256, SPLIT_SEED,
    STAGE0_SEAL_SHA256, TRAJECTORIES_PER_FILE, StageBlocked, TrajectoryAccessLedger,
    assert_rank_admissible, basis_cap_record, center_train, empty_access_counters,
    index_digest, open_role_file, request_trajectories, sha256_file, split_family,
    train_paths_from_seal,
    validate_access_counters, verify_protocol_and_seal, write_artifact)

STAGE = "A"

# ----------------------------------------------------------------------------------------
# stage A
# ----------------------------------------------------------------------------------------

def load_train_pairs(ordered_paths, splits, ledger):
    """Gather only the sealed TRAIN trajectories, never the whole dataset."""
    import h5py

    xs, ys, reads = [], [], 0
    for family_index, (family, path, _) in enumerate(ordered_paths):
        ids = request_trajectories(ledger, family_index, splits[family]["train"],
                                   role="train", stage=STAGE)
        with h5py.File(path, "r") as handle:
            dataset = handle[list(handle.keys())[0]]["u"]
            if int(dataset.shape[0]) != TRAJECTORIES_PER_FILE:
                raise StageBlocked(f"{path.name}: expected {TRAJECTORIES_PER_FILE} trajectories")
            taken = dataset[ids]                       # sorted gather, only the sealed ids
        reads += len(ids)
        order = np.argsort(np.argsort(np.asarray(splits[family]["train"], dtype=np.int64)))
        taken = taken[order]                           # restore the sealed order
        if taken.shape[1] - 1 != PAIRS_PER_TRAJECTORY:
            raise StageBlocked(f"{path.name}: expected {PAIRS_PER_TRAJECTORY} pairs")
        d = int(taken.shape[-1] * taken.shape[-2])
        xs.append(taken[:, :-1].reshape(-1, d).astype(np.float32))
        ys.append(taken[:, 1:].reshape(-1, d).astype(np.float32))
    return np.concatenate(xs, 0), np.concatenate(ys, 0), reads


def stage_a(train_dir, seal_path, output_dir, largest_arm=2):
    started = time.perf_counter()
    protocol_sha, seal = verify_protocol_and_seal(seal_path)
    ordered = train_paths_from_seal(seal, train_dir)

    splits = {}
    for family_index, family in enumerate(FAMILIES):
        train, validation, unused = split_family(SPLIT_SEED, family_index)
        splits[family] = {
            "family_index": family_index, "train": train.tolist(),
            "permutation_sha256": index_digest(np.concatenate([train, validation, unused])),
            "train_sha256": index_digest(train),
            "validation_sha256": index_digest(validation),
            "unused_sha256": index_digest(unused)}

    ledger = TrajectoryAccessLedger()
    x_train, y_train, reads = load_train_pairs(ordered, splits, ledger)
    counters = empty_access_counters()
    counters["hdf5_field_reads"]["train"] = reads
    validate_access_counters(counters, STAGE)

    mean, centered, centering = center_train(y_train)
    train_rows = int(centered.shape[0])
    decomposition = canonical_parent_basis(centered, basis_cap=train_rows - 1)
    if decomposition["status"] != "ok" or decomposition["basis"] is None:
        raise StageBlocked(f"parent basis failed: {decomposition['status']}")

    basis = np.ascontiguousarray(decomposition["basis"], dtype=np.float32)
    resolved = int(decomposition["rank"]["r_gram_resolved"])
    assert_rank_admissible(resolved, train_rows, largest_arm)
    if int(decomposition["selector_rank"]) != resolved:
        raise StageBlocked(
            f"selector_rank {decomposition['selector_rank']} != r_gram_resolved {resolved}")
    cap = basis_cap_record(train_rows, resolved)

    output_dir = pathlib.Path(output_dir)
    arrays = {"parent_basis_float32.npy": basis,
              "train_mean_float32.npy": mean,
              "train_eigenvalues_float64.npy": np.asarray(decomposition["eigenvalues"])}
    output_dir.mkdir(parents=True, exist_ok=True)
    array_digests = {}
    for name, array in arrays.items():
        target = output_dir / name
        if target.exists():
            raise FileExistsError(f"{target} exists; outputs are create-only")
        np.save(target, array)
        array_digests[name] = sha256_file(target)

    record = {
        "protocol_id": PROTOCOL_ID, "stage": STAGE, "protocol_sha256": protocol_sha,
        "stage0_seal_sha256": STAGE0_SEAL_SHA256,
        "families": list(FAMILIES), "split_seed": SPLIT_SEED,
        "splits": {str(k): {kk: vv for kk, vv in v.items() if kk != "train"}
                   for k, v in splits.items()},
        "train_rows": train_rows, "input_columns": int(x_train.shape[1]),
        "r_gram_resolved": resolved, "R": resolved,
        "orthonormality_residual": orthonormality_residual(decomposition["basis"]),
        "centering": centering, "basis_cap": cap,
        "arrays": array_digests,
        "train_file_digests": {f: d for f, _, d in
                               [(p.name, p, dg) for _, p, dg in ordered]},
        "access_counters": counters, "trajectory_access": ledger.as_record(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "elapsed_seconds": float(time.perf_counter() - started),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STAGE_A_COMPLETE",
    }
    write_artifact(output_dir / "STAGE_A.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    record = stage_a(args.train_dir, args.seal, args.output)
    print(json.dumps({k: record[k] for k in
                      ("status", "train_rows", "R", "orthonormality_residual",
                       "basis_cap", "access_counters")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
