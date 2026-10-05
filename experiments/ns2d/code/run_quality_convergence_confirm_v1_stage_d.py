"""QUALITY_CONVERGENCE_CONFIRM_V1 Stage D -- the two compressed arms, and the pre-TEST seal.

Trains `K_prop` and `K_next` at seeds 0, 1 and 2 under the same stopping rule the dense reference
used, then seals every model, rank, index set and digest. TEST is not opened, and on this host the
TEST files are not even present.

Dense is **not retrained**. Its three checkpoints were sealed by Stage B and are carried into the
seal by digest, so the denominator of every Stage E ratio is the same object that Stage B produced.
Retraining it here would silently give the comparison two different denominators.

Each compressed arm is built from the exact basis rows Stage C recorded, `gain_order[:K]`, never a
leading prefix. Training follows Stage C's own semantics: the target is the supervision projected
into those rows and lifted back, and the prediction is confined to the same subspace. Validation is
monitored on the raw field, as everywhere in this protocol.

After this stage the pre-TEST seal is closed. Nothing downstream may change a rank, an index set, a
checkpoint, the tolerance or the stopping policy.

Protocol: `protocols/QUALITY_CONVERGENCE_CONFIRM_V1.md`
Upstream: Stage C, and through it Stages B, A and the Stage 0 data seal.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_quality_convergence_confirm_v1 import (  # noqa: E402
    FAMILIES, MAX_EPOCHS, MIN_DELTA, PAIRS_PER_TRAJECTORY, PATIENCE, PROTOCOL_ID,
    PROTOCOL_SHA256, REFERENCE_TRACE_FIELDS, SPLIT_SEED, TOLERANCE, TRAJECTORIES_PER_FILE,
    StageBlocked, TrajectoryAccessLedger, arm_indices, empty_access_counters, index_digest,
    open_role_file, replay_stopping, request_trajectories, require_digest,
    require_reference_convergence, sha256_file, split_family, validate_access_counters,
    validate_arm_trace, validate_reference_seal, validation_true_field_mse,
    verify_protocol_and_seal, write_artifact)

STAGE = "D"
SEEDS = (0, 1, 2)
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 512


def load_role_pairs(train_paths, role, ledger):
    import h5py

    xs, ys, reads = [], [], 0
    for family_index, (_, path) in enumerate(train_paths):
        train, validation, _ = split_family(SPLIT_SEED, family_index)
        wanted = train if role == "train" else validation
        ids = request_trajectories(ledger, family_index, wanted, role=role, stage=STAGE)
        with h5py.File(path, "r") as handle:
            dataset = handle[list(handle.keys())[0]]["u"]
            if int(dataset.shape[0]) != TRAJECTORIES_PER_FILE:
                raise StageBlocked(f"{path.name}: unexpected trajectory count")
            taken = dataset[ids]
        reads += len(ids)
        taken = taken[np.argsort(np.argsort(np.asarray(wanted, dtype=np.int64)))]
        if taken.shape[1] - 1 != PAIRS_PER_TRAJECTORY:
            raise StageBlocked(f"{path.name}: unexpected pair count")
        d = int(taken.shape[-1] * taken.shape[-2])
        xs.append(taken[:, :-1].reshape(-1, d).astype(np.float32))
        ys.append(taken[:, 1:].reshape(-1, d).astype(np.float32))
    return np.concatenate(xs, 0), np.concatenate(ys, 0), reads


def _state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(
            tensor.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return digest.hexdigest()


def train_arm(arm, seed, rows_basis, mean, x_train, y_train, x_validation, y_validation,
              output_dir, device="cpu", progress=25):
    """One compressed cell. Same stopping rule and the same raw-field validation as Stage B."""
    import torch
    import torch.nn as nn
    from e2e_cost import MLP

    started = time.perf_counter()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    basis_np = np.ascontiguousarray(rows_basis, dtype=np.float32)
    mean_np = np.ascontiguousarray(mean, dtype=np.float32)
    coefficients = (y_train.astype(np.float32) - mean_np) @ basis_np.T

    x_g = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32)).to(device)
    xv_g = torch.from_numpy(np.ascontiguousarray(x_validation, dtype=np.float32)).to(device)
    target = torch.from_numpy(np.ascontiguousarray(coefficients, dtype=np.float32))
    basis_g = torch.from_numpy(basis_np).to(device)
    mean_g = torch.from_numpy(mean_np).to(device)
    rows, k = int(target.shape[0]), int(target.shape[1])

    model = MLP(int(x_train.shape[1]), k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    loss_function = nn.MSELoss()

    train_curve, validation_curve, permutations = [], [], []
    nonfinite = 0
    running_best = None
    selected_epoch, selected_state, selected_digest = None, None, None
    raw_best, raw_best_epoch, raw_best_digest = None, None, None
    counter, stop_epoch, stop_reason = 0, None, "CEILING_REACHED"

    for epoch in range(1, MAX_EPOCHS + 1):
        generator = torch.Generator().manual_seed(int(seed) * 1000 + epoch - 1)
        permutation = torch.randperm(rows, generator=generator)
        permutations.append(permutation.numpy().astype(np.int32))
        model.train()
        total = torch.zeros((), dtype=torch.float64, device=device)
        for start in range(0, rows, BATCH_SIZE):
            index = permutation[start:start + BATCH_SIZE]
            batch = target[index].to(device)
            optimizer.zero_grad()
            target_field = batch @ basis_g + mean_g
            predicted_field = model(x_g[index.to(device)]) @ basis_g + mean_g
            loss = loss_function(predicted_field, target_field)
            loss.backward()
            optimizer.step()
            total += loss.detach().double() * index.shape[0]
        train_loss = float((total / rows).item())

        model.eval()
        with torch.no_grad():
            chunks = [model(xv_g[i:i + EVALUATION_BATCH_SIZE]).cpu().numpy()
                      for i in range(0, xv_g.shape[0], EVALUATION_BATCH_SIZE)]
        validation = validation_true_field_mse(np.concatenate(chunks, 0), basis_np, mean_np,
                                               y_validation)
        if not (np.isfinite(train_loss) and np.isfinite(validation)):
            nonfinite += 1
        train_curve.append(train_loss)
        validation_curve.append(validation)

        digest = _state_digest(model)
        if raw_best is None or validation < raw_best:
            raw_best, raw_best_epoch, raw_best_digest = validation, epoch, digest
        if running_best is None:
            running_best, selected_epoch = validation, epoch
            selected_state, selected_digest = copy.deepcopy(model.state_dict()), digest
        elif (running_best - validation) / running_best >= MIN_DELTA:
            running_best, selected_epoch, counter = validation, epoch, 0
            selected_state, selected_digest = copy.deepcopy(model.state_dict()), digest
        else:
            counter += 1
            if counter >= PATIENCE:
                stop_epoch, stop_reason = epoch, "PATIENCE"
                break

        if progress and epoch % progress == 0:
            print(f"  [{arm} s{seed}] epoch {epoch:3d} train {train_loss:.6e} "
                  f"val {validation:.6e} sel@{selected_epoch}", flush=True)

    if stop_epoch is None:
        stop_epoch = len(validation_curve)

    replay = replay_stopping(validation_curve, patience=PATIENCE, min_delta=MIN_DELTA)
    observed = {"selected_epoch": selected_epoch, "raw_best_epoch": raw_best_epoch,
                "stop_epoch": stop_epoch, "stop_reason": stop_reason}
    for field, got in observed.items():
        if replay[field] != got:
            raise StageBlocked(
                f"{arm} seed {seed}: in-loop {field}={got} != replay {replay[field]}")

    checkpoint = pathlib.Path(output_dir) / f"{arm}_seed{int(seed)}.pt"
    if checkpoint.exists():
        raise FileExistsError(f"{checkpoint} exists; outputs are create-only")
    torch.save(selected_state, checkpoint)

    trace = {
        "arm": arm, "seed": int(seed), "k": k,
        "train_loss_per_epoch": train_curve,
        "validation_true_field_mse_per_epoch": validation_curve,
        "raw_best_epoch": int(raw_best_epoch), "raw_best_validation_mse": float(raw_best),
        "raw_best_checkpoint_sha256": raw_best_digest,
        "selected_epoch": int(selected_epoch),
        "selected_validation_mse": float(running_best),
        "selected_checkpoint_sha256": selected_digest,
        "selected_checkpoint_path": checkpoint.name,
        "selected_checkpoint_file_sha256": sha256_file(checkpoint),
        "stop_epoch": int(stop_epoch), "stop_reason": stop_reason,
        "converged": stop_reason == "PATIENCE",
        "batch_order_sha256": hashlib.sha256(
            np.concatenate(permutations).astype(np.int32).tobytes()).hexdigest(),
        "nonfinite_count": int(nonfinite),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    validate_arm_trace(trace, REFERENCE_TRACE_FIELDS)
    validate_reference_seal(trace)
    if nonfinite:
        raise StageBlocked(f"{arm} seed {seed}: {nonfinite} non-finite epochs")
    return trace


def stage_d(train_dir, seal_path, stage_a_dir, stage_b_dir, stage_c_dir, output_dir,
            device="cpu", progress=25):
    started = time.perf_counter()
    protocol_sha, _ = verify_protocol_and_seal(seal_path)

    records = {}
    for name, directory in (("A", stage_a_dir), ("B", stage_b_dir), ("C", stage_c_dir)):
        path = pathlib.Path(directory) / f"STAGE_{name}.json"
        if not path.is_file():
            raise StageBlocked(f"stage {name} artifact missing at {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("protocol_sha256") != PROTOCOL_SHA256:
            raise StageBlocked(f"stage {name} was written against a different protocol")
        if record.get("status") != f"STAGE_{name}_COMPLETE":
            raise StageBlocked(f"stage {name} status {record.get('status')}")
        records[name] = (record, sha256_file(path))
    stage_a, stage_a_sha = records["A"]
    stage_b, stage_b_sha = records["B"]
    stage_c, stage_c_sha = records["C"]
    require_reference_convergence(stage_b["traces"])

    for name, digest in stage_a["arrays"].items():
        require_digest(pathlib.Path(stage_a_dir) / name, digest)
    basis = np.load(pathlib.Path(stage_a_dir) / "parent_basis_float32.npy")
    mean = np.load(pathlib.Path(stage_a_dir) / "train_mean_float32.npy")
    if basis.shape[0] != int(stage_a["R"]):
        raise StageBlocked("parent basis rank does not match Stage A")

    dense_seal = {}
    for seed_key, entry in stage_b["reference_checkpoints"].items():
        checkpoint = pathlib.Path(stage_b_dir) / entry["path"]
        require_digest(checkpoint, entry["file_sha256"])
        dense_seal[str(int(seed_key))] = dict(entry, source="STAGE_B", retrained=False)

    train_paths = []
    for family in FAMILIES:
        name = next(n for n in stage_a["train_file_digests"] if f"_train_{family}_" in n)
        path = open_role_file(pathlib.Path(train_dir) / name, "train", STAGE)
        require_digest(path, stage_a["train_file_digests"][name])
        train_paths.append((family, path))

    ledger = TrajectoryAccessLedger()
    x_train, y_train, train_reads = load_role_pairs(train_paths, "train", ledger)
    x_validation, y_validation, validation_reads = load_role_pairs(train_paths, "validation",
                                                                   ledger)
    counters = empty_access_counters()
    counters["hdf5_field_reads"]["train"] = train_reads
    counters["hdf5_field_reads"]["validation"] = validation_reads
    validate_access_counters(counters, STAGE)

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces, index_records = [], {}
    for arm in ("K_prop", "K_next"):
        indices = arm_indices(stage_c, arm)
        if index_digest(indices) != stage_c["index_digests"][arm]:
            raise StageBlocked(f"{arm}: index set does not match the Stage C digest")
        rows_basis = np.ascontiguousarray(basis[np.asarray(indices, dtype=np.int64)])
        index_records[arm] = {"k": len(indices), "indices": indices,
                              "index_sha256": index_digest(indices),
                              "basis_rows_sha256": hashlib.sha256(
                                  np.ascontiguousarray(rows_basis,
                                                       dtype=np.float32).tobytes()).hexdigest(),
                              "is_leading_prefix": indices == list(range(len(indices)))}
        for seed in SEEDS:
            print(f"[{arm} k={len(indices)} seed={seed}]", flush=True)
            traces.append(train_arm(arm, seed, rows_basis, mean, x_train, y_train,
                                    x_validation, y_validation, output_dir,
                                    device=device, progress=progress))
            _release(device)

    expected = len(SEEDS) * 2
    if len(traces) != expected:
        raise StageBlocked(f"{len(traces)} traces for {expected} cells")
    stalled = [t for t in traces if t["stop_reason"] == "CEILING_REACHED"]
    status = "STAGE_D_CEILING_REACHED" if stalled else "STAGE_D_COMPLETE"

    pretest = {
        "tolerance": TOLERANCE, "patience": PATIENCE, "min_delta": MIN_DELTA,
        "max_epochs": MAX_EPOCHS, "R": int(stage_a["R"]),
        "K_prop": int(stage_c["K_prop"]), "K_next": int(stage_c["K_next"]),
        "effective_ladder": stage_c["effective_ladder"],
        "index_digests": stage_c["index_digests"],
        "parent_basis_sha256": stage_a["arrays"]["parent_basis_float32.npy"],
        "train_mean_sha256": stage_a["arrays"]["train_mean_float32.npy"],
        "dense_checkpoints": dense_seal,
        "compressed_checkpoints": {f"{t['arm']}_seed{t['seed']}":
                                   {"state_sha256": t["selected_checkpoint_sha256"],
                                    "file_sha256": t["selected_checkpoint_file_sha256"],
                                    "epoch": t["selected_epoch"]} for t in traces},
        "frozen_after_this_seal": [
            "the two ranks and their index sets", "every checkpoint",
            "the parent basis and mean", "the tolerance",
            "the stopping policy", "the Stage E statistic and its bootstrap"],
    }

    record = {
        "protocol_id": PROTOCOL_ID, "stage": STAGE, "protocol_sha256": protocol_sha,
        "stage_a_sha256": stage_a_sha, "stage_b_sha256": stage_b_sha,
        "stage_c_sha256": stage_c_sha,
        "stage0_seal_sha256": stage_a["stage0_seal_sha256"],
        "arms": {"dense": "reused from Stage B, not retrained",
                 "K_prop": index_records["K_prop"]["k"],
                 "K_next": index_records["K_next"]["k"]},
        "index_records": index_records,
        "seeds": list(SEEDS), "traces": traces,
        "pretest_seal": pretest,
        "pretest_seal_sha256": hashlib.sha256(
            json.dumps(pretest, sort_keys=True).encode("utf-8")).hexdigest(),
        "runtime": {"optimizer": "Adam", "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "evaluation_batch_size": EVALUATION_BATCH_SIZE,
                    "validation_metric": "raw_full_field_mse_against_y"},
        "access_counters": counters, "trajectory_access": ledger.as_record(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "elapsed_seconds": float(time.perf_counter() - started),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "downstream_note": ("Stage E opens TEST exactly once and computes quality and the floor "
                            "over the index sets sealed here. Nothing above may change."),
    }
    write_artifact(output_dir / "STAGE_D.json", record)
    return record


def _release(device):
    import gc

    gc.collect()
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("--train-dir", "--seal", "--stage-a", "--stage-b", "--stage-c", "--output"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress", type=int, default=25)
    args = parser.parse_args(argv)
    record = stage_d(args.train_dir, args.seal, args.stage_a, args.stage_b, args.stage_c,
                     args.output, device=args.device, progress=args.progress)
    print(json.dumps({
        "status": record["status"], "arms": record["arms"],
        "access_counters": record["access_counters"],
        "pretest_seal_sha256": record["pretest_seal_sha256"],
        "is_leading_prefix": {k: v["is_leading_prefix"]
                              for k, v in record["index_records"].items()},
        "cells": [{k: t[k] for k in ("arm", "seed", "selected_epoch", "raw_best_epoch",
                                     "stop_epoch", "stop_reason", "selected_validation_mse")}
                  for t in record["traces"]]}, indent=2, sort_keys=True))
    return 0 if record["status"] == "STAGE_D_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
