"""QUALITY_CONVERGENCE_CONFIRM_V1 Stage B -- the three dense reference models.

Trains `reference_dense` at seeds 0, 1 and 2 to the frozen stopping rule and seals the
checkpoints. Nothing else. The Stage C arms are not trained here and SELECTION is not opened.

Two things this stage exists to get right.

  the split inside one file   TRAIN and VALIDATION live in the same physical train HDF5. Every
                              dataset access goes through the trajectory guard, so a VALIDATION
                              id cannot reach a gradient and an UNUSED id cannot be read at all.

  which checkpoint survives   `min_delta` governs the kept checkpoint, not only the patience
                              counter, so the saved weights are the last significant
                              improvement. The raw validation argmin is recorded as a digest for
                              diagnosis and is never written to disk, because there must be no
                              file for a later stage to pick up by mistake.

VALIDATION is monitored as MSE against the raw field `y`, never against a projection, and never
contributes a gradient.

Protocol: `protocols/QUALITY_CONVERGENCE_CONFIRM_V1.md`
Upstream: Stage A, and through it the Stage 0 data seal.
"""
from __future__ import annotations

import argparse
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
    PROTOCOL_SHA256, REFERENCE_TRACE_FIELDS, SPLIT_SEED, TRAJECTORIES_PER_FILE, StageBlocked,
    TrajectoryAccessLedger, empty_access_counters, open_role_file, replay_stopping,
    request_trajectories, require_digest, sha256_file, split_family, stage_b_status,
    validate_access_counters, validate_arm_trace, validate_reference_seal,
    validation_true_field_mse, verify_protocol_and_seal, write_artifact)

STAGE = "B"
ARM = "reference_dense"
SEEDS = (0, 1, 2)

#: Protocol-fixed training setup: Adam with an MLP of hidden width 256.
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 512


# ----------------------------------------------------------------------------------------
# data, through the trajectory guard
# ----------------------------------------------------------------------------------------

def load_role_pairs(train_paths, role, ledger):
    """Gather only the sealed trajectories for `role`, in the sealed order.

    Both TRAIN and VALIDATION come out of the same file. The guard, not the caller, decides
    which ids are permissible for this stage and role.
    """
    import h5py

    xs, ys, reads = [], [], 0
    for family_index, (family, path) in enumerate(train_paths):
        _, validation, _ = split_family(SPLIT_SEED, family_index)
        train, _, _ = split_family(SPLIT_SEED, family_index)
        wanted = train if role == "train" else validation
        ids = request_trajectories(ledger, family_index, wanted, role=role, stage=STAGE)
        with h5py.File(path, "r") as handle:
            dataset = handle[list(handle.keys())[0]]["u"]
            if int(dataset.shape[0]) != TRAJECTORIES_PER_FILE:
                raise StageBlocked(f"{path.name}: unexpected trajectory count")
            taken = dataset[ids]
        reads += len(ids)
        order = np.argsort(np.argsort(np.asarray(wanted, dtype=np.int64)))
        taken = taken[order]
        if taken.shape[1] - 1 != PAIRS_PER_TRAJECTORY:
            raise StageBlocked(f"{path.name}: unexpected pair count")
        d = int(taken.shape[-1] * taken.shape[-2])
        xs.append(taken[:, :-1].reshape(-1, d).astype(np.float32))
        ys.append(taken[:, 1:].reshape(-1, d).astype(np.float32))
    return np.concatenate(xs, 0), np.concatenate(ys, 0), reads


def train_paths_from_stage_a(stage_a_record, train_dir):
    """Re-verify each train shard against the digest Stage A recorded."""
    digests = stage_a_record["train_file_digests"]
    out = []
    for family in FAMILIES:
        matches = [name for name in digests if f"_train_{family}_" in name]
        if len(matches) != 1:
            raise StageBlocked(f"family {family} is not uniquely present in the Stage A record")
        name = matches[0]
        path = open_role_file(pathlib.Path(train_dir) / name, "train", STAGE)
        if not path.is_file():
            raise StageBlocked(f"{name} missing at {path}")
        actual = sha256_file(path)
        if actual != digests[name]:
            raise StageBlocked(f"{name} changed since Stage A: {actual} != {digests[name]}")
        out.append((family, path))
    return out


# ----------------------------------------------------------------------------------------
# one reference model
# ----------------------------------------------------------------------------------------

def _state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(
            tensor.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return digest.hexdigest()


def train_reference(seed, x_train, y_train, x_validation, y_validation,
                    output_dir, device="cpu", progress=25):
    """Adam on the dense full-field target, monitored on the raw validation field."""
    import copy

    import torch
    import torch.nn as nn
    from e2e_cost import MLP

    started = time.perf_counter()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    x_g = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32)).to(device)
    y_g = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))
    xv_g = torch.from_numpy(np.ascontiguousarray(x_validation, dtype=np.float32)).to(device)
    rows = int(y_g.shape[0])

    model = MLP(int(x_train.shape[1]), int(y_train.shape[1])).to(device)
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
            optimizer.zero_grad()
            loss = loss_function(model(x_g[index.to(device)]), y_g[index].to(device))
            loss.backward()
            optimizer.step()
            total += loss.detach().double() * index.shape[0]
        train_loss = float((total / rows).item())

        model.eval()
        with torch.no_grad():
            chunks = [model(xv_g[i:i + EVALUATION_BATCH_SIZE]).cpu().numpy()
                      for i in range(0, xv_g.shape[0], EVALUATION_BATCH_SIZE)]
        validation = validation_true_field_mse(np.concatenate(chunks, 0), None, None,
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
            print(f"  [{ARM} s{seed}] epoch {epoch:3d} train {train_loss:.6e} "
                  f"val {validation:.6e} sel@{selected_epoch}", flush=True)

    if stop_epoch is None:
        stop_epoch = len(validation_curve)

    # cross-check the in-loop bookkeeping against the frozen replay of the same rule
    replay = replay_stopping(validation_curve, patience=PATIENCE, min_delta=MIN_DELTA)
    for field in ("selected_epoch", "raw_best_epoch", "stop_epoch", "stop_reason"):
        got = {"selected_epoch": selected_epoch, "raw_best_epoch": raw_best_epoch,
               "stop_epoch": stop_epoch, "stop_reason": stop_reason}[field]
        if replay[field] != got:
            raise StageBlocked(
                f"seed {seed}: in-loop {field}={got} disagrees with the replay {replay[field]}")

    checkpoint = pathlib.Path(output_dir) / f"{ARM}_seed{int(seed)}.pt"
    if checkpoint.exists():
        raise FileExistsError(f"{checkpoint} exists; outputs are create-only")
    torch.save(selected_state, checkpoint)

    trace = {
        "arm": ARM, "seed": int(seed),
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
        raise StageBlocked(f"seed {seed}: {nonfinite} non-finite epochs")
    return trace


# ----------------------------------------------------------------------------------------
# stage B
# ----------------------------------------------------------------------------------------

def stage_b(train_dir, seal_path, stage_a_dir, output_dir, device="cpu", progress=25):
    started = time.perf_counter()
    protocol_sha, _ = verify_protocol_and_seal(seal_path)

    stage_a_path = pathlib.Path(stage_a_dir) / "STAGE_A.json"
    if not stage_a_path.is_file():
        raise StageBlocked(f"Stage A artifact missing at {stage_a_path}")
    stage_a_record = json.loads(stage_a_path.read_text(encoding="utf-8"))
    if stage_a_record.get("status") != "STAGE_A_COMPLETE":
        raise StageBlocked(f"Stage A status {stage_a_record.get('status')}")
    if stage_a_record.get("protocol_sha256") != PROTOCOL_SHA256:
        raise StageBlocked("Stage A was written against a different protocol")
    stage_a_sha = sha256_file(stage_a_path)
    for name, digest in stage_a_record["arrays"].items():
        require_digest(pathlib.Path(stage_a_dir) / name, digest)

    paths = train_paths_from_stage_a(stage_a_record, train_dir)
    ledger = TrajectoryAccessLedger()
    x_train, y_train, train_reads = load_role_pairs(paths, "train", ledger)
    x_validation, y_validation, validation_reads = load_role_pairs(paths, "validation", ledger)

    counters = empty_access_counters()
    counters["hdf5_field_reads"]["train"] = train_reads
    counters["hdf5_field_reads"]["validation"] = validation_reads
    validate_access_counters(counters, STAGE)
    print(f"TRAIN {x_train.shape}  VALIDATION {x_validation.shape}", flush=True)

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    for seed in SEEDS:
        print(f"[{ARM} seed={seed}]", flush=True)
        traces.append(train_reference(seed, x_train, y_train, x_validation, y_validation,
                                      output_dir, device=device, progress=progress))
        _release(device)

    if len(traces) != len(SEEDS):
        raise StageBlocked(f"{len(traces)} traces for {len(SEEDS)} seeds")
    status = stage_b_status(traces)

    record = {
        "protocol_id": PROTOCOL_ID, "stage": STAGE, "protocol_sha256": protocol_sha,
        "stage_a_sha256": stage_a_sha, "stage0_seal_sha256": stage_a_record["stage0_seal_sha256"],
        "arm": ARM, "seeds": list(SEEDS),
        "convergence_policy": {"patience": PATIENCE, "min_delta": MIN_DELTA,
                               "max_epochs": MAX_EPOCHS,
                               "min_delta_semantics": "relative_and_governs_the_checkpoint",
                               "source": "CONVERGENCE_POLICY_PILOT_V1"},
        "runtime": {"optimizer": "Adam", "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "evaluation_batch_size": EVALUATION_BATCH_SIZE,
                    "validation_metric": "raw_full_field_mse_against_y"},
        "train_rows": int(x_train.shape[0]), "validation_rows": int(x_validation.shape[0]),
        "traces": traces,
        "reference_checkpoints": {str(t["seed"]): {
            "path": t["selected_checkpoint_path"],
            "state_sha256": t["selected_checkpoint_sha256"],
            "file_sha256": t["selected_checkpoint_file_sha256"],
            "epoch": t["selected_epoch"]} for t in traces},
        "access_counters": counters, "trajectory_access": ledger.as_record(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "elapsed_seconds": float(time.perf_counter() - started),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "downstream_note": ("Stage C consumes reference_checkpoints from this record only. It "
                            "does not scan the directory and does not re-select a checkpoint."),
    }
    write_artifact(output_dir / "STAGE_B.json", record)
    return record


def _release(device):
    import gc

    gc.collect()
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--stage-a", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress", type=int, default=25)
    args = parser.parse_args(argv)
    record = stage_b(args.train_dir, args.seal, args.stage_a, args.output,
                     device=args.device, progress=args.progress)
    print(json.dumps({
        "status": record["status"],
        "access_counters": record["access_counters"],
        "per_seed": [{k: t[k] for k in ("seed", "selected_epoch", "selected_validation_mse",
                                        "raw_best_epoch", "stop_epoch", "stop_reason")}
                     for t in record["traces"]]}, indent=2, sort_keys=True))
    return 0 if record["status"] == "STAGE_B_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
