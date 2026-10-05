"""QUALITY_CONVERGENCE_CONFIRM_V1 Stage E -- TEST, opened exactly once.

Evaluates the nine sealed checkpoints on the held-out shards, computes the selected-subspace floor
for each rank, and writes the verdict. Everything it uses was fixed by the Stage D pre-TEST seal;
this module chooses nothing.

The one-shot marker is consumed **before** TEST is opened. That ordering is the point: if the
process dies after the marker is written, the marker survives and a second attempt is refused. A
recovery path conditioned on `STAGE_E.json` being absent would reopen TEST, so no such path exists.
Failures are preserved, not retried.

The floor is `L_perp(S)` over the exact index sets Stage C sealed, re-verified here by index digest
and basis-row digest. On this cohort neither set is a leading prefix, so a prefix substitution
would change the number and with it the verdict.

Protocol: `protocols/QUALITY_CONVERGENCE_CONFIRM_V1.md`
Upstream: the Stage D pre-TEST seal.
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
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, FAMILIES, PAIRS_PER_TRAJECTORY, PROTOCOL_ID,
    PROTOCOL_SHA256, TOLERANCE, OneShotTestEvaluator, StageBlocked, arm_indices,
    assert_role_openable, empty_access_counters, floor_perp, floor_perp_per_trajectory,
    floor_point, index_digest, open_role_file, quality_point, paired_bootstrap, require_digest,
    sha256_file, validate_access_counters, validate_stage_e, verdict, verify_protocol_and_seal,
    write_artifact)

STAGE = "E"
SEEDS = (0, 1, 2)
TEST_TRAJECTORIES = 25
TEST_OFFSET = 200
EVALUATION_BATCH_SIZE = 512


def test_paths(data_dir):
    out = []
    for family in FAMILIES:
        name = f"NavierStokes2D_test_{family + TEST_OFFSET}_0.50000.h5"
        path = open_role_file(pathlib.Path(data_dir) / name, "test", STAGE)
        if not path.is_file():
            raise StageBlocked(f"{name} missing at {path}")
        out.append((family, path))
    return out


def load_test(paths, seal):
    """Local ids 0..24 of each test file, verified against the Stage 0 seal before reading."""
    import h5py

    by_name = {r["filename"]: r for r in seal["files"] if r["role"] == "test"}
    xs, ys, groups, reads, ledger = [], [], [], 0, []
    trajectory = 0
    for family, path in paths:
        record = by_name.get(path.name)
        if record is None:
            raise StageBlocked(f"{path.name} is not a sealed test file")
        if sha256_file(path) != record["sha256"]:
            raise StageBlocked(f"{path.name} changed since the seal")
        with h5py.File(path, "r") as handle:
            dataset = handle[list(handle.keys())[0]]["u"]
            if int(dataset.shape[0]) < TEST_TRAJECTORIES:
                raise StageBlocked(f"{path.name}: fewer than {TEST_TRAJECTORIES}")
            taken = dataset[:TEST_TRAJECTORIES]
        reads += TEST_TRAJECTORIES
        if taken.shape[1] - 1 != PAIRS_PER_TRAJECTORY:
            raise StageBlocked(f"{path.name}: unexpected pair count")
        d = int(taken.shape[-1] * taken.shape[-2])
        xs.append(taken[:, :-1].reshape(-1, d).astype(np.float32))
        ys.append(taken[:, 1:].reshape(-1, d).astype(np.float32))
        for _ in range(TEST_TRAJECTORIES):
            groups.extend([trajectory] * PAIRS_PER_TRAJECTORY)
            trajectory += 1
        ledger.append({"file": path.name, "role": "test", "mode": "field_read",
                       "trajectories": TEST_TRAJECTORIES, "family": family})
    return (np.concatenate(xs, 0), np.concatenate(ys, 0),
            np.asarray(groups, dtype=np.int64), reads, ledger)


def evaluate(checkpoint, x_test, y_test, groups, out_dim, basis_rows, mean, device):
    """Per-trajectory TEST MSE in the raw field, for one sealed checkpoint."""
    import torch
    from e2e_cost import MLP

    model = MLP(int(x_test.shape[1]), int(out_dim)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(x_test, dtype=np.float32)).to(device)
    with torch.no_grad():
        chunks = [model(x[i:i + EVALUATION_BATCH_SIZE]).cpu().numpy()
                  for i in range(0, x.shape[0], EVALUATION_BATCH_SIZE)]
    prediction = np.concatenate(chunks, 0).astype(np.float64)
    if basis_rows is not None:
        prediction = prediction @ np.asarray(basis_rows, dtype=np.float64) \
            + np.asarray(mean, dtype=np.float64)
    per_row = ((prediction - np.asarray(y_test, dtype=np.float64)) ** 2).mean(axis=1)
    per_trajectory = np.array([per_row[groups == g].mean() for g in np.unique(groups)])
    state = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        state.update(name.encode("utf-8"))
        state.update(np.ascontiguousarray(
            tensor.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return per_trajectory, float(per_row.mean()), state.hexdigest()


def stage_e(data_dir, seal_path, stage_a_dir, stage_c_dir, stage_d_dir, stage_b_dir,
            output_dir, device="cpu"):
    started = time.perf_counter()
    protocol_sha, seal = verify_protocol_and_seal(seal_path)

    stage_d_path = pathlib.Path(stage_d_dir) / "STAGE_D.json"
    stage_c_path = pathlib.Path(stage_c_dir) / "STAGE_C.json"
    for path in (stage_d_path, stage_c_path):
        if not path.is_file():
            raise StageBlocked(f"upstream artifact missing at {path}")
    stage_d = json.loads(stage_d_path.read_text(encoding="utf-8"))
    stage_c = json.loads(stage_c_path.read_text(encoding="utf-8"))
    if stage_d.get("status") != "STAGE_D_COMPLETE":
        raise StageBlocked(f"Stage D status {stage_d.get('status')}")
    if stage_d.get("protocol_sha256") != PROTOCOL_SHA256:
        raise StageBlocked("Stage D was written against a different protocol")
    pretest = stage_d["pretest_seal"]
    recomputed = hashlib.sha256(json.dumps(pretest, sort_keys=True).encode("utf-8")).hexdigest()
    if recomputed != stage_d["pretest_seal_sha256"]:
        raise StageBlocked("the pre-TEST seal does not match its recorded digest")

    basis = np.load(pathlib.Path(stage_a_dir) / "parent_basis_float32.npy")
    mean = np.load(pathlib.Path(stage_a_dir) / "train_mean_float32.npy")
    require_digest(pathlib.Path(stage_a_dir) / "parent_basis_float32.npy",
                   pretest["parent_basis_sha256"])
    require_digest(pathlib.Path(stage_a_dir) / "train_mean_float32.npy",
                   pretest["train_mean_sha256"])

    arms = {}
    for arm in ("K_prop", "K_next"):
        indices = arm_indices(stage_c, arm)
        if index_digest(indices) != pretest["index_digests"][arm]:
            raise StageBlocked(f"{arm}: index digest differs from the pre-TEST seal")
        rows = np.ascontiguousarray(basis[np.asarray(indices, dtype=np.int64)])
        recorded = stage_d["index_records"][arm]["basis_rows_sha256"]
        if hashlib.sha256(np.ascontiguousarray(rows, dtype=np.float32).tobytes()).hexdigest() \
                != recorded:
            raise StageBlocked(f"{arm}: basis rows differ from the pre-TEST seal")
        arms[arm] = {"indices": indices, "rows": rows, "k": len(indices),
                     "is_leading_prefix": indices == list(range(len(indices)))}

    checkpoints = {}
    for seed in SEEDS:
        entry = pretest["dense_checkpoints"][str(seed)]
        path = pathlib.Path(stage_b_dir) / entry["path"]
        require_digest(path, entry["file_sha256"])
        checkpoints[("dense", seed)] = (path, None, entry["state_sha256"])
    for arm in ("K_prop", "K_next"):
        for seed in SEEDS:
            entry = pretest["compressed_checkpoints"][f"{arm}_seed{seed}"]
            path = pathlib.Path(stage_d_dir) / f"{arm}_seed{seed}.pt"
            require_digest(path, entry["file_sha256"])
            checkpoints[(arm, seed)] = (path, arm, entry["state_sha256"])

    # ---- the marker is consumed before TEST is opened, and never after ----
    assert_role_openable("test", STAGE)
    output_dir = pathlib.Path(output_dir)
    marker = OneShotTestEvaluator(output_dir / "TEST_ACCESS_CONSUMED.json", {
        "protocol_sha256": protocol_sha, "stage0_seal_sha256": pretest.get("stage0_seal_sha256")
        or stage_d.get("stage0_seal_sha256"),
        "stage_d_sha256": sha256_file(stage_d_path),
        "pretest_seal_sha256": stage_d["pretest_seal_sha256"],
        "stage_e_executor_sha256": sha256_file(pathlib.Path(__file__)),
        "expected_test_digests": {r["filename"]: r["sha256"] for r in seal["files"]
                                  if r["role"] == "test"},
        "attempt": output_dir.name})
    marker.open()

    paths = test_paths(data_dir)
    x_test, y_test, groups, reads, ledger = load_test(paths, seal)
    counters = empty_access_counters()
    counters["hdf5_field_reads"]["test"] = reads
    validate_access_counters(counters, STAGE)

    per_trajectory, per_seed_mse, loaded = {}, {}, {}
    for (arm, seed), (path, which, expected_state) in sorted(checkpoints.items()):
        rows = arms[which]["rows"] if which else None
        out_dim = arms[which]["k"] if which else int(y_test.shape[1])
        traj, overall, state = evaluate(path, x_test, y_test, groups, out_dim, rows, mean,
                                        device)
        if state != expected_state:
            raise StageBlocked(f"{arm} seed {seed}: loaded state digest != sealed")
        if not np.all(np.isfinite(traj)):
            raise StageBlocked(f"{arm} seed {seed}: non-finite TEST loss")
        per_trajectory[(arm, seed)] = traj
        per_seed_mse.setdefault(arm, {})[seed] = overall
        loaded[f"{arm}_seed{seed}"] = state
        print(f"  {arm:7s} s{seed} TEST mse {overall:.8e}", flush=True)

    floors, floor_traj = {}, {}
    for arm in ("K_prop", "K_next"):
        floors[arm] = floor_perp(basis, arms[arm]["indices"], y_test, mean)
        floor_traj[arm] = floor_perp_per_trajectory(basis, arms[arm]["indices"], y_test, mean,
                                                    groups)
        print(f"  floor({arm}) = {floors[arm]:.8e}", flush=True)

    dense = per_seed_mse["dense"]
    results = {}
    for arm in ("K_prop", "K_next"):
        q_seeds = {s: per_seed_mse[arm][s] / dense[s] for s in SEEDS}
        f_seeds = {s: floors[arm] / dense[s] for s in SEEDS}
        block = {
            "k": arms[arm]["k"], "indices_sha256": index_digest(arms[arm]["indices"]),
            "is_leading_prefix": arms[arm]["is_leading_prefix"],
            "point": quality_point(per_seed_mse[arm], dense),
            "floor_point": floor_point(floors[arm], dense),
            "l_perp": floors[arm],
            "per_seed_ratios": {str(s): q_seeds[s] for s in SEEDS},
            "per_seed_floor_ratios": {str(s): f_seeds[s] for s in SEEDS},
            "per_seed_test_mse": {str(s): per_seed_mse[arm][s] for s in SEEDS},
            "functional": "median_of_per_seed_ratios",
        }
        block["verdict"] = verdict(block["point"], block["floor_point"], TOLERANCE)
        results[arm] = block

    boot = paired_bootstrap(per_trajectory, floor_traj, replicates=BOOTSTRAP_REPLICATES,
                            seed=BOOTSTRAP_SEED)
    for arm in ("K_prop", "K_next"):
        results[arm]["interval"] = boot[f"quality_{arm}"]["interval"]
        results[arm]["floor_interval"] = boot[f"floor_{arm}"]["interval"]
        validate_stage_e(results[arm])

    raw = output_dir / "TEST_PER_TRAJECTORY.npz"
    np.savez(raw, **{f"{a}_seed{s}": v for (a, s), v in per_trajectory.items()},
             **{f"floor_{a}": v for a, v in floor_traj.items()},
             groups=np.unique(groups))

    record = {
        "protocol_id": PROTOCOL_ID, "stage": STAGE, "protocol_sha256": protocol_sha,
        "stage_c_sha256": sha256_file(stage_c_path),
        "stage_d_sha256": sha256_file(stage_d_path),
        "pretest_seal_sha256": stage_d["pretest_seal_sha256"],
        "stage_e_executor_sha256": sha256_file(pathlib.Path(__file__)),
        "tolerance": TOLERANCE, "seeds": list(SEEDS),
        "dense_per_seed_test_mse": {str(s): dense[s] for s in SEEDS},
        "results": results,
        "bootstrap": {k: v for k, v in boot.items() if not isinstance(v, dict)},
        "bootstrap_blocks": {k: v for k, v in boot.items() if isinstance(v, dict)},
        "raw_per_trajectory": raw.name,
        "raw_per_trajectory_sha256": sha256_file(raw),
        "loaded_state_digests": loaded,
        "test_rows": int(x_test.shape[0]), "test_trajectories": int(np.unique(groups).size),
        "access_counters": counters, "access_ledger": ledger,
        "one_shot_marker": marker.marker_path.name,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "elapsed_seconds": float(time.perf_counter() - started),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STAGE_E_COMPLETE",
    }
    write_artifact(output_dir / "STAGE_E.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("--data-dir", "--seal", "--stage-a", "--stage-b", "--stage-c", "--stage-d",
                 "--output"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    record = stage_e(args.data_dir, args.seal, args.stage_a, args.stage_c, args.stage_d,
                     args.stage_b, args.output, device=args.device)
    print(json.dumps({"status": record["status"],
                      "access_counters": record["access_counters"],
                      "dense_per_seed_test_mse": record["dense_per_seed_test_mse"],
                      "results": {a: {k: b[k] for k in
                                      ("k", "point", "interval", "floor_point",
                                       "floor_interval", "per_seed_ratios", "verdict",
                                       "is_leading_prefix")}
                                  for a, b in record["results"].items()}},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
