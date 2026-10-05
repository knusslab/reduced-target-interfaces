"""QUALITY_CONVERGENCE_CONFIRM_V1 Stage C -- the proposed rank.

Opens SELECTION for the first time, evaluates the sealed reference on it, computes the predictive
gains in the Stage A parent basis, and resolves `K_prop` and `K_next` through the canonical
selector. No arm is trained here and TEST is neither present on this host nor referenced.

Three constraints worth naming, because each one has already gone wrong somewhere in this line.

  the reference is named, not found   the only route to a checkpoint is the `reference_checkpoints`
                                      block of `STAGE_B.json`. This module never lists a directory
                                      and never re-picks a "best" checkpoint.

  the gain spans the whole basis      a gain vector narrower than `R` caps the ladder silently.
                                      `assert_gain_width` refuses that.

  `R` is a boundary, not a rung       `tail[R] = 0` for any data, so a rung there is admitted
                                      unconditionally. That is how V8 arrived at an
                                      uninformative `K_pred = 256`.

If the selector returns any censor reason, or lands on the largest effective rung, the run stops
before Stage D rather than producing a rank that carries no information.

Protocol: `protocols/QUALITY_CONVERGENCE_CONFIRM_V1.md`
Upstream: Stage B, and through it Stage A and the Stage 0 data seal.
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

from bigdata_predictive_gain_selector_v1 import (  # noqa: E402
    canonical_predictive_gain_selector)
from run_quality_convergence_confirm_v1 import (  # noqa: E402
    FAMILIES, PAIRS_PER_TRAJECTORY, PROTOCOL_ID, PROTOCOL_SHA256, StageBlocked,
    assert_gain_width, assert_role_openable, effective_ladder, empty_access_counters,
    index_digest, open_role_file, reference_checkpoint_for_stage_c, require_digest,
    require_reference_convergence, resolve_ranks, sha256_file, validate_access_counters,
    verify_protocol_and_seal, write_artifact)

STAGE = "C"
SELECTION_TRAJECTORIES = 25
SELECTION_OFFSET = 100
TAU = 0.05
EVALUATION_BATCH_SIZE = 512
PRIMARY_SELECTOR_SEED = 0


def selection_paths(data_dir):
    """The eight official valid shards, by family. Refused if any test file is named."""
    out = []
    for family in FAMILIES:
        name = f"NavierStokes2D_valid_{family + SELECTION_OFFSET}_0.50000.h5"
        path = open_role_file(pathlib.Path(data_dir) / name, "selection", STAGE)
        if not path.is_file():
            raise StageBlocked(f"{name} missing at {path}")
        out.append((family, path))
    return out


def load_selection(paths, seal):
    """Local ids 0..24 of each valid file, in family order, verified against the Stage 0 seal."""
    import h5py

    by_name = {r["filename"]: r for r in seal["files"] if r["role"] == "selection"}
    xs, ys, reads, ledger = [], [], 0, []
    for family, path in paths:
        record = by_name.get(path.name)
        if record is None:
            raise StageBlocked(f"{path.name} is not a sealed selection file")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise StageBlocked(f"{path.name} changed since the seal")
        with h5py.File(path, "r") as handle:
            dataset = handle[list(handle.keys())[0]]["u"]
            if int(dataset.shape[0]) < SELECTION_TRAJECTORIES:
                raise StageBlocked(f"{path.name}: fewer than {SELECTION_TRAJECTORIES}")
            taken = dataset[:SELECTION_TRAJECTORIES]
        reads += SELECTION_TRAJECTORIES
        if taken.shape[1] - 1 != PAIRS_PER_TRAJECTORY:
            raise StageBlocked(f"{path.name}: unexpected pair count")
        d = int(taken.shape[-1] * taken.shape[-2])
        xs.append(taken[:, :-1].reshape(-1, d).astype(np.float32))
        ys.append(taken[:, 1:].reshape(-1, d).astype(np.float32))
        ledger.append({"file": path.name, "role": "selection", "mode": "field_read",
                       "trajectories": SELECTION_TRAJECTORIES, "family": family})
    return np.concatenate(xs, 0), np.concatenate(ys, 0), reads, ledger


def predict_with_reference(checkpoint_path, x_selection, input_dim, output_dim, device):
    """Load exactly the checkpoint the Stage B seal names, and run it once."""
    import torch
    from e2e_cost import MLP

    model = MLP(int(input_dim), int(output_dim)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(x_selection, dtype=np.float32)).to(device)
    with torch.no_grad():
        chunks = [model(x[i:i + EVALUATION_BATCH_SIZE]).cpu().numpy()
                  for i in range(0, x.shape[0], EVALUATION_BATCH_SIZE)]
    state = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        state.update(name.encode("utf-8"))
        state.update(np.ascontiguousarray(
            tensor.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return np.concatenate(chunks, 0), state.hexdigest()


def gains_in_basis(prediction, target, basis, mean):
    """`q_j - e_j` per basis direction: target content minus what the reference already misses."""
    residual = np.asarray(target, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    coeff_target = residual @ np.asarray(basis, dtype=np.float64).T
    coeff_error = ((np.asarray(prediction, dtype=np.float64)
                    - np.asarray(target, dtype=np.float64))
                   @ np.asarray(basis, dtype=np.float64).T)
    return (coeff_target ** 2).mean(0) - (coeff_error ** 2).mean(0)


def stage_c(data_dir, seal_path, stage_a_dir, stage_b_dir, output_dir, device="cpu"):
    started = time.perf_counter()
    protocol_sha, seal = verify_protocol_and_seal(seal_path)

    stage_a_path = pathlib.Path(stage_a_dir) / "STAGE_A.json"
    stage_b_path = pathlib.Path(stage_b_dir) / "STAGE_B.json"
    for path, stage in ((stage_a_path, "A"), (stage_b_path, "B")):
        if not path.is_file():
            raise StageBlocked(f"stage {stage} artifact missing at {path}")
    stage_a = json.loads(stage_a_path.read_text(encoding="utf-8"))
    stage_b = json.loads(stage_b_path.read_text(encoding="utf-8"))
    for record, stage in ((stage_a, "A"), (stage_b, "B")):
        if record.get("protocol_sha256") != PROTOCOL_SHA256:
            raise StageBlocked(f"stage {stage} was written against a different protocol")
    if stage_b.get("status") != "STAGE_B_COMPLETE":
        raise StageBlocked(f"Stage B status {stage_b.get('status')}")
    require_reference_convergence(stage_b["traces"])
    for name, digest in stage_a["arrays"].items():
        require_digest(pathlib.Path(stage_a_dir) / name, digest)

    basis = np.load(pathlib.Path(stage_a_dir) / "parent_basis_float32.npy")
    mean = np.load(pathlib.Path(stage_a_dir) / "train_mean_float32.npy")
    R = int(stage_a["R"])
    if basis.shape[0] != R:
        raise StageBlocked(f"parent basis rank {basis.shape[0]} != recorded R {R}")

    assert_role_openable("selection", STAGE)
    paths = selection_paths(data_dir)
    x_selection, y_selection, reads, ledger = load_selection(paths, seal)
    counters = empty_access_counters()
    counters["hdf5_field_reads"]["selection"] = reads
    validate_access_counters(counters, STAGE)
    print(f"SELECTION {x_selection.shape}", flush=True)

    per_seed, gains_primary = {}, None
    for seed_key, entry in sorted(stage_b["reference_checkpoints"].items(),
                                  key=lambda kv: int(kv[0])):
        seed = int(seed_key)
        trace = next(t for t in stage_b["traces"] if int(t["seed"]) == seed)
        expected_state = reference_checkpoint_for_stage_c(trace)
        checkpoint = pathlib.Path(stage_b_dir) / entry["path"]
        require_digest(checkpoint, entry["file_sha256"])

        prediction, state_digest = predict_with_reference(
            checkpoint, x_selection, x_selection.shape[1], y_selection.shape[1], device)
        if state_digest != expected_state:
            raise StageBlocked(
                f"seed {seed}: loaded state digest {state_digest} != sealed "
                f"{expected_state}")

        gains = gains_in_basis(prediction, y_selection, basis, mean)
        assert_gain_width(gains.size, R)
        sigma2 = float(((prediction.astype(np.float64)
                         - y_selection.astype(np.float64)) ** 2).mean())
        threshold = TAU * float(y_selection.shape[1]) * sigma2
        selection = canonical_predictive_gain_selector(
            gains, effective_ladder(R), threshold, basis_rank=R)
        per_seed[seed] = {
            "seed": seed, "checkpoint_state_sha256": state_digest,
            "selected_k": int(selection["selected_k"]),
            "censor_reasons": list(selection["censor_reasons"]),
            "feasible_rungs": [int(k) for k in selection["feasible_rungs"]],
            "used_max_rung_fallback": bool(selection["used_max_rung_fallback"]),
            "selected_slack": float(selection["selected_slack"]),
            "tail_at_selected": float(selection["tail_at_selected"]),
            "tail_by_rung": {k: float(v) for k, v in selection["tail_by_rung"].items()},
            "threshold": threshold, "sigma2": sigma2,
            "reference_mse_on_selection": sigma2,
            "order_sha256": index_digest(selection["order"]),
        }
        if seed == PRIMARY_SELECTOR_SEED:
            gains_primary = (gains, selection)
        print(f"  seed {seed}: K={selection['selected_k']} "
              f"censor={selection['censor_reasons']} slack={selection['selected_slack']:.4g}",
              flush=True)

    if gains_primary is None:
        raise StageBlocked(f"primary selector seed {PRIMARY_SELECTOR_SEED} absent")
    gains, selection = gains_primary
    ranks = resolve_ranks(selection, R)
    order = np.asarray(selection["order"], dtype=np.int64)
    index_sets = {"K_prop": order[:ranks["K_prop"]].tolist(),
                  "K_next": order[:ranks["K_next"]].tolist()}

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gains_path = output_dir / "selection_gains_float64.npy"
    if gains_path.exists():
        raise FileExistsError(f"{gains_path} exists; outputs are create-only")
    np.save(gains_path, np.asarray(gains, dtype=np.float64))

    record = {
        "protocol_id": PROTOCOL_ID, "stage": STAGE, "protocol_sha256": protocol_sha,
        "stage_a_sha256": sha256_file(stage_a_path),
        "stage_b_sha256": sha256_file(stage_b_path),
        "stage0_seal_sha256": stage_a["stage0_seal_sha256"],
        "R": R, "effective_ladder": ranks["effective_ladder"], "tau": TAU,
        "primary_selector_seed": PRIMARY_SELECTOR_SEED,
        "K_prop": ranks["K_prop"], "K_next": ranks["K_next"],
        "index_sets": index_sets,
        "index_digests": {k: index_digest(v) for k, v in index_sets.items()},
        "gain_order_sha256": index_digest(order),
        "gains_sha256": sha256_file(gains_path),
        "per_seed": per_seed,
        "selection_rows": int(x_selection.shape[0]),
        "access_counters": counters, "access_ledger": ledger,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "elapsed_seconds": float(time.perf_counter() - started),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STAGE_C_COMPLETE",
        "downstream_note": ("Stage D trains dense, K_prop and K_next using index_sets from this "
                            "record. The floor at Stage E is computed over these same index "
                            "sets, never over a POD prefix."),
    }
    write_artifact(output_dir / "STAGE_C.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--stage-a", required=True)
    parser.add_argument("--stage-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    record = stage_c(args.data_dir, args.seal, args.stage_a, args.stage_b, args.output,
                     device=args.device)
    print(json.dumps({k: record[k] for k in
                      ("status", "R", "K_prop", "K_next", "effective_ladder",
                       "selection_rows", "access_counters")}, indent=2, sort_keys=True))
    print(json.dumps({s: {k: v[k] for k in
                          ("selected_k", "censor_reasons", "selected_slack", "threshold",
                           "sigma2", "used_max_rung_fallback")}
                      for s, v in record["per_seed"].items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
