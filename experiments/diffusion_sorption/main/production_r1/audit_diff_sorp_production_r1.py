from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ADMISSION_ROOT = HERE.parent
PACKAGE_CODE = ADMISSION_ROOT / "package" / "code"
sys.path.insert(0, str(PACKAGE_CODE))

import diff_sorp_second_contract_v1 as core  # noqa: E402


LINEAGE_ID = "DIFF_SORP_SECOND_CONTRACT_V1_PRODUCTION_R1"


class AuditFailure(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditFailure("JSON object required: %s" % path)
    return value


def write_json_create_only(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def verify_bound_files(root: Path, manifest: Mapping[str, Mapping]) -> bool:
    root = root.resolve()
    for relative, expected in sorted(manifest.items()):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise AuditFailure("bound path escape: %s" % relative) from exc
        if not path.is_file():
            raise AuditFailure("bound file missing: %s" % relative)
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(expected.get("bytes", -1)) or digest != expected.get("sha256"):
            raise AuditFailure("bound file mismatch: %s" % relative)
    return True


def replay_stopping(curve, patience: int, min_delta: float) -> dict:
    values = [float(value) for value in curve]
    if not values or not all(np.isfinite(values)) or values[0] <= 0:
        raise AuditFailure("invalid validation curve")
    retained = values[0]
    selected_epoch = 1
    counter = 0
    stop_epoch = len(values)
    stop_reason = "CEILING_REACHED"
    for epoch, value in enumerate(values[1:], start=2):
        if (retained - value) / retained >= float(min_delta):
            retained = value
            selected_epoch = epoch
            counter = 0
        else:
            counter += 1
            if counter >= int(patience):
                stop_epoch = epoch
                stop_reason = "PATIENCE"
                break
    ran = values[:stop_epoch]
    raw_best = min(ran)
    return {
        "selected_epoch": selected_epoch,
        "selected_validation_mse": retained,
        "raw_best_epoch": ran.index(raw_best) + 1,
        "raw_best_validation_mse": raw_best,
        "stop_epoch": stop_epoch,
        "stop_reason": stop_reason,
        "converged": stop_reason == "PATIENCE",
    }


def state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return digest.hexdigest()


def _array_sha(values, dtype) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def _selector(y, prediction, basis, mean, ladder, tau) -> dict:
    target64 = np.asarray(y, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    basis64 = np.asarray(basis, dtype=np.float64)
    mean64 = np.asarray(mean, dtype=np.float64).reshape(-1)
    coefficients = (target64 - mean64) @ basis64.T
    predicted_coefficients = (prediction64 - mean64) @ basis64.T
    q = np.mean(coefficients * coefficients, axis=0)
    e = np.mean((coefficients - predicted_coefficients) ** 2, axis=0)
    gain = q - e
    positive = np.clip(gain, 0.0, None)
    order = np.argsort(-positive, kind="stable").astype(np.int64)
    reference_mse = float(np.mean((prediction64 - target64) ** 2))
    threshold = float(tau) * target64.shape[1] * reference_mse
    selected = None
    tails = {}
    for raw_k in ladder:
        k = int(raw_k)
        tail = 0.0 if k == target64.shape[1] else float(positive[order[k:]].sum())
        tails[str(k)] = tail
        if selected is None and tail <= threshold:
            selected = k
    if selected is None:
        selected = target64.shape[1]
    indices = np.ascontiguousarray(order[:selected], dtype=np.int64)
    return {
        "K_prop": int(selected),
        "indices": indices,
        "order": order,
        "gain": gain,
        "q": q,
        "e": e,
        "reference_mse": reference_mse,
        "threshold": threshold,
        "tail_by_k": tails,
        "indices_sha256": _array_sha(indices, np.int64),
        "order_sha256": _array_sha(order, np.int64),
        "gain_sha256": _array_sha(gain, np.float64),
        "q_sha256": _array_sha(q, np.float64),
        "e_sha256": _array_sha(e, np.float64),
    }


def _audit_basis(root: Path, seal: Mapping) -> None:
    basis32 = np.load(root / seal["files"]["basis_float32"]["path"])
    basis64 = np.load(root / seal["files"]["basis_float64"]["path"])
    mean32 = np.load(root / seal["files"]["mean_float32"]["path"])
    mean64 = np.load(root / seal["files"]["mean_float64"]["path"])
    eigenvalues = np.load(root / seal["files"]["eigenvalues_float64"]["path"])
    gram32 = np.load(root / seal["files"]["gram_float32"]["path"])
    if basis32.shape != (core.FIELD_DIM, core.FIELD_DIM) or basis32.dtype != np.float32:
        raise AuditFailure("deployed basis shape/dtype mismatch")
    if basis64.shape != (core.FIELD_DIM, core.FIELD_DIM) or basis64.dtype != np.float64:
        raise AuditFailure("canonical basis shape/dtype mismatch")
    if mean32.shape != (core.FIELD_DIM,) or mean64.shape != (core.FIELD_DIM,):
        raise AuditFailure("mean shape mismatch")
    if eigenvalues.shape != (core.FIELD_DIM,) or gram32.shape != (core.FIELD_DIM, core.FIELD_DIM):
        raise AuditFailure("basis companion shape mismatch")
    checks = {
        "basis_sha256": _array_sha(basis32, np.float32),
        "basis64_sha256": _array_sha(basis64, np.float64),
        "mean_sha256": _array_sha(mean32, np.float32),
        "mean64_sha256": _array_sha(mean64, np.float64),
        "eigenvalues_sha256": _array_sha(eigenvalues, np.float64),
        "gram_sha256": _array_sha(gram32, np.float32),
    }
    for key, value in checks.items():
        if seal.get(key) != value:
            raise AuditFailure("basis seal mismatch: %s" % key)
    residual64 = float(np.max(np.abs(basis64 @ basis64.T - np.eye(core.FIELD_DIM))))
    if residual64 != float(seal["orthonormality_residual_float64_max_abs"]):
        raise AuditFailure("float64 orthonormality residual mismatch")


def _audit_traces(root: Path, pretest: Mapping) -> None:
    expected = {(arm, seed) for arm in ("dense", "compressed") for seed in core.SEEDS}
    observed = set()
    for relative in pretest["trace_paths"]:
        trace = read_json(root / relative)
        arm = str(trace.get("arm"))
        seed = int(trace.get("seed", -1))
        observed.add((arm, seed))
        replay = replay_stopping(
            trace.get("validation_mse_per_epoch", []),
            patience=core.PATIENCE,
            min_delta=core.MIN_DELTA,
        )
        for key in (
            "selected_epoch",
            "selected_validation_mse",
            "raw_best_epoch",
            "raw_best_validation_mse",
            "stop_epoch",
            "stop_reason",
            "converged",
        ):
            if trace.get(key) != replay[key]:
                raise AuditFailure("trace replay mismatch: %s seed %s %s" % (arm, seed, key))
        if trace.get("stop_reason") != "PATIENCE" or int(trace.get("nonfinite_count", -1)) != 0:
            raise AuditFailure("training gate failed: %s seed %s" % (arm, seed))
        checkpoint = root / trace["checkpoint_relative_path"]
        if sha256_file(checkpoint) != trace.get("checkpoint_file_sha256"):
            raise AuditFailure("checkpoint file mismatch: %s seed %s" % (arm, seed))
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        if state_digest(state) != trace.get("checkpoint_state_sha256"):
            raise AuditFailure("checkpoint state mismatch: %s seed %s" % (arm, seed))
        expected_output = core.FIELD_DIM if arm == "dense" else int(pretest["selection"]["K_prop"])
        expected_parameters = core.mlp_parameter_count(core.FIELD_DIM, core.DENSE_HIDDEN, expected_output)
        if int(trace.get("parameter_count", -1)) != expected_parameters:
            raise AuditFailure("parameter count mismatch: %s seed %s" % (arm, seed))
    if observed != expected:
        raise AuditFailure("trace cell set mismatch")


def _audit_selection(root: Path, pretest: Mapping) -> None:
    material = pretest["selection_material"]
    y = np.load(root / material["target_path"])
    basis = np.load(root / pretest["basis"]["files"]["basis_float64"]["path"])
    mean = np.load(root / pretest["basis"]["files"]["mean_float64"]["path"])
    ladder = [int(value) for value in pretest["selection"]["ladder"]]
    tau = float(pretest["selection"]["tau"])
    recomputed = {}
    for seed in core.SEEDS:
        prediction = np.load(root / material["prediction_paths"][str(seed)])
        recomputed[seed] = _selector(y, prediction, basis, mean, ladder, tau)
    primary = recomputed[0]
    selection = pretest["selection"]
    for key in (
        "K_prop",
        "indices_sha256",
        "order_sha256",
        "gain_sha256",
    ):
        if selection.get(key) != primary[key]:
            raise AuditFailure("selection mismatch: %s" % key)
    stored_indices = np.load(root / material["indices_path"])
    stored_order = np.load(root / material["order_path"])
    stored_gain = np.load(root / material["gain_path"])
    if not np.array_equal(stored_indices, primary["indices"]):
        raise AuditFailure("selected indices bytes mismatch")
    if not np.array_equal(stored_order, primary["order"]):
        raise AuditFailure("selector order bytes mismatch")
    if not np.array_equal(stored_gain, primary["gain"]):
        raise AuditFailure("gain vector bytes mismatch")
    for seed in (1, 2):
        diagnostic = selection["stability_diagnostics"][str(seed)]
        for key in ("K_prop", "indices_sha256", "order_sha256", "gain_sha256"):
            if diagnostic.get(key) != recomputed[seed][key]:
                raise AuditFailure("stability diagnostic mismatch: seed %s %s" % (seed, key))


def audit_pretest(root: Path, pretest_path: Path, run_seal_path: Path) -> dict:
    root = root.resolve()
    pretest = read_json(pretest_path)
    run_seal = read_json(run_seal_path)
    if pretest.get("execution_lineage") != LINEAGE_ID or pretest.get("status") != "PRETEST_SEAL_READY":
        raise AuditFailure("pretest lineage/status mismatch")
    if pretest.get("run_seal_sha256") != sha256_file(run_seal_path):
        raise AuditFailure("run seal binding mismatch")
    current_auditor_sha = sha256_file(Path(__file__).resolve())
    if pretest.get("auditor_source_sha256") != current_auditor_sha:
        raise AuditFailure("auditor source binding mismatch")
    if run_seal.get("status") != "RUN_SEALED" or run_seal.get("execution_lineage") != LINEAGE_ID:
        raise AuditFailure("RUN_SEAL invalid")
    verify_bound_files(root, pretest.get("bound_files", {}))
    ledger = read_json(root / pretest["access_ledger_path"])
    counts = ledger.get("trajectory_counts", {})
    if int(counts.get("test", -1)) != 0 or int(counts.get("unused", -1)) != 0:
        raise AuditFailure("forbidden pretest role access")
    if int(counts.get("selection", -1)) != 200:
        raise AuditFailure("selection access count mismatch")
    _audit_basis(root, pretest["basis"])
    _audit_traces(root, pretest)
    _audit_selection(root, pretest)
    return {
        "execution_lineage": LINEAGE_ID,
        "status": "PASS",
        "audit_phase": "PRETEST",
        "pretest_seal_sha256": sha256_file(pretest_path),
        "run_seal_sha256": sha256_file(run_seal_path),
        "auditor_source_sha256": current_auditor_sha,
        "test_access": 0,
        "unused_access": 0,
        "verified_at_utc": utc_now(),
    }


def _quality_point(raw: Mapping[str, np.ndarray]) -> tuple[float, dict, dict, dict]:
    dense = {}
    compressed = {}
    ratios = {}
    for seed in core.SEEDS:
        dense[seed] = float(np.asarray(raw["dense_seed%s" % seed], dtype=np.float64).mean())
        compressed[seed] = float(np.asarray(raw["compressed_seed%s" % seed], dtype=np.float64).mean())
        ratios[str(seed)] = compressed[seed] / dense[seed]
    return float(np.median(list(ratios.values()))), dense, compressed, ratios


def _bootstrap(raw: Mapping[str, np.ndarray]) -> dict:
    n = len(raw["dense_seed0"])
    rng = np.random.Generator(np.random.PCG64(core.BOOTSTRAP_SEED))
    draws = rng.integers(0, n, size=(core.BOOTSTRAP_REPLICATES, n), dtype=np.int64)
    values = np.empty(core.BOOTSTRAP_REPLICATES, dtype=np.float64)
    for row, indices in enumerate(draws):
        ratios = []
        for seed in core.SEEDS:
            dense = float(raw["dense_seed%s" % seed][indices].mean())
            compressed = float(raw["compressed_seed%s" % seed][indices].mean())
            ratios.append(compressed / dense)
        values[row] = np.median(ratios)
    interval = np.quantile(values, core.BOOTSTRAP_QUANTILES)
    return {
        "point": _quality_point(raw)[0],
        "interval": [float(interval[0]), float(interval[1])],
        "resample_digest": hashlib.sha256(draws.tobytes()).hexdigest(),
    }


def audit_final(root: Path, pretest_audit_path: Path) -> dict:
    root = root.resolve()
    pretest_audit = read_json(pretest_audit_path)
    if pretest_audit.get("status") != "PASS" or int(pretest_audit.get("test_access", -1)) != 0:
        raise AuditFailure("pretest audit is not a PASS")
    marker = root / "test" / "TEST_ACCESS_CONSUMED.json"
    raw_path = root / "test" / "TEST_PER_TRAJECTORY.npz"
    result_path = root / "test" / "TEST_RESULT.json"
    if not marker.is_file() or not raw_path.is_file() or not result_path.is_file():
        raise AuditFailure("terminal TEST artifacts missing")
    with np.load(raw_path) as archive:
        raw = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}
    expected_names = {"%s_seed%s" % (arm, seed) for arm in ("dense", "compressed") for seed in core.SEEDS}
    if set(raw) != expected_names:
        raise AuditFailure("TEST per-trajectory cell set mismatch")
    if any(values.shape != (200,) or not np.all(np.isfinite(values)) for values in raw.values()):
        raise AuditFailure("TEST per-trajectory array invalid")
    q_value, dense, compressed, ratios = _quality_point(raw)
    result = read_json(result_path)
    if result.get("Q_TEST") != q_value or result.get("per_seed_ratios") != ratios:
        raise AuditFailure("TEST point estimate mismatch")
    if result.get("dense_per_seed_mse") != {str(k): v for k, v in dense.items()}:
        raise AuditFailure("dense TEST means mismatch")
    if result.get("compressed_per_seed_mse") != {str(k): v for k, v in compressed.items()}:
        raise AuditFailure("compressed TEST means mismatch")
    expected_verdict = "PASS" if q_value <= core.QUALITY_TOLERANCE else "FAIL"
    if result.get("verdict") != expected_verdict:
        raise AuditFailure("TEST verdict mismatch")
    boot = _bootstrap(raw)
    reported = result.get("bootstrap", {})
    if reported.get("point") != boot["point"] or reported.get("interval") != boot["interval"]:
        raise AuditFailure("bootstrap statistic mismatch")
    ledger = read_json(root / "ACCESS_LEDGER_FINAL.json")
    counts = ledger.get("trajectory_counts", {})
    if int(counts.get("test", -1)) != 200 or int(counts.get("unused", -1)) != 0:
        raise AuditFailure("final role access mismatch")
    return {
        "execution_lineage": LINEAGE_ID,
        "status": "PASS",
        "audit_phase": "FINAL",
        "verdict": expected_verdict,
        "Q_TEST": q_value,
        "test_result_sha256": sha256_file(result_path),
        "test_per_trajectory_sha256": sha256_file(raw_path),
        "test_marker_sha256": sha256_file(marker),
        "pretest_audit_sha256": sha256_file(pretest_audit_path),
        "test_access": 200,
        "unused_access": 0,
        "verified_at_utc": utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)
    pre = sub.add_parser("pretest")
    pre.add_argument("--root", required=True)
    pre.add_argument("--pretest-seal", required=True)
    pre.add_argument("--run-seal", required=True)
    pre.add_argument("--output", required=True)
    final = sub.add_parser("final")
    final.add_argument("--root", required=True)
    final.add_argument("--pretest-audit", required=True)
    final.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.phase == "pretest":
        value = audit_pretest(Path(args.root), Path(args.pretest_seal), Path(args.run_seal))
    else:
        value = audit_final(Path(args.root), Path(args.pretest_audit))
    write_json_create_only(Path(args.output), value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
