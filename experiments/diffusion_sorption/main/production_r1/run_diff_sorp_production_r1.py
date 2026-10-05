from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest
import torch


HERE = Path(__file__).resolve().parent
ADMISSION_ROOT = HERE.parent
PACKAGE_ROOT = ADMISSION_ROOT / "package"
PACKAGE_CODE = PACKAGE_ROOT / "code"
sys.path.insert(0, str(PACKAGE_CODE))

import diff_sorp_second_contract_v1 as core  # noqa: E402
import produce_diff_sorp_second_contract_v1 as primitives  # noqa: E402
import run_diff_sorp_second_contract_v1 as admission  # noqa: E402


LINEAGE_ID = "DIFF_SORP_SECOND_CONTRACT_V1_PRODUCTION_R1"
EXPECTED_PACKAGE_SHA256 = "42e23784d9b973907f3ae05c2fedfe70868b644c44f31e1b86a136e7b5851177"
EXPECTED_DATA_SHA256 = "8e48ab3efd39ab63524d92e85a4e0db46347b72e7c5b05edf81e1c9637a3ab2d"
AUDITOR_PATH = HERE / "audit_diff_sorp_production_r1.py"
DATA_PATH = ADMISSION_ROOT / "data" / core.DATA_FILENAME
RECEIPTS = ADMISSION_ROOT / "admission_receipts"


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


def file_record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("JSON object required: %s" % path)
    return value


def write_json_create_only(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text_create_only(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def save_npy_create_only(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)


def _array_sha(values, dtype) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def _selector_one(y, prediction, basis, mean, ladder, tau) -> dict:
    y64 = np.asarray(y, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    basis64 = np.asarray(basis, dtype=np.float64)
    mean64 = np.asarray(mean, dtype=np.float64).reshape(-1)
    coefficients = (y64 - mean64) @ basis64.T
    predicted_coefficients = (prediction64 - mean64) @ basis64.T
    q = np.mean(coefficients * coefficients, axis=0)
    e = np.mean((coefficients - predicted_coefficients) ** 2, axis=0)
    gain = q - e
    positive = np.clip(gain, 0.0, None)
    order = np.argsort(-positive, kind="stable").astype(np.int64)
    reference_mse = float(np.mean((prediction64 - y64) ** 2))
    threshold = float(tau) * y64.shape[1] * reference_mse
    selected = None
    tails = {}
    for raw_k in ladder:
        k = int(raw_k)
        tail = 0.0 if k == y64.shape[1] else float(positive[order[k:]].sum())
        tails[str(k)] = tail
        if selected is None and tail <= threshold:
            selected = k
    if selected is None:
        selected = y64.shape[1]
    indices = np.ascontiguousarray(order[:selected], dtype=np.int64)
    return {
        "K_prop": int(selected),
        "indices": indices,
        "order": order,
        "gain": gain,
        "q": q,
        "e": e,
        "indices_sha256": _array_sha(indices, np.int64),
        "order_sha256": _array_sha(order, np.int64),
        "gain_sha256": _array_sha(gain, np.float64),
        "q_sha256": _array_sha(q, np.float64),
        "e_sha256": _array_sha(e, np.float64),
        "reference_mse": reference_mse,
        "threshold": threshold,
        "tail_by_k": tails,
        "positive_gain_directions": int((gain > 0).sum()),
    }


def selection_material(y, predictions_by_seed, basis, mean, ladder=core.LADDER, tau=core.PRIMARY_TAU) -> dict:
    if sorted(predictions_by_seed) != list(core.SEEDS):
        raise RuntimeError("dense selection seed set changed")
    frozen_record = primitives.build_selection_record(
        y, predictions_by_seed, basis, mean, ladder=ladder, tau=tau
    )
    computed = {
        seed: _selector_one(y, predictions_by_seed[seed], basis, mean, ladder, tau)
        for seed in core.SEEDS
    }
    primary = computed[0]
    for key in ("K_prop", "indices_sha256", "order_sha256", "gain_sha256"):
        if frozen_record.get(key) != primary[key]:
            raise RuntimeError("selector implementation disagreement: %s" % key)
    for seed in (1, 2):
        for key in ("K_prop", "indices_sha256", "order_sha256", "gain_sha256"):
            if frozen_record["stability_diagnostics"][str(seed)].get(key) != computed[seed][key]:
                raise RuntimeError("selector diagnostic disagreement: seed %s %s" % (seed, key))
    return {
        "record": frozen_record,
        "indices": primary["indices"],
        "order": primary["order"],
        "gain": primary["gain"],
        "q": primary["q"],
        "e": primary["e"],
        "computed": computed,
    }


def assert_test_gate(audit_path: Path, pretest_sha256: str, auditor_sha256: str) -> bool:
    audit = read_json(audit_path)
    valid = (
        audit.get("status") == "PASS"
        and audit.get("audit_phase") in (None, "PRETEST")
        and audit.get("pretest_seal_sha256") == pretest_sha256
        and audit.get("auditor_source_sha256") == auditor_sha256
        and int(audit.get("test_access", -1)) == 0
        and int(audit.get("unused_access", -1)) == 0
    )
    if not valid:
        raise RuntimeError("independent pre-TEST audit gate failed")
    return True


def _current_environment() -> dict:
    return {
        "executable": str(Path(sys.executable).resolve()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            "h5py": h5py.__version__,
            "numpy": np.__version__,
            "pytest": pytest.__version__,
            "torch": torch.__version__,
        },
        "torch_num_threads_before_binding": int(torch.get_num_threads()),
    }


def _source_paths() -> Sequence[Path]:
    paths = []
    for folder in (PACKAGE_ROOT / "protocols", PACKAGE_ROOT / "code", PACKAGE_ROOT / "tests", HERE / "tests"):
        paths.extend(path for path in folder.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    paths.extend([Path(__file__).resolve(), AUDITOR_PATH.resolve()])
    return sorted(set(path.resolve() for path in paths))


def _relative_to_admission(path: Path) -> str:
    return str(path.resolve().relative_to(ADMISSION_ROOT.resolve()))


def _verify_junit(path: Path) -> dict:
    root = ET.parse(str(path)).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    if tests < 38 or failures or errors or skipped:
        raise RuntimeError(
            "qualification tests incomplete: tests=%s failures=%s errors=%s skipped=%s"
            % (tests, failures, errors, skipped)
        )
    return {"tests": tests, "failures": failures, "errors": errors, "skipped": skipped}


def _verify_admission() -> dict:
    stop = read_json(RECEIPTS / "MANDATORY_STOP_AFTER_SCHEMA.json")
    raw = read_json(RECEIPTS / "RAW_BYTE_SEAL.json")
    schema = read_json(RECEIPTS / "SCHEMA_SEAL.json")
    independent = read_json(RECEIPTS / "SCHEMA_SEAL_INDEPENDENT_VERIFICATION.json")
    if stop.get("status") != "ADMISSION_COMPLETE_MANDATORY_STOP":
        raise RuntimeError("schema admission stop receipt invalid")
    if any(stop.get(key) != "NO" for key in ("TRAIN_STARTED", "SELECTION_OPENED", "TEST_OPENED", "UNUSED_OPENED")):
        raise RuntimeError("schema admission role history invalid")
    if raw.get("data_sha256") != EXPECTED_DATA_SHA256 or raw.get("publisher_md5") != core.PUBLISHER_MD5:
        raise RuntimeError("raw byte seal invalid")
    if schema.get("status") != "SCHEMA_ADMITTED" or independent.get("status") != "PASS":
        raise RuntimeError("schema seal invalid")
    admission._validate_raw_seal(raw)
    core.validate_schema_record(schema["schema"])
    return {
        name: file_record(RECEIPTS / name)
        for name in (
            "ADMISSION_MANIFEST.json",
            "MANDATORY_STOP_AFTER_SCHEMA.json",
            "RAW_BYTE_SEAL.json",
            "SCHEMA_SEAL.json",
            "SCHEMA_SEAL_INDEPENDENT_VERIFICATION.json",
            "ZERO_DATA_TEST_RECEIPT.json",
            "ENVIRONMENT_RECEIPT.json",
            "HOST_RECEIPT.json",
            "PACKAGE_SHA_RECEIPT.json",
        )
    }


def create_run_seal(test_xml: Path, output: Path, result_root: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if result_root.exists():
        raise FileExistsError("fresh result namespace required: %s" % result_root)
    qualification = _verify_junit(test_xml)
    admission_receipts = _verify_admission()
    package_zip = ADMISSION_ROOT / "incoming" / "DIFF_SORP_SECOND_CONTRACT_V1_PREPRODUCTION_V3_20260815.zip"
    if sha256_file(package_zip) != EXPECTED_PACKAGE_SHA256:
        raise RuntimeError("preproduction package digest mismatch")
    data_sha = sha256_file(DATA_PATH)
    if data_sha != EXPECTED_DATA_SHA256:
        raise RuntimeError("data bytes drifted after admission")
    frozen_environment = read_json(RECEIPTS / "ENVIRONMENT_RECEIPT.json")
    current = _current_environment()
    if current["executable"] != str(Path(frozen_environment["executable"]).resolve()):
        raise RuntimeError("Python executable differs from admission environment")
    if current["packages"] != frozen_environment["packages"]:
        raise RuntimeError("package versions differ from admission environment")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("production R1 is bound to the admitted Mac arm64 numerical path")
    sources = {_relative_to_admission(path): file_record(path) for path in _source_paths()}
    seal = {
        "execution_lineage": LINEAGE_ID,
        "status": "RUN_SEALED",
        "protocol_id": core.PROTOCOL_ID,
        "protocol_sha256": core.PROTOCOL_SHA256,
        "addendum_sha256": core.ADDENDUM_SHA256,
        "addendum_v1_2_sha256": core.ADDENDUM_V1_2_SHA256,
        "addendum_v1_3_sha256": core.ADDENDUM_V1_3_SHA256,
        "addendum_v1_4_sha256": core.ADDENDUM_V1_4_SHA256,
        "preproduction_package": file_record(package_zip),
        "data": {"path": str(DATA_PATH.resolve()), **file_record(DATA_PATH), "publisher_md5": core.PUBLISHER_MD5},
        "admission_receipts": admission_receipts,
        "source_files": sources,
        "qualification": {"junit": file_record(test_xml), **qualification},
        "environment": current,
        "device_contract": {
            "device": "cpu",
            "dtype": "float32",
            "torch_num_threads": 1,
            "deterministic_algorithms": True,
            "gpu_required": False,
        },
        "result_root": str(result_root.resolve()),
        "test_access": 0,
        "unused_access": 0,
        "sealed_at_utc": utc_now(),
    }
    write_json_create_only(output, seal)
    return seal


def verify_run_seal(path: Path, expected_result_root: Path) -> dict:
    seal = read_json(path)
    if seal.get("status") != "RUN_SEALED" or seal.get("execution_lineage") != LINEAGE_ID:
        raise RuntimeError("RUN_SEAL lineage/status invalid")
    if seal.get("result_root") != str(expected_result_root.resolve()):
        raise RuntimeError("RUN_SEAL result namespace mismatch")
    if expected_result_root.exists():
        raise FileExistsError("fresh result namespace required")
    if sha256_file(DATA_PATH) != seal["data"]["sha256"]:
        raise RuntimeError("RUN_SEAL data binding mismatch")
    for relative, expected in seal["source_files"].items():
        path_item = ADMISSION_ROOT / relative
        if file_record(path_item) != expected:
            raise RuntimeError("RUN_SEAL source drift: %s" % relative)
    current = _current_environment()
    for key in ("executable", "python", "platform", "machine", "packages"):
        if current[key] != seal["environment"][key]:
            raise RuntimeError("RUN_SEAL environment drift: %s" % key)
    return seal


class StageEvents:
    def __init__(self, root: Path):
        self.root = root / "events"
        self.count = 0

    def add(self, stage: str, status: str, **details) -> None:
        self.count += 1
        value = {"index": self.count, "stage": stage, "status": status, "at_utc": utc_now()}
        value.update(details)
        write_json_create_only(self.root / ("%03d_%s.json" % (self.count, stage.lower())), value)


def _load_model(path: Path, output_dim: int) -> core.MLP:
    model = core.MLP(core.FIELD_DIM, int(output_dim), hidden=core.DENSE_HIDDEN)
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _trace_path(root: Path, arm: str, seed: int) -> Path:
    return root / "traces" / ("%s_seed%s.json" % (arm, seed))


def _train_arm(
    root: Path,
    arm: str,
    x_train,
    target_train,
    x_validation,
    y_validation,
    output_dim: int,
    basis_rows,
    mean,
) -> list:
    traces = []
    checkpoint_dir = root / "checkpoints" / arm
    for seed in core.SEEDS:
        trace = primitives.train_cell(
            arm,
            seed,
            x_train,
            target_train,
            x_validation,
            y_validation,
            checkpoint_dir,
            hidden=core.DENSE_HIDDEN,
            dout=output_dim,
            basis_rows=basis_rows,
            mean=mean,
        )
        trace["checkpoint_relative_path"] = str(
            (checkpoint_dir / trace["checkpoint_path"]).relative_to(root)
        )
        write_json_create_only(_trace_path(root, arm, seed), trace)
        traces.append(trace)
        if not trace["converged"] or trace["stop_reason"] != "PATIENCE" or trace["nonfinite_count"] != 0:
            raise core.TrainingGateFail("%s seed %s failed convergence gate" % (arm, seed))
    return traces


def _basis_artifacts(root: Path, basis: Mapping) -> dict:
    folder = root / "basis"
    arrays = {
        "mean_float64": ("TRAIN_MEAN_FLOAT64.npy", basis["mean_float64"]),
        "mean_float32": ("TRAIN_MEAN_FLOAT32.npy", basis["mean_float32"]),
        "basis_float64": ("BASIS_FLOAT64.npy", basis["basis_float64"]),
        "basis_float32": ("BASIS_FLOAT32.npy", basis["basis_float32"]),
        "eigenvalues_float64": ("EIGENVALUES_FLOAT64.npy", basis["eigenvalues_float64"]),
        "gram_float32": ("GRAM_FLOAT32.npy", basis["gram_float32"]),
    }
    files = {}
    for key, (name, value) in arrays.items():
        path = folder / name
        save_npy_create_only(path, value)
        files[key] = {"path": str(path.relative_to(root)), **file_record(path)}
    seal = {
        "status": "BASIS_SEALED",
        "protocol_sha256": core.PROTOCOL_SHA256,
        "addendum_v1_4_sha256": core.ADDENDUM_V1_4_SHA256,
        "selector_basis": "float64_canonical",
        "deployment_basis": "float32_realised_gram",
        "files": files,
    }
    for key in (
        "mean_sha256",
        "basis_sha256",
        "mean64_sha256",
        "basis64_sha256",
        "eigenvalues_sha256",
        "gram_sha256",
        "orthonormality_residual_max_abs",
        "orthonormality_residual_float64_max_abs",
        "k_energy",
        "covariance_denominator",
        "field_dim",
    ):
        seal[key] = basis[key]
    write_json_create_only(folder / "BASIS_SEAL.json", seal)
    return seal


def _selection_artifacts(root: Path, y, predictions, material: Mapping) -> tuple[dict, dict]:
    folder = root / "selection"
    paths = {
        "target_path": folder / "SELECTION_TARGETS_FLOAT32.npy",
        "indices_path": folder / "SELECTED_INDICES_INT64.npy",
        "order_path": folder / "ORDER_INT64.npy",
        "gain_path": folder / "GAIN_FLOAT64.npy",
        "q_path": folder / "Q_FLOAT64.npy",
        "e_path": folder / "E_FLOAT64.npy",
    }
    save_npy_create_only(paths["target_path"], np.asarray(y, dtype=np.float32))
    save_npy_create_only(paths["indices_path"], material["indices"])
    save_npy_create_only(paths["order_path"], material["order"])
    save_npy_create_only(paths["gain_path"], material["gain"])
    save_npy_create_only(paths["q_path"], material["q"])
    save_npy_create_only(paths["e_path"], material["e"])
    prediction_paths = {}
    for seed in core.SEEDS:
        path = folder / ("DENSE_SEED%s_PREDICTIONS_FLOAT32.npy" % seed)
        save_npy_create_only(path, np.asarray(predictions[seed], dtype=np.float32))
        prediction_paths[str(seed)] = str(path.relative_to(root))
    record = dict(material["record"])
    record["materialized_files"] = {
        key: {"path": str(path.relative_to(root)), **file_record(path)} for key, path in paths.items()
    }
    record["prediction_files"] = {
        seed: {"path": prediction_paths[seed], **file_record(root / prediction_paths[seed])}
        for seed in prediction_paths
    }
    write_json_create_only(folder / "SELECTION_SEAL.json", record)
    locator = {
        "target_path": str(paths["target_path"].relative_to(root)),
        "indices_path": str(paths["indices_path"].relative_to(root)),
        "order_path": str(paths["order_path"].relative_to(root)),
        "gain_path": str(paths["gain_path"].relative_to(root)),
        "q_path": str(paths["q_path"].relative_to(root)),
        "e_path": str(paths["e_path"].relative_to(root)),
        "prediction_paths": prediction_paths,
    }
    return record, locator


def _bound_files(root: Path, relative_paths: Sequence[str]) -> dict:
    return {relative: file_record(root / relative) for relative in sorted(set(relative_paths))}


def _run_auditor(arguments: Sequence[str], stdout_path: Path, stderr_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(AUDITOR_PATH), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )
    write_text_create_only(stdout_path, completed.stdout)
    write_text_create_only(stderr_path, completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError("independent auditor failed with return code %s" % completed.returncode)


def execute(run_seal_path: Path, result_root: Path) -> dict:
    run_seal = verify_run_seal(run_seal_path, result_root)
    result_root.mkdir(parents=True, exist_ok=False)
    events = StageEvents(result_root)
    stage = "START"
    start = {
        "execution_lineage": LINEAGE_ID,
        "status": "STARTED",
        "run_seal_path": str(run_seal_path.resolve()),
        "run_seal_sha256": sha256_file(run_seal_path),
        "producer_source_sha256": sha256_file(Path(__file__).resolve()),
        "auditor_source_sha256": sha256_file(AUDITOR_PATH),
        "started_at_utc": utc_now(),
        "test_access": 0,
        "unused_access": 0,
    }
    write_json_create_only(result_root / "START.json", start)
    events.add(stage, "PASS")
    trajectory_counts = {role: 0 for role in ("train", "validation", "selection", "test", "unused")}
    access_events = []
    try:
        primitives.configure_cpu_determinism()
        stage = "BASIS"
        x_train, y_train, _ = admission.load_role_pairs(DATA_PATH, "train", stage="BASIS_TRAIN")
        trajectory_counts["train"] = 480
        access_events.append({"role": "train", "stage": "BASIS_TRAIN", "trajectories": 480})
        basis = core.fit_train_basis(y_train)
        basis_seal = _basis_artifacts(result_root, basis)
        events.add(stage, "PASS")

        stage = "DENSE"
        x_validation, y_validation, _ = admission.load_role_pairs(
            DATA_PATH, "validation", stage="DENSE_TRAIN"
        )
        trajectory_counts["validation"] = 160
        access_events.append({"role": "validation", "stage": "DENSE_TRAIN", "trajectories": 160})
        dense_traces = _train_arm(
            result_root,
            "dense",
            x_train,
            y_train,
            x_validation,
            y_validation,
            core.FIELD_DIM,
            None,
            None,
        )
        events.add(stage, "PASS", cells=3)

        stage = "SELECTION"
        x_selection, y_selection, _ = admission.load_role_pairs(DATA_PATH, "selection", stage="SELECT")
        trajectory_counts["selection"] = 200
        access_events.append({"role": "selection", "stage": "SELECT", "trajectories": 200})
        predictions = {}
        for seed in core.SEEDS:
            model = _load_model(result_root / "checkpoints" / "dense" / ("dense_seed%s.pt" % seed), core.FIELD_DIM)
            predictions[seed] = np.ascontiguousarray(
                primitives._physical_prediction(model, x_selection), dtype=np.float32
            )
        material = selection_material(
            y_selection,
            predictions,
            basis["basis_float64"],
            basis["mean_float64"],
        )
        selection_record, selection_locator = _selection_artifacts(
            result_root, y_selection, predictions, material
        )
        events.add(stage, "PASS")

        stage = "COMPRESSED"
        indices = material["indices"]
        basis_rows = np.ascontiguousarray(basis["basis_float32"][indices], dtype=np.float32)
        centred_train = np.ascontiguousarray(
            np.asarray(y_train, dtype=np.float32) - basis["mean_float32"], dtype=np.float32
        )
        coefficient_targets = np.ascontiguousarray(centred_train @ basis_rows.T, dtype=np.float32)
        compressed_traces = _train_arm(
            result_root,
            "compressed",
            x_train,
            coefficient_targets,
            x_validation,
            y_validation,
            int(selection_record["K_prop"]),
            basis_rows,
            basis["mean_float32"],
        )
        events.add(stage, "PASS", cells=3)

        stage = "PRETEST_SEAL"
        access_ledger = {
            "execution_lineage": LINEAGE_ID,
            "status": "PRETEST_ROLE_LEDGER",
            "trajectory_counts": trajectory_counts,
            "events": access_events,
            "test_access": 0,
            "unused_access": 0,
            "sealed_at_utc": utc_now(),
        }
        ledger_path = result_root / "ACCESS_LEDGER_PRETEST.json"
        write_json_create_only(ledger_path, access_ledger)
        traces = dense_traces + compressed_traces
        base = core.build_pretest_seal(
            traces,
            selection=selection_record,
            data_sha256=run_seal["data"]["sha256"],
            basis_sha256=basis["basis_sha256"],
            mean_sha256=basis["mean_sha256"],
            basis64_sha256=basis["basis64_sha256"],
            mean64_sha256=basis["mean64_sha256"],
            gram_sha256=basis["gram_sha256"],
            evaluator_sha256=sha256_file(PACKAGE_CODE / "produce_diff_sorp_second_contract_v1.py"),
            bootstrap_sha256=sha256_file(PACKAGE_CODE / "diff_sorp_second_contract_v1.py"),
        )
        trace_paths = [str(_trace_path(result_root, arm, seed).relative_to(result_root))
                       for arm in ("dense", "compressed") for seed in core.SEEDS]
        checkpoint_paths = [trace["checkpoint_relative_path"] for trace in traces]
        basis_paths = [item["path"] for item in basis_seal["files"].values()] + ["basis/BASIS_SEAL.json"]
        selection_paths = [
            selection_locator["target_path"], selection_locator["indices_path"],
            selection_locator["order_path"], selection_locator["gain_path"],
            selection_locator["q_path"], selection_locator["e_path"],
            *selection_locator["prediction_paths"].values(), "selection/SELECTION_SEAL.json",
        ]
        base.update(
            {
                "execution_lineage": LINEAGE_ID,
                "run_seal_sha256": sha256_file(run_seal_path),
                "producer_source_sha256": sha256_file(Path(__file__).resolve()),
                "auditor_source_sha256": sha256_file(AUDITOR_PATH),
                "data_bytes": run_seal["data"]["bytes"],
                "schema_seal_sha256": sha256_file(RECEIPTS / "SCHEMA_SEAL.json"),
                "admission_receipts": run_seal["admission_receipts"],
                "basis": basis_seal,
                "selection": selection_record,
                "selection_material": selection_locator,
                "trace_paths": trace_paths,
                "access_ledger_path": str(ledger_path.relative_to(result_root)),
                "bound_files": _bound_files(
                    result_root,
                    [*trace_paths, *checkpoint_paths, *basis_paths, *selection_paths,
                     str(ledger_path.relative_to(result_root))],
                ),
                "sealed_at_utc": utc_now(),
            }
        )
        pretest_path = result_root / "PRETEST_SEAL.json"
        write_json_create_only(pretest_path, base)
        events.add(stage, "PASS")

        stage = "PRETEST_AUDIT"
        pretest_audit_path = result_root / "audits" / "PRETEST_AUDIT.json"
        _run_auditor(
            [
                "pretest", "--root", str(result_root), "--pretest-seal", str(pretest_path),
                "--run-seal", str(run_seal_path), "--output", str(pretest_audit_path),
            ],
            result_root / "audits" / "PRETEST_AUDITOR_STDOUT.log",
            result_root / "audits" / "PRETEST_AUDITOR_STDERR.log",
        )
        assert_test_gate(pretest_audit_path, sha256_file(pretest_path), sha256_file(AUDITOR_PATH))
        for relative, expected in base["bound_files"].items():
            if file_record(result_root / relative) != expected:
                raise RuntimeError("pre-TEST TOCTOU binding mismatch: %s" % relative)
        events.add(stage, "PASS")

        stage = "ONE_SHOT_TEST"
        test_result = primitives.test_evaluation(
            pretest_path,
            result_root / "test",
            data_file=DATA_PATH,
            dense_dir=result_root / "checkpoints" / "dense",
            compressed_dir=result_root / "checkpoints" / "compressed",
            basis_dir=result_root / "basis",
            selection_dir=result_root / "selection",
        )
        trajectory_counts["test"] = 200
        final_ledger = {
            "execution_lineage": LINEAGE_ID,
            "status": "FINAL_ROLE_LEDGER",
            "trajectory_counts": trajectory_counts,
            "events": access_events + [{"role": "test", "stage": "TEST_EVAL", "trajectories": 200}],
            "test_marker_sha256": sha256_file(result_root / "test" / "TEST_ACCESS_CONSUMED.json"),
            "unused_access": 0,
            "sealed_at_utc": utc_now(),
        }
        write_json_create_only(result_root / "ACCESS_LEDGER_FINAL.json", final_ledger)
        events.add(stage, "PASS")

        stage = "FINAL_AUDIT"
        final_audit_path = result_root / "audits" / "FINAL_AUDIT.json"
        _run_auditor(
            ["final", "--root", str(result_root), "--pretest-audit", str(pretest_audit_path),
             "--output", str(final_audit_path)],
            result_root / "audits" / "FINAL_AUDITOR_STDOUT.log",
            result_root / "audits" / "FINAL_AUDITOR_STDERR.log",
        )
        final_audit = read_json(final_audit_path)
        if final_audit.get("status") != "PASS":
            raise RuntimeError("final independent audit failed")
        events.add(stage, "PASS")
        complete = {
            "execution_lineage": LINEAGE_ID,
            "status": "SCIENTIFIC_RUN_COMPLETE",
            "run_seal_sha256": sha256_file(run_seal_path),
            "pretest_audit_sha256": sha256_file(pretest_audit_path),
            "final_audit_sha256": sha256_file(final_audit_path),
            "test_result_sha256": sha256_file(result_root / "test" / "TEST_RESULT.json"),
            "test_verdict": test_result["verdict"],
            "test_access": 200,
            "unused_access": 0,
            "completed_at_utc": utc_now(),
        }
        write_json_create_only(result_root / "COMPLETE.json", complete)
        return complete
    except BaseException as exc:
        marker_exists = (result_root / "test" / "TEST_ACCESS_CONSUMED.json").exists()
        failure_path = result_root / (
            "FAILURE_AFTER_TEST_ACCESS.json" if marker_exists else "FAILURE_PRETEST.json"
        )
        if not failure_path.exists():
            write_json_create_only(
                failure_path,
                {
                    "execution_lineage": LINEAGE_ID,
                    "status": "FAILED",
                    "stage": stage,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "test_marker_exists": marker_exists,
                    "unused_access": 0,
                    "failed_at_utc": utc_now(),
                },
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--test-xml", required=True)
    seal.add_argument("--output", required=True)
    seal.add_argument("--result-root", required=True)
    run = sub.add_parser("run")
    run.add_argument("--run-seal", required=True)
    run.add_argument("--result-root", required=True)
    args = parser.parse_args()
    if args.command == "seal":
        value = create_run_seal(Path(args.test_xml), Path(args.output), Path(args.result_root))
    else:
        value = execute(Path(args.run_seal), Path(args.result_root))
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
