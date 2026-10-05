from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Mapping
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest
import torch


HERE = Path(__file__).resolve().parent
ADMISSION_ROOT = HERE.parent
R1 = ADMISSION_ROOT / "production_r1"
R1_RESULTS = R1 / "results_r1"
PACKAGE_CODE = ADMISSION_ROOT / "package" / "code"
sys.path.insert(0, str(PACKAGE_CODE))

import diff_sorp_second_contract_v1 as core  # noqa: E402
import produce_diff_sorp_second_contract_v1 as evaluator  # noqa: E402


LINEAGE_ID = "DIFF_SORP_SECOND_CONTRACT_V1_INFRA_CORRECTION_R2"
R1_LINEAGE_ID = "DIFF_SORP_SECOND_CONTRACT_V1_PRODUCTION_R1"
ADDENDUM = HERE / "DIFF_SORP_SECOND_CONTRACT_V1_INFRA_CORRECTION_R2_ADDENDUM.md"
R1_RUN_SEAL = R1 / "qualification_r1/RUN_SEAL_R1.json"
R1_PRETEST = R1_RESULTS / "PRETEST_SEAL.json"
R1_LEDGER = R1_RESULTS / "ACCESS_LEDGER_PRETEST.json"
R1_FAILURE = R1_RESULTS / "FAILURE_PRETEST.json"
R1_POSTMORTEM = R1 / "postmortem_r1/R1_PRETEST_AUDITOR_INFRA_FAILURE_AUDIT.json"
R1_AUDITOR = R1 / "audit_diff_sorp_production_r1.py"
R1_PRODUCER = R1 / "run_diff_sorp_production_r1.py"
DATA_PATH = ADMISSION_ROOT / "data" / core.DATA_FILENAME
INHERITED_MANIFEST = HERE / "qualification_r2/INHERITED_R1_BYTES.json"


class R2GateFailure(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R2GateFailure("JSON object required: %s" % path)
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


def verify_file_manifest(root: Path, manifest: Mapping[str, Mapping]) -> bool:
    root = root.resolve()
    for relative, expected in sorted(manifest.items()):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise R2GateFailure("inherited path escape: %s" % relative) from exc
        if not path.is_file() or file_record(path) != dict(expected):
            raise R2GateFailure("inherited byte mismatch: %s" % relative)
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
    }


def verify_cpu_launch_environment() -> dict:
    expected = {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    observed = {key: os.environ.get(key) for key in expected}
    if observed != expected:
        if observed.get("CUDA_VISIBLE_DEVICES") != "":
            raise R2GateFailure("CUDA must be hidden for the frozen CPU path")
        raise R2GateFailure("CPU launch thread environment mismatch")
    if torch.cuda.is_available():
        raise R2GateFailure("CUDA remains available despite CPU-only launch")
    return {"variables": observed, "gpu_visible": False, "torch_num_threads": 1}


def build_inherited_manifest() -> dict:
    run_seal = read_json(R1_RUN_SEAL)
    pretest = read_json(R1_PRETEST)
    ledger = read_json(R1_LEDGER)
    failure = read_json(R1_FAILURE)
    postmortem = read_json(R1_POSTMORTEM)
    if run_seal.get("execution_lineage") != R1_LINEAGE_ID:
        raise R2GateFailure("R1 RUN seal lineage mismatch")
    if pretest.get("execution_lineage") != R1_LINEAGE_ID:
        raise R2GateFailure("R1 PRETEST lineage mismatch")
    if postmortem.get("status") != "R1_CLOSED_NO_TEST":
        raise R2GateFailure("R1 is not closed")
    counts = ledger.get("trajectory_counts", {})
    if int(counts.get("test", -1)) != 0 or int(counts.get("unused", -1)) != 0:
        raise R2GateFailure("R1 role firewall mismatch")
    if failure.get("test_marker_exists") is not False:
        raise R2GateFailure("R1 TEST marker state mismatch")
    primary = pretest["selection"]["primary"]
    controls = {
        "production_r1/qualification_r1/RUN_SEAL_R1.json": file_record(R1_RUN_SEAL),
        "production_r1/results_r1/PRETEST_SEAL.json": file_record(R1_PRETEST),
        "production_r1/results_r1/ACCESS_LEDGER_PRETEST.json": file_record(R1_LEDGER),
        "production_r1/results_r1/FAILURE_PRETEST.json": file_record(R1_FAILURE),
        "production_r1/postmortem_r1/R1_PRETEST_AUDITOR_INFRA_FAILURE_AUDIT.json": file_record(R1_POSTMORTEM),
        "production_r1/audit_diff_sorp_production_r1.py": file_record(R1_AUDITOR),
        "production_r1/run_diff_sorp_production_r1.py": file_record(R1_PRODUCER),
        "package/code/produce_diff_sorp_second_contract_v1.py": file_record(Path(evaluator.__file__).resolve()),
        "package/code/diff_sorp_second_contract_v1.py": file_record(Path(core.__file__).resolve()),
    }
    return {
        "schema": "DIFF_SORP_R2_INHERITED_R1_BYTES_V1",
        "r1_status": "R1_CLOSED_NO_TEST",
        "r1_lineage": R1_LINEAGE_ID,
        "r2_lineage": LINEAGE_ID,
        "r1_control_files": controls,
        "r1_pretest_bound_files": pretest["bound_files"],
        "raw_data": dict(run_seal["data"]),
        "source_bindings": {
            "auditor_source_sha256": pretest["auditor_source_sha256"],
            "evaluator_source_sha256": pretest["evaluator_source_sha256"],
            "bootstrap_source_sha256": pretest["bootstrap_source_sha256"],
            "producer_source_sha256": pretest["producer_source_sha256"],
        },
        "scientific_state": {
            "K_prop": primary["K_prop"],
            "selected_indices": primary["indices"],
            "split_digests": pretest["split_digests"],
            "seeds": list(core.SEEDS),
            "quality_tolerance": core.QUALITY_TOLERANCE,
            "bootstrap_replicates": core.BOOTSTRAP_REPLICATES,
            "bootstrap_seed": core.BOOTSTRAP_SEED,
            "test_access": 0,
            "unused_access": 0,
            "trajectory_counts": counts,
        },
        "r1_pretest_seal_sha256": sha256_file(R1_PRETEST),
        "r1_run_seal_sha256": sha256_file(R1_RUN_SEAL),
    }


def verify_inherited_manifest(manifest: Mapping) -> bool:
    if manifest.get("schema") != "DIFF_SORP_R2_INHERITED_R1_BYTES_V1":
        raise R2GateFailure("inherited manifest schema mismatch")
    verify_file_manifest(ADMISSION_ROOT, manifest["r1_control_files"])
    verify_file_manifest(R1_RESULTS, manifest["r1_pretest_bound_files"])
    raw = manifest["raw_data"]
    if DATA_PATH.stat().st_size != int(raw["bytes"]) or sha256_file(DATA_PATH) != raw["sha256"]:
        raise R2GateFailure("raw HDF5 bytes drifted")
    if raw.get("publisher_md5") != core.PUBLISHER_MD5:
        raise R2GateFailure("publisher MD5 binding drifted")
    source = manifest["source_bindings"]
    if source["auditor_source_sha256"] != sha256_file(R1_AUDITOR):
        raise R2GateFailure("R1 auditor source drifted")
    if source["evaluator_source_sha256"] != sha256_file(Path(evaluator.__file__).resolve()):
        raise R2GateFailure("frozen evaluator source drifted")
    if source["bootstrap_source_sha256"] != sha256_file(Path(core.__file__).resolve()):
        raise R2GateFailure("frozen bootstrap source drifted")
    scientific = manifest["scientific_state"]
    if scientific["K_prop"] != 8 or scientific["selected_indices"] != [0, 1, 2, 3, 4, 5, 6, 8]:
        raise R2GateFailure("selector binding drifted")
    if scientific["test_access"] != 0 or scientific["unused_access"] != 0:
        raise R2GateFailure("pre-TEST role binding drifted")
    if (R1_RESULTS / "test/TEST_ACCESS_CONSUMED.json").exists():
        raise R2GateFailure("R1 TEST marker unexpectedly exists")
    if (R1_RESULTS / "test/TEST_RESULT.json").exists():
        raise R2GateFailure("R1 TEST result unexpectedly exists")
    return True


def write_inherited_manifest(output: Path) -> dict:
    value = build_inherited_manifest()
    verify_inherited_manifest(value)
    write_json_create_only(output, value)
    return value


def _verify_junit(path: Path) -> dict:
    root = ET.parse(str(path)).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    record = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if record != {"tests": 45, "failures": 0, "errors": 0, "skipped": 0}:
        raise R2GateFailure("qualification mismatch: %s" % record)
    return record


def create_run_seal(test_xml: Path, output: Path, result_root: Path) -> dict:
    if output.exists() or result_root.exists():
        raise FileExistsError("fresh R2 seal and result namespace required")
    launch = verify_cpu_launch_environment()
    manifest = read_json(INHERITED_MANIFEST)
    verify_inherited_manifest(manifest)
    r1_environment = read_json(R1_RUN_SEAL)["environment"]
    current = _current_environment()
    for key in ("executable", "python", "platform", "machine", "packages"):
        if current[key] != r1_environment[key]:
            raise R2GateFailure("R1 numerical environment drift: %s" % key)
    qualification = _verify_junit(test_xml)
    seal = {
        "schema": "DIFF_SORP_R2_RUN_SEAL_V1",
        "status": "RUN_SEALED_PRETEST_CLOSED",
        "execution_lineage": LINEAGE_ID,
        "correction_class": "POST_SELECTION_PRETEST_AUDITOR_LAUNCH_INFRASTRUCTURE_CORRECTION",
        "addendum": {"path": str(ADDENDUM), **file_record(ADDENDUM)},
        "r2_launcher": file_record(Path(__file__).resolve()),
        "r2_tests": file_record(HERE / "tests/test_infra_correction_r2.py"),
        "qualification": {"junit": file_record(test_xml), **qualification},
        "inherited_manifest": {"path": str(INHERITED_MANIFEST), **file_record(INHERITED_MANIFEST)},
        "r1_run_seal_sha256": manifest["r1_run_seal_sha256"],
        "r1_pretest_seal_sha256": manifest["r1_pretest_seal_sha256"],
        "r1_auditor_source_sha256": manifest["source_bindings"]["auditor_source_sha256"],
        "r1_evaluator_source_sha256": manifest["source_bindings"]["evaluator_source_sha256"],
        "r1_bootstrap_source_sha256": manifest["source_bindings"]["bootstrap_source_sha256"],
        "launch_environment": launch,
        "numerical_environment": current,
        "device_contract": {
            "device": "cpu",
            "dtype": "float32",
            "deterministic_algorithms": True,
            "torch_num_threads": 1,
            "gpu_required": False,
        },
        "result_root": str(result_root.resolve()),
        "new_training": 0,
        "reselection": 0,
        "test_access": 0,
        "unused_access": 0,
        "sealed_at_utc": utc_now(),
    }
    write_json_create_only(output, seal)
    return seal


def verify_run_seal(path: Path, result_root: Path) -> dict:
    seal = read_json(path)
    if seal.get("status") != "RUN_SEALED_PRETEST_CLOSED" or seal.get("execution_lineage") != LINEAGE_ID:
        raise R2GateFailure("R2 RUN seal lineage/status mismatch")
    if seal.get("result_root") != str(result_root.resolve()):
        raise R2GateFailure("R2 result namespace mismatch")
    if seal["addendum"] != {"path": str(ADDENDUM), **file_record(ADDENDUM)}:
        raise R2GateFailure("R2 addendum drift")
    if seal["r2_launcher"] != file_record(Path(__file__).resolve()):
        raise R2GateFailure("R2 launcher drift")
    if seal["r2_tests"] != file_record(HERE / "tests/test_infra_correction_r2.py"):
        raise R2GateFailure("R2 tests drift")
    if seal["inherited_manifest"] != {"path": str(INHERITED_MANIFEST), **file_record(INHERITED_MANIFEST)}:
        raise R2GateFailure("inherited manifest drift")
    verify_cpu_launch_environment()
    verify_inherited_manifest(read_json(INHERITED_MANIFEST))
    return seal


def verify_standalone_pretest_audit(path: Path, run_seal: Mapping) -> bool:
    audit = read_json(path)
    valid = (
        audit.get("status") == "PASS"
        and audit.get("audit_phase") == "PRETEST"
        and audit.get("pretest_seal_sha256") == run_seal["r1_pretest_seal_sha256"]
        and audit.get("run_seal_sha256") == run_seal["r1_run_seal_sha256"]
        and audit.get("auditor_source_sha256") == run_seal["r1_auditor_source_sha256"]
        and int(audit.get("test_access", -1)) == 0
        and int(audit.get("unused_access", -1)) == 0
    )
    if not valid:
        raise R2GateFailure("standalone pre-TEST auditor receipt mismatch")
    return True


def record_pretest_gate(
    run_seal_path: Path,
    pretest_audit_path: Path,
    result_root: Path,
    standalone_job_id: str,
    standalone_command_digest: str,
) -> dict:
    seal = verify_run_seal(run_seal_path, result_root)
    verify_standalone_pretest_audit(pretest_audit_path, seal)
    verify_inherited_manifest(read_json(INHERITED_MANIFEST))
    if (result_root / "test/TEST_ACCESS_CONSUMED.json").exists():
        raise R2GateFailure("TEST opened before R2 pretest gate")
    start = {
        "execution_lineage": LINEAGE_ID,
        "status": "STARTED_PRETEST_CLOSED",
        "run_seal_sha256": sha256_file(run_seal_path),
        "standalone_pretest_auditor_job_id": standalone_job_id,
        "standalone_pretest_auditor_command_digest": standalone_command_digest,
        "test_access": 0,
        "unused_access": 0,
        "recorded_at_utc": utc_now(),
    }
    gate = {
        "execution_lineage": LINEAGE_ID,
        "status": "PASS",
        "audit_launch": "STANDALONE_TOP_LEVEL_PROCESS",
        "pretest_audit_sha256": sha256_file(pretest_audit_path),
        "run_seal_sha256": sha256_file(run_seal_path),
        "inherited_manifest_sha256": sha256_file(INHERITED_MANIFEST),
        "test_access": 0,
        "unused_access": 0,
        "verified_at_utc": utc_now(),
    }
    write_json_create_only(result_root / "START.json", start)
    write_json_create_only(result_root / "PRETEST_AUDIT_GATE.json", gate)
    return gate


def _verify_recorded_pretest_gate(
    result_root: Path, run_seal_path: Path, pretest_audit_path: Path, seal: Mapping
) -> None:
    verify_standalone_pretest_audit(pretest_audit_path, seal)
    gate = read_json(result_root / "PRETEST_AUDIT_GATE.json")
    expected = {
        "pretest_audit_sha256": sha256_file(pretest_audit_path),
        "run_seal_sha256": sha256_file(run_seal_path),
        "inherited_manifest_sha256": sha256_file(INHERITED_MANIFEST),
    }
    if gate.get("status") != "PASS" or any(gate.get(k) != v for k, v in expected.items()):
        raise R2GateFailure("recorded pre-TEST gate drift")


def execute_one_shot_test(
    run_seal_path: Path, pretest_audit_path: Path, result_root: Path
) -> dict:
    seal = verify_run_seal(run_seal_path, result_root)
    _verify_recorded_pretest_gate(result_root, run_seal_path, pretest_audit_path, seal)
    verify_inherited_manifest(read_json(INHERITED_MANIFEST))
    marker = result_root / "test/TEST_ACCESS_CONSUMED.json"
    if marker.exists() or (result_root / "test/TEST_RESULT.json").exists():
        raise R2GateFailure("one-shot TEST was already opened")
    if (result_root / "R2_CLOSED_NO_TEST.json").exists() or (result_root / "FAILURE_AFTER_TEST_ACCESS.json").exists():
        raise R2GateFailure("R2 is terminal and cannot be retried")
    evaluator.configure_cpu_determinism()
    try:
        result = evaluator.test_evaluation(
            R1_PRETEST,
            result_root / "test",
            data_file=DATA_PATH,
            dense_dir=R1_RESULTS / "checkpoints/dense",
            compressed_dir=R1_RESULTS / "checkpoints/compressed",
            basis_dir=R1_RESULTS / "basis",
            selection_dir=R1_RESULTS / "selection",
        )
        ledger = {
            "execution_lineage": LINEAGE_ID,
            "status": "FINAL_ROLE_LEDGER",
            "trajectory_counts": {
                "train": 480,
                "validation": 160,
                "selection": 200,
                "test": 200,
                "unused": 0,
            },
            "inherited_pretest_access": True,
            "new_training": 0,
            "reselection": 0,
            "test_marker_sha256": sha256_file(marker),
            "unused_access": 0,
            "sealed_at_utc": utc_now(),
        }
        write_json_create_only(result_root / "ACCESS_LEDGER_FINAL.json", ledger)
        complete = {
            "execution_lineage": LINEAGE_ID,
            "status": "TEST_EXECUTION_COMPLETE_AUDIT_PENDING",
            "test_result_sha256": sha256_file(result_root / "test/TEST_RESULT.json"),
            "test_per_trajectory_sha256": sha256_file(result_root / "test/TEST_PER_TRAJECTORY.npz"),
            "test_marker_sha256": sha256_file(marker),
            "Q_TEST": result["Q_TEST"],
            "verdict": result["verdict"],
            "test_access": 200,
            "unused_access": 0,
            "completed_at_utc": utc_now(),
        }
        write_json_create_only(result_root / "TEST_EXECUTION_COMPLETE.json", complete)
        return complete
    except BaseException as exc:
        after = marker.exists()
        path = result_root / ("FAILURE_AFTER_TEST_ACCESS.json" if after else "R2_CLOSED_NO_TEST.json")
        if not path.exists():
            write_json_create_only(
                path,
                {
                    "execution_lineage": LINEAGE_ID,
                    "status": "FAILED_TERMINAL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "test_marker_exists": after,
                    "unused_access": 0,
                    "failed_at_utc": utc_now(),
                },
            )
        raise


def finalize_after_standalone_audit(
    run_seal_path: Path,
    pretest_audit_path: Path,
    final_audit_path: Path,
    result_root: Path,
    standalone_job_id: str,
    standalone_command_digest: str,
) -> dict:
    seal = verify_run_seal(run_seal_path, result_root)
    _verify_recorded_pretest_gate(result_root, run_seal_path, pretest_audit_path, seal)
    test_complete = read_json(result_root / "TEST_EXECUTION_COMPLETE.json")
    final = read_json(final_audit_path)
    if final.get("status") != "PASS" or final.get("audit_phase") != "FINAL":
        raise R2GateFailure("standalone final auditor did not PASS")
    if int(final.get("test_access", -1)) != 200 or int(final.get("unused_access", -1)) != 0:
        raise R2GateFailure("standalone final role audit mismatch")
    if final.get("Q_TEST") != test_complete.get("Q_TEST") or final.get("verdict") != test_complete.get("verdict"):
        raise R2GateFailure("standalone final scientific result mismatch")
    raw_path = result_root / "test/TEST_PER_TRAJECTORY.npz"
    result_path = result_root / "test/TEST_RESULT.json"
    evaluator.verify_test_result(read_json(result_path), raw_path)
    complete = {
        "execution_lineage": LINEAGE_ID,
        "status": "SCIENTIFIC_RUN_COMPLETE",
        "run_seal_sha256": sha256_file(run_seal_path),
        "pretest_audit_sha256": sha256_file(pretest_audit_path),
        "final_audit_sha256": sha256_file(final_audit_path),
        "standalone_final_auditor_job_id": standalone_job_id,
        "standalone_final_auditor_command_digest": standalone_command_digest,
        "test_result_sha256": sha256_file(result_path),
        "test_per_trajectory_sha256": sha256_file(raw_path),
        "Q_TEST": final["Q_TEST"],
        "verdict": final["verdict"],
        "test_access": 200,
        "unused_access": 0,
        "new_training": 0,
        "reselection": 0,
        "completed_at_utc": utc_now(),
    }
    write_json_create_only(result_root / "COMPLETE.json", complete)
    return complete


def analyze_result(result_root: Path) -> dict:
    complete = read_json(result_root / "COMPLETE.json")
    result = read_json(result_root / "test/TEST_RESULT.json")
    if complete.get("status") != "SCIENTIFIC_RUN_COMPLETE":
        raise R2GateFailure("analysis requires terminal audited completion")
    q_value = float(result["Q_TEST"])
    interpretation = (
        "CONDITIONALLY_SUPPORTED_ON_PREREGISTERED_DIFFUSION_SORPTION_TARGET"
        if result["verdict"] == "PASS"
        else "REFUTED_ON_PREREGISTERED_DIFFUSION_SORPTION_TARGET"
    )
    value = {
        "schema": "DIFF_SORP_R2_ANALYZE_V1",
        "status": "ANALYSIS_COMPLETE",
        "Q_TEST": q_value,
        "threshold": core.QUALITY_TOLERANCE,
        "verdict": result["verdict"],
        "per_seed_ratios": result["per_seed_ratios"],
        "dense_per_seed_mse": result["dense_per_seed_mse"],
        "compressed_per_seed_mse": result["compressed_per_seed_mse"],
        "bootstrap": result["bootstrap"],
        "bootstrap_replicates": result["bootstrap_replicates"],
        "interpretation": interpretation,
        "claim_boundary": [
            "one preregistered diffusion-sorption target family",
            "evaluated MLP family and frozen training contract",
            "no universal K or intrinsic-rank claim",
            "no cross-backbone or architecture-causality claim",
        ],
        "analyzed_at_utc": utc_now(),
    }
    write_json_create_only(result_root / "ANALYZE_R2.json", value)
    return value


def verify_result(result_root: Path, run_seal_path: Path) -> dict:
    complete = read_json(result_root / "COMPLETE.json")
    analysis = read_json(result_root / "ANALYZE_R2.json")
    final = read_json(result_root / "audits/FINAL_AUDIT.json")
    ledger = read_json(result_root / "ACCESS_LEDGER_FINAL.json")
    verify_run_seal(run_seal_path, result_root)
    evaluator.verify_test_result(
        read_json(result_root / "test/TEST_RESULT.json"),
        result_root / "test/TEST_PER_TRAJECTORY.npz",
    )
    checks = {
        "complete_status": complete.get("status") == "SCIENTIFIC_RUN_COMPLETE",
        "final_auditor_pass": final.get("status") == "PASS",
        "analysis_matches_complete": analysis.get("Q_TEST") == complete.get("Q_TEST"),
        "test_access_200": ledger["trajectory_counts"].get("test") == 200,
        "unused_access_0": ledger["trajectory_counts"].get("unused") == 0,
        "new_training_0": ledger.get("new_training") == 0,
        "reselection_0": ledger.get("reselection") == 0,
        "r1_remains_closed": read_json(R1_POSTMORTEM).get("status") == "R1_CLOSED_NO_TEST",
        "r1_test_marker_absent": not (R1_RESULTS / "test/TEST_ACCESS_CONSUMED.json").exists(),
    }
    if not all(checks.values()):
        raise R2GateFailure("terminal verification failed: %s" % checks)
    value = {
        "schema": "DIFF_SORP_R2_VERIFY_V1",
        "status": "VERIFIED",
        "checks": checks,
        "run_seal_sha256": sha256_file(run_seal_path),
        "test_result_sha256": sha256_file(result_root / "test/TEST_RESULT.json"),
        "final_audit_sha256": sha256_file(result_root / "audits/FINAL_AUDIT.json"),
        "verified_at_utc": utc_now(),
    }
    write_json_create_only(result_root / "VERIFY_R2.json", value)
    return value


def write_review(result_root: Path) -> dict:
    verification = read_json(result_root / "VERIFY_R2.json")
    analysis = read_json(result_root / "ANALYZE_R2.json")
    if verification.get("status") != "VERIFIED":
        raise R2GateFailure("review requires VERIFIED evidence")
    review = f"""# Blind review — DIFF-SORP R2

## Verdict

The preregistered diffusion-sorption TEST verdict is `{analysis['verdict']}` at
`Q_TEST={analysis['Q_TEST']:.10g}` against the frozen `1.05` threshold.

## Strongest supported statement

This result is {analysis['interpretation'].lower().replace('_', ' ')}. It is a
prospective second-target measurement under one frozen MLP/training contract.

## Adversarial checks

- R1 remains a preserved `R1_CLOSED_NO_TEST` infrastructure failure.
- R2 inherited all basis, selector, and checkpoint bytes; training and
  reselection were both zero.
- TEST was opened once, after a standalone pre-TEST auditor PASS.
- The final audit independently replayed the point estimate and 10,000-draw
  paired-trajectory bootstrap.
- UNUSED access remained zero.

## Claims that remain unsupported

The result does not identify an intrinsic rank, a universal or minimum K,
cross-backbone transfer, cross-dataset universality, or an architectural cause.
It does not erase the R1 launch failure. Any manuscript use must preserve these
limits and report the exact target/model/training scope.
"""
    path = result_root / "BLIND_REVIEW_R2.md"
    write_text_create_only(path, review)
    value = {
        "status": "REVIEW_COMPLETE",
        "review_sha256": sha256_file(path),
        "Q_TEST": analysis["Q_TEST"],
        "verdict": analysis["verdict"],
        "reviewed_at_utc": utc_now(),
    }
    write_json_create_only(result_root / "REVIEW_R2.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--output", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--test-xml", required=True)
    seal.add_argument("--output", required=True)
    seal.add_argument("--result-root", required=True)
    pre = sub.add_parser("verify-pretest")
    pre.add_argument("--run-seal", required=True)
    pre.add_argument("--pretest-audit", required=True)
    pre.add_argument("--result-root", required=True)
    pre.add_argument("--standalone-job-id", required=True)
    pre.add_argument("--standalone-command-digest", required=True)
    test = sub.add_parser("test")
    test.add_argument("--run-seal", required=True)
    test.add_argument("--pretest-audit", required=True)
    test.add_argument("--result-root", required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--run-seal", required=True)
    final.add_argument("--pretest-audit", required=True)
    final.add_argument("--final-audit", required=True)
    final.add_argument("--result-root", required=True)
    final.add_argument("--standalone-job-id", required=True)
    final.add_argument("--standalone-command-digest", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--result-root", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--result-root", required=True)
    verify.add_argument("--run-seal", required=True)
    review = sub.add_parser("review")
    review.add_argument("--result-root", required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        value = write_inherited_manifest(Path(args.output))
    elif args.command == "seal":
        value = create_run_seal(Path(args.test_xml), Path(args.output), Path(args.result_root))
    elif args.command == "verify-pretest":
        value = record_pretest_gate(
            Path(args.run_seal), Path(args.pretest_audit), Path(args.result_root),
            args.standalone_job_id, args.standalone_command_digest,
        )
    elif args.command == "test":
        value = execute_one_shot_test(Path(args.run_seal), Path(args.pretest_audit), Path(args.result_root))
    elif args.command == "finalize":
        value = finalize_after_standalone_audit(
            Path(args.run_seal), Path(args.pretest_audit), Path(args.final_audit),
            Path(args.result_root), args.standalone_job_id, args.standalone_command_digest,
        )
    elif args.command == "analyze":
        value = analyze_result(Path(args.result_root))
    elif args.command == "verify":
        value = verify_result(Path(args.result_root), Path(args.run_seal))
    else:
        value = write_review(Path(args.result_root))
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
