from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ADMISSION_ROOT = ROOT.parent
R1 = ADMISSION_ROOT / "production_r1"
sys.path.insert(0, str(ROOT))

import run_diff_sorp_infra_correction_r2 as r2  # noqa: E402


def _record(path: Path) -> dict:
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_r2_source_has_no_child_auditor_launch() -> None:
    tree = ast.parse((ROOT / "run_diff_sorp_infra_correction_r2.py").read_text())
    imports = {
        node.names[0].name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and node.names
    }
    assert "subprocess" not in imports
    assert "_run_auditor" not in (ROOT / "run_diff_sorp_infra_correction_r2.py").read_text()


def test_inherited_manifest_exactly_covers_r1_pretest_bound_files() -> None:
    pretest_path = R1 / "results_r1/PRETEST_SEAL.json"
    if not pretest_path.is_file():
        pytest.skip("archived R1 terminal state is intentionally not bundled in the public package")
    pretest = json.loads(pretest_path.read_text())
    manifest = r2.build_inherited_manifest()
    assert manifest["r1_pretest_bound_files"] == pretest["bound_files"]
    assert manifest["scientific_state"]["K_prop"] == 8
    assert manifest["scientific_state"]["selected_indices"] == [0, 1, 2, 3, 4, 5, 6, 8]
    assert manifest["scientific_state"]["test_access"] == 0
    assert manifest["scientific_state"]["unused_access"] == 0


def test_inherited_verifier_rejects_one_byte_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"frozen")
    manifest = {"artifact.bin": _record(artifact)}
    assert r2.verify_file_manifest(tmp_path, manifest)
    artifact.write_bytes(b"drifted")
    with pytest.raises(r2.R2GateFailure, match="artifact.bin"):
        r2.verify_file_manifest(tmp_path, manifest)


def test_cpu_launch_contract_requires_hidden_cuda_and_one_thread(monkeypatch) -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert r2.verify_cpu_launch_environment()["gpu_visible"] is False
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(r2.R2GateFailure, match="CUDA"):
        r2.verify_cpu_launch_environment()


def test_pretest_gate_requires_exact_standalone_auditor_receipt(tmp_path: Path) -> None:
    run_seal = {
        "r1_pretest_seal_sha256": "a" * 64,
        "r1_auditor_source_sha256": "b" * 64,
        "r1_run_seal_sha256": "c" * 64,
    }
    receipt = tmp_path / "PRETEST_AUDIT.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "PASS",
                "audit_phase": "PRETEST",
                "pretest_seal_sha256": "a" * 64,
                "auditor_source_sha256": "b" * 64,
                "run_seal_sha256": "c" * 64,
                "test_access": 0,
                "unused_access": 0,
            }
        )
    )
    assert r2.verify_standalone_pretest_audit(receipt, run_seal)
    bad = tmp_path / "BAD.json"
    bad.write_text(receipt.read_text().replace('"PASS"', '"FAIL"'))
    with pytest.raises(r2.R2GateFailure, match="standalone"):
        r2.verify_standalone_pretest_audit(bad, run_seal)


def test_test_command_fails_closed_before_marker_without_audit(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, r2.R2GateFailure)):
        r2.execute_one_shot_test(
            run_seal_path=tmp_path / "RUN_SEAL_R2.json",
            pretest_audit_path=tmp_path / "PRETEST_AUDIT.json",
            result_root=tmp_path / "results_r2",
        )
    assert not (tmp_path / "results_r2/test/TEST_ACCESS_CONSUMED.json").exists()


def test_r1_closed_artifacts_remain_present_and_test_unopened() -> None:
    if not (R1 / "results_r1/FAILURE_PRETEST.json").is_file():
        pytest.skip("archived R1 terminal state is intentionally not bundled in the public package")
    assert (R1 / "results_r1/FAILURE_PRETEST.json").is_file()
    assert (R1 / "postmortem_r1/R1_PRETEST_AUDITOR_INFRA_FAILURE_AUDIT.json").is_file()
    assert not (R1 / "results_r1/test/TEST_ACCESS_CONSUMED.json").exists()
    assert not (R1 / "results_r1/test/TEST_RESULT.json").exists()
