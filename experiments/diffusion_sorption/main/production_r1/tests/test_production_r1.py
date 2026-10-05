from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audit_diff_sorp_production_r1 as auditor  # noqa: E402
import run_diff_sorp_production_r1 as producer  # noqa: E402


def test_json_artifacts_are_create_only(tmp_path: Path) -> None:
    path = tmp_path / "seal.json"
    producer.write_json_create_only(path, {"status": "PASS"})
    with pytest.raises(FileExistsError):
        producer.write_json_create_only(path, {"status": "MUTATED"})


def test_selection_material_is_seed0_only_and_byte_bound() -> None:
    rng = np.random.default_rng(8)
    q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    basis = q.T.astype(np.float64)
    mean = np.zeros(7, dtype=np.float64)
    y = rng.normal(size=(24, 7)).astype(np.float32)
    pred0 = y.copy()
    pred0[:, 4:] += 0.8
    pred1 = y.copy()
    pred1[:, :2] += 4.0
    pred2 = y.copy()
    pred2[:, 2:5] += 2.0

    material = producer.selection_material(
        y,
        {0: pred0, 1: pred1, 2: pred2},
        basis,
        mean,
        ladder=(1, 2, 3, 4, 5, 6, 7),
        tau=0.05,
    )
    primary_only = producer.selection_material(
        y,
        {0: pred0, 1: pred0, 2: pred0},
        basis,
        mean,
        ladder=(1, 2, 3, 4, 5, 6, 7),
        tau=0.05,
    )

    assert material["record"]["proposal_seed"] == 0
    assert material["record"]["K_prop"] == primary_only["record"]["K_prop"]
    assert np.array_equal(material["indices"], primary_only["indices"])
    assert material["record"]["gain_sha256"] == hashlib.sha256(
        np.ascontiguousarray(material["gain"], dtype=np.float64).tobytes()
    ).hexdigest()


def test_independent_bound_file_verifier_rejects_mutation(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"sealed bytes")
    manifest = {
        "artifact.bin": {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    }
    assert auditor.verify_bound_files(tmp_path, manifest)
    path.write_bytes(b"mutated")
    with pytest.raises(auditor.AuditFailure, match="artifact.bin"):
        auditor.verify_bound_files(tmp_path, manifest)


def test_test_gate_requires_exact_independent_audit(tmp_path: Path) -> None:
    pretest_sha = "a" * 64
    auditor_sha = "b" * 64
    path = tmp_path / "PRETEST_AUDIT.json"
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "pretest_seal_sha256": pretest_sha,
                "auditor_source_sha256": auditor_sha,
                "test_access": 0,
                "unused_access": 0,
            }
        )
    )
    assert producer.assert_test_gate(path, pretest_sha, auditor_sha)
    bad = tmp_path / "BAD_AUDIT.json"
    bad.write_text(path.read_text().replace('"PASS"', '"FAIL"'))
    with pytest.raises(RuntimeError, match="audit"):
        producer.assert_test_gate(bad, pretest_sha, auditor_sha)


def test_independent_stopping_replay_matches_frozen_semantics() -> None:
    curve = [1.0, 0.9995] + [0.9995] * 19
    replay = auditor.replay_stopping(curve, patience=20, min_delta=1e-3)
    assert replay["selected_epoch"] == 1
    assert replay["stop_epoch"] == 21
    assert replay["stop_reason"] == "PATIENCE"
