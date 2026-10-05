from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_public_verifier(*relative_args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, *relative_args],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_ns2d_v033_public_layout_selftest() -> None:
    result = run_public_verifier(
        "experiments/ns2d/controls/v033/run_portable.py",
        "selftest",
    )
    assert "SELFTEST_PASS" in result.stdout


def test_broadband_reference_implementation_verifier() -> None:
    result = run_public_verifier(
        "experiments/broadband3d/reference_implementation/verify_reference_implementation.py"
    )
    assert '"status": "PASS"' in result.stdout
