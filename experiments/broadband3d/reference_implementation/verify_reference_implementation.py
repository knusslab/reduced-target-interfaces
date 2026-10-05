"""Offline verifier for the Broadband 3D protocol reference implementation.

This verifier checks the preserved contract mechanics without presenting the reference implementation as the byte-identical historical training/evaluation executor.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
BROADBAND = ROOT.parent
MOD_PATH = ROOT / "broadband3d_protocol_reference.py"


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def load_module():
    spec = importlib.util.spec_from_file_location("bb_reimpl", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    m = load_module()
    checks = []
    boundary = (ROOT / "REFERENCE_IMPLEMENTATION.md").read_text(encoding="utf-8")
    require("protocol reference implementation" in boundary.lower(), "reference-implementation label missing")
    checks.append("boundary_label")

    solver = BROADBAND / "solver" / "ns3d_spectral.py"
    digest = hashlib.sha256(solver.read_bytes()).hexdigest()
    require(digest == m.SOLVER_SHA256, "exact solver SHA mismatch")
    checks.append("solver_sha")

    require(len(m.role_seeds("TRAIN")) == 96, "TRAIN role count")
    require(len(m.role_seeds("VALIDATION")) == 32, "VALIDATION role count")
    require(len(m.role_seeds("SELECTION")) == 32, "SELECTION role count")
    require(len(m.role_seeds("TEST")) == 32, "TEST role count")
    require(len(set().union(*(set(m.role_seeds(r)) for r in m.ROLE_RANGES))) == 192, "role overlap")
    checks.append("role_identity_partition")

    ds = m.dense_segments()
    cs = m.compact_microsegments()
    require([(x.first_epoch, x.last_epoch) for x in ds] == [(0,9),(10,19),(20,29),(30,39),(40,49)], "dense segment schedule")
    require(cs[0].first_epoch == 0 and cs[-1].last_epoch == 49 and len(cs) == 25, "compact microsegment schedule")
    require([e for x in cs for e in range(x.first_epoch, x.last_epoch + 1)] == list(range(50)), "compact epoch coverage")
    checks.append("segmentation_semantics")

    require(m.epoch_permutation_seed(0, 0) == 1_000_000, "epoch seed 0")
    require(m.epoch_permutation_seed(2, 49) == 1_002_049, "epoch seed terminal")
    checks.append("epoch_order_seed_rule")

    if m.torch is None:
        raise RuntimeError("PyTorch required for architecture verification")
    model = m.FrozenUNet3D()
    require(m.trainable_parameter_count(model) == 46_707, "3D U-Net parameter count")
    x = m.torch.zeros((1,3,32,32,32), dtype=m.torch.float32)
    with m.torch.no_grad():
        y = model(x)
    require(tuple(y.shape) == (1,3,32,32,32), "3D U-Net shape")
    checks.append("unet_architecture_46707")

    # Generator contract on a non-confirmation identity only.
    solver_spec = importlib.util.spec_from_file_location("exact_ns3d", solver)
    solver_mod = importlib.util.module_from_spec(solver_spec)
    solver_spec.loader.exec_module(solver_mod)
    sim = solver_mod.NS3D(N=32, nu=5e-4)
    u1, h1 = m.broadband_initial_condition(123456, sim)
    u2, h2 = m.broadband_initial_condition(123456, sim)
    require(np.array_equal(u1, u2) and np.array_equal(h1, h2), "PCG64 IC determinism")
    energy_norm = float(np.mean(np.sum(u1*u1, axis=0)))
    require(abs(energy_norm - 1.0) < 1e-12, f"IC normalization {energy_norm}")
    require(sim.max_divergence(h1) < 1e-11, "IC divergence")
    checks.append("broadband_ic_contract_nonrole_seed")

    # Stable gain ordering and selection mechanics on deterministic synthetic arrays.
    rng = np.random.default_rng(7)
    target = rng.normal(size=(40,95))
    pred = target.copy()
    pred[:,20:] += rng.normal(scale=0.2, size=(40,75))
    gain, order = m.predictive_gain_order(target, pred)
    require(order.shape == (95,) and sorted(order.tolist()) == list(range(95)), "gain order permutation")
    require(np.all(np.maximum(gain[order[:-1]],0) >= np.maximum(gain[order[1:]],0)), "gain order descending")
    checks.append("selector_order")

    # Exact Pythagorean decomposition on an orthonormal synthetic basis.
    a = rng.normal(size=(13,17))
    q, _ = np.linalg.qr(rng.normal(size=(17,17)))
    b = q[:,:6].T
    pred = rng.normal(size=(13,17))
    d = m.squared_error_decomposition(a, pred, b)
    require(d["closure_error"] < 1e-12, f"decomposition closure {d['closure_error']}")
    checks.append("two_bottleneck_identity")

    dense = [rng.uniform(0.9,1.1,size=32) for _ in range(3)]
    compact = [dense[i] * (1.1 + 0.01*i) for i in range(3)]
    b1 = m.bootstrap_median_ratio(dense, compact, replicates=300, seed=m.BOOTSTRAP_SEED)
    b2 = m.bootstrap_median_ratio(dense, compact, replicates=300, seed=m.BOOTSTRAP_SEED)
    require(np.array_equal(b1,b2), "bootstrap PCG64 determinism")
    checks.append("shared_bootstrap_determinism")

    payload = {"flag": np.bool_(True), "i": np.int64(3), "x": np.float32(1.5), "a": np.array([1,2])}
    safe = m.json_safe(payload)
    encoded = json.dumps(safe)
    require('"flag": true' in encoded and safe["i"] == 3 and safe["a"] == [1,2], "JSON-safe numpy conversion")
    checks.append("v1_5_json_safe_conversion")

    out = {
        "status": "PASS",
        "scope": "offline structural/algebraic verification only; no scientific role generation/training/TEST access",
        "label": m.REIMPLEMENTATION_LABEL,
        "checks": checks,
        "check_count": len(checks),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
