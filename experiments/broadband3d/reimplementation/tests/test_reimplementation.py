"""Deterministic unit tests for the Broadband 3D clean REIMPLEMENTATION.
No frozen scientific role trajectory is generated or read here.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "broadband3d_protocol_reimplementation.py"
spec = importlib.util.spec_from_file_location("bb_reimpl_test", P)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_role_counts_and_disjointness():
    assert [len(m.role_seeds(r)) for r in ("TRAIN","VALIDATION","SELECTION","TEST")] == [96,32,32,32]
    sets = [set(m.role_seeds(r)) for r in m.ROLE_RANGES]
    assert len(set.union(*sets)) == sum(map(len, sets))


def test_segment_schedules_cover_each_epoch_once():
    assert [(s.first_epoch,s.last_epoch) for s in m.dense_segments()] == [(0,9),(10,19),(20,29),(30,39),(40,49)]
    assert [(e) for s in m.compact_microsegments() for e in range(s.first_epoch,s.last_epoch+1)] == list(range(50))


def test_unet_parameter_count_and_shape():
    model = m.FrozenUNet3D()
    assert m.trainable_parameter_count(model) == 46707
    x = m.torch.zeros((1,3,32,32,32))
    with m.torch.no_grad():
        assert tuple(model(x).shape) == tuple(x.shape)


def test_basis_sign_canonicalization():
    b = np.array([[-1., 0.2, 0.1], [0.1, -2., 0.3]])
    c = m.canonicalize_basis_rows(b)
    assert c[0,0] > 0 and c[1,1] > 0


def test_two_bottleneck_identity():
    rng = np.random.default_rng(11)
    q, _ = np.linalg.qr(rng.normal(size=(12,12)))
    b = q[:,:5].T
    y = rng.normal(size=(20,12))
    p = rng.normal(size=(20,12))
    d = m.squared_error_decomposition(y,p,b)
    assert d["closure_error"] < 1e-12


def test_bootstrap_is_deterministic():
    rng = np.random.default_rng(12)
    dense = [rng.uniform(0.5,1.5,size=32) for _ in range(3)]
    compact = [dense[i] * (1.02 + 0.01*i) for i in range(3)]
    a = m.bootstrap_median_ratio(dense,compact,replicates=100,seed=m.BOOTSTRAP_SEED)
    b = m.bootstrap_median_ratio(dense,compact,replicates=100,seed=m.BOOTSTRAP_SEED)
    assert np.array_equal(a,b)


def test_json_safe_converts_numpy_bool():
    x = m.json_safe({"pass": np.bool_(True), "n": np.int64(4)})
    assert x == {"pass": True, "n": 4}
    assert type(x["pass"]) is bool and type(x["n"]) is int
