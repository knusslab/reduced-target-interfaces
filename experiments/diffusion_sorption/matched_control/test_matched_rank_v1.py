from pathlib import Path
import hashlib, importlib.util, json, tempfile
import numpy as np, torch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run", ROOT / "run_matched_rank_v1.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_protocol_hash():
    assert (
        run.sha(ROOT / "DIFF_SORP_MATCHED_RANK_GAIN_VS_POD_V1.md")
        == run.PROTOCOL_SHA256
    )


def test_seed_disjointness():
    inherited = (
        set(range(10000, 10064))
        | set(range(20000, 20016))
        | set(range(30000, 30024))
        | set(range(40000, 40024))
        | set(range(50000, 50024))
    )
    assert set(run.TEST_SEEDS).isdisjoint(inherited)
    assert run.QUALIFICATION_SEED not in inherited | set(run.TEST_SEEDS)


def test_matched_initialization():
    run.configure()
    torch.manual_seed(0)
    a = run.core.MLP(1024, 8, 256)
    da = run.state_digest(a.state_dict())
    torch.manual_seed(0)
    b = run.core.MLP(1024, 8, 256)
    db = run.state_digest(b.state_dict())
    assert da == db


def test_projection_loss_identity():
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(32, 8)))
    B = torch.tensor(q.T, dtype=torch.float32)
    p = torch.randn(10, 8)
    t = torch.randn(10, 8)
    got = run.core.projected_field_mse_from_coefficients(p, t, B)
    explicit = torch.mean(((p - t) @ B) ** 2)
    assert torch.allclose(got, explicit, rtol=1e-5, atol=1e-7)


def test_selector_constructs_pod_prefix():
    rng = np.random.default_rng(8)
    Y = rng.normal(size=(50, 16))
    P = Y + 0.1 * rng.normal(size=Y.shape)
    basis = np.eye(16)
    mu = np.zeros(16)
    old = run.core.LADDER
    run.core.LADDER = (2, 4, 8, 16)
    try:
        s = run.selector(Y, P, basis, mu)
        K = s["K"]
        pod = np.arange(K)
        assert len(s["indices"]) == K
        assert np.array_equal(pod, np.arange(K))
    finally:
        run.core.LADDER = old


def test_test_gate_refuses_without_marker():
    with tempfile.TemporaryDirectory() as td:
        try:
            run.generate_test(Path(td) / "missing_marker.json", Path(td) / "test.npz")
        except PermissionError:
            pass
        else:
            raise AssertionError("TEST generator opened without marker")


def test_cache_digests_are_well_formed():
    assert set(run.CACHE_SHA) == {"train", "validation", "selection"}
    for digest in run.CACHE_SHA.values():
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
