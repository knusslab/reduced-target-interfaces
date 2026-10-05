from pathlib import Path
import json, hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "run_v1_1"
RES = json.load(open(ROOT / "test" / "TEST_RESULT.json"))
MARK = json.load(open(ROOT / "test" / "TEST_ACCESS_ONCE.json"))
SEAL = json.load(open(ROOT / "PRETEST_SEAL.json"))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


assert MARK["pretest_seal_sha256"] == sha(ROOT / "PRETEST_SEAL.json")
with np.load(ROOT / "test" / "TEST_ROWS.npz") as z:
    X = np.asarray(z["X"])
    Y = np.asarray(z["Y"])
    traj = np.asarray(z["traj"])
    seeds = np.asarray(z["seeds"])
assert (
    X.shape == Y.shape == (480, 1024)
    and traj.shape == (480,)
    and seeds.tolist() == list(range(70000, 70024))
)
assert np.isfinite(X).all() and np.isfinite(Y).all()
u, c = np.unique(traj, return_counts=True)
assert u.tolist() == list(range(24)) and np.all(c == 20)
with np.load(ROOT / "test" / "TEST_PER_TRAJECTORY.npz") as z:
    raw = {k: np.asarray(z[k], dtype=np.float64) for k in z.files}
for arm in ("dense", "gain", "pod"):
    for s in (0, 1, 2):
        assert raw[f"{arm}_seed{s}"].shape == (24,)
dense = {s: raw[f"dense_seed{s}"].mean() for s in (0, 1, 2)}
gain = {s: raw[f"gain_seed{s}"].mean() for s in (0, 1, 2)}
pod = {s: raw[f"pod_seed{s}"].mean() for s in (0, 1, 2)}
rg = {s: gain[s] / dense[s] for s in (0, 1, 2)}
rp = {s: pod[s] / dense[s] for s in (0, 1, 2)}
dd = {s: gain[s] / pod[s] for s in (0, 1, 2)}
Qg = float(np.median(list(rg.values())))
Qp = float(np.median(list(rp.values())))
assert abs(Qg - RES["Q_gain"]) < 1e-15 and abs(Qp - RES["Q_pod"]) < 1e-15
for s in (0, 1, 2):
    assert abs(rg[s] - RES["r_gain"][str(s)]) < 1e-15
    assert abs(rp[s] - RES["r_pod"][str(s)]) < 1e-15
    assert abs(dd[s] - RES["gain_over_pod"][str(s)]) < 1e-15
verdict = (
    "GAIN_ONLY_PASS"
    if Qg <= 1.05 < Qp
    else (
        "POD_ONLY_PASS"
        if Qp <= 1.05 < Qg
        else "BOTH_PASS" if Qg <= 1.05 and Qp <= 1.05 else "BOTH_FAIL"
    )
)
assert verdict == RES["verdict"] == "BOTH_PASS"
rng = np.random.Generator(np.random.PCG64(20260817))
draws = np.empty((10000, 3))
for b in range(10000):
    idx = rng.integers(0, 24, size=24)
    a = []
    p = []
    d = []
    for s in (0, 1, 2):
        dm = raw[f"dense_seed{s}"][idx].mean()
        gm = raw[f"gain_seed{s}"][idx].mean()
        pm = raw[f"pod_seed{s}"][idx].mean()
        a.append(gm / dm)
        p.append(pm / dm)
        d.append(gm / pm)
    draws[b] = [np.median(a), np.median(p), np.median(d)]
ci = {
    name: [float(x) for x in np.quantile(draws[:, i], [0.025, 0.975])]
    for i, name in enumerate(("Q_gain", "Q_pod", "gain_over_pod"))
}
for k in ci:
    assert np.allclose(ci[k], RES["bootstrap_ci95"][k], rtol=0, atol=1e-15)
out = {
    "status": "PASS",
    "producer_result_sha256": sha(ROOT / "test" / "TEST_RESULT.json"),
    "test_rows_sha256": sha(ROOT / "test" / "TEST_ROWS.npz"),
    "per_trajectory_sha256": sha(ROOT / "test" / "TEST_PER_TRAJECTORY.npz"),
    "marker_sha256": sha(ROOT / "test" / "TEST_ACCESS_ONCE.json"),
    "Q_gain": Qg,
    "Q_pod": Qp,
    "gain_over_pod_median": float(np.median(list(dd.values()))),
    "verdict": verdict,
    "ci95": ci,
    "test_seeds": seeds.tolist(),
    "rows": 480,
    "trajectory_count": 24,
    "test_access_replay": "artifact-only; no generator rerun",
}
(HERE / "VERIFY_TEST_V1_2.json").write_text(
    json.dumps(out, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(out, indent=2))
