from __future__ import annotations
import argparse, copy, hashlib, json, os, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.sparse import diags

HERE = Path(__file__).resolve().parent
CORE_DIR = HERE.parent / "main" / "package" / "code"
sys.path.insert(0, str(CORE_DIR))
import diff_sorp_second_contract_v1 as core

PROTOCOL_ID = "DIFF_SORP_MATCHED_RANK_GAIN_VS_POD_V1"
PROTOCOL_SHA256 = "414ede932e2a4b5376449b5fd570e946838c068eb1028f7827c453f3c37475e1"
PINNED_PDEBENCH_COMMIT = "4ff3e3a4aa1561721b5571fa3a048a0a463e0568"
CACHE_SHA = {
    "train": "1218ec0651dcb8f4b3e4b10e978492d87425ec61a65734ad3675fd4c6d29f4be",
    "validation": "deffc3daca734cc9794dcbc1195221784ef21d800f2d6f865bae474811f7441c",
    "selection": "aaaf420f3e92bebd17619299f67d6c4097e1da47b83021aca0c4c12b5fe5e688",
}
TEST_SEEDS = tuple(range(70000, 70024))
QUALIFICATION_SEED = 69999
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_REPS = 10000
QUALITY_TOLERANCE = 1.05


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path, block=1 << 20):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def save_npy(path, a):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.save(path, a, allow_pickle=False)


def state_digest(state):
    h = hashlib.sha256()
    for n, t in sorted(state.items()):
        h.update(n.encode())
        h.update(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def configure():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
        "deterministic_algorithms": True,
    }


def load_cache(path: Path, expected_sha: str, rows: int):
    if sha(path) != expected_sha:
        raise RuntimeError(f"cache sha mismatch {path}")
    with np.load(path) as z:
        if set(z.files) != {"X", "Y", "traj"}:
            raise RuntimeError(f"cache keys {z.files}")
        X = np.asarray(z["X"], dtype=np.float32)
        Y = np.asarray(z["Y"], dtype=np.float32)
        traj = np.asarray(z["traj"], dtype=np.int64)
    if X.shape != (rows, 1024) or Y.shape != (rows, 1024) or traj.shape != (rows,):
        raise RuntimeError(
            f"cache shape mismatch {path}: {X.shape} {Y.shape} {traj.shape}"
        )
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise RuntimeError("nonfinite cache")
    vals, counts = np.unique(traj, return_counts=True)
    if not np.all(counts == 20):
        raise RuntimeError("not 20 rows/trajectory")
    return X, Y, traj


def physical_prediction(model, X, basis_rows=None, mean=None, batch=512):
    model.eval()
    xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    out = []
    B = (
        None
        if basis_rows is None
        else torch.from_numpy(np.ascontiguousarray(basis_rows, dtype=np.float32))
    )
    mu = (
        None
        if mean is None
        else torch.from_numpy(np.ascontiguousarray(mean, dtype=np.float32))
    )
    with torch.no_grad():
        for s in range(0, len(xt), batch):
            z = model(xt[s : s + batch])
            if B is not None:
                z = z @ B + mu
            out.append(z.detach().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def validation_mse(model, X, Y, basis_rows=None, mean=None):
    p = physical_prediction(model, X, basis_rows, mean)
    d = p - np.asarray(Y, dtype=np.float64)
    return float(np.mean(d * d))


def train_arm(label, seed, Xtr, Ctr, Xv, Yv, basis_rows, mean, outdir):
    configure()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    x = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float32))
    target = torch.from_numpy(np.ascontiguousarray(Ctr, dtype=np.float32))
    B = (
        None
        if basis_rows is None
        else torch.from_numpy(np.ascontiguousarray(basis_rows, dtype=np.float32))
    )
    dout = int(target.shape[1])
    model = core.MLP(1024, dout, hidden=256)
    init_digest = state_digest(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    vals = []
    train = []
    perms = []
    selected_state = None
    selected_epoch = None
    running_best = None
    selected_digest = None
    counter = 0
    raw_best = None
    raw_best_epoch = None
    stop_reason = "CEILING_REACHED"
    stop_epoch = None
    nonfinite = 0
    started = time.perf_counter()
    n = len(x)
    for epoch in range(1, 301):
        perm_np = core.batch_permutation(n, seed=seed, epoch=epoch).astype(np.int64)
        perms.append(perm_np.astype(np.int32))
        perm = torch.from_numpy(perm_np)
        model.train()
        total = torch.zeros((), dtype=torch.float64)
        for start in range(0, n, 64):
            idx = perm[start : start + 64]
            opt.zero_grad()
            pred = model(x[idx])
            if label == "dense":
                loss = torch.mean((pred - target[idx]) ** 2)
            else:
                loss = core.projected_field_mse_from_coefficients(pred, target[idx], B)
            loss.backward()
            opt.step()
            total += loss.detach().double() * idx.shape[0]
        tl = float((total / n).item())
        val = validation_mse(
            model,
            Xv,
            Yv,
            None if label == "dense" else basis_rows,
            None if label == "dense" else mean,
        )
        train.append(tl)
        vals.append(val)
        if not (np.isfinite(tl) and np.isfinite(val)):
            nonfinite += 1
            stop_reason = "NONFINITE"
            stop_epoch = epoch
            break
        dig = state_digest(model.state_dict())
        if raw_best is None or val < raw_best:
            raw_best = val
            raw_best_epoch = epoch
        if running_best is None:
            running_best = val
            selected_epoch = epoch
            selected_state = copy.deepcopy(model.state_dict())
            selected_digest = dig
        elif (running_best - val) / running_best >= 1e-3:
            running_best = val
            selected_epoch = epoch
            selected_state = copy.deepcopy(model.state_dict())
            selected_digest = dig
            counter = 0
        else:
            counter += 1
            if counter >= 20:
                stop_reason = "PATIENCE"
                stop_epoch = epoch
                break
    if stop_epoch is None:
        stop_epoch = len(vals)
    if stop_reason != "NONFINITE":
        rp = core.replay_stopping(vals, patience=20, min_delta=1e-3)
        if (
            rp["selected_epoch"],
            rp["stop_epoch"],
            rp["stop_reason"],
            rp["raw_best_epoch"],
        ) != (selected_epoch, stop_epoch, stop_reason, raw_best_epoch):
            raise RuntimeError(f"stopping replay mismatch {label} {seed}")
    if stop_reason != "PATIENCE" or nonfinite:
        raise RuntimeError(f"training gate {label} {seed}: {stop_reason}")
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / f"{label}_seed{seed}.pt"
    if ckpt.exists():
        raise FileExistsError(ckpt)
    torch.save(selected_state, ckpt)
    return {
        "label": label,
        "seed": int(seed),
        "output_dim": dout,
        "hidden": 256,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_state_sha256": init_digest,
        "checkpoint_state_sha256": selected_digest,
        "checkpoint_file_sha256": sha(ckpt),
        "checkpoint": str(ckpt),
        "selected_epoch": int(selected_epoch),
        "selected_validation_mse": float(running_best),
        "raw_best_epoch": int(raw_best_epoch),
        "raw_best_validation_mse": float(raw_best),
        "stop_epoch": int(stop_epoch),
        "stop_reason": stop_reason,
        "nonfinite_count": nonfinite,
        "batch_order_sha256": hashlib.sha256(
            np.concatenate(perms).astype(np.int32).tobytes()
        ).hexdigest(),
        "train_loss_per_epoch": train,
        "validation_mse_per_epoch": vals,
        "runtime_seconds": time.perf_counter() - started,
    }


def load_model(path: Path, dout: int):
    m = core.MLP(1024, dout, hidden=256)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m


def selector(Y, pred, basis, mean):
    y = np.asarray(Y, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    B = np.asarray(basis, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    c = (y - mu) @ B.T
    chat = (p - mu) @ B.T
    q = np.mean(c * c, 0)
    e = np.mean((c - chat) ** 2, 0)
    gain = q - e
    mse = float(np.mean((p - y) ** 2))
    threshold = 0.05 * 1024 * mse
    sel = core.select_budget(gain, ladder=core.LADDER, threshold=threshold)
    return {
        "K": int(sel["selected_k"]),
        "indices": sel["indices"].tolist(),
        "order": sel["order"].tolist(),
        "gain": gain,
        "q": q,
        "e": e,
        "reference_mse": mse,
        "threshold": threshold,
        "tail_by_k": {str(k): v for k, v in sel["tail_by_k"].items()},
        "gain_sha256": core.float64_array_digest(gain),
        "indices_sha256": sel["indices_sha256"],
        "order_sha256": sel["order_sha256"],
    }


class Simulator:
    def __init__(self, seed):
        self.D = 5e-4
        self.por = 0.29
        self.rho_s = 2880.0
        self.k_f = 3.5e-4
        self.n_f = 0.874
        self.sol = 1.0
        self.T = 500.0
        self.Nx = 1024
        self.Nt = 501
        self.X0 = 0.0
        self.X1 = 1.0
        self.seed = int(seed)
        self.dx = (self.X1 - self.X0) / self.Nx
        self.x = np.linspace(self.X0 + self.dx / 2, self.X1 - self.dx / 2, self.Nx)
        self.t = np.linspace(0, self.T, self.Nt)

    def rc_ode(self, t, y):
        left_BC = self.sol
        right_BC = (y[-2] - y[-1]) / self.dx * self.D
        retardation = 1 + (
            (1 - self.por) / self.por
        ) * self.rho_s * self.k_f * self.n_f * (y + 1e-6) ** (self.n_f - 1)
        self.rhs[0] = self.D / retardation[0] / (self.dx**2) * left_BC
        self.rhs[-1] = self.D / retardation[-1] / (self.dx**2) * right_BC
        return self.D / retardation * (self.lap @ y) + self.rhs

    def generate(self):
        g = np.random.default_rng(self.seed)
        u0 = np.ones(self.Nx) * g.uniform(0, 0.2)
        main = -2 * np.ones(self.Nx) / self.dx**2
        side = np.ones(self.Nx - 1) / self.dx**2
        self.lap = diags([main, side, side], [0, -1, 1])
        self.rhs = np.zeros(self.Nx)
        prob = solve_ivp(self.rc_ode, (0, self.T), u0, t_eval=self.t)
        if not prob.success:
            raise RuntimeError(f"solve_ivp failed seed {self.seed}: {prob.message}")
        a = np.expand_dims(prob.y.T, axis=-1)
        if a.shape != (501, 1024, 1) or not np.isfinite(a).all():
            raise RuntimeError(
                f"bad generated trajectory seed {self.seed} shape={a.shape} finite={np.isfinite(a).all()}"
            )
        return np.asarray(a, dtype=np.float64)


def rows_for_seed(seed):
    a = Simulator(seed).generate()
    idx = np.arange(0, 100, 5, dtype=np.int64)
    X = np.ascontiguousarray(a[idx, :, 0], dtype=np.float32)
    Y = np.ascontiguousarray(a[idx + 1, :, 0], dtype=np.float32)
    return int(seed), X, Y


def qualify_worker(seed):
    return rows_for_seed(seed)


def generate_test(marker: Path, out: Path):
    if not marker.exists():
        raise PermissionError("TEST marker missing")
    if out.exists():
        raise FileExistsError(out)
    rows = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        fut = {ex.submit(rows_for_seed, s): s for s in TEST_SEEDS}
        for f in as_completed(fut):
            seed, X, Y = f.result()
            rows[seed] = (X, Y)
    X = np.concatenate([rows[s][0] for s in TEST_SEEDS])
    Y = np.concatenate([rows[s][1] for s in TEST_SEEDS])
    traj = np.concatenate(
        [np.full(20, i, dtype=np.int64) for i, _ in enumerate(TEST_SEEDS)]
    )
    np.savez(out, X=X, Y=Y, traj=traj, seeds=np.asarray(TEST_SEEDS, dtype=np.int64))
    return X, Y, traj


def prefinal(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    env = configure()
    protocol = HERE / "DIFF_SORP_MATCHED_RANK_GAIN_VS_POD_V1.md"
    code = Path(__file__)
    legacy = HERE / "legacy_core.py"
    if sha(protocol) != PROTOCOL_SHA256:
        raise RuntimeError("protocol hash mismatch")
    Xtr, Ytr, Ttr = load_cache(Path(args.train), CACHE_SHA["train"], 1280)
    Xv, Yv, Tv = load_cache(Path(args.validation), CACHE_SHA["validation"], 320)
    Xs, Ys, Ts = load_cache(Path(args.selection), CACHE_SHA["selection"], 480)
    basis = core.fit_train_basis(Ytr)
    bdir = out / "basis"
    save_npy(bdir / "TRAIN_MEAN_FLOAT32.npy", basis["mean_float32"])
    save_npy(bdir / "BASIS_FLOAT32.npy", basis["basis_float32"])
    save_npy(bdir / "EIGENVALUES_FLOAT64.npy", basis["eigenvalues_float64"])
    dense = []
    for s in core.SEEDS:
        tr = train_arm(
            "dense", s, Xtr, Ytr, Xv, Yv, None, None, out / "checkpoints" / "dense"
        )
        write_json(out / "traces" / f"dense_seed{s}.json", tr)
        dense.append(tr)
    predictions = {}
    refs = {}
    for s in core.SEEDS:
        m = load_model(Path(dense[s]["checkpoint"]), 1024)
        p = physical_prediction(m, Xs)
        predictions[s] = p
        refs[s] = selector(Ys, p, basis["basis_float64"], basis["mean_float64"])
    primary = refs[0]
    K = primary["K"]
    gain_idx = np.asarray(primary["indices"], dtype=np.int64)
    pod_idx = np.arange(K, dtype=np.int64)
    srec = {
        "status": "SELECTION_SEALED",
        "K": K,
        "gain_indices": gain_idx.tolist(),
        "pod_indices": pod_idx.tolist(),
        "sets_identical": bool(np.array_equal(gain_idx, pod_idx)),
        "references": {
            str(s): {
                k: v for k, v in refs[s].items() if k not in ("gain", "q", "e", "order")
            }
            for s in core.SEEDS
        },
        "primary_gain_sha256": refs[0]["gain_sha256"],
        "primary_order_sha256": refs[0]["order_sha256"],
        "sealed_at_utc": utc(),
    }
    save_npy(out / "selection" / "GAIN_FLOAT64.npy", primary["gain"])
    save_npy(out / "selection" / "GAIN_INDICES_INT64.npy", gain_idx)
    save_npy(out / "selection" / "POD_INDICES_INT64.npy", pod_idx)
    write_json(out / "selection" / "SELECTION_SEAL.json", srec)
    if K >= 1024 or np.array_equal(gain_idx, pod_idx):
        status = (
            "NO_REDUCED_BUDGET"
            if K >= 1024
            else "SETS_IDENTICAL_NO_ORDERING_TEST_NEEDED"
        )
        write_json(
            out / "TERMINAL_PREFINAL.json",
            {"status": status, "K": K, "test_access": 0, "at_utc": utc()},
        )
        return
    arms = {}
    for label, idxs in [("gain", gain_idx), ("pod", pod_idx)]:
        B = np.ascontiguousarray(basis["basis_float32"][idxs], dtype=np.float32)
        coeff = np.ascontiguousarray(
            (Ytr - basis["mean_float32"]) @ B.T, dtype=np.float32
        )
        traces = []
        for s in core.SEEDS:
            tr = train_arm(
                label,
                s,
                Xtr,
                coeff,
                Xv,
                Yv,
                B,
                basis["mean_float32"],
                out / "checkpoints" / label,
            )
            write_json(out / "traces" / f"{label}_seed{s}.json", tr)
            traces.append(tr)
        arms[label] = traces
    for s in core.SEEDS:
        if (
            arms["gain"][s]["initial_state_sha256"]
            != arms["pod"][s]["initial_state_sha256"]
        ):
            raise RuntimeError(f"initialization mismatch seed{s}")
        if (
            arms["gain"][s]["batch_order_sha256"]
            != arms["pod"][s]["batch_order_sha256"]
        ):
            raise RuntimeError(f"batch order mismatch seed{s}")
    seal = {
        "status": "PREFINAL_SEALED",
        "protocol_sha256": PROTOCOL_SHA256,
        "code_sha256": sha(code),
        "legacy_core_sha256": sha(legacy),
        "cache_sha256": CACHE_SHA,
        "environment": env,
        "basis": {
            k: v
            for k, v in basis.items()
            if k.endswith("sha256")
            or k in ("k_energy", "orthonormality_residual_max_abs", "field_dim")
        },
        "selection": srec,
        "traces": {"dense": dense, "gain": arms["gain"], "pod": arms["pod"]},
        "test_seed_list": list(TEST_SEEDS),
        "test_access": 0,
        "sealed_at_utc": utc(),
        "runtime_seconds": time.time() - start,
    }
    # Keep the seal compact; trace files bind the full curves.
    for arm in seal["traces"]:
        for t in seal["traces"][arm]:
            t.pop("train_loss_per_epoch", None)
            t.pop("validation_mse_per_epoch", None)
    write_json(out / "PRETEST_SEAL.json", seal)
    print(
        json.dumps(
            {
                "status": "PREFINAL_SEALED",
                "K": K,
                "gain_indices": gain_idx.tolist(),
                "pod_indices": pod_idx.tolist(),
                "runtime_s": time.time() - start,
            },
            indent=2,
        )
    )


def test_stage(args):
    root = Path(args.root)
    seal = json.load(open(root / "PRETEST_SEAL.json"))
    marker = root / "test" / "TEST_ACCESS_ONCE.json"
    write_json(
        marker,
        {
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "pretest_seal_sha256": sha(root / "PRETEST_SEAL.json"),
            "test_seeds": list(TEST_SEEDS),
            "consumed_at_utc": utc(),
        },
    )
    test_npz = root / "test" / "TEST_ROWS.npz"
    X, Y, traj = generate_test(marker, test_npz)
    basis = np.load(root / "basis" / "BASIS_FLOAT32.npy")
    mean = np.load(root / "basis" / "TRAIN_MEAN_FLOAT32.npy")
    gain_idx = np.load(root / "selection" / "GAIN_INDICES_INT64.npy")
    pod_idx = np.load(root / "selection" / "POD_INDICES_INT64.npy")
    K = len(gain_idx)
    per = {
        arm: {s: np.zeros(24, dtype=np.float64) for s in core.SEEDS}
        for arm in ("dense", "gain", "pod")
    }
    for arm, idx in [("dense", None), ("gain", gain_idx), ("pod", pod_idx)]:
        B = None if idx is None else np.ascontiguousarray(basis[idx], dtype=np.float32)
        for s in core.SEEDS:
            m = load_model(
                root / "checkpoints" / arm / f"{arm}_seed{s}.pt",
                1024 if arm == "dense" else K,
            )
            p = physical_prediction(m, X, B, None if arm == "dense" else mean)
            for t in range(24):
                mask = traj == t
                d = p[mask] - Y[mask].astype(np.float64)
                per[arm][s][t] = np.mean(d * d)
    dense = {s: float(per["dense"][s].mean()) for s in core.SEEDS}
    gain = {s: float(per["gain"][s].mean()) for s in core.SEEDS}
    pod = {s: float(per["pod"][s].mean()) for s in core.SEEDS}
    rg = {s: gain[s] / dense[s] for s in core.SEEDS}
    rp = {s: pod[s] / dense[s] for s in core.SEEDS}
    d = {s: gain[s] / pod[s] for s in core.SEEDS}
    Qg = float(np.median(list(rg.values())))
    Qp = float(np.median(list(rp.values())))
    if Qg <= QUALITY_TOLERANCE and Qp > QUALITY_TOLERANCE:
        verdict = "GAIN_ONLY_PASS"
    elif Qg > QUALITY_TOLERANCE and Qp <= QUALITY_TOLERANCE:
        verdict = "POD_ONLY_PASS"
    elif Qg <= QUALITY_TOLERANCE and Qp <= QUALITY_TOLERANCE:
        verdict = "BOTH_PASS"
    else:
        verdict = "BOTH_FAIL"
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = np.empty((BOOTSTRAP_REPS, 3), dtype=np.float64)
    for b in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, 24, size=24)
        rg_b = []
        rp_b = []
        d_b = []
        for s in core.SEEDS:
            dm = per["dense"][s][idx].mean()
            gm = per["gain"][s][idx].mean()
            pm = per["pod"][s][idx].mean()
            rg_b.append(gm / dm)
            rp_b.append(pm / dm)
            d_b.append(gm / pm)
        draws[b] = [np.median(rg_b), np.median(rp_b), np.median(d_b)]
    ci = {
        name: [float(x) for x in np.quantile(draws[:, i], [0.025, 0.975])]
        for i, name in enumerate(("Q_gain", "Q_pod", "gain_over_pod"))
    }
    raw = {f"{arm}_seed{s}": per[arm][s] for arm in per for s in core.SEEDS}
    np.savez(root / "test" / "TEST_PER_TRAJECTORY.npz", **raw)
    result = {
        "status": "TEST_COMPLETE",
        "verdict": verdict,
        "K": K,
        "gain_indices": gain_idx.tolist(),
        "pod_indices": pod_idx.tolist(),
        "dense_mse": {str(k): v for k, v in dense.items()},
        "gain_mse": {str(k): v for k, v in gain.items()},
        "pod_mse": {str(k): v for k, v in pod.items()},
        "r_gain": {str(k): v for k, v in rg.items()},
        "r_pod": {str(k): v for k, v in rp.items()},
        "gain_over_pod": {str(k): v for k, v in d.items()},
        "Q_gain": Qg,
        "Q_pod": Qp,
        "quality_tolerance": QUALITY_TOLERANCE,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_ci95": ci,
        "test_rows_sha256": sha(test_npz),
        "per_trajectory_sha256": sha(root / "test" / "TEST_PER_TRAJECTORY.npz"),
        "completed_at_utc": utc(),
    }
    write_json(root / "test" / "TEST_RESULT.json", result)
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prefinal")
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)
    t = sp.add_parser("test")
    t.add_argument("--root", required=True)
    q = sp.add_parser("qualify-generator")
    q.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "prefinal":
        prefinal(a)
    elif a.cmd == "test":
        test_stage(a)
    else:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        seed = QUALIFICATION_SEED
        t = time.time()
        s1, X1, Y1 = rows_for_seed(seed)
        with ProcessPoolExecutor(max_workers=1) as ex:
            s2, X2, Y2 = ex.submit(rows_for_seed, seed).result()
        rec = {
            "seed": seed,
            "shape_X": list(X1.shape),
            "shape_Y": list(Y1.shape),
            "finite": bool(np.isfinite(X1).all() and np.isfinite(Y1).all()),
            "sequential_worker_equal": bool(
                np.array_equal(X1, X2) and np.array_equal(Y1, Y2)
            ),
            "X_sha256": hashlib.sha256(X1.tobytes()).hexdigest(),
            "Y_sha256": hashlib.sha256(Y1.tobytes()).hexdigest(),
            "runtime_seconds": time.time() - t,
        }
        write_json(out, rec)
        print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
