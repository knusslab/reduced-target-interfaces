"""E2E_COST_V1 -- what a rank budget costs in a training pipeline, measured.

Protocol: protocols/E2E_COST_V1.md

The source paper argues for a rank budget on accuracy grounds and accounts for storage
arithmetically. This driver measures the pipeline instead. Four measurement boundaries are
fixed here because the result depends on them, and a table that leaves them vague is a
table whose numbers cannot be checked:

  1. Host memory is the peak resident set of the **whole process tree**, sampled while the
     arm runs, and every arm runs in its **own process**. A single reading at the end
     reports what is resident then, not the peak; `peak_wset` in a shared process reports
     the largest arm seen so far, not the arm being measured. Both were tried and both were
     wrong, which is why the sampler and the process isolation exist.
  2. Device memory is reported as peak **allocated** and peak **reserved**, reset per arm,
     read after a synchronize. Reserved is what decides whether the next allocation fails.
  3. Input time is split into **read**, **transfer** and **decode**, because a compressed
     arm reads less and then computes, and one number called "loading" hides which. Epoch
     time is reported for the **cold** epoch and for the **steady** state separately: a
     saving that appears only on the first epoch is an ingestion saving, not a training
     speed-up, and must not be written as one.
  4. Storage is the size of the file actually written -- coefficients, basis, mean, the
     retained index set, dtypes, shapes and container overhead. Comparing bare coefficients
     against a dense tensor would be unfair to the dense arm.

`load_traj` is taken from the source paper's `ns2d_kenergy_ext.py` so both papers read the
same tensors the same way.

Run:  python code/e2e_cost.py --ns2d_root ../ns2d --ts_csv ../ts/electricity/electricity.csv
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import torch
import torch.nn as nn

PROTOCOL_ID = "E2E_COST_V2"


def sha256_of_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha256_int32(values):
    """Canonical digest for a retained-mode index vector (little host-independent int32 bytes)."""
    a = np.ascontiguousarray(values, dtype="<i4")
    return hashlib.sha256(a.tobytes()).hexdigest()


def sha256_float64(values):
    """Canonical digest for numeric selector evidence (little-endian float64 bytes)."""
    a = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(a.tobytes()).hexdigest()


# --------------------------------------------------------------------------- measurement
class TreeRssSampler:
    """Peak resident set of this process and its children, sampled during one arm."""

    def __init__(self, interval=0.02):
        self.interval, self.peak = interval, 0
        self._stop = threading.Event()
        self._t = None

    def _read(self, proc):
        total = proc.memory_info().rss
        for c in proc.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
        return total

    def _run(self):
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self._read(proc))
            except psutil.Error:
                pass
            self._stop.wait(self.interval)
        try:
            self.peak = max(self.peak, self._read(proc))
        except psutil.Error:
            pass

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2.0)
        return self.peak


# -------------------------------------------------------------------------------- data
def load_traj(root, tag, max_traj):
    """Verbatim from the source paper's ns2d_kenergy_ext.py."""
    import h5py
    us = []
    for f in sorted(glob.glob(f"{root}/*{tag}*.h5")):
        g = h5py.File(f, "r")
        t = g[list(g.keys())[0]]
        us.append(t["u"][:])
        if sum(x.shape[0] for x in us) >= max_traj:
            break
    return np.concatenate(us)[:max_traj]


def ns2d_pairs(root, max_traj, pairing="roll"):
    """NS2D next-step rows (X, Y) under one of two pairing rules.

    "roll" (frozen E2E_COST_V2 default): a GLOBAL np.roll over the concatenated trajectory stack,
    so each trajectory's first frame is paired with the previous trajectory's last frame and row 0
    wraps -- ~1/T of rows are cross-trajectory. Kept as the default so existing results reproduce.
    "traj" (protocol 7c): per-trajectory pairing as the source paper's pod_head_ns2d.py does it
    (X = u[:, :-1], Y = u[:, 1:]) -- no cross-trajectory pair, no wraparound.
    """
    u = load_traj(root, "train", max_traj)
    Ng = u.shape[-1]
    D = Ng * Ng
    if pairing == "traj":
        return (u[:, :-1].reshape(-1, D).astype(np.float32),
                u[:, 1:].reshape(-1, D).astype(np.float32))
    if pairing != "roll":
        raise ValueError(f"unknown pairing: {pairing}")
    Y = u.reshape(-1, D).astype(np.float32)
    return np.roll(Y, 1, axis=0).astype(np.float32), Y


def load_ts(csv_path, L, H, n_win, stride, train_frac=0.8,
            scaling="legacy_raw_fraction", return_meta=False):
    import csv as _csv
    with open(csv_path, newline="") as fh:
        rows = list(_csv.reader(fh))
    cols = [i for i in range(len(rows[0])) if i > 0]
    arr = np.array([[float(r[i]) for i in cols] for r in rows[1:]], dtype=np.float32)
    starts = list(range(0, len(arr) - L - H, stride))[:n_win]
    if not starts:
        raise ValueError("time-series geometry produced no windows")
    if scaling == "legacy_raw_fraction":
        fit_start, fit_stop = 0, int(train_frac * len(arr))
        n_train_windows = None
    elif scaling == "train_window_union":
        n_train_windows = int(round(float(train_frac) * len(starts)))
        if not (0 < n_train_windows < len(starts)):
            raise ValueError(f"invalid training-window count {n_train_windows}/{len(starts)}")
        fit_start = int(starts[0])
        fit_stop = int(starts[n_train_windows - 1] + L + H)
    else:
        raise ValueError(f"unknown scaling mode: {scaling}")
    if not (0 <= fit_start < fit_stop <= len(arr)):
        raise ValueError(f"invalid scaler fit range [{fit_start},{fit_stop}) for {len(arr)} rows")
    fit = arr[fit_start:fit_stop]
    mu = fit.mean(0, keepdims=True).astype(np.float32)
    sd = (fit.std(0, keepdims=True) + 1e-8).astype(np.float32)
    if not np.isfinite(mu).all() or not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError("non-finite or non-positive time-series scaler")
    arr = (arr - mu) / sd
    X = np.stack([arr[s:s + L].T for s in starts])
    Y = np.stack([arr[s + L:s + L + H].T for s in starts])
    out = (X.reshape(-1, L), Y.reshape(-1, H))
    if not return_meta:
        return out
    meta = {
        "mode": scaling, "train_window_frac": float(train_frac),
        "n_windows": int(len(starts)),
        "n_train_windows": (int(n_train_windows) if n_train_windows is not None else None),
        "fit_raw_start": int(fit_start), "fit_raw_stop_exclusive": int(fit_stop),
        "fit_raw_count": int(fit_stop - fit_start), "stride": int(stride),
        "L": int(L), "H": int(H),
        "mean": [float(x) for x in mu.reshape(-1)],
        "std": [float(x) for x in sd.reshape(-1)],
        "mean_sha256": sha256_float64(mu.reshape(-1).astype(np.float64)),
        "std_sha256": sha256_float64(sd.reshape(-1).astype(np.float64)),
    }
    return out + (meta,)


def ranks_from_calibration(path, target):
    """Read the stored/oracle ranks for `target` from a calibration run's JSON (protocol 7d(ii)).

    Keeps the two roles separate and auditable instead of typing numbers into a command line:
      budget arm  <- `K_pred`           read on the SELECTION split (never on test)
      oracle arm  <- `K_oracle_ladder`  a diagnostic comparator DEFINED on test; it must never
                                        influence what the budget arm stores.
    Returns (K_pred, K_oracle, provenance) where provenance records the file, its sha256 and the
    field names, so the artifact says where each rank came from.
    """
    d = json.load(open(path))
    cell = d["per_target"][target]
    kp, ko = int(cell["K_pred"]), int(cell["K_oracle_ladder"])
    prov = {"source_json": os.path.abspath(path), "source_sha256": sha256_of_file(path),
            "protocol_id": d.get("protocol_id"), "target": target,
            "budget_rank": {"value": kp, "field": "K_pred", "read_on": "selection split",
                            "role": "what the budget arm stores"},
            "oracle_rank": {"value": ko, "field": "K_oracle_ladder", "read_on": "test split",
                            "role": "diagnostic comparator only; not used to choose storage"}}
    return kp, ko, prov


def rank_selection_from_calibration_v10(path, target):
    """Load and verify the complete V10 selector output used by the system-cost run.

    A scalar K is not enough for an arbitrary predictive subset.  V10 transfers the full gain
    order, verifies both index digests, and derives every stored subset from that order.  A corrupt
    or legacy artifact is a hard error; there is no prefix fallback.
    """
    with open(path) as fh:
        d = json.load(fh)
    cell = d["per_target"][target]
    kp, ko = int(cell["K_pred"]), int(cell["K_oracle_ladder"])
    order = np.asarray(cell["gain_order"], dtype=np.int32)
    keep = np.asarray(cell["kept_idx"], dtype=np.int32)
    assert order.ndim == 1 and len(order) > 0, "gain_order must be a non-empty vector"
    assert len(np.unique(order)) == len(order), "gain_order contains duplicate modes"
    assert int(order.min()) >= 0, "gain_order contains a negative mode"
    assert 0 < kp <= len(order), (kp, len(order))
    assert 0 < ko <= len(order), (ko, len(order))
    assert cell["gain_order_sha256"] == sha256_int32(order), "gain_order digest mismatch"
    assert cell["kept_idx_sha256"] == sha256_int32(keep), "kept_idx digest mismatch"
    assert np.array_equal(keep, order[:kp]), "kept_idx is not gain_order[:K_pred]"
    provenance = {
        "source_json": os.path.abspath(path),
        "source_sha256": sha256_of_file(path),
        "protocol_id": d.get("protocol_id"),
        "target": target,
        "gain_order_sha256": sha256_int32(order),
        "budget_rank": {"value": kp, "field": "K_pred", "read_on": "selection split",
                        "role": "rank and exact modes stored by the budget arm"},
        "oracle_rank": {"value": ko, "field": "K_oracle_ladder", "read_on": "test split",
                        "role": "diagnostic comparator only; not used to choose budget storage"},
    }
    return {"K_pred": kp, "K_oracle_ladder": ko, "gain_order": order,
            "budget_keep_idx": order[:kp].copy(), "oracle_keep_idx": order[:ko].copy(),
            "provenance": provenance}


def kept_idx_for_system_scheme(scheme, K, selection):
    """Exact V10 mode indices for a system-cost store; None means the defined POD-prefix path."""
    if scheme not in ("budget", "oracle"):
        return None
    assert selection is not None, f"V10 {scheme} store requires selector provenance"
    order = np.asarray(selection["gain_order"], dtype=np.int32)
    assert 0 < K <= len(order), (scheme, K, len(order))
    expected = selection["K_pred"] if scheme == "budget" else selection["K_oracle_ladder"]
    assert K == expected, (scheme, K, expected)
    return order[:K].copy()


def make_split_v9_3way(name, N, fracs, embargo):
    """Train / held-out rows taken from the V9 leak-free 3-WAY split: train = first block,
    held-out = the TEST block (the selection block is not used by this driver -- it exists so the
    calibration run can read K_pred without touching test). Used by protocol 7d(ii) so the
    system-cost table shares the split, not just the pairing, with the calibration primary.
    """
    from e2e_cost_v9_2d import three_way          # local import: v9_2d imports this module
    tr, se, te = three_way(N, fracs, embargo=embargo)
    return tr, te, se


def make_split(N, frac=0.8):
    """Contiguous, deterministic 80/20 split over rows (E2E_COST_V2 item 7).

    Contiguous rather than random because both targets are temporal: a random split would
    place a row's temporal neighbours on both sides. Independent of the seed, so every arm
    and every seed sees the same rows.
    """
    n_train = int(frac * N)
    return np.arange(n_train), np.arange(n_train, N)


def split_digest(train_idx, heldout_idx):
    """Digests of the two index arrays, so the split can be audited from the artifact."""
    return {"n_train": int(len(train_idx)), "n_heldout": int(len(heldout_idx)),
            "train_sha256": hashlib.sha256(
                np.ascontiguousarray(train_idx, dtype=np.int64).tobytes()).hexdigest(),
            "heldout_sha256": hashlib.sha256(
                np.ascontiguousarray(heldout_idx, dtype=np.int64).tobytes()).hexdigest()}


def split_digest3(train_idx, selection_idx, test_idx):
    """Complete V10 three-way split provenance; no truncated or self-authored PASS fields."""
    def one(x):
        return hashlib.sha256(np.ascontiguousarray(x, dtype="<i8").tobytes()).hexdigest()
    return {"n_train": int(len(train_idx)), "n_selection": int(len(selection_idx)),
            "n_test": int(len(test_idx)), "train_sha256": one(train_idx),
            "selection_sha256": one(selection_idx), "test_sha256": one(test_idx)}


def spectrum_and_basis(Sc, kmax):
    """Top-`kmax` right singular vectors and the full spectrum, via the smaller Gram.

    A direct `svd` on the NS2D supervision materialises an 11200-square U and takes hours;
    on the forecasting target the left Gram would be 642000-square, three terabytes. The
    side is chosen by shape. Verified against `np.linalg.svd` in both orientations.
    """
    N, D = Sc.shape
    if D <= N:
        w, V = np.linalg.eigh(Sc.T @ Sc)
        w = np.clip(w[::-1], 0.0, None)
        V = V[:, ::-1]
        return w, V[:, :min(kmax, D)].T.astype(np.float32)
    w, U = np.linalg.eigh(Sc @ Sc.T)
    w = np.clip(w[::-1], 0.0, None)
    U = U[:, ::-1]
    sv = np.sqrt(w)
    k = min(kmax, int((sv > sv[0] * 1e-12).sum()))
    V = (Sc.T @ U[:, :k]) / np.maximum(sv[:k], 1e-300)
    return w, V.T.astype(np.float32)


def pod_basis_lean(Ytr, kmax, col_block=None):
    """Top-`kmax` POD basis + full spectrum of the centered training supervision, WITHOUT
    ever holding a second float64 copy of `Ytr`.

    At D = 128**3 a float64 copy of the supervision is 25 GB and `Sc = Ytr - mu` another 25 GB,
    so `spectrum_and_basis`'s `Sc @ Sc.T` OOMs a 33 GB host. This uses the same small left
    Gram (N x N) but centers it algebraically and accumulates in float64 over column blocks,
    so only one float64 block of `Ytr` exists at a time:

        Sc Sc^T[i,j] = G0[i,j] - r[i] - r[j] + <mu,mu> ,  G0 = Ytr Ytr^T ,  r = Ytr mu

    Then V = Sc^T U / sv = (Ytr^T U - mu (1^T U)) / sv, also column-blocked. Verified to match
    `np.linalg.svd` to eig-relerr < 1e-9 and subspace |cos| > 1 - 1e-6 (test_pod_branch.py).
    Only the D > N orientation is provided; the 3D targets always have D >> N.
    """
    N, D = Ytr.shape
    cb = col_block or max(1, D // 16)
    mu = np.zeros(D, np.float64)
    for j in range(0, D, cb):
        mu[j:j + cb] = Ytr[:, j:j + cb].sum(0, dtype=np.float64)
    mu /= N
    G0 = np.zeros((N, N), np.float64)
    r = np.zeros(N, np.float64)
    for j in range(0, D, cb):
        blk = Ytr[:, j:j + cb].astype(np.float64)        # one column block only
        G0 += blk @ blk.T
        r += blk @ mu[j:j + cb]
    G = G0 - r[:, None] - r[None, :] + float(mu @ mu)     # centered Gram (N x N)
    w, U = np.linalg.eigh(G)
    w = np.clip(w[::-1], 0.0, None); U = U[:, ::-1]
    sv = np.sqrt(w)
    k = min(kmax, int((sv > sv[0] * 1e-12).sum()) if sv[0] > 0 else kmax)
    oneU = U[:, :k].sum(0)
    V = np.empty((D, k), np.float32)
    for j in range(0, D, cb):
        blk = Ytr[:, j:j + cb].astype(np.float64)
        Vj = blk.T @ U[:, :k] - np.outer(mu[j:j + cb], oneU)
        V[j:j + cb] = (Vj / np.maximum(sv[:k], 1e-300)).astype(np.float32)
    return w, V.T, mu.astype(np.float32).reshape(1, D)   # (eigs, Vt(k,D), mu(1,D))


class MLP(nn.Module):
    def __init__(self, din, dout, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, dout))

    def forward(self, x):
        return self.net(x)


def shuffle_within_blocks(Y, blocks, shuffle_seed):
    """Negative control (protocol 7d): permute supervision rows INSIDE each split block.

    The control must destroy the input->target correspondence WITHOUT moving supervision across
    the leak-free split boundaries: a single global permutation would place test-block targets into
    the train block, so the control would no longer run on the same split geometry as the real
    arms. Each block gets its own independent RNG stream (`shuffle_seed + block index`), so the
    per-block target marginal is exactly preserved and inputs are untouched.

    blocks: sequence of index arrays (train, selection, test); rows outside every block are left
    as they are (they are unused by the caller).
    """
    Ys = Y.copy()
    for i, idx in enumerate(blocks):
        idx = np.asarray(idx)
        rng = np.random.default_rng(shuffle_seed + i)
        Ys[idx] = Y[idx][rng.permutation(idx.shape[0])]
    return Ys


def write_store(path, scheme, Y, Vt, mu, K, keep_idx=None):
    """`Y` here is the TRAINING supervision only; the held-out rows are never stored.

    Write the supervision as it would actually be stored, and return its size. The compressed
    form carries everything a reader needs: coefficients, the basis, the training mean and the
    retained index set. Omitting any of them would understate it.

    `keep_idx` selects WHICH K modes are retained. When None (the historical default, used by
    every pre-V9 driver) the first K POD modes are kept -- the energy prefix {0..K-1}. When an
    explicit index array is given (V9 budget arm) the retained set is that arbitrary subset,
    e.g. the top-K modes by predictive gain, which need not be an energy prefix. The chosen
    indices are stored in `kept_idx` so a reader reconstructs `C @ Vt[kept_idx] + mu`; the
    index map is |keep_idx| int32s, a negligible addition to the coefficient store.
    """
    if scheme == "dense":
        np.savez(path, Y=Y.astype(np.float32))
    else:
        idx = (np.arange(K, dtype=np.int32) if keep_idx is None
               else np.asarray(keep_idx, dtype=np.int32))
        assert idx.ndim == 1 and len(idx) == K and len(np.unique(idx)) == K, \
            f"keep_idx must be K={K} distinct mode indices, got {idx!r}"
        # no silent clipping: asking for modes the basis does not contain is a contract error
        assert idx.max() < Vt.shape[0], \
            f"requested mode index {int(idx.max())} but basis has only {Vt.shape[0]} rows (K={K})"
        Vk = Vt[idx].astype(np.float32)
        C = ((Y - mu) @ Vk.T).astype(np.float32)
        np.savez(path, C=C, Vk=Vk, mu=mu.astype(np.float32), kept_idx=idx)
    return os.path.getsize(path)


# ----------------------------------------------------------------------------- one arm
def run_single(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = TreeRssSampler().start()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    t_open = time.perf_counter()
    with np.load(a.store) as _z:      # materialise, then close: Windows locks open handles
        store = {k: _z[k] for k in _z.files}
    open_s = time.perf_counter() - t_open
    X = np.load(a.x_path)
    host = {k: torch.from_numpy(v) for k, v in store.items()}
    Xg = torch.from_numpy(X.astype(np.float32)).to(dev)
    dense = "Y" in host
    if dense:
        N, D = host["Y"].shape
    else:
        Vk_g, mu_g = host["Vk"].to(dev), host["mu"].to(dev)
        N, D = host["C"].shape[0], host["Vk"].shape[1]

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    model = MLP(X.shape[1], D).to(dev)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    lossf = nn.MSELoss()
    nb = (N + a.bs - 1) // a.bs
    ep_rows = []

    for ep in range(a.epochs + a.warmup):
        rec_ep = ep >= a.warmup
        t_ep = time.perf_counter()
        acc = {"read": 0.0, "transfer": 0.0, "decode": 0.0, "step": 0.0}
        g = torch.Generator().manual_seed(a.seed * 1000 + ep)
        perm = torch.randperm(N, generator=g)
        model.train()
        for bi in range(nb):
            idx = perm[bi * a.bs:(bi + 1) * a.bs]
            t0 = time.perf_counter()
            cpu_b = host["Y"][idx] if dense else host["C"][idx]
            t1 = time.perf_counter()
            gb = cpu_b.to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            if dense:
                # no reconstruction happens, so the phase costs exactly nothing; timing it
                # would attribute the timer's own overhead to a step that does not exist
                yb, t3 = gb, t2
            else:
                yb = gb @ Vk_g + mu_g
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
            xb = Xg[idx.to(dev)]
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t4 = time.perf_counter()
            if rec_ep:
                acc["read"] += t1 - t0
                acc["transfer"] += t2 - t1
                acc["decode"] += t3 - t2
                acc["step"] += t4 - t3
        if rec_ep:
            acc["epoch"] = time.perf_counter() - t_ep
            ep_rows.append(acc)

    peak_alloc = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0
    peak_resv = int(torch.cuda.max_memory_reserved(dev)) if dev.type == "cuda" else 0
    peak_rss = sampler.stop()

    cold = ep_rows[0]
    steady = ep_rows[1:] or ep_rows
    med = lambda k: float(np.median([r[k] for r in steady]))
    rec = {"scheme": a.scheme, "K": a.K, "seed": a.seed,
           "bytes_on_disk": int(os.path.getsize(a.store)),
           "store_open_seconds": open_s,
           "supervision_bytes_in_host": int(sum(v.nbytes for v in store.values())),
           "peak_host_rss": int(peak_rss),
           "peak_device_allocated": peak_alloc, "peak_device_reserved": peak_resv,
           "cold_epoch_seconds": cold["epoch"],
           "steady_epoch_seconds": med("epoch"),
           "steady_epoch_range": [float(min(r["epoch"] for r in steady)),
                                  float(max(r["epoch"] for r in steady))],
           "n_steady_epochs": len(steady),
           "read_seconds": med("read"), "transfer_seconds": med("transfer"),
           "decode_seconds": med("decode"), "step_seconds": med("step")}
    rec["input_seconds"] = (rec["read_seconds"] + rec["transfer_seconds"]
                            + rec["decode_seconds"])

    # scoring happens only after the memory measurement is closed, and only on rows the
    # model never trained on (E2E_COST_V2 item 7)
    Xho = np.load(a.x_heldout).astype(np.float32)
    Yho = np.load(a.y_heldout)
    Xho_g = torch.from_numpy(Xho).to(dev)
    model.eval()
    with torch.no_grad():
        se = 0.0
        for i in range(0, Xho.shape[0], 512):
            p = model(Xho_g[i:i + 512]).cpu().numpy()
            se += float(((p - Yho[i:i + 512]) ** 2).sum())
    rec["heldout_error"] = se / Yho.size
    rec["n_heldout"] = int(Yho.shape[0])
    print("__JSON__" + json.dumps(rec), flush=True)




def run_single_v4(a):
    """V4: the head width IS the rank, and every arm trains on FIELD MSE. E2E_COST_V4.

    Faithful to the source PODHead3, whose forward reconstructs the K coefficients to the
    field and trains on field MSE. Training all arms in field space keeps the loss identical
    across arms (rule on arm-equivalence): a coefficient-space loss for budget would differ
    from dense's field loss both in space and in scale.

    dense  -> D-output head. forward IS the field. loss = MSE(field, stored field Y).
    budget -> K-output head + FIXED basis V_K. forward = head(x) @ V_K + mu (a field).
              loss = MSE(that field, the stored rank-K field C @ V_K + mu). Both sides pass
              through the same fixed basis, so this is field MSE, matching the source.

    The only trainable D-sized object is dense's final layer (hidden x D), which is what
    exhausts memory at large D; budget's final layer is hidden x K and its basis V_K is a
    fixed buffer (no gradient, no optimiser state). The basis must be resident to
    reconstruct, so a rank so large that V_K itself does not fit is recorded as OOM -- which
    is a faithful outcome, not an artefact.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = TreeRssSampler().start()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    t_open = time.perf_counter()
    with np.load(a.store) as _z:
        store = {k: _z[k] for k in _z.files}
    open_s = time.perf_counter() - t_open
    X = np.load(a.x_path)
    Xg = torch.from_numpy(X.astype(np.float32)).to(dev)
    dense = "Y" in store
    if dense:
        target = torch.from_numpy(store["Y"])          # (N, D) field, trained on directly
        N, out_dim = target.shape
        Vk_g = mu_g = None
    else:
        target = torch.from_numpy(store["C"])          # (N, K) coefficients of the field
        Vk_g = torch.from_numpy(store["Vk"]).to(dev)   # (K, D) FIXED basis, no gradient
        mu_g = torch.from_numpy(store["mu"]).to(dev)   # (1, D)
        N, out_dim = target.shape                       # out_dim == K (the head width)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    model = MLP(X.shape[1], out_dim).to(dev)            # head outputs D (dense) or K (budget)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    lossf = nn.MSELoss()
    nb = (N + a.bs - 1) // a.bs
    ep_rows = []

    for ep in range(a.epochs + a.warmup):
        rec_ep = ep >= a.warmup
        t_ep = time.perf_counter()
        acc = {"read": 0.0, "transfer": 0.0, "decode": 0.0, "step": 0.0}
        g = torch.Generator().manual_seed(a.seed * 1000 + ep)
        perm = torch.randperm(N, generator=g)
        model.train()
        for bi in range(nb):
            idx = perm[bi * a.bs:(bi + 1) * a.bs]
            t0 = time.perf_counter()
            cpu_b = target[idx]
            t1 = time.perf_counter()
            tb = cpu_b.to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            opt.zero_grad()
            if dense:
                yb, t3 = tb, t2                                      # no target decode
            else:
                yb = tb @ Vk_g + mu_g                                # target field decode
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
            pred = model(Xg[idx.to(dev)])
            out_field = pred if dense else pred @ Vk_g + mu_g        # source-faithful field loss
            lossf(out_field, yb).backward()
            opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t4 = time.perf_counter()
            if rec_ep:
                acc["read"] += t1 - t0
                acc["transfer"] += t2 - t1
                acc["decode"] += t3 - t2
                acc["step"] += t4 - t3
        if rec_ep:
            acc["epoch"] = time.perf_counter() - t_ep
            ep_rows.append(acc)

    peak_alloc = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0
    peak_resv = int(torch.cuda.max_memory_reserved(dev)) if dev.type == "cuda" else 0
    peak_rss = sampler.stop()

    cold = ep_rows[0]
    steady = ep_rows[1:] or ep_rows
    med = lambda k: float(np.median([r[k] for r in steady]))
    rec = {"scheme": a.scheme, "K": a.K, "seed": a.seed, "head_out_dim": int(out_dim),
           "optimization_epochs": int(a.epochs + a.warmup),
           "recorded_epochs": int(a.epochs), "warmup_epochs": int(a.warmup),
           "bytes_on_disk": int(os.path.getsize(a.store)),
           "store_open_seconds": open_s,
           "supervision_bytes_in_host": int(sum(v.nbytes for v in store.values())),
           "peak_host_rss": int(peak_rss),
           "peak_device_allocated": peak_alloc, "peak_device_reserved": peak_resv,
           "cold_epoch_seconds": cold["epoch"], "steady_epoch_seconds": med("epoch"),
           "steady_epoch_range": [float(min(r["epoch"] for r in steady)),
                                  float(max(r["epoch"] for r in steady))],
           "n_steady_epochs": len(steady),
           "read_seconds": med("read"), "transfer_seconds": med("transfer"),
           "decode_seconds": med("decode"), "step_seconds": med("step")}
    rec["input_seconds"] = (rec["read_seconds"] + rec["transfer_seconds"]
                            + rec["decode_seconds"])

    # held-out field error: reconstruct to D whatever the head emitted, then compare fields
    Xho = np.load(a.x_heldout).astype(np.float32)
    Yho = np.load(a.y_heldout)                          # (M, D) raw field
    Xho_g = torch.from_numpy(Xho).to(dev)
    model.eval()
    with torch.no_grad():
        se = 0.0
        for i in range(0, Xho.shape[0], 512):
            out = model(Xho_g[i:i + 512])
            if dense:
                pred = out.cpu().numpy()
            else:
                pred = (out @ Vk_g + mu_g).cpu().numpy()  # lift K coeffs to the field
            se += float(((pred - Yho[i:i + 512]) ** 2).sum())
    rec["heldout_error"] = se / Yho.size
    rec["n_heldout"] = int(Yho.shape[0])
    print("__JSON__" + json.dumps(rec), flush=True)


class PODBranchND(nn.Module):
    """The source PODDeepONet branch (published_heads.py, arXiv:2111.05512 sec 3.1.4),
    generalised from 2 to n spatial dimensions so the SAME architecture serves 2D and 3D.

    A Conv stack whose parameters are D-independent (channel widths are fixed; only the
    spatial extent of the activations grows with D), then AdaptiveAvgPool to (B, 4*width),
    then a single Linear whose output width is the head:

        dense  arm: out_dim = D  -> the head IS the field (a trainable D-sized layer)
        budget arm: out_dim = K  -> K coefficients, lifted to the field by a FIXED basis

    That final Linear is the only layer whose width scales, which is exactly the paper's
    claim -- the rank is the width of the output head. The input field enters as a D-sized
    ACTIVATION (reshaped to (B, cin, *spatial)), never as a trainable D-sized layer, so the
    branch itself costs the same at every rank.
    """

    def __init__(self, spatial, cin, out_dim, width=32):
        super().__init__()
        self.spatial = tuple(int(s) for s in spatial)
        self.cin = int(cin)
        nd = len(self.spatial)
        Conv = getattr(nn, f"Conv{nd}d")
        Pool = getattr(nn, f"AdaptiveAvgPool{nd}d")
        c = width
        self.branch = nn.Sequential(
            Conv(cin, c, 3, stride=2, padding=1), nn.ReLU(),
            Conv(c, 2 * c, 3, stride=2, padding=1), nn.ReLU(),
            Conv(2 * c, 4 * c, 3, stride=2, padding=1), nn.ReLU(),
            Conv(4 * c, 4 * c, 3, stride=2, padding=1), nn.ReLU(),
            Pool(1), nn.Flatten(), nn.Linear(4 * c, out_dim))

    def forward(self, x):
        # x: (B, D) flattened field -> (B, cin, *spatial); D == cin * prod(spatial)
        b = x.shape[0]
        return self.branch(x.reshape(b, self.cin, *self.spatial))


def run_single_v5(a):
    """V5: the source PODDeepONet CNN branch (D-independent params) replaces the MLP, so the
    ONLY layer whose width scales is the output head. E2E_COST_V5.

    Identical fidelity rules to V4 (field MSE for every arm, fixed basis for budget/energy),
    but the backbone is now 3D-scalable: an MLP's input layer nn.Linear(D, hidden) is itself
    D-sized and OOMs every arm at D=128**3; the Conv branch has no such layer.

    Every D-sized object is streamed a minibatch at a time (the input field, the target
    field, the held-out field), so the arm never materialises the full N x D input on the
    device -- another way the MLP path silently OOM'd at large D.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = TreeRssSampler().start()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    spatial = tuple(int(s) for s in a.spatial.split(","))
    cin = getattr(a, "cin", 1)

    t_open = time.perf_counter()
    with np.load(a.store) as _z:
        store = {k: _z[k] for k in _z.files}
    open_s = time.perf_counter() - t_open
    # mmap the input: at D=128**3 the full training input is 15-38 GB, larger than host RAM.
    # Only the minibatch is ever copied into memory (and to the device).
    Xnp = np.load(a.x_path, mmap_mode="r")                        # (N, D) on disk, streamed
    assert Xnp.shape[1] == cin * int(np.prod(spatial)), (
        f"input width {Xnp.shape[1]} != cin*prod(spatial) {cin * int(np.prod(spatial))}")
    dense = "Y" in store
    if dense:
        target = torch.from_numpy(store["Y"])          # (N, D) field, trained on directly
        N, out_dim = target.shape
        Vk_g = mu_g = None
    else:
        target = torch.from_numpy(store["C"])          # (N, K) coefficients of the field
        Vk_g = torch.from_numpy(store["Vk"]).to(dev)   # (K, D) FIXED basis, no gradient
        mu_g = torch.from_numpy(store["mu"]).to(dev)   # (1, D)
        N, out_dim = target.shape                       # out_dim == K (the head width)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    model = PODBranchND(spatial, cin, out_dim).to(dev)  # head outputs D (dense) or K (budget)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    lossf = nn.MSELoss()
    nb = (N + a.bs - 1) // a.bs
    ep_rows = []

    for ep in range(a.epochs + a.warmup):
        rec_ep = ep >= a.warmup
        t_ep = time.perf_counter()
        acc = {"read": 0.0, "transfer": 0.0, "step": 0.0}
        g = torch.Generator().manual_seed(a.seed * 1000 + ep)
        perm = torch.randperm(N, generator=g)
        model.train()
        for bi in range(nb):
            idx = perm[bi * a.bs:(bi + 1) * a.bs]
            t0 = time.perf_counter()
            cpu_t = target[idx]
            cpu_x = torch.from_numpy(np.ascontiguousarray(Xnp[idx.numpy()]).astype(np.float32))
            t1 = time.perf_counter()
            tb, xb = cpu_t.to(dev), cpu_x.to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            opt.zero_grad()
            pred = model(xb)
            if dense:
                yb, out_field = tb, pred                              # already the field
            else:
                # reconstruct BOTH sides through the fixed basis: field MSE, source-faithful
                yb = tb @ Vk_g + mu_g                                # target field
                out_field = pred @ Vk_g + mu_g                       # predicted field
            lossf(out_field, yb).backward()
            opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            if rec_ep:
                acc["read"] += t1 - t0
                acc["transfer"] += t2 - t1
                acc["step"] += t3 - t2
        if rec_ep:
            acc["epoch"] = time.perf_counter() - t_ep
            ep_rows.append(acc)

    peak_alloc = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0
    peak_resv = int(torch.cuda.max_memory_reserved(dev)) if dev.type == "cuda" else 0
    peak_rss = sampler.stop()

    cold = ep_rows[0]
    steady = ep_rows[1:] or ep_rows
    med = lambda k: float(np.median([r[k] for r in steady]))
    rec = {"scheme": a.scheme, "K": a.K, "seed": a.seed, "head_out_dim": int(out_dim),
           "backbone": "pod_branch_nd", "spatial": list(spatial), "cin": int(cin),
           "bytes_on_disk": int(os.path.getsize(a.store)),
           "store_open_seconds": open_s,
           "supervision_bytes_in_host": int(sum(v.nbytes for v in store.values())),
           "peak_host_rss": int(peak_rss),
           "peak_device_allocated": peak_alloc, "peak_device_reserved": peak_resv,
           "cold_epoch_seconds": cold["epoch"], "steady_epoch_seconds": med("epoch"),
           "steady_epoch_range": [float(min(r["epoch"] for r in steady)),
                                  float(max(r["epoch"] for r in steady))],
           "n_steady_epochs": len(steady),
           "read_seconds": med("read"), "transfer_seconds": med("transfer"),
           "step_seconds": med("step")}
    rec["input_seconds"] = rec["read_seconds"] + rec["transfer_seconds"]

    # held-out field error: reconstruct to D whatever the head emitted, then compare fields.
    # Streamed a chunk at a time so the held-out input is never fully resident on the device.
    Xho = np.load(a.x_heldout, mmap_mode="r")           # (M, D) on disk, streamed in chunks
    Yho = np.load(a.y_heldout, mmap_mode="r")           # (M, D) raw field, streamed
    model.eval()
    with torch.no_grad():
        se = 0.0
        for i in range(0, Xho.shape[0], 128):
            xb = torch.from_numpy(np.ascontiguousarray(Xho[i:i + 128]).astype(np.float32))
            out = model(xb.to(dev))
            pred = out.cpu().numpy() if dense else (out @ Vk_g + mu_g).cpu().numpy()
            se += float(((pred - np.asarray(Yho[i:i + 128])) ** 2).sum())
    rec["heldout_error"] = se / Yho.size
    rec["n_heldout"] = int(Yho.shape[0])
    print("__JSON__" + json.dumps(rec), flush=True)


class UNet3D(nn.Module):
    """A D-independent, FIELD-OUTPUT surrogate: a one-level U-Net (down/up with a skip).

    Direct measurement (this project, 2026-07-23) established that the POD-coefficient head
    (PODBranchND) cannot predict 3D field->field maps at all -- 0% variance explained on
    next-step turbulence, on a cross-channel operator, and even on a deterministic smooth
    elliptic operator (3D Poisson) -- because predicting K global POD coefficients through a
    pooled/compressed head is intractable. This field predictor learns the same maps (+27%
    on turb64 next-step, +66% on 3D Poisson), so the predictable rank can be read from a
    surrogate that actually predicts. Parameters are D-independent (fixed channel widths;
    only activation extent grows with D). The spatial size must be even (one stride-2 level).
    """

    def __init__(self, spatial, cin=1, width=16):
        super().__init__()
        self.spatial = tuple(int(s) for s in spatial)
        self.cin = int(cin)
        nd = len(self.spatial)
        Conv = getattr(nn, f"Conv{nd}d")
        ConvT = getattr(nn, f"ConvTranspose{nd}d")
        w = width
        self.in_conv = nn.Sequential(Conv(cin, w, 3, 1, 1), nn.GELU())
        self.down = nn.Sequential(Conv(w, w, 3, 2, 1), nn.GELU(), Conv(w, w, 3, 1, 1), nn.GELU())
        self.up = ConvT(w, w, 4, 2, 1)
        self.out_conv = nn.Sequential(Conv(2 * w, w, 3, 1, 1), nn.GELU(), Conv(w, cin, 3, 1, 1))

    def forward(self, x):
        b = x.shape[0]
        h0 = self.in_conv(x.reshape(b, self.cin, *self.spatial))
        u = self.up(self.down(h0))
        return self.out_conv(torch.cat([u, h0], dim=1)).reshape(b, -1)


def run_single_v6(a):
    """V6: a FIELD-OUTPUT U-Net surrogate; the rank lives in the SUPERVISION, not the head.

    The head-width feasibility of V5 is unreachable in 3D (the POD-coefficient head predicts
    nothing there), so V6 keeps the honest, measurable claim: read the predictable rank once
    from a surrogate that actually predicts the field, then train against rank-K SUPERVISION
    and measure the accuracy it costs.

        dense  -> supervision is the full field Y. loss = MSE(unet(X), Y).
        budget -> supervision is the rank-K field  C @ V_K + mu. loss = MSE(unet(X), that).
        energy -> as budget at the 99.9% reconstructable rank.

    Every arm uses the identical U-Net (same size, same seed init); only the target field's
    rank differs, so device memory is the same across arms (there is no feasibility divide,
    and none is claimed). Held-out error is always scored against the TRUE field Y. Every
    D-sized array is streamed from a memmap.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = TreeRssSampler().start()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    spatial = tuple(int(s) for s in a.spatial.split(","))
    cin = getattr(a, "cin", 1)

    t_open = time.perf_counter()
    with np.load(a.store) as _z:
        store = {k: _z[k] for k in _z.files}
    open_s = time.perf_counter() - t_open
    Xnp = np.load(a.x_path, mmap_mode="r")               # streamed input field
    assert Xnp.shape[1] == cin * int(np.prod(spatial)), (
        f"input width {Xnp.shape[1]} != cin*prod(spatial) {cin * int(np.prod(spatial))}")
    dense = "Y" in store
    if dense:
        target = torch.from_numpy(store["Y"])            # (N, D) full field supervision
        N = target.shape[0]
        Vk_g = mu_g = None
        sup_rank = target.shape[1]
    else:
        target = torch.from_numpy(store["C"])            # (N, K) coefficients of the field
        Vk_g = torch.from_numpy(store["Vk"]).to(dev)     # (K, D) FIXED basis, no gradient
        mu_g = torch.from_numpy(store["mu"]).to(dev)     # (1, D)
        N = target.shape[0]
        sup_rank = target.shape[1]

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    model = UNet3D(spatial, cin).to(dev)                 # field output, same size for all arms
    opt = torch.optim.Adam(model.parameters(), a.lr)
    lossf = nn.MSELoss()
    nb = (N + a.bs - 1) // a.bs
    ep_rows = []

    for ep in range(a.epochs + a.warmup):
        rec_ep = ep >= a.warmup
        t_ep = time.perf_counter()
        acc = {"read": 0.0, "transfer": 0.0, "step": 0.0}
        g = torch.Generator().manual_seed(a.seed * 1000 + ep)
        perm = torch.randperm(N, generator=g)
        model.train()
        for bi in range(nb):
            idx = perm[bi * a.bs:(bi + 1) * a.bs]
            t0 = time.perf_counter()
            cpu_t = target[idx]
            cpu_x = torch.from_numpy(np.ascontiguousarray(Xnp[idx.numpy()]).astype(np.float32))
            t1 = time.perf_counter()
            tb, xb = cpu_t.to(dev), cpu_x.to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            opt.zero_grad()
            # the target field is the full field (dense) or the rank-K field (budget/energy);
            # the model always predicts a field, so the loss is field MSE for every arm
            yb = tb if dense else tb @ Vk_g + mu_g
            lossf(model(xb), yb).backward()
            opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            if rec_ep:
                acc["read"] += t1 - t0
                acc["transfer"] += t2 - t1
                acc["step"] += t3 - t2
        if rec_ep:
            acc["epoch"] = time.perf_counter() - t_ep
            ep_rows.append(acc)

    peak_alloc = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0
    peak_resv = int(torch.cuda.max_memory_reserved(dev)) if dev.type == "cuda" else 0
    peak_rss = sampler.stop()

    cold = ep_rows[0]
    steady = ep_rows[1:] or ep_rows
    med = lambda k: float(np.median([r[k] for r in steady]))
    rec = {"scheme": a.scheme, "K": a.K, "seed": a.seed, "supervision_rank": int(sup_rank),
           "backbone": "unet3d", "head_out_dim": int(cin * int(np.prod(spatial))),
           "spatial": list(spatial), "cin": int(cin),
           "bytes_on_disk": int(os.path.getsize(a.store)),
           "store_open_seconds": open_s,
           "supervision_bytes_in_host": int(sum(v.nbytes for v in store.values())),
           "peak_host_rss": int(peak_rss),
           "peak_device_allocated": peak_alloc, "peak_device_reserved": peak_resv,
           "cold_epoch_seconds": cold["epoch"], "steady_epoch_seconds": med("epoch"),
           "steady_epoch_range": [float(min(r["epoch"] for r in steady)),
                                  float(max(r["epoch"] for r in steady))],
           "n_steady_epochs": len(steady),
           "read_seconds": med("read"), "transfer_seconds": med("transfer"),
           "step_seconds": med("step")}
    rec["input_seconds"] = rec["read_seconds"] + rec["transfer_seconds"]

    # held-out error is ALWAYS scored against the true field Y (never the rank-K target):
    # we want the real accuracy cost of compressing the supervision, streamed a chunk at a time
    Xho = np.load(a.x_heldout, mmap_mode="r")
    Yho = np.load(a.y_heldout, mmap_mode="r")            # (M, D) raw field
    model.eval()
    with torch.no_grad():
        se = 0.0
        for i in range(0, Xho.shape[0], 128):
            xb = torch.from_numpy(np.ascontiguousarray(Xho[i:i + 128]).astype(np.float32))
            pred = model(xb.to(dev)).cpu().numpy()
            se += float(((pred - np.asarray(Yho[i:i + 128])) ** 2).sum())
    rec["heldout_error"] = se / Yho.size
    rec["n_heldout"] = int(Yho.shape[0])
    print("__JSON__" + json.dumps(rec), flush=True)


def run_single_v8(a):
    """V8: the I/O-bound regime. Identical to run_single_v6 (field-output U-Net, rank in the
    supervision) EXCEPT the dense arm's supervision is a raw .npy memmap that is STREAMED a
    minibatch at a time, never held in host RAM. On a target whose full-field supervision is
    larger than host RAM (turb64 stacked, ~36 GB > 33 GB), the OS page cache cannot hold it,
    so every epoch re-reads it from disk -- a genuine I/O-bound regime. The compressed
    budget/energy supervision fits the cache and is read once. `read_seconds` (already split
    out per epoch) is what carries the I/O the compression saves; this is a MEASUREMENT of the
    I/O-bound payoff, not a projection.

    Dense store: a raw `.npy` path (mmap). Budget/energy store: a `.npz` (compressed, small).
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = TreeRssSampler().start()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    spatial = tuple(int(s) for s in a.spatial.split(","))
    cin = getattr(a, "cin", 1)

    dense = a.store.endswith(".npy")
    t_open = time.perf_counter()
    if dense:
        target = np.load(a.store, mmap_mode="r")         # (N, D) full field, STREAMED per batch
        N = target.shape[0]
        Vk_g = mu_g = None
        sup_rank = target.shape[1]
        supervision_bytes = int(os.path.getsize(a.store))
    else:
        with np.load(a.store) as _z:
            store = {k: _z[k] for k in _z.files}
        target = torch.from_numpy(store["C"])            # (N, K) coefficients, fits RAM/cache
        Vk_g = torch.from_numpy(store["Vk"]).to(dev)
        mu_g = torch.from_numpy(store["mu"]).to(dev)
        N = target.shape[0]
        sup_rank = target.shape[1]
        supervision_bytes = int(sum(v.nbytes for v in store.values()))
    open_s = time.perf_counter() - t_open
    Xnp = np.load(a.x_path, mmap_mode="r")               # input always streamed
    assert Xnp.shape[1] == cin * int(np.prod(spatial))

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    model = UNet3D(spatial, cin).to(dev)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    lossf = nn.MSELoss()
    nb = (N + a.bs - 1) // a.bs
    ep_rows = []

    proc = psutil.Process()   # process-wide I/O counters, to attribute the read-time gap to bytes
    for ep in range(a.epochs + a.warmup):
        rec_ep = ep >= a.warmup
        t_ep = time.perf_counter()
        try:
            io0 = proc.io_counters(); mf0 = proc.memory_info()
        except Exception:
            io0 = mf0 = None
        acc = {"read": 0.0, "transfer": 0.0, "step": 0.0}
        g = torch.Generator().manual_seed(a.seed * 1000 + ep)
        perm = torch.randperm(N, generator=g)
        model.train()
        for bi in range(nb):
            idx = perm[bi * a.bs:(bi + 1) * a.bs]
            t0 = time.perf_counter()
            if dense:
                # the I/O the compression saves is HERE: a cold read of the full-field target
                cpu_t = torch.from_numpy(
                    np.ascontiguousarray(target[idx.numpy()]).astype(np.float32))
            else:
                cpu_t = target[idx]
            cpu_x = torch.from_numpy(np.ascontiguousarray(Xnp[idx.numpy()]).astype(np.float32))
            t1 = time.perf_counter()
            tb, xb = cpu_t.to(dev), cpu_x.to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            opt.zero_grad()
            yb = tb if dense else tb @ Vk_g + mu_g
            lossf(model(xb), yb).backward()
            opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            if rec_ep:
                acc["read"] += t1 - t0
                acc["transfer"] += t2 - t1
                acc["step"] += t3 - t2
        if rec_ep:
            acc["epoch"] = time.perf_counter() - t_ep
            if io0 is not None:
                io1, mf1 = proc.io_counters(), proc.memory_info()
                acc["read_bytes"] = int(io1.read_bytes - io0.read_bytes)
                acc["read_count"] = int(io1.read_count - io0.read_count)
                acc["page_faults"] = int(getattr(mf1, "num_page_faults", 0)
                                         - getattr(mf0, "num_page_faults", 0))
            ep_rows.append(acc)

    peak_alloc = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0
    peak_resv = int(torch.cuda.max_memory_reserved(dev)) if dev.type == "cuda" else 0
    peak_rss = sampler.stop()

    cold = ep_rows[0]
    steady = ep_rows[1:] or ep_rows
    med = lambda k: float(np.median([r[k] for r in steady]))
    rec = {"scheme": a.scheme, "K": a.K, "seed": a.seed, "supervision_rank": int(sup_rank),
           "backbone": "unet3d", "head_out_dim": int(cin * int(np.prod(spatial))),
           "spatial": list(spatial), "cin": int(cin),
           "bytes_on_disk": int(os.path.getsize(a.store)),
           "store_open_seconds": open_s, "supervision_bytes_in_host": supervision_bytes,
           "peak_host_rss": int(peak_rss),
           "peak_device_allocated": peak_alloc, "peak_device_reserved": peak_resv,
           "cold_epoch_seconds": cold["epoch"], "steady_epoch_seconds": med("epoch"),
           "steady_epoch_range": [float(min(r["epoch"] for r in steady)),
                                  float(max(r["epoch"] for r in steady))],
           "n_steady_epochs": len(steady),
           "read_seconds": med("read"), "transfer_seconds": med("transfer"),
           "step_seconds": med("step")}
    rec["input_seconds"] = rec["read_seconds"] + rec["transfer_seconds"]
    if "read_bytes" in steady[0]:   # process-wide I/O attribution (io instrumentation on)
        rec["read_bytes_median"] = float(np.median([r["read_bytes"] for r in steady]))
        rec["read_count_median"] = float(np.median([r["read_count"] for r in steady]))
        rec["page_faults_median"] = float(np.median([r["page_faults"] for r in steady]))

    Xho = np.load(a.x_heldout, mmap_mode="r")
    Yho = np.load(a.y_heldout, mmap_mode="r")
    model.eval()
    with torch.no_grad():
        se = 0.0
        for i in range(0, Xho.shape[0], 128):
            xb = torch.from_numpy(np.ascontiguousarray(Xho[i:i + 128]).astype(np.float32))
            pred = model(xb.to(dev)).cpu().numpy()
            se += float(((pred - np.asarray(Yho[i:i + 128])) ** 2).sum())
    rec["heldout_error"] = se / Yho.size
    rec["n_heldout"] = int(Yho.shape[0])
    print("__JSON__" + json.dumps(rec), flush=True)


def summarise(rows, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in rows]
        out[k] = {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}
    return out


def build_system_arm_command(a, scheme, K, seed, store, x_path, y_path,
                             x_heldout, y_heldout, python=None, driver=None):
    """Build the exact child command used by the system-cost parent (shell-free and testable)."""
    cmd = [python or sys.executable, driver or os.path.abspath(__file__), "--single"]
    if getattr(a, "rank_contract", "legacy") in ("v11", "v12"):
        cmd.append("--v4")
    cmd += ["--scheme", scheme, "--K", str(K), "--seed", str(seed),
            "--store", store, "--x_path", x_path, "--y_path", y_path,
            "--x_heldout", x_heldout, "--y_heldout", y_heldout,
            "--epochs", str(a.epochs), "--warmup", str(a.warmup),
            "--bs", str(a.bs), "--lr", str(a.lr)]
    return cmd


def validate_system_head_dim(row, scheme, K, D, rank_contract):
    """V11 hard gate: dense head is D; every compressed supervision head is exactly K."""
    if rank_contract not in ("v11", "v12"):
        return
    expected = D if scheme == "dense" else K
    assert int(row.get("head_out_dim", -1)) == int(expected), \
        (scheme, row.get("seed"), row.get("head_out_dim"), expected)


# ------------------------------------------------------------------------------ parent
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--v4", action="store_true")
    ap.add_argument("--v5", action="store_true")
    ap.add_argument("--v6", action="store_true")
    ap.add_argument("--v8", action="store_true")
    ap.add_argument("--spatial", help="comma-separated spatial shape, e.g. 128,128,128")
    ap.add_argument("--cin", type=int, default=1)
    ap.add_argument("--scheme")
    ap.add_argument("--K", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--store")
    ap.add_argument("--x_path")
    ap.add_argument("--y_path")
    ap.add_argument("--x_heldout")
    ap.add_argument("--y_heldout")
    ap.add_argument("--ns2d_root", default="../ns2d")
    ap.add_argument("--ts_csv", default="../ts/electricity/electricity.csv")
    ap.add_argument("--traj", type=int, default=800)
    ap.add_argument("--ns2d_pairing", choices=["roll", "traj"], default="roll",
                    help="roll = frozen V2 global-roll pairing; traj = source-paper "
                         "per-trajectory pairing (protocol 7c)")
    ap.add_argument("--ts_windows", type=int, default=2000)
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ns2d_budget", type=int, default=48)
    ap.add_argument("--ns2d_oracle", type=int, default=48)
    ap.add_argument("--ts_budget", type=int, default=24)
    ap.add_argument("--ts_oracle", type=int, default=24)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--ns2d_rank_source", default=None,
                    help="protocol 7d(ii): read --ns2d_budget/--ns2d_oracle from this calibration "
                         "JSON (K_pred on selection / K_oracle_ladder on test) instead of the CLI")
    ap.add_argument("--ts_rank_source", default=None,
                    help="V10: calibration JSON carrying electricity K_pred and exact gain_order")
    ap.add_argument("--rank_contract", choices=["legacy", "v10", "v11", "v12"], default="legacy",
                    help="v10 requires exact selector indices; v11/v12 additionally require K heads")
    ap.add_argument("--protocol_id", default=PROTOCOL_ID,
                    help="artifact protocol ID; V10 runs must set this explicitly")
    ap.add_argument("--provenance_code", nargs="*", default=[],
                    help="V10 executed code files whose current sha256 is stored")
    ap.add_argument("--provenance_data", nargs="*", default=[],
                    help="V10 input data files whose current sha256 is stored")
    ap.add_argument("--split_mode", choices=["2way", "v9_3way"], default="2way",
                    help="2way = frozen V2 contiguous train_frac split; v9_3way = the V9 leak-free "
                         "3-way geometry with train=block1, held-out=test block (protocol 7d(ii))")
    ap.add_argument("--fracs", type=float, nargs=3, default=[0.6, 0.2, 0.2],
                    help="used only when --split_mode v9_3way")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--boot_seed", type=int, default=20260722)
    ap.add_argument("--store_dir", default="store_tmp")
    ap.add_argument("--protocol", default="protocols/E2E_COST_V2.md")
    ap.add_argument("--out", default="results/e2e_cost_v2/e2e_cost.json")
    a = ap.parse_args()
    if a.single:
        if getattr(a, "v8", False):
            return run_single_v8(a)
        if getattr(a, "v6", False):
            return run_single_v6(a)
        if getattr(a, "v5", False):
            return run_single_v5(a)
        return run_single_v4(a) if getattr(a, "v4", False) else run_single(a)

    os.makedirs(a.store_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    targets = {}
    rank_selections = {}
    X2, Y2 = ns2d_pairs(a.ns2d_root, a.traj, getattr(a, "ns2d_pairing", "roll"))
    rank_provenance = {}
    ns2d_kb, ns2d_ks = a.ns2d_budget, a.ns2d_oracle
    if a.ns2d_rank_source:          # 7d(ii): ranks come from the matching calibration run
        if a.rank_contract in ("v10", "v11"):
            sel = rank_selection_from_calibration_v10(a.ns2d_rank_source, "ns2d")
            ns2d_kb, ns2d_ks = sel["K_pred"], sel["K_oracle_ladder"]
            rank_selections["ns2d"] = sel
            rank_provenance["ns2d"] = sel["provenance"]
        else:
            ns2d_kb, ns2d_ks, rank_provenance["ns2d"] = ranks_from_calibration(
                a.ns2d_rank_source, "ns2d")
        print(f"[ns2d] ranks from {a.ns2d_rank_source}: budget K_pred={ns2d_kb} "
              f"(selection), oracle K_oracle_ladder={ns2d_ks} (test, diagnostic only)", flush=True)
    targets["ns2d"] = (X2, Y2, ns2d_kb, ns2d_ks)
    Xt, Yt = load_ts(a.ts_csv, a.L, a.H, a.ts_windows, 4)
    ts_kb, ts_ks = a.ts_budget, a.ts_oracle
    if a.ts_rank_source:
        if a.rank_contract in ("v10", "v11"):
            sel = rank_selection_from_calibration_v10(a.ts_rank_source, "electricity")
            ts_kb, ts_ks = sel["K_pred"], sel["K_oracle_ladder"]
            rank_selections["electricity"] = sel
            rank_provenance["electricity"] = sel["provenance"]
        else:
            ts_kb, ts_ks, rank_provenance["electricity"] = ranks_from_calibration(
                a.ts_rank_source, "electricity")
    targets["electricity"] = (Xt, Yt, ts_kb, ts_ks)
    if a.rank_contract in ("v10", "v11"):
        assert a.protocol_id.startswith(f"E2E_COST_{a.rank_contract.upper()}"), a.protocol_id
        assert a.split_mode == "v9_3way", "V10 system cost requires the three-way split"
        assert a.ns2d_pairing == "traj", "V10 NS2D requires per-trajectory pairing"
        assert set(rank_selections) == set(targets), \
            f"V10 requires calibration index provenance for every target: {set(targets)-set(rank_selections)}"
        assert a.provenance_code and a.provenance_data, \
            "V10 requires explicit executed code and input data paths"

    KEYS = ["bytes_on_disk", "supervision_bytes_in_host", "peak_host_rss",
            "peak_device_allocated", "peak_device_reserved", "cold_epoch_seconds",
            "steady_epoch_seconds", "read_seconds", "transfer_seconds",
            "decode_seconds", "step_seconds", "input_seconds", "heldout_error"]
    per_target, failures = {}, []
    splits = {}
    for name, (X, Y, kb, ks) in targets.items():
        if a.split_mode == "v9_3way":
            from e2e_cost_v9_2d import boundary_embargo
            emb = boundary_embargo(a, name, Y.shape[0])
            tr, ho, unused_sel = make_split_v9_3way(name, Y.shape[0], a.fracs, emb)
            print(f"[{name}] split_mode=v9_3way embargo={emb} "
                  f"(selection block of {len(unused_sel)} rows not used here)", flush=True)
            splits[name] = split_digest3(tr, unused_sel, ho)
        else:
            tr, ho = make_split(Y.shape[0], a.train_frac)
            splits[name] = split_digest(tr, ho)
        print(f"[{name}] {Y.shape} fp32 supervision {Y.nbytes/1e6:.0f} MB | "
              f"train {len(tr)} / held-out {len(ho)}", flush=True)
        # basis, mean and energy rank come from TRAINING rows only (item 7)
        S = Y[tr].astype(np.float64)
        mu = S.mean(0, keepdims=True)
        t0 = time.perf_counter()
        ev, Vt = spectrum_and_basis(S - mu, min(S.shape))
        k_energy = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.999) + 1)
        print(f"[{name}] spectrum {time.perf_counter()-t0:.0f}s | 99.9% rank {k_energy} "
              f"(training rows only)", flush=True)
        Ytr, Xtr = Y[tr], X[tr]
        xp = os.path.join(a.store_dir, f"{name}_Xtr.npy")
        yp = os.path.join(a.store_dir, f"{name}_Ytr.npy")
        xh = os.path.join(a.store_dir, f"{name}_Xho.npy")
        yh = os.path.join(a.store_dir, f"{name}_Yho.npy")
        np.save(xp, Xtr.astype(np.float32))
        np.save(yp, Ytr.astype(np.float32))
        np.save(xh, X[ho].astype(np.float32))
        np.save(yh, Y[ho].astype(np.float32))

        schemes = [("dense", Ytr.shape[1]), ("budget", kb)]
        if ks != kb:
            schemes.append(("oracle", ks))
        if k_energy < Ytr.shape[1]:
            schemes.append(("energy", k_energy))
        cell = {}
        for sch, K in schemes:
            sp = os.path.join(a.store_dir, f"{name}_{sch}.npz")
            keep_idx = kept_idx_for_system_scheme(
                sch, K, rank_selections.get(name)) if a.rank_contract in ("v10", "v11") else None
            write_store(sp, sch, Ytr, Vt, mu, K, keep_idx=keep_idx)
            stored_idx_sha256 = None
            if sch != "dense":
                with np.load(sp) as z:
                    stored_idx = np.asarray(z["kept_idx"], dtype=np.int32)
                if keep_idx is not None:
                    assert np.array_equal(stored_idx, keep_idx), (name, sch, stored_idx, keep_idx)
                stored_idx_sha256 = sha256_int32(stored_idx)
            rows = []
            for seed in a.seeds:
                cmd = build_system_arm_command(a, sch, K, seed, sp, xp, yp, xh, yh)
                r = subprocess.run(cmd, capture_output=True, text=True)
                line = [ln for ln in r.stdout.splitlines() if ln.startswith("__JSON__")]
                if not line:
                    failures.append({"target": name, "scheme": sch, "seed": seed,
                                     "stderr": r.stderr[-400:]})
                    print(f"  [{name}/{sch}/{seed}] FAILED: {r.stderr.strip()[-160:]}",
                          flush=True)
                    continue
                rec = json.loads(line[-1][len("__JSON__"):])
                validate_system_head_dim(rec, sch, K, Ytr.shape[1], a.rank_contract)
                rows.append(rec)
                print(f"  [{name}/{sch:7s} s{seed}] disk {rec['bytes_on_disk']/1e6:7.1f}MB "
                      f"rss {rec['peak_host_rss']/1e9:5.2f}GB "
                      f"vram {rec['peak_device_reserved']/1e6:6.0f}MB "
                      f"read {rec['read_seconds']:5.2f} tx {rec['transfer_seconds']:5.2f} "
                      f"dec {rec['decode_seconds']:5.2f} step {rec['step_seconds']:5.2f} "
                      f"cold {rec['cold_epoch_seconds']:5.2f} steady "
                      f"{rec['steady_epoch_seconds']:5.2f} err {rec['heldout_error']:.4e}",
                      flush=True)
            if rows:
                cell[sch] = {"K": K, "rows": rows, "n_ok": len(rows),
                             "expected_seeds": list(a.seeds),
                             "kept_idx_sha256": stored_idx_sha256,
                             **summarise(rows, KEYS)}
            os.remove(sp)
        cell["_notes"] = {"K_budget": kb, "K_oracle": ks, "K_energy_999": k_energy,
                          "oracle_collapsed_into_budget": bool(ks == kb)}
        cell["_split"] = splits[name]
        per_target[name] = cell
        for f in (xp, yp, xh, yh):
            os.remove(f)

    ratios = {}
    for name, cell in per_target.items():
        if "dense" not in cell or "budget" not in cell:
            continue
        d, b = cell["dense"], cell["budget"]
        ratios[name] = {k: (d[k]["median"] / b[k]["median"] if b[k]["median"] else None)
                        for k in ("bytes_on_disk", "supervision_bytes_in_host",
                                  "peak_host_rss", "input_seconds",
                                  "cold_epoch_seconds", "steady_epoch_seconds")}
        ratios[name]["error_excess_budget_vs_dense"] = (
            b["heldout_error"]["median"] / d["heldout_error"]["median"] - 1.0)

    # item 19: a speed-up is only called one when the seed-level bootstrap interval for
    # the ratio excludes 1. Seeds are the resampling unit; 10,000 replicates.
    rng = np.random.default_rng(a.boot_seed)
    boot = {}
    for name in ratios:
        d_rows = per_target[name]["dense"]["rows"]
        b_rows = per_target[name]["budget"]["rows"]
        n = min(len(d_rows), len(b_rows))
        cell = {}
        for key in ("steady_epoch_seconds", "input_seconds", "peak_host_rss",
                    "bytes_on_disk"):
            dv = np.array([r[key] for r in d_rows[:n]], dtype=float)
            bv = np.array([r[key] for r in b_rows[:n]], dtype=float)
            ratio = float(np.median(dv) / max(np.median(bv), 1e-30))
            if n < 2:
                # one seed resamples one value, so the interval collapses to a point and
                # would read as a certainty it is not. Report it as not computable.
                cell[key] = {"ratio": ratio, "ci_lo": None, "ci_hi": None,
                             "excludes_one": None,
                             "note": f"only {n} seed(s); no interval is computable"}
                continue
            # paired: the same resampled seed indices are used for both arms
            idx = rng.integers(0, n, size=(a.n_boot, n))
            reps = np.median(dv[idx], axis=1) / np.maximum(np.median(bv[idx], axis=1), 1e-30)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            cell[key] = {"ratio": ratio, "ci_lo": float(lo), "ci_hi": float(hi),
                         "n_seeds": int(n), "n_boot": int(a.n_boot),
                         "excludes_one": bool(lo > 1.0 or hi < 1.0)}
        boot[name] = cell

    h0a = (all((r.get("peak_host_rss") or 0) < 1.05 and
               (r.get("steady_epoch_seconds") or 0) < 1.05 for r in ratios.values())
           if ratios else False)
    h0b = any(per_target[n]["budget"]["input_seconds"]["median"] >
              per_target[n]["dense"]["input_seconds"]["median"] for n in ratios)
    # item 2, H0-c: the budget rank must stay within tolerance on held-out rows
    h0c = {n: bool(per_target[n]["budget"]["heldout_error"]["median"] >
                   (1 + a.tau) * per_target[n]["dense"]["heldout_error"]["median"])
           for n in ratios}

    out = {"protocol_id": a.protocol_id, "protocol_sha256": sha256_of_file(a.protocol),
           "code_sha256": {os.path.abspath(p): sha256_of_file(p) for p in a.provenance_code},
           "data_sha256": {os.path.abspath(p): sha256_of_file(p) for p in a.provenance_data},
           "config": vars(a),
           "measurement_boundaries": {
               "host_memory": "peak RSS of the process tree, sampled at 20 ms, one process "
                              "per arm",
               "device_memory": "torch peak allocated and peak reserved, reset per arm, "
                                "read after synchronize",
               "input_time": "read (host slice) + transfer (host to device) + decode "
                             "(reconstruction), reported separately",
               "epoch_time": "cold (first recorded epoch) and steady (median of the rest) "
                             "reported separately",
               "storage": "size of the written container including basis, mean, retained "
                          "index set and format overhead"},
           "env": {"python": platform.python_version(), "torch": torch.__version__,
                   "numpy": np.__version__, "platform": platform.platform(),
                   "gpu": dev_name, "host_ram_bytes": int(psutil.virtual_memory().total)},
           "split": splits, "rank_provenance": rank_provenance,
           "per_target": per_target, "ratios": ratios, "bootstrap_ci": boot,
           "failures": failures,
           "verdicts": {"H0a_fired": bool(h0a), "H0b_fired": bool(h0b),
                        "H0c_fired_per_target": h0c,
                        "H0c_fired": bool(any(h0c.values()))}}
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nratios (dense / budget):")
    for n, r in ratios.items():
        print(f"  {n}: disk {r['bytes_on_disk']:.1f}x  "
              f"host-sup {r['supervision_bytes_in_host']:.1f}x  "
              f"peak-RSS {r['peak_host_rss']:.2f}x  input {r['input_seconds']:.2f}x  "
              f"cold {r['cold_epoch_seconds']:.2f}x  "
              f"steady {r['steady_epoch_seconds']:.2f}x  "
              f"error {r['error_excess_budget_vs_dense']*100:+.1f}%")
    for n, c in boot.items():
        e = c["steady_epoch_seconds"]
        if e["excludes_one"] is None:
            print(f"  {n}: steady speed-up {e['ratio']:.2f}x  (no interval: {e['note']})")
        else:
            print(f"  {n}: steady speed-up {e['ratio']:.2f}x "
                  f"[{e['ci_lo']:.2f},{e['ci_hi']:.2f}] "
                  f"{'interval excludes 1' if e['excludes_one'] else 'interval includes 1'}")
    print("H0a fired:", h0a, "| H0b fired:", h0b, "| H0c fired:", h0c)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
