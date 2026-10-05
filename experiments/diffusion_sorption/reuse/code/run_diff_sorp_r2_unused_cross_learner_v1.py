from __future__ import annotations

# Deterministic CUDA settings must be established before importing torch.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import hashlib
import io
import json
import math
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROTOCOL_ID = "DIFF_SORP_R2_UNUSED_CROSS_LEARNER_PORTABILITY_V1"
PROTOCOL_SHA256 = "52275a1302092801b8bd16d5818cf24ce5212f3bee4df3428e30bee1d2e66dba"
PINNED_PDEBENCH_COMMIT = "4ff3e3a4aa1561721b5571fa3a048a0a463e0568"
DATA_FILENAME = "1D_diff-sorp_NA_NA.h5"
DATA_URL = "https://darus.uni-stuttgart.de/api/access/datafile/133020"
DATA_BYTES = 4217044280
DATA_SHA256 = "8e48ab3efd39ab63524d92e85a4e0db46347b72e7c5b05edf81e1c9637a3ab2d"
DATA_MD5 = "9d466d1213065619d087319e16d9a938"

DEVELOPMENT_SPLIT_SEED = 2221047658708269114
SPLIT_DIGESTS = {
    "train": "bdd5fe2abc0dc9706ddec253f702b9ebdd6751f69bbd5d9f5005d663cc60fcd2",
    "validation": "823b6f6a0ce08224ef14153babd0a52c54c1d2212492629fd6551d7c923e3ee6",
    "unused": "060451f6180354ffe574c8095da4bf0349a5032ea5cd8003102d77c9ac0242da",
    "selection": "d4ab1c03f1d2eb2a96baaeb1fea61ca42e73272a9f4ca5816df5198901290f44",
}
ROLE_COUNTS = {"train": 480, "validation": 160, "unused": 160, "selection": 200}
PAIR_INDICES = np.arange(0, 100, 5, dtype=np.int64)
FIELD_DIM = 1024
K = 8
SEEDS = (10, 11, 12, 13, 14)
QUALITY_TOLERANCE = 1.05
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 13859889106505471460

BASIS_SHA256 = "3958eceb871c779cda4c3f5545b47cdff99f4e7ee32f2a7a5b4c830d58275f50"
MEAN_SHA256 = "b6c164c35b1506596a1a94bc7f7c788475498fa356eba8539cc6d77c0545fce3"
INDICES_SHA256 = "dc940ab47c70520db150fcd27e3297f03f689c3d4baf251fea07d4d8ec725894"
INDICES_EXPECTED = np.array([0, 1, 2, 3, 4, 5, 6, 8], dtype=np.int64)

MLP_HIDDEN = 256
MLP_MAX_EPOCHS = 300
MLP_PATIENCE = 20
MLP_BATCH_SIZE = 64
FNO_MODES = 16
FNO_WIDTH = 32
FNO_MAX_EPOCHS = 120
FNO_PATIENCE = 12
FNO_BATCH_SIZE = 32
MIN_DELTA = 1e-3
LEARNING_RATE = 1e-3
FNO_EXPECTED_PARAMS = 74209
PREDICT_BATCH_MLP = 512
PREDICT_BATCH_FNO = 128


class ProtocolError(RuntimeError):
    pass


class PreHoldoutNoGo(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode("utf-8"))
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str(arr.dtype).encode("ascii"))
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(arr.tobytes())
    return h.hexdigest()


def write_json_atomic(path: Path, obj: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_json_create_only(path: Path, obj: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def save_torch_atomic(path: Path, state: Mapping[str, torch.Tensor]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return sha256_file(path)


def key_digest(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(k) for k in keys).encode("utf-8")).hexdigest()


def frozen_splits() -> Dict[str, list[str]]:
    development = [f"{i:04d}" for i in range(9000)]
    order = np.random.Generator(np.random.PCG64(DEVELOPMENT_SPLIT_SEED)).permutation(9000)
    splits = {
        "train": [development[int(i)] for i in order[:480]],
        "validation": [development[int(i)] for i in order[480:640]],
        "unused": [development[int(i)] for i in order[640:800]],
        "selection": [development[int(i)] for i in order[800:1000]],
    }
    for role, keys in splits.items():
        if len(keys) != ROLE_COUNTS[role]:
            raise ProtocolError(f"{role}: count drift")
        got = key_digest(keys)
        if got != SPLIT_DIGESTS[role]:
            raise ProtocolError(f"{role}: split digest mismatch {got}")
    all_keys: list[str] = []
    for role in ("train", "validation", "unused", "selection"):
        all_keys.extend(splits[role])
    if len(set(all_keys)) != len(all_keys):
        raise ProtocolError("development roles overlap")
    return splits


def verify_inherited(package_dir: Path) -> dict:
    inherited = package_dir / "inherited"
    paths = {
        "basis": inherited / "BASIS_FLOAT32.npy",
        "mean": inherited / "TRAIN_MEAN_FLOAT32.npy",
        "indices": inherited / "SELECTED_INDICES_INT64.npy",
    }
    expected = {"basis": BASIS_SHA256, "mean": MEAN_SHA256, "indices": INDICES_SHA256}
    for key, path in paths.items():
        if not path.is_file():
            raise ProtocolError(f"missing inherited {key}: {path}")
        got = sha256_file(path)
        if got != expected[key]:
            raise ProtocolError(f"{key} hash mismatch: {got}")
    basis = np.load(paths["basis"], allow_pickle=False)
    mean = np.load(paths["mean"], allow_pickle=False)
    indices = np.load(paths["indices"], allow_pickle=False)
    if basis.shape != (FIELD_DIM, FIELD_DIM) or basis.dtype != np.float32:
        raise ProtocolError(f"basis schema drift: {basis.shape} {basis.dtype}")
    if mean.shape != (FIELD_DIM,) or mean.dtype != np.float32:
        raise ProtocolError(f"mean schema drift: {mean.shape} {mean.dtype}")
    if indices.dtype != np.int64 or not np.array_equal(indices, INDICES_EXPECTED):
        raise ProtocolError(f"selected indices drift: {indices}")
    basis_rows = np.ascontiguousarray(basis[indices], dtype=np.float32)
    gram = basis_rows @ basis_rows.T
    gram_resid = float(np.max(np.abs(gram - np.eye(K, dtype=np.float32))))
    if gram_resid > 2e-5:
        raise ProtocolError(f"realized selected-basis Gram residual too large: {gram_resid}")
    return {
        "basis_path": str(paths["basis"]),
        "mean_path": str(paths["mean"]),
        "indices_path": str(paths["indices"]),
        "basis_sha256": BASIS_SHA256,
        "mean_sha256": MEAN_SHA256,
        "indices_sha256": INDICES_SHA256,
        "indices": indices.tolist(),
        "K": K,
        "selected_basis_gram_residual_max_abs": gram_resid,
    }


def verify_package_seal(package_dir: Path) -> dict:
    seal_path = package_dir / "PACKAGE_SEAL.json"
    if not seal_path.is_file():
        raise ProtocolError("PACKAGE_SEAL.json missing")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("protocol_id") != PROTOCOL_ID or seal.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ProtocolError("package protocol identity mismatch")
    protocol_path = package_dir / "DIFF_SORP_R2_UNUSED_CROSS_LEARNER_PORTABILITY_V1.md"
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ProtocolError("protocol file hash mismatch")
    bound = seal.get("bound_files", {})
    required = [
        "DIFF_SORP_R2_UNUSED_CROSS_LEARNER_PORTABILITY_V1.md",
        "code/run_diff_sorp_r2_unused_cross_learner_v1.py",
        "code/audit_diff_sorp_r2_unused_cross_learner_v1.py",
        "tests/test_diff_sorp_r2_unused_cross_learner_v1.py",
        "inherited/BASIS_FLOAT32.npy",
        "inherited/TRAIN_MEAN_FLOAT32.npy",
        "inherited/SELECTED_INDICES_INT64.npy",
    ]
    for rel in required:
        p = package_dir / rel
        if rel not in bound or not p.is_file():
            raise ProtocolError(f"package bound file missing: {rel}")
        got = sha256_file(p)
        if got != bound[rel]["sha256"]:
            raise ProtocolError(f"package file hash mismatch: {rel}")
    if sha256_file(Path(__file__).resolve()) != bound["code/run_diff_sorp_r2_unused_cross_learner_v1.py"]["sha256"]:
        raise ProtocolError("executed runner does not match package seal")
    return seal


def configure_device(device_name: str) -> torch.device:
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise PreHoldoutNoGo("CUDA requested but unavailable")
        device = torch.device("cuda")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("device must be cuda or cpu")
    return device


def seed_everything(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def verify_data_bytes(data_file: Path) -> dict:
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    size = data_file.stat().st_size
    if size != DATA_BYTES:
        raise ProtocolError(f"data bytes mismatch: {size}")
    sha = sha256_file(data_file)
    if sha != DATA_SHA256:
        raise ProtocolError(f"data sha256 mismatch: {sha}")
    md5 = md5_file(data_file)
    if md5 != DATA_MD5:
        raise ProtocolError(f"data md5 mismatch: {md5}")
    return {
        "filename": data_file.name,
        "bytes": size,
        "sha256": sha,
        "md5": md5,
        "publisher_md5_matches": True,
        "verified_at_utc": utc_now(),
        "scientific_tensor_reads": 0,
    }


def verify_data_schema(data_file: Path) -> dict:
    # This opens HDF5 metadata only. No dataset values are indexed here.
    expected_names = [f"{i:04d}" for i in range(10000)]
    with h5py.File(data_file, "r") as h:
        names = sorted(str(k) for k in h.keys())
        if names != expected_names:
            raise ProtocolError("HDF5 group names mismatch")
        first = None
        for name in names:
            g = h[name]
            if set(map(str, g.keys())) != {"data", "grid"}:
                raise ProtocolError(f"{name}: top-level keys mismatch")
            if set(map(str, g["grid"].keys())) != {"x", "t"}:
                raise ProtocolError(f"{name}: grid keys mismatch")
            cur = (
                tuple(g["data"].shape), str(g["data"].dtype),
                tuple(g["grid"]["x"].shape), str(g["grid"]["x"].dtype),
                tuple(g["grid"]["t"].shape), str(g["grid"]["t"].dtype),
            )
            if first is None:
                first = cur
            elif cur != first:
                raise ProtocolError(f"{name}: schema differs from first group")
    expected = ((101, 1024, 1), "float32", (1024,), "float32", (101,), "float32")
    if first != expected:
        raise ProtocolError(f"schema mismatch: {first}")
    return {
        "group_count": 10000,
        "group_names_sha256": key_digest(expected_names),
        "data_shape": [101, 1024, 1],
        "data_dtype": "float32",
        "x_shape": [1024],
        "x_dtype": "float32",
        "t_shape": [101],
        "t_dtype": "float32",
        "scientific_tensor_reads": 0,
    }


def materialize_role(data_file: Path, keys: Sequence[str], role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if role not in ("train", "validation", "unused"):
        raise ProtocolError(f"role read forbidden in this successor: {role}")
    if role == "unused":
        raise ProtocolError("UNUSED must be loaded only through materialize_unused_after_marker")
    expected = frozen_splits()[role]
    if list(keys) != expected:
        raise ProtocolError(f"{role}: key list differs from frozen split")
    needed = np.sort(np.unique(np.concatenate([PAIR_INDICES, PAIR_INDICES + 1]))).astype(np.int64)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    groups: list[int] = []
    grid_x: np.ndarray | None = None
    with h5py.File(data_file, "r") as h:
        for trajectory_id, key in enumerate(keys):
            g = h[key]
            if grid_x is None:
                grid_x = np.asarray(g["grid"]["x"][:], dtype=np.float32)
                if grid_x.shape != (FIELD_DIM,):
                    raise ProtocolError("grid/x shape mismatch")
            block = np.asarray(g["data"][needed, :, :], dtype=np.float32)
            if block.shape != (40, FIELD_DIM, 1):
                raise ProtocolError(f"{role} {key}: required frame block shape mismatch {block.shape}")
            positions = {int(t): i for i, t in enumerate(needed.tolist())}
            x = np.stack([block[positions[int(t)], :, 0] for t in PAIR_INDICES], axis=0)
            y = np.stack([block[positions[int(t + 1)], :, 0] for t in PAIR_INDICES], axis=0)
            xs.append(np.ascontiguousarray(x, dtype=np.float32))
            ys.append(np.ascontiguousarray(y, dtype=np.float32))
            groups.extend([trajectory_id] * len(PAIR_INDICES))
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    T = np.asarray(groups, dtype=np.int64)
    assert grid_x is not None
    if X.shape != (len(keys) * 20, FIELD_DIM) or Y.shape != X.shape or T.shape != (len(keys) * 20,):
        raise ProtocolError(f"{role}: materialized shape mismatch")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y)) and np.all(np.isfinite(grid_x))):
        raise ProtocolError(f"{role}: nonfinite values")
    return X, Y, T, grid_x


def materialize_unused_after_marker(data_file: Path, out: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    marker = out / "UNUSED_ACCESS_ONCE.json"
    if not marker.is_file():
        raise ProtocolError("UNUSED marker missing")
    keys = frozen_splits()["unused"]
    needed = np.sort(np.unique(np.concatenate([PAIR_INDICES, PAIR_INDICES + 1]))).astype(np.int64)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    groups: list[int] = []
    with h5py.File(data_file, "r") as h:
        for trajectory_id, key in enumerate(keys):
            block = np.asarray(h[key]["data"][needed, :, :], dtype=np.float32)
            positions = {int(t): i for i, t in enumerate(needed.tolist())}
            x = np.stack([block[positions[int(t)], :, 0] for t in PAIR_INDICES], axis=0)
            y = np.stack([block[positions[int(t + 1)], :, 0] for t in PAIR_INDICES], axis=0)
            xs.append(np.ascontiguousarray(x, dtype=np.float32))
            ys.append(np.ascontiguousarray(y, dtype=np.float32))
            groups.extend([trajectory_id] * len(PAIR_INDICES))
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    T = np.asarray(groups, dtype=np.int64)
    if X.shape != (3200, FIELD_DIM) or Y.shape != X.shape or T.shape != (3200,):
        raise ProtocolError("UNUSED materialized shape mismatch")
    if len(np.unique(T)) != 160:
        raise ProtocolError("UNUSED trajectory count mismatch")
    return X, Y, T


class MLP(nn.Module):
    def __init__(self, din: int, dout: int, hidden: int = MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        xft = torch.fft.rfft(x)
        out = torch.zeros(
            b, self.out_channels, x.size(-1) // 2 + 1,
            device=x.device, dtype=torch.cfloat,
        )
        out[:, :, : self.modes1] = torch.einsum(
            "bix,iox->box", xft[:, :, : self.modes1], self.weights1
        )
        return torch.fft.irfft(out, n=x.size(-1))


class FNO1d(nn.Module):
    def __init__(self, num_channels: int = 1, modes: int = FNO_MODES, width: int = FNO_WIDTH, initial_step: int = 1):
        super().__init__()
        self.modes1 = modes
        self.width = width
        self.padding = 2
        self.fc0 = nn.Linear(initial_step * num_channels + 1, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        # x [B, D, 1], grid [1, D, 1]
        x = torch.cat((x, grid.expand(x.shape[0], -1, -1)), dim=-1)
        x = self.fc0(x).permute(0, 2, 1)
        x = F.pad(x, [0, self.padding])
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)
        x = x[..., : -self.padding].permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


def param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def state_shapes(model: nn.Module) -> dict[str, list[int]]:
    return {k: list(v.shape) for k, v in model.state_dict().items()}


def batch_permutation(n: int, seed: int, epoch: int) -> np.ndarray:
    # Epoch is 1-based, matching original diffusion-sorption contract style.
    rs = 1_000_000 + 1000 * int(seed) + int(epoch)
    return np.random.Generator(np.random.PCG64(rs)).permutation(n).astype(np.int64)


def coeff_targets(Y: np.ndarray, mean: np.ndarray, basis_rows: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((Y.astype(np.float32) - mean[None, :]) @ basis_rows.T, dtype=np.float32)


def decode_coeff(C: np.ndarray, mean: np.ndarray, basis_rows: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(C.astype(np.float32) @ basis_rows + mean[None, :], dtype=np.float32)


def raw_to_coeff(raw: torch.Tensor, mean_t: torch.Tensor, basis_rows_t: torch.Tensor) -> torch.Tensor:
    return (raw - mean_t) @ basis_rows_t.T


def projected_field_loss_coeff(pred_coeff: torch.Tensor, target_coeff: torch.Tensor, basis_rows_t: torch.Tensor) -> torch.Tensor:
    err = pred_coeff - target_coeff
    gram = basis_rows_t @ basis_rows_t.T
    return torch.einsum("bi,ij,bj->", err, gram, err) / (err.shape[0] * FIELD_DIM)


def predict_mlp(model: MLP, X: np.ndarray, device: torch.device, batch: int = PREDICT_BATCH_MLP) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[start:start+batch], dtype=np.float32)).to(device)
            outs.append(model(xb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def predict_fno(model: FNO1d, X: np.ndarray, grid_t: torch.Tensor, device: torch.device, batch: int = PREDICT_BATCH_FNO) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[start:start+batch], dtype=np.float32)).to(device).unsqueeze(-1)
            outs.append(model(xb, grid_t).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def validation_mse_mlp(model: MLP, arm: str, Xv: np.ndarray, Yv: np.ndarray, device: torch.device,
                        mean: np.ndarray, basis_rows: np.ndarray) -> float:
    raw = predict_mlp(model, Xv, device)
    if arm == "dense":
        pred = raw
    else:
        pred = decode_coeff(raw, mean, basis_rows)
    diff = pred.astype(np.float64) - Yv.astype(np.float64)
    return float(np.mean(diff * diff))


def validation_mse_fno(model: FNO1d, arm: str, Xv: np.ndarray, Yv: np.ndarray, grid_t: torch.Tensor,
                        device: torch.device, mean: np.ndarray, basis_rows: np.ndarray) -> float:
    raw = predict_fno(model, Xv, grid_t, device)
    if arm == "dense":
        pred = raw
    else:
        c = (raw - mean[None, :]) @ basis_rows.T
        pred = decode_coeff(c, mean, basis_rows)
    diff = pred.astype(np.float64) - Yv.astype(np.float64)
    return float(np.mean(diff * diff))


def train_mlp_cell(arm: str, seed: int, Xtr: np.ndarray, Ytr: np.ndarray, Ctr: np.ndarray,
                   Xv: np.ndarray, Yv: np.ndarray, mean: np.ndarray, basis_rows: np.ndarray,
                   device: torch.device, out: Path) -> dict:
    seed_everything(seed)
    dout = FIELD_DIM if arm == "dense" else K
    model = MLP(FIELD_DIM, dout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    basis_t = torch.from_numpy(basis_rows).to(device)
    target_np = Ytr if arm == "dense" else Ctr
    target_all = torch.from_numpy(np.ascontiguousarray(target_np, dtype=np.float32))
    best = None
    selected_state = None
    selected_epoch = None
    selected_state_digest = None
    counter = 0
    trace = []
    nonfinite = 0
    stop_reason = "CEILING_REACHED"
    start_time = time.perf_counter()
    for epoch in range(1, MLP_MAX_EPOCHS + 1):
        model.train()
        perm_np = batch_permutation(len(Xtr), seed, epoch)
        total = 0.0
        for start in range(0, len(perm_np), MLP_BATCH_SIZE):
            idx_np = perm_np[start:start+MLP_BATCH_SIZE]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[idx_np], dtype=np.float32)).to(device)
            tb = target_all[idx_np].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            if arm == "dense":
                loss = torch.mean((pred - tb) ** 2)
            else:
                loss = projected_field_loss_coeff(pred, tb, basis_t)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(idx_np)
        train_loss = total / len(Xtr)
        val = validation_mse_mlp(model, arm, Xv, Yv, device, mean, basis_rows)
        if not (math.isfinite(train_loss) and math.isfinite(val)):
            nonfinite += 1
            stop_reason = "NONFINITE"
            trace.append({"epoch": epoch, "train_loss": train_loss, "validation_mse": val})
            break
        improved = best is None or (best - val) / best >= MIN_DELTA
        if improved:
            best = val
            selected_epoch = epoch
            selected_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            selected_state_digest = state_sha256(selected_state)
            counter = 0
        else:
            counter += 1
        trace.append({
            "epoch": epoch, "train_loss": train_loss, "validation_mse": val,
            "selected_validation_mse": best, "patience_counter": counter,
        })
        if counter >= MLP_PATIENCE:
            stop_reason = "PATIENCE"
            break
    if selected_state is None or selected_epoch is None or best is None:
        raise PreHoldoutNoGo(f"MLP {arm} seed {seed}: no valid checkpoint")
    model.load_state_dict(selected_state)
    ck = out / "checkpoints" / "mlp" / arm / f"{arm}_seed{seed}.pt"
    cksha = save_torch_atomic(ck, selected_state)
    info = {
        "family": "MLP", "arm": arm, "seed": seed,
        "input_dim": FIELD_DIM, "output_dim": dout, "hidden": MLP_HIDDEN,
        "parameter_count": param_count(model),
        "max_epochs": MLP_MAX_EPOCHS, "patience": MLP_PATIENCE,
        "relative_min_delta": MIN_DELTA, "batch_size": MLP_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "selected_epoch": selected_epoch, "selected_validation_mse": best,
        "stop_epoch": trace[-1]["epoch"], "stop_reason": stop_reason,
        "nonfinite_count": nonfinite,
        "checkpoint_state_sha256": selected_state_digest,
        "checkpoint_file_sha256": cksha,
        "checkpoint_relative_path": str(ck.relative_to(out)),
        "runtime_seconds": time.perf_counter() - start_time,
        "trace": trace,
    }
    write_json_atomic(out / "traces" / "mlp" / f"{arm}_seed{seed}.json", info)
    return info


def fno_initial_state(seed: int) -> tuple[dict[str, torch.Tensor], str, dict[str, list[int]]]:
    seed_everything(seed)
    model = FNO1d()
    if param_count(model) != FNO_EXPECTED_PARAMS:
        raise PreHoldoutNoGo(f"FNO parameter count drift: {param_count(model)}")
    state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    return state, state_sha256(state), state_shapes(model)


def train_fno_cell(arm: str, seed: int, initial_state: Mapping[str, torch.Tensor], init_digest: str,
                   Xtr: np.ndarray, Ytr: np.ndarray, Ctr: np.ndarray,
                   Xv: np.ndarray, Yv: np.ndarray, mean: np.ndarray, basis_rows: np.ndarray,
                   grid_t: torch.Tensor, device: torch.device, out: Path) -> dict:
    seed_everything(seed)
    model = FNO1d().to(device)
    model.load_state_dict(initial_state)
    loaded_init_digest = state_sha256({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if loaded_init_digest != init_digest:
        raise PreHoldoutNoGo(f"FNO {arm} seed {seed}: paired init digest mismatch")
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mean_t = torch.from_numpy(mean).to(device)
    basis_t = torch.from_numpy(basis_rows).to(device)
    target_all = torch.from_numpy(np.ascontiguousarray(Ytr if arm == "dense" else Ctr, dtype=np.float32))
    best = None
    selected_state = None
    selected_epoch = None
    selected_state_digest = None
    counter = 0
    nonfinite = 0
    stop_reason = "CEILING_REACHED"
    trace = []
    start_time = time.perf_counter()
    for epoch in range(1, FNO_MAX_EPOCHS + 1):
        model.train()
        perm_np = batch_permutation(len(Xtr), seed, epoch)
        total = 0.0
        for start in range(0, len(perm_np), FNO_BATCH_SIZE):
            idx_np = perm_np[start:start+FNO_BATCH_SIZE]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[idx_np], dtype=np.float32)).to(device).unsqueeze(-1)
            tb = target_all[idx_np].to(device)
            opt.zero_grad(set_to_none=True)
            raw = model(xb, grid_t)
            if arm == "dense":
                loss = torch.mean((raw - tb) ** 2)
            else:
                cp = raw_to_coeff(raw, mean_t, basis_t)
                loss = projected_field_loss_coeff(cp, tb, basis_t)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(idx_np)
        train_loss = total / len(Xtr)
        val = validation_mse_fno(model, arm, Xv, Yv, grid_t, device, mean, basis_rows)
        if not (math.isfinite(train_loss) and math.isfinite(val)):
            nonfinite += 1
            stop_reason = "NONFINITE"
            trace.append({"epoch": epoch, "train_loss": train_loss, "validation_mse": val})
            break
        improved = best is None or (best - val) / best >= MIN_DELTA
        if improved:
            best = val
            selected_epoch = epoch
            selected_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            selected_state_digest = state_sha256(selected_state)
            counter = 0
        else:
            counter += 1
        trace.append({
            "epoch": epoch, "train_loss": train_loss, "validation_mse": val,
            "selected_validation_mse": best, "patience_counter": counter,
        })
        if counter >= FNO_PATIENCE:
            stop_reason = "PATIENCE"
            break
    if selected_state is None or selected_epoch is None or best is None:
        raise PreHoldoutNoGo(f"FNO {arm} seed {seed}: no valid checkpoint")
    model.load_state_dict(selected_state)
    ck = out / "checkpoints" / "fno" / arm / f"{arm}_seed{seed}.pt"
    cksha = save_torch_atomic(ck, selected_state)
    info = {
        "family": "FNO", "arm": arm, "seed": seed,
        "modes": FNO_MODES, "width": FNO_WIDTH, "initial_step": 1,
        "parameter_count": param_count(model),
        "state_shapes": state_shapes(model),
        "initial_state_sha256": init_digest,
        "max_epochs": FNO_MAX_EPOCHS, "patience": FNO_PATIENCE,
        "relative_min_delta": MIN_DELTA, "batch_size": FNO_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "selected_epoch": selected_epoch, "selected_validation_mse": best,
        "stop_epoch": trace[-1]["epoch"], "stop_reason": stop_reason,
        "nonfinite_count": nonfinite,
        "checkpoint_state_sha256": selected_state_digest,
        "checkpoint_file_sha256": cksha,
        "checkpoint_relative_path": str(ck.relative_to(out)),
        "runtime_seconds": time.perf_counter() - start_time,
        "trace": trace,
    }
    write_json_atomic(out / "traces" / "fno" / f"{arm}_seed{seed}.json", info)
    return info


def load_mlp_checkpoint(path: Path, arm: str, device: torch.device) -> MLP:
    dout = FIELD_DIM if arm == "dense" else K
    model = MLP(FIELD_DIM, dout).to(device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def load_fno_checkpoint(path: Path, device: torch.device) -> FNO1d:
    model = FNO1d().to(device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if param_count(model) != FNO_EXPECTED_PARAMS:
        raise ProtocolError("FNO parameter count drift while loading")
    return model


def prepare(package_dir: Path, data_file: Path, out: Path, device_name: str) -> dict:
    if out.exists():
        raise FileExistsError(f"prepare output already exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    try:
        package_seal = verify_package_seal(package_dir)
        inherited = verify_inherited(package_dir)
        splits = frozen_splits()
        start = {
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "status": "PREHOLDOUT_STARTED",
            "started_at_utc": utc_now(),
            "device_requested": device_name,
            "selection_access": 0,
            "original_test_access": 0,
            "unused_access": 0,
            "new_seeds": list(SEEDS),
        }
        write_json_create_only(out / "START.json", start)

        raw = verify_data_bytes(data_file)
        write_json_create_only(out / "RAW_BYTE_SEAL.json", raw)
        schema = verify_data_schema(data_file)
        write_json_create_only(out / "SCHEMA_SEAL.json", schema)
        role_record = {
            role: {"count": len(keys), "sha256": key_digest(keys)} for role, keys in splits.items()
        }
        write_json_create_only(out / "ROLE_MANIFEST.json", role_record)

        device = configure_device(device_name)
        environment = {
            "python": os.sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "h5py": h5py.__version__,
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "deterministic_algorithms": True,
        }
        write_json_create_only(out / "ENVIRONMENT.json", environment)

        Xtr, Ytr, Ttr, grid_x_train = materialize_role(data_file, splits["train"], "train")
        Xv, Yv, Tv, grid_x_val = materialize_role(data_file, splits["validation"], "validation")
        if not np.array_equal(grid_x_train, grid_x_val):
            raise PreHoldoutNoGo("TRAIN/VALIDATION grid mismatch")
        np.savez(out / "TRAIN_ROWS.npz", X=Xtr, Y=Ytr, traj=Ttr)
        np.savez(out / "VALIDATION_ROWS.npz", X=Xv, Y=Yv, traj=Tv)
        np.save(out / "GRID_X_FLOAT32.npy", grid_x_train)
        dev = {
            "train": {"trajectories": 480, "rows": 9600, "X_sha256": array_sha256(Xtr), "Y_sha256": array_sha256(Ytr), "traj_sha256": array_sha256(Ttr), "file_sha256": sha256_file(out / "TRAIN_ROWS.npz")},
            "validation": {"trajectories": 160, "rows": 3200, "X_sha256": array_sha256(Xv), "Y_sha256": array_sha256(Yv), "traj_sha256": array_sha256(Tv), "file_sha256": sha256_file(out / "VALIDATION_ROWS.npz")},
            "grid_x_sha256": array_sha256(grid_x_train),
            "grid_file_sha256": sha256_file(out / "GRID_X_FLOAT32.npy"),
            "selection_access": 0, "original_test_access": 0, "unused_access": 0,
        }
        write_json_create_only(out / "DEVELOPMENT_MATERIALIZATION.json", dev)

        basis = np.load(package_dir / "inherited" / "BASIS_FLOAT32.npy", allow_pickle=False)
        mean = np.load(package_dir / "inherited" / "TRAIN_MEAN_FLOAT32.npy", allow_pickle=False)
        indices = np.load(package_dir / "inherited" / "SELECTED_INDICES_INT64.npy", allow_pickle=False)
        basis_rows = np.ascontiguousarray(basis[indices], dtype=np.float32)
        Ctr = coeff_targets(Ytr, mean, basis_rows)
        persistence_val = float(np.mean((Xv.astype(np.float64) - Yv.astype(np.float64)) ** 2))
        if not math.isfinite(persistence_val) or persistence_val <= 0:
            raise PreHoldoutNoGo("invalid persistence validation MSE")

        mlp_info: dict[str, dict[str, dict]] = {"dense": {}, "compact": {}}
        for seed in SEEDS:
            mlp_info["dense"][str(seed)] = train_mlp_cell("dense", seed, Xtr, Ytr, Ctr, Xv, Yv, mean, basis_rows, device, out)
            mlp_info["compact"][str(seed)] = train_mlp_cell("compact", seed, Xtr, Ytr, Ctr, Xv, Yv, mean, basis_rows, device, out)

        grid_t = torch.from_numpy(grid_x_train.astype(np.float32)).to(device).view(1, FIELD_DIM, 1)
        fno_info: dict[str, dict[str, dict]] = {"dense": {}, "compact": {}}
        init_records = {}
        for seed in SEEDS:
            init_state, init_digest, shapes = fno_initial_state(seed)
            init_records[str(seed)] = {"sha256": init_digest, "state_shapes": shapes}
            fno_info["dense"][str(seed)] = train_fno_cell("dense", seed, init_state, init_digest, Xtr, Ytr, Ctr, Xv, Yv, mean, basis_rows, grid_t, device, out)
            fno_info["compact"][str(seed)] = train_fno_cell("compact", seed, init_state, init_digest, Xtr, Ytr, Ctr, Xv, Yv, mean, basis_rows, grid_t, device, out)

        failures = []
        for family_name, info in (("MLP", mlp_info), ("FNO", fno_info)):
            for arm in ("dense", "compact"):
                for seed in SEEDS:
                    cell = info[arm][str(seed)]
                    if cell["nonfinite_count"] != 0:
                        failures.append(f"{family_name} {arm} seed {seed}: nonfinite")
                    if cell["stop_reason"] != "PATIENCE":
                        failures.append(f"{family_name} {arm} seed {seed}: stop_reason={cell['stop_reason']}")
            for seed in SEEDS:
                val = info["dense"][str(seed)]["selected_validation_mse"]
                if not (val < persistence_val):
                    failures.append(f"{family_name} dense seed {seed}: validation {val} not better than persistence {persistence_val}")

        for seed in SEEDS:
            d = fno_info["dense"][str(seed)]
            c = fno_info["compact"][str(seed)]
            if d["parameter_count"] != FNO_EXPECTED_PARAMS or c["parameter_count"] != FNO_EXPECTED_PARAMS:
                failures.append(f"FNO seed {seed}: parameter count mismatch")
            if d["state_shapes"] != c["state_shapes"]:
                failures.append(f"FNO seed {seed}: state shape mismatch")
            if d["initial_state_sha256"] != c["initial_state_sha256"]:
                failures.append(f"FNO seed {seed}: paired initial state mismatch")

        # Validate prediction and decoding shapes for every checkpoint before TEST.
        validation_smoke = {"mlp": {}, "fno": {}}
        for seed in SEEDS:
            md = load_mlp_checkpoint(out / mlp_info["dense"][str(seed)]["checkpoint_relative_path"], "dense", device)
            mc = load_mlp_checkpoint(out / mlp_info["compact"][str(seed)]["checkpoint_relative_path"], "compact", device)
            pd = predict_mlp(md, Xv[:64], device)
            cc = predict_mlp(mc, Xv[:64], device)
            pc = decode_coeff(cc, mean, basis_rows)
            if not (np.all(np.isfinite(pd)) and np.all(np.isfinite(pc))):
                failures.append(f"MLP seed {seed}: validation prediction smoke nonfinite")
            validation_smoke["mlp"][str(seed)] = {"dense_sha256": array_sha256(pd), "compact_decoded_sha256": array_sha256(pc)}

            fd = load_fno_checkpoint(out / fno_info["dense"][str(seed)]["checkpoint_relative_path"], device)
            fc = load_fno_checkpoint(out / fno_info["compact"][str(seed)]["checkpoint_relative_path"], device)
            pfd = predict_fno(fd, Xv[:64], grid_t, device)
            rawc = predict_fno(fc, Xv[:64], grid_t, device)
            cfc = (rawc - mean[None, :]) @ basis_rows.T
            pfc = decode_coeff(cfc, mean, basis_rows)
            if not (np.all(np.isfinite(pfd)) and np.all(np.isfinite(pfc))):
                failures.append(f"FNO seed {seed}: validation prediction smoke nonfinite")
            validation_smoke["fno"][str(seed)] = {"dense_sha256": array_sha256(pfd), "compact_decoded_sha256": array_sha256(pfc)}

        access_ledger = {
            "status": "PREHOLDOUT_ROLE_LEDGER",
            "protocol_id": PROTOCOL_ID,
            "train": 480, "validation": 160,
            "selection": 0, "original_test": 0, "unused": 0,
            "sealed_at_utc": utc_now(),
        }
        write_json_create_only(out / "ACCESS_LEDGER_PREHOLDOUT.json", access_ledger)

        checkpoint_files = {}
        for family, info in (("mlp", mlp_info), ("fno", fno_info)):
            for arm in ("dense", "compact"):
                for seed in SEEDS:
                    rel = info[arm][str(seed)]["checkpoint_relative_path"]
                    checkpoint_files[rel] = {"sha256": sha256_file(out / rel), "bytes": (out / rel).stat().st_size}

        pre = {
            "status": "PREHOLDOUT_NO_GO" if failures else "PREHOLDOUT_READY",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "package_seal_sha256": sha256_file(package_dir / "PACKAGE_SEAL.json"),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "data": raw,
            "schema": schema,
            "roles": role_record,
            "inherited": inherited,
            "environment": environment,
            "new_seeds": list(SEEDS),
            "quality_tolerance": QUALITY_TOLERANCE,
            "persistence_validation_mse": persistence_val,
            "mlp": mlp_info,
            "fno": fno_info,
            "fno_initial_states": init_records,
            "validation_smoke": validation_smoke,
            "checkpoint_files": checkpoint_files,
            "access_ledger_sha256": sha256_file(out / "ACCESS_LEDGER_PREHOLDOUT.json"),
            "development_materialization_sha256": sha256_file(out / "DEVELOPMENT_MATERIALIZATION.json"),
            "failures": failures,
            "unused_marker_exists": (out / "UNUSED_ACCESS_ONCE.json").exists(),
            "created_at_utc": utc_now(),
        }
        write_json_create_only(out / "PREHOLDOUT_SEAL.json", pre)
        if failures:
            write_json_create_only(out / "PREHOLDOUT_NO_GO.json", {"status": "PREHOLDOUT_NO_GO", "failures": failures, "preholdout_seal_sha256": sha256_file(out / "PREHOLDOUT_SEAL.json")})
            raise PreHoldoutNoGo("pre-holdout gates failed; UNUSED remains unopened")
        return pre
    except Exception as exc:
        # Failure receipt never opens UNUSED; marker creation exists only in evaluate().
        failure_path = out / "PREHOLDOUT_FAILURE.json"
        if not failure_path.exists():
            write_json_atomic(failure_path, {
                "status": "PREHOLDOUT_FAILED",
                "protocol_id": PROTOCOL_ID,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "unused_marker_exists": (out / "UNUSED_ACCESS_ONCE.json").exists(),
                "failed_at_utc": utc_now(),
            })
        raise


def verify_prehodout_audit(out: Path) -> dict:
    verify_path = out / "PREHOLDOUT_VERIFY.json"
    if not verify_path.is_file():
        raise ProtocolError("PREHOLDOUT_VERIFY.json missing")
    verify = json.loads(verify_path.read_text(encoding="utf-8"))
    if verify.get("status") != "PREHOLDOUT_VERIFY_PASS":
        raise ProtocolError("pre-holdout independent verification did not pass")
    seal_sha = sha256_file(out / "PREHOLDOUT_SEAL.json")
    if verify.get("preholdout_seal_sha256") != seal_sha:
        raise ProtocolError("pre-holdout verifier binds different seal")
    return verify


def exclusive_create_unused_marker(out: Path) -> dict:
    seal = json.loads((out / "PREHOLDOUT_SEAL.json").read_text(encoding="utf-8"))
    if seal.get("status") != "PREHOLDOUT_READY":
        raise ProtocolError("pre-holdout seal not ready")
    verify = verify_prehodout_audit(out)
    marker_path = out / "UNUSED_ACCESS_ONCE.json"
    payload = {
        "class": "one_shot_unused_access_marker",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "unused_split_sha256": SPLIT_DIGESTS["unused"],
        "unused_trajectories": 160,
        "preholdout_seal_sha256": sha256_file(out / "PREHOLDOUT_SEAL.json"),
        "preholdout_verify_sha256": sha256_file(out / "PREHOLDOUT_VERIFY.json"),
        "consumed_at_utc": utc_now(),
    }
    try:
        write_json_create_only(marker_path, payload)
    except FileExistsError as exc:
        raise ProtocolError("UNUSED marker already exists; V1 holdout is already consumed") from exc
    return payload


def per_trajectory_mse(pred: np.ndarray, y: np.ndarray, groups: np.ndarray, n_groups: int = 160) -> np.ndarray:
    if pred.shape != y.shape:
        raise ValueError("prediction/target shape mismatch")
    err = (pred.astype(np.float64) - y.astype(np.float64)) ** 2
    out = np.empty(n_groups, dtype=np.float64)
    for i in range(n_groups):
        rows = err[groups == i]
        if rows.shape != (20, FIELD_DIM):
            raise ProtocolError(f"trajectory {i}: expected 20x{FIELD_DIM} error block, got {rows.shape}")
        out[i] = float(rows.mean())
    return out


def aggregate_from_per_trajectory(per: Mapping[str, np.ndarray]) -> dict:
    family_seed_ratios: dict[str, dict[str, float]] = {"mlp": {}, "fno": {}}
    for family in ("mlp", "fno"):
        for seed in SEEDS:
            d = np.asarray(per[f"{family}_dense_seed{seed}"], dtype=np.float64)
            c = np.asarray(per[f"{family}_compact_seed{seed}"], dtype=np.float64)
            if d.shape != (160,) or c.shape != (160,):
                raise ProtocolError(f"{family} seed {seed}: per-trajectory shape mismatch")
            if not (np.all(np.isfinite(d)) and np.all(np.isfinite(c)) and np.all(d > 0) and np.all(c >= 0)):
                raise ProtocolError(f"{family} seed {seed}: invalid per-trajectory risks")
            family_seed_ratios[family][str(seed)] = float(c.mean() / d.mean())
    q_mlp = float(np.median(list(family_seed_ratios["mlp"].values())))
    q_fno = float(np.median(list(family_seed_ratios["fno"].values())))
    q_port = float(max(q_mlp, q_fno))
    all_ratios = [*family_seed_ratios["mlp"].values(), *family_seed_ratios["fno"].values()]
    seed_uniform = bool(all(r <= QUALITY_TOLERANCE for r in all_ratios))
    max_seed_ratio = float(max(all_ratios))

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    b_mlp = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    b_fno = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    b_port = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for b in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, 160, size=160)
        qf = {}
        for family in ("mlp", "fno"):
            ratios = []
            for seed in SEEDS:
                d = np.asarray(per[f"{family}_dense_seed{seed}"], dtype=np.float64)
                c = np.asarray(per[f"{family}_compact_seed{seed}"], dtype=np.float64)
                ratios.append(float(c[draw].mean() / d[draw].mean()))
            qf[family] = float(np.median(ratios))
        b_mlp[b] = qf["mlp"]
        b_fno[b] = qf["fno"]
        b_port[b] = max(qf["mlp"], qf["fno"])
    intervals = {
        "Q_MLP": np.quantile(b_mlp, [0.025, 0.975]).tolist(),
        "Q_FNO": np.quantile(b_fno, [0.025, 0.975]).tolist(),
        "Q_port": np.quantile(b_port, [0.025, 0.975]).tolist(),
    }
    return {
        "Q_MLP": q_mlp,
        "Q_FNO": q_fno,
        "Q_port": q_port,
        "threshold": QUALITY_TOLERANCE,
        "verdict": "PORTABILITY_PASS" if q_port <= QUALITY_TOLERANCE else "PORTABILITY_FAIL",
        "seed_ratios": family_seed_ratios,
        "seed_uniform_pass": seed_uniform,
        "max_seed_ratio": max_seed_ratio,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval95": intervals,
        },
    }


def evaluate_holdout(package_dir: Path, data_file: Path, out: Path, device_name: str) -> dict:
    if not out.is_dir():
        raise FileNotFoundError(out)
    verify_package_seal(package_dir)
    verify_inherited(package_dir)
    verify_data_bytes(data_file)
    verify_prehodout_audit(out)
    if (out / "FINAL_RESULT.json").exists():
        raise ProtocolError("FINAL_RESULT already exists")
    device = configure_device(device_name)
    pre = json.loads((out / "PREHOLDOUT_SEAL.json").read_text(encoding="utf-8"))
    if pre.get("status") != "PREHOLDOUT_READY":
        raise ProtocolError("pre-holdout seal not ready")

    # Before marker: load all checkpoints, validate their bound hashes, and execute a tiny no-holdout shape smoke.
    mean = np.load(package_dir / "inherited" / "TRAIN_MEAN_FLOAT32.npy", allow_pickle=False)
    basis = np.load(package_dir / "inherited" / "BASIS_FLOAT32.npy", allow_pickle=False)
    indices = np.load(package_dir / "inherited" / "SELECTED_INDICES_INT64.npy", allow_pickle=False)
    basis_rows = np.ascontiguousarray(basis[indices], dtype=np.float32)
    grid_x = np.load(out / "GRID_X_FLOAT32.npy", allow_pickle=False).astype(np.float32)
    grid_t = torch.from_numpy(grid_x).to(device).view(1, FIELD_DIM, 1)
    models: dict[str, dict[str, dict[int, nn.Module]]] = {
        "mlp": {"dense": {}, "compact": {}},
        "fno": {"dense": {}, "compact": {}},
    }
    for family in ("mlp", "fno"):
        for arm in ("dense", "compact"):
            for seed in SEEDS:
                cell = pre[family][arm][str(seed)]
                ck = out / cell["checkpoint_relative_path"]
                if sha256_file(ck) != cell["checkpoint_file_sha256"]:
                    raise ProtocolError(f"checkpoint hash mismatch: {ck}")
                if family == "mlp":
                    models[family][arm][seed] = load_mlp_checkpoint(ck, arm, device)
                else:
                    models[family][arm][seed] = load_fno_checkpoint(ck, device)

    marker = exclusive_create_unused_marker(out)
    consumed = True
    try:
        X, Y, groups = materialize_unused_after_marker(data_file, out)
        per: dict[str, np.ndarray] = {}
        for seed in SEEDS:
            md = models["mlp"]["dense"][seed]
            mc = models["mlp"]["compact"][seed]
            pd = predict_mlp(md, X, device)
            cc = predict_mlp(mc, X, device)
            pc = decode_coeff(cc, mean, basis_rows)
            per[f"mlp_dense_seed{seed}"] = per_trajectory_mse(pd, Y, groups)
            per[f"mlp_compact_seed{seed}"] = per_trajectory_mse(pc, Y, groups)

            fd = models["fno"]["dense"][seed]
            fc = models["fno"]["compact"][seed]
            pfd = predict_fno(fd, X, grid_t, device)
            rawc = predict_fno(fc, X, grid_t, device)
            cfc = (rawc - mean[None, :]) @ basis_rows.T
            pfc = decode_coeff(cfc, mean, basis_rows)
            per[f"fno_dense_seed{seed}"] = per_trajectory_mse(pfd, Y, groups)
            per[f"fno_compact_seed{seed}"] = per_trajectory_mse(pfc, Y, groups)

        per_path = out / "UNUSED_PER_TRAJECTORY_ERRORS.npz"
        np.savez(per_path, **per)
        aggregate = aggregate_from_per_trajectory(per)
        ledger = {
            "status": "FINAL_ROLE_LEDGER",
            "protocol_id": PROTOCOL_ID,
            "train": 480, "validation": 160,
            "selection": 0, "original_test": 0, "unused": 160,
            "unused_rows": 3200,
            "unused_split_sha256": SPLIT_DIGESTS["unused"],
            "unused_access_marker_sha256": sha256_file(out / "UNUSED_ACCESS_ONCE.json"),
            "sealed_at_utc": utc_now(),
        }
        write_json_create_only(out / "ACCESS_LEDGER_FINAL.json", ledger)
        result = {
            "status": "UNUSED_EVALUATION_COMPLETE",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "holdout_role": "original_R2_UNUSED",
            "holdout_trajectories": 160,
            "holdout_rows": 3200,
            "unused_split_sha256": SPLIT_DIGESTS["unused"],
            "target_K": K,
            "target_indices": INDICES_EXPECTED.tolist(),
            "new_seeds": list(SEEDS),
            **aggregate,
            "per_trajectory_sha256": sha256_file(per_path),
            "access_ledger_final_sha256": sha256_file(out / "ACCESS_LEDGER_FINAL.json"),
            "preholdout_seal_sha256": sha256_file(out / "PREHOLDOUT_SEAL.json"),
            "preholdout_verify_sha256": sha256_file(out / "PREHOLDOUT_VERIFY.json"),
            "unused_marker_sha256": sha256_file(out / "UNUSED_ACCESS_ONCE.json"),
            "completed_at_utc": utc_now(),
        }
        write_json_create_only(out / "FINAL_RESULT.json", result)
        return result
    except Exception as exc:
        # Marker has already been consumed. This V1 cannot be retried.
        failure = {
            "status": "UNUSED_CONSUMED_NO_SCIENTIFIC_VERDICT",
            "protocol_id": PROTOCOL_ID,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "unused_marker_sha256": sha256_file(out / "UNUSED_ACCESS_ONCE.json") if (out / "UNUSED_ACCESS_ONCE.json").exists() else None,
            "failed_at_utc": utc_now(),
        }
        if not (out / "UNUSED_CONSUMED_FAILURE.json").exists():
            write_json_create_only(out / "UNUSED_CONSUMED_FAILURE.json", failure)
        raise


def replay(out: Path) -> dict:
    result_path = out / "FINAL_RESULT.json"
    per_path = out / "UNUSED_PER_TRAJECTORY_ERRORS.npz"
    if not result_path.is_file() or not per_path.is_file():
        raise FileNotFoundError("final result/per-trajectory artifact missing")
    with np.load(per_path, allow_pickle=False) as z:
        per = {k: np.asarray(z[k], dtype=np.float64) for k in z.files}
    agg = aggregate_from_per_trajectory(per)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    keys = ["Q_MLP", "Q_FNO", "Q_port", "threshold", "verdict", "seed_ratios", "seed_uniform_pass", "max_seed_ratio", "bootstrap"]
    matches = all(result.get(k) == agg.get(k) for k in keys)
    receipt = {
        "status": "REPLAY_PASS" if matches else "REPLAY_FAIL",
        "matches": matches,
        "protocol_id": PROTOCOL_ID,
        "recomputed": agg,
        "source_final_result_sha256": sha256_file(result_path),
        "source_per_trajectory_sha256": sha256_file(per_path),
        "replayed_at_utc": utc_now(),
    }
    write_json_atomic(out / "VERIFY_RESULT_REPLAY_RUNNER.json", receipt)
    return receipt


def download_data(dest: Path, out: Path, url: str = DATA_URL) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        receipt = verify_data_bytes(dest)
        receipt.update({"status": "DATA_ALREADY_PRESENT_VERIFIED", "url": url})
        write_json_atomic(out / "DOWNLOAD_RECEIPT.json", receipt)
        return receipt
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    existing = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "portable-research-runner/1.0"})
    if existing > 0:
        req.add_header("Range", f"bytes={existing}-")
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        status = getattr(resp, "status", None)
        append = existing > 0 and status == 206
        if existing > 0 and not append:
            existing = 0
        mode = "ab" if append else "wb"
        with part.open(mode) as f:
            while True:
                chunk = resp.read(8 << 20)
                if not chunk:
                    break
                f.write(chunk)
                f.flush()
    if part.stat().st_size != DATA_BYTES:
        raise ProtocolError(f"download incomplete: {part.stat().st_size}/{DATA_BYTES}")
    os.replace(part, dest)
    receipt = verify_data_bytes(dest)
    receipt.update({
        "status": "DOWNLOAD_COMPLETE_VERIFIED",
        "url": url,
        "elapsed_seconds": time.perf_counter() - start,
        "hdf5_open_count": 0,
        "scientific_tensor_reads": 0,
    })
    write_json_atomic(out / "DOWNLOAD_RECEIPT.json", receipt)
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download-data")
    p.add_argument("--dest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--url", default=DATA_URL)

    p = sub.add_parser("prepare")
    p.add_argument("--package-dir", required=True)
    p.add_argument("--data-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    p = sub.add_parser("evaluate-holdout")
    p.add_argument("--package-dir", required=True)
    p.add_argument("--data-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    p = sub.add_parser("replay")
    p.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "download-data":
        result = download_data(Path(args.dest), Path(args.out), args.url)
    elif args.cmd == "prepare":
        result = prepare(Path(args.package_dir), Path(args.data_file), Path(args.out), args.device)
    elif args.cmd == "evaluate-holdout":
        result = evaluate_holdout(Path(args.package_dir), Path(args.data_file), Path(args.out), args.device)
    else:
        result = replay(Path(args.out))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
