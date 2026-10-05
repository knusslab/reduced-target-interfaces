"""REIMPLEMENTATION — NOT HISTORICAL SOURCE.

Clean, protocol-faithful implementation of the preserved Broadband 3D contract.
This module was written after the historical orchestration/training/evaluator Python
bytes could not be recovered. It must never be represented as the original executor.

The module contains deterministic scientific building blocks and protocol constants.
Its offline verifier does not generate protected role data or rerun the historical
training/TEST endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - lets non-torch users inspect constants/helpers
    torch = None
    nn = None


REIMPLEMENTATION_LABEL = "REIMPLEMENTATION — NOT HISTORICAL SOURCE"
SOLVER_SHA256 = "84fa125162228bc17ca8aee0970f0a5b7867ce1a95140075c75f9297aca91fa9"
N = 32
NU = 5e-4
DT = 0.01
N_STEPS = 25
T_TARGET = 0.25
CHANNELS = 3
D = CHANNELS * N**3
MODEL_SEEDS = (0, 1, 2)
ADAM_LR = 1e-3
BATCH_SIZE = 8
EPOCHS = 50
TRAIN_THREADS = 5
K_LADDER = (2, 4, 8, 16, 32, 48, 64, 80, 88, 92, 95)
TAU = 0.05
FROZEN_SELECTED_K = 80
QUALITY_LIMIT = 1.05
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 5427810623496981202
ROLE_RANGES = {
    "TRAIN": (920000, 920095),
    "VALIDATION": (930000, 930031),
    "SELECTION": (940000, 940031),
    "TEST": (950000, 950031),
}


@dataclass(frozen=True)
class Segment:
    index: int
    first_epoch: int
    last_epoch: int


def role_seeds(role: str) -> tuple[int, ...]:
    lo, hi = ROLE_RANGES[role.upper()]
    return tuple(range(lo, hi + 1))


def dense_segments() -> tuple[Segment, ...]:
    """V1.2: five create-only ten-epoch segments covering epochs 0..49."""
    return tuple(Segment(i, 10 * i, 10 * i + 9) for i in range(5))


def compact_microsegments() -> tuple[Segment, ...]:
    """V1.3: 25 create-only two-epoch microsegments covering epochs 0..49."""
    return tuple(Segment(i, 2 * i, 2 * i + 1) for i in range(25))


def epoch_permutation_seed(model_seed: int, epoch: int) -> int:
    if model_seed not in MODEL_SEEDS:
        raise ValueError(f"unexpected model seed: {model_seed}")
    if not 0 <= epoch < EPOCHS:
        raise ValueError(f"epoch out of frozen range: {epoch}")
    return 1_000_000 + 1000 * model_seed + epoch


def broadband_initial_condition(seed: int, solver) -> tuple[np.ndarray, np.ndarray]:
    """Reimplement the frozen PCG64 broadband initial-condition family.

    Returns (physical velocity, projected/dealiased Fourier velocity).
    The supplied solver is expected to be the exact recovered NS3D implementation.
    """
    if getattr(solver, "N", None) != N:
        raise ValueError(f"frozen confirmation requires N={N}")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    u0 = rng.standard_normal((3, N, N, N))
    uh = solver.fft(u0)
    k = np.sqrt(solver.k2)
    envelope = np.zeros_like(k, dtype=np.float64)
    nonzero = k > 0
    envelope[nonzero] = k[nonzero] / np.sqrt(1.0 + (k[nonzero] / 3.0) ** (17.0 / 3.0))
    uh *= envelope[None, ...]
    uh[:, 0, 0, 0] = 0.0
    uh *= solver.dealias[None, ...]
    uh = solver.project(uh)
    u = solver.ifft(uh)
    mean_speed_sq = float(np.mean(np.sum(u * u, axis=0)))
    if not np.isfinite(mean_speed_sq) or mean_speed_sq <= 0.0:
        raise RuntimeError("invalid broadband normalization energy")
    scale = mean_speed_sq ** -0.5
    u *= scale
    uh = solver.fft(u)
    uh *= solver.dealias[None, ...]
    uh = solver.project(uh)
    u = solver.ifft(uh)
    return u.astype(np.float64, copy=False), uh


def generate_pair(seed: int, solver) -> tuple[np.ndarray, np.ndarray]:
    """Generate one frozen t=0 -> 0.25 pair for an explicitly supplied identity.

    This function is provided for protocol completeness. The package verifier never calls
    it on frozen confirmation role identities, so it does not reopen scientific endpoints.
    """
    u0, uh = broadband_initial_condition(seed, solver)
    for _ in range(N_STEPS):
        uh = solver.step_rk4(uh, DT)
    y = solver.ifft(uh)
    return u0.astype(np.float32), y.astype(np.float32)


if nn is not None:
    class FrozenUNet3D(nn.Module):
        """One-level 3D U-Net specified by the frozen Broadband 3D contract."""

        def __init__(self):
            super().__init__()
            self.conv_in = nn.Conv3d(3, 16, 3, 1, 1)
            self.down = nn.Conv3d(16, 16, 3, 2, 1)
            self.down_refine = nn.Conv3d(16, 16, 3, 1, 1)
            self.up = nn.ConvTranspose3d(16, 16, 4, 2, 1)
            self.merge = nn.Conv3d(32, 16, 3, 1, 1)
            self.out = nn.Conv3d(16, 3, 3, 1, 1)
            self.act = nn.GELU()

        def forward(self, x):
            skip = self.act(self.conv_in(x))
            h = self.act(self.down(skip))
            h = self.act(self.down_refine(h))
            h = self.up(h)
            h = torch.cat((h, skip), dim=1)
            h = self.act(self.merge(h))
            return self.out(h)


def trainable_parameter_count(model) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def canonicalize_basis_rows(rows: np.ndarray) -> np.ndarray:
    """Largest-absolute component positive; first-index tie break."""
    b = np.asarray(rows, dtype=np.float64).copy()
    for j in range(b.shape[0]):
        pivot = int(np.argmax(np.abs(b[j])))
        if b[j, pivot] < 0:
            b[j] *= -1.0
    return b


def predictive_gain_order(target_coeff: np.ndarray, pred_coeff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_coeff = np.asarray(target_coeff, dtype=np.float64)
    pred_coeff = np.asarray(pred_coeff, dtype=np.float64)
    if target_coeff.shape != pred_coeff.shape or target_coeff.ndim != 2:
        raise ValueError("coefficient arrays must be matching [samples,directions]")
    q = np.mean(target_coeff**2, axis=0)
    e = np.mean((target_coeff - pred_coeff) ** 2, axis=0)
    gain = q - e
    positive = np.maximum(gain, 0.0)
    # stable descending sort, preserving ascending index for exact ties
    order = np.argsort(-positive, kind="stable")
    return gain, order.astype(np.int64)


def select_k_from_gain(gain: np.ndarray, dense_selection_mse: float) -> int:
    positive = np.maximum(np.asarray(gain, dtype=np.float64), 0.0)
    order = np.argsort(-positive, kind="stable")
    ordered = positive[order]
    total = float(np.sum(ordered))
    prefix = np.concatenate(([0.0], np.cumsum(ordered)))
    threshold = TAU * D * float(dense_selection_mse)
    for k in K_LADDER:
        if k > ordered.size:
            continue
        tail = total - float(prefix[k])
        if tail <= threshold:
            return int(k)
    raise RuntimeError("no frozen candidate rung satisfies gain-tail rule")


def compact_loss_from_coeff_error(coeff_error, basis_rows):
    """Field-normalized projected MSE; supports NumPy or Torch arrays."""
    if torch is not None and torch.is_tensor(coeff_error):
        gram = basis_rows @ basis_rows.T
        return torch.einsum("bi,ij,bj->", coeff_error, gram, coeff_error) / (coeff_error.shape[0] * D)
    e = np.asarray(coeff_error)
    b = np.asarray(basis_rows)
    gram = b @ b.T
    return float(np.einsum("bi,ij,bj->", e, gram, e) / (e.shape[0] * D))


def squared_error_decomposition(target_centered: np.ndarray, prediction_centered: np.ndarray,
                                basis_rows: np.ndarray) -> Mapping[str, float]:
    """Compute R_compact = R_perp + R_parallel for an orthonormal retained basis."""
    y = np.asarray(target_centered, dtype=np.float64)
    pred = np.asarray(prediction_centered, dtype=np.float64)
    b = np.asarray(basis_rows, dtype=np.float64)
    if y.shape != pred.shape or y.ndim != 2 or b.ndim != 2 or b.shape[1] != y.shape[1]:
        raise ValueError("shape mismatch")
    coeff_y = y @ b.T
    coeff_p = pred @ b.T
    y_parallel = coeff_y @ b
    p_parallel = coeff_p @ b
    y_perp = y - y_parallel
    compact_pred = p_parallel
    r_compact = float(np.mean((y - compact_pred) ** 2))
    r_perp = float(np.mean(y_perp**2))
    r_parallel = float(np.mean((y_parallel - p_parallel) ** 2))
    return {
        "R_compact": r_compact,
        "R_perp": r_perp,
        "R_parallel": r_parallel,
        "closure_error": abs(r_compact - r_perp - r_parallel),
    }


def bootstrap_median_ratio(dense_sqerr_by_seed: Sequence[np.ndarray],
                           compact_sqerr_by_seed: Sequence[np.ndarray],
                           replicates: int = BOOTSTRAP_REPLICATES,
                           seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Shared trajectory-cluster resampling across all arms/seeds.

    Inputs are per-trajectory aggregate squared-error values for the same trajectories.
    Each bootstrap replicate computes one compact/dense ratio per optimization seed and
    then the median across the three seeds, matching the frozen finite-seed statistic.
    """
    dense = [np.asarray(x, dtype=np.float64) for x in dense_sqerr_by_seed]
    compact = [np.asarray(x, dtype=np.float64) for x in compact_sqerr_by_seed]
    if len(dense) != len(MODEL_SEEDS) or len(compact) != len(MODEL_SEEDS):
        raise ValueError("frozen endpoint requires exactly three model seeds")
    n = len(dense[0])
    if n == 0 or any(len(x) != n for x in dense + compact):
        raise ValueError("all per-trajectory arrays must have equal nonzero length")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    out = np.empty(int(replicates), dtype=np.float64)
    for r in range(int(replicates)):
        idx = rng.integers(0, n, size=n)
        ratios = [float(np.mean(compact[s][idx]) / np.mean(dense[s][idx])) for s in range(3)]
        out[r] = float(np.median(ratios))
    return out


def json_safe(value):
    """V1.5-style recursive conversion of NumPy scalar containers for JSON output."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def dump_json_safe(obj, path: Path | str) -> None:
    Path(path).write_text(json.dumps(json_safe(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")
