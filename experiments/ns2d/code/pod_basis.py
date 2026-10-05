"""Canonical parent POD decomposition for the fresh 2D/3D successors, contract V2.

Implements the canonical POD numerical contract used by the staged NS2D experiment. The current version names the resolved Gram-path rank `r_gram_resolved`, keeps the
direct-SVD tolerance as a diagnostic only, and computes the energy rank against the FULL
centred Frobenius energy instead of renormalising the truncated spectrum.

Raw `eigh`
rows must not reach the selector: an arbitrary rotation inside an indistinguishable
eigenspace changes per-coordinate gain, the selected indices, the POD prefix and every arm
store.  This module fixes the numerical rank first, then canonicalizes each degenerate
cluster deterministically, so the same SELECTION values give the same basis across BLAS
backends.

One parent decomposition serves both the selector basis (first min(cap, R_num) rows) and the
energy basis (first K_energy rows) -- design addendum A8.
"""
from __future__ import annotations

import sys

import numpy as np


EPS64 = sys.float_info.epsilon
DEGENERACY_RELATIVE_TOLERANCE = 1e-10
PIVOT_TIE_TOLERANCE = 1e-12


def gram_eigendecomposition(centered):
    """Right-basis rows from the smaller Gram, as `e2e_cost.spectrum_and_basis` does.

    Returns eigenvalues (descending, clipped at zero), singular values, and the raw right
    basis rows `(rank, D)` before canonicalization.
    """
    Sc = np.asarray(centered, dtype=np.float64)
    if Sc.ndim != 2 or min(Sc.shape) < 1:
        raise ValueError("centered matrix must be 2-D and non-empty")
    n_rows, n_cols = Sc.shape
    if n_cols <= n_rows:
        w, V = np.linalg.eigh(Sc.T @ Sc)
        w = np.clip(w[::-1], 0.0, None)
        return w, np.sqrt(w), np.ascontiguousarray(V[:, ::-1].T)
    w, U = np.linalg.eigh(Sc @ Sc.T)
    w = np.clip(w[::-1], 0.0, None)
    U = U[:, ::-1]
    sv = np.sqrt(w)
    positive = int((sv > 0).sum())
    V = (Sc.T @ U[:, :positive]) / np.maximum(sv[:positive], 1e-300)
    return w, sv, np.ascontiguousarray(V.T)


def gram_resolved_rank(singular_values, n_rows, n_cols):
    """Rank the Gram path can actually resolve, plus the direct-SVD tolerance as diagnostic.

    Contract V2 section 2.  `r_num` is deliberately not used as a name: it reads as the
    universal numerical rank of the matrix, while this is the rank trustworthy under the Gram
    algorithm we actually run.

    `r_svd_nominal_diagnostic` is the count obtained by applying the direct-SVD nominal
    threshold to the singular-value spectrum RECOVERED FROM THE GRAM.  No direct SVD is
    performed anywhere in this pipeline, so it is not a rank measured by a direct SVD; it
    exists only so the gap against the audit contract stays visible and auditable.
    """
    sv = np.asarray(singular_values, dtype=np.float64)
    if sv.ndim != 1 or sv.size < 1:
        raise ValueError("singular values must be a non-empty 1-D array")
    if np.any(~np.isfinite(sv)) or np.any(sv < 0):
        raise ValueError("singular values must be finite and non-negative")
    if np.any(np.diff(sv) > 0):
        raise ValueError("singular values must be sorted in descending order")
    sigma_max = float(sv[0])
    if not np.isfinite(sigma_max) or sigma_max <= 0:
        return {"gram_lambda_tolerance": None, "gram_sigma_tolerance": None,
                "lambda_max": None, "r_gram_resolved": 0, "r_svd_nominal_diagnostic": 0,
                "sigma_max": None, "status": "failed_invalid_sigma_max",
                "svd_nominal_sigma_tolerance": None}
    shape_factor = max(int(n_rows), int(n_cols))
    lambda_max = sigma_max * sigma_max
    gram_sigma_tolerance = sigma_max * float(np.sqrt(shape_factor * EPS64))
    svd_nominal = sigma_max * shape_factor * EPS64
    resolved = int((sv > gram_sigma_tolerance).sum())
    return {"gram_lambda_tolerance": float(lambda_max * shape_factor * EPS64),
            "gram_sigma_tolerance": float(gram_sigma_tolerance),
            "lambda_max": float(lambda_max), "r_gram_resolved": resolved,
            "r_svd_nominal_diagnostic": int((sv > svd_nominal).sum()),
            "sigma_max": sigma_max,
            "status": "ok" if resolved > 0 else "failed_zero_rank",
            "svd_nominal_sigma_tolerance": float(svd_nominal)}


def energy_closure(centered, eigenvalues, r_gram_resolved, threshold=0.999):
    """Energy rank against the FULL centred Frobenius energy -- contract V2 section 3.

    Renormalising the truncated spectrum would redefine the rule as "99.9% of what the Gram
    path could resolve", which is not the 99.9% energy rule.  If the resolved spectrum cannot
    account for the threshold, the run fails instead of silently rescaling.
    """
    Sc = np.asarray(centered, dtype=np.float64)
    w = np.asarray(eigenvalues, dtype=np.float64)[:int(r_gram_resolved)]
    total = float(np.einsum("ij,ij->", Sc, Sc, optimize=False))
    if not np.isfinite(total) or total <= 0:
        return {"k_energy": None, "status": "failed_zero_energy",
                "total_frobenius_energy": total}
    resolved = float(np.sum(np.clip(w, 0.0, None)))
    fraction = resolved / total
    all_eigen = float(np.sum(np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)))
    record = {"discarded_energy_fraction": 1.0 - fraction,
              "energy_closure_residual": abs(all_eigen / total - 1.0),
              "resolved_eigen_energy": resolved, "resolved_energy_fraction": fraction,
              "total_frobenius_energy": total}
    if fraction < float(threshold):
        return {**record, "k_energy": None, "status": "failed_energy_rank_unresolved"}
    cumulative = np.cumsum(np.clip(w, 0.0, None)) / total
    reached = np.flatnonzero(cumulative >= float(threshold))
    if reached.size == 0:                     # unreachable given fraction >= threshold
        return {**record, "k_energy": None, "status": "failed_energy_rank_unresolved"}
    k_energy = int(reached[0]) + 1
    return {**record, "k_energy": k_energy,
            "k_energy_at_gram_resolution": bool(k_energy == int(r_gram_resolved)),
            "status": "exact"}


def degeneracy_clusters(eigenvalues, r_num,
                        relative_tolerance=DEGENERACY_RELATIVE_TOLERANCE):
    """Group adjacent retained modes with `|l_i - l_{i+1}| / l_0 <= tol`."""
    w = np.asarray(eigenvalues, dtype=np.float64)[:int(r_num)]
    if w.size < 1:
        raise ValueError("no retained eigenvalues")
    lam0 = float(w[0])
    if not np.isfinite(lam0) or lam0 <= 0:
        raise ValueError("leading eigenvalue must be finite and positive")
    clusters, start = [], 0
    for index in range(1, w.size):
        if abs(w[index - 1] - w[index]) / lam0 > float(relative_tolerance):
            clusters.append((start, index))
            start = index
    clusters.append((start, int(w.size)))
    return clusters


def _canonicalize_cluster(block, tie_tolerance=PIVOT_TIE_TOLERANCE):
    """Deterministic pivoted modified Gram-Schmidt in the cluster's coefficient space.

    `block` is `(m, D)`.  If the raw rows are rotated as `B' = O B`, the computed `Q' = O Q`,
    so `Q'^T B' = Q^T B` -- the canonical rows are rotation-invariant.
    """
    B = np.ascontiguousarray(np.asarray(block, dtype=np.float64))
    m = B.shape[0]
    if m == 1:
        # singleton: fix the sign so the maximum-absolute coordinate is positive
        column = int(np.argmax(np.abs(B[0])))
        sign = 1.0 if B[0, column] >= 0 else -1.0
        return B * sign, [column]
    Q = np.zeros((m, m), dtype=np.float64)
    pivots = []
    residual = B.copy()
    for step in range(m):
        norms = np.einsum("ij,ij->j", residual, residual, optimize=False)
        maximum = float(norms.max())
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("degenerate cluster lost rank during canonicalization")
        threshold = maximum - float(tie_tolerance) * max(maximum, 1.0)
        pivot = int(np.flatnonzero(norms >= threshold)[0])   # ties -> minimum index
        vector = residual[:, pivot] / np.sqrt(norms[pivot])
        if step:                                             # second orthogonalization pass
            vector -= Q[:, :step] @ (Q[:, :step].T @ vector)
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                raise ValueError("degenerate cluster lost rank during reorthogonalization")
            vector /= norm
        Q[:, step] = vector
        pivots.append(pivot)
        residual = B - Q[:, :step + 1] @ (Q[:, :step + 1].T @ B)
    return Q.T @ B, pivots


def canonical_parent_basis(centered, *, basis_cap):
    """Full canonical parent decomposition and the two derived bases.

    Returns eigenvalues, the numerical-rank record, the canonical rows `(R_num, D)`, the
    per-cluster pivot record, and the selector cut `min(basis_cap, R_num)`.  The energy basis
    is the first `K_energy` rows of the very same array -- never a second eigendecomposition.
    """
    Sc = np.asarray(centered, dtype=np.float64)
    n_rows, n_cols = Sc.shape
    w, sv, raw = gram_eigendecomposition(Sc)
    rank = gram_resolved_rank(sv, n_rows, n_cols)
    if rank["status"] != "ok":
        return {"basis": None, "clusters": None, "eigenvalues": w, "energy": None,
                "rank": rank, "selector_rank": 0, "singular_values": sv,
                "status": rank["status"]}
    r_num = min(rank["r_gram_resolved"], raw.shape[0])
    clusters = degeneracy_clusters(w, r_num)
    basis = np.empty((r_num, Sc.shape[1]), dtype=np.float64)
    pivot_record = []
    for start, stop in clusters:
        canonical, pivots = _canonicalize_cluster(raw[start:stop])
        basis[start:stop] = canonical
        pivot_record.append({"pivots": [int(p) for p in pivots],
                             "size": int(stop - start), "start": int(start)})
    energy = energy_closure(Sc, w, r_num)
    return {"basis": basis, "clusters": pivot_record, "decomposition_path": "gram_eigh",
            "eigenvalues": w[:r_num], "energy": energy,
            "rank": {**rank, "r_gram_resolved": r_num},
            "selector_rank": int(min(int(basis_cap), r_num)),
            "singular_values": sv[:r_num],
            "status": "ok" if energy["status"] in ("exact",) else energy["status"]}


def orthonormality_residual(basis):
    """max |B B^T - I|, reported so the artifact says how orthonormal the basis actually is."""
    B = np.asarray(basis, dtype=np.float64)
    gram = B @ B.T
    return float(np.abs(gram - np.eye(gram.shape[0])).max())
