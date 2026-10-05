"""QUALITY_CONVERGENCE_CONFIRM_V1 -- contracts shared by every stage.

Splits, the two access firewalls, centring, rank admissibility and stage provenance. The Stage E
statistics are deliberately **not** here yet: the role firewall must be closed by the structure of
the code before a TEST evaluator exists, so that no import or shared helper can create an
accidental path to the held-out data.

Protocol: `protocols/QUALITY_CONVERGENCE_CONFIRM_V1.md`
"""
from __future__ import annotations

import argparse
import hashlib
import math
import json
import pathlib
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bigdata_pod_canonical_v2 import (  # noqa: E402
    canonical_parent_basis, orthonormality_residual)

PROTOCOL_ID = "QUALITY_CONVERGENCE_CONFIRM_V1"
PROTOCOL_PATH = "protocols/QUALITY_CONVERGENCE_CONFIRM_V1.md"
PROTOCOL_SHA256 = "4e3ee2da43720ec208b9e0a481823bbb26d3bdf5aa0f97c738f2b199b6dec6ca"
STAGE0_SEAL_SHA256 = "5bc35c2d840e31244ab9509cde1bde9db19e2509a4179ebc64fe73e658143045"
STAGE = "A"

FAMILIES = (61749, 61991, 97473, 433143, 445636, 452245, 453445, 471651)
SPLIT_SEED = 776131
TRAJECTORIES_PER_FILE = 100
TRAIN_TRAJECTORIES = 60
VALIDATION_TRAJECTORIES = 20
PAIRS_PER_TRAJECTORY = 13

#: pre-fixed multiplier on float64 eps, from `run_bigdata_quality_2d_v6_stage_a.py`
CENTERING_RESIDUAL_MULTIPLIER = 1024.0

#: which roles each stage may open at all. `unused` appears nowhere on purpose.
ROLE_PERMISSIONS = {
    "A": {"train"},
    "B": {"train", "validation"},
    "C": {"train", "validation", "selection"},
    "D": {"train", "validation", "selection"},
    "E": {"train", "validation", "selection", "test"},
}

ROLES = ("train", "validation", "unused", "selection", "test")


class StageBlocked(RuntimeError):
    """A preregistered stop. Never caught and downgraded."""


# ----------------------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------------------

def split_family(seed, family_index, n_trajectories=TRAJECTORIES_PER_FILE):
    family_index = int(family_index)
    if family_index < 0:
        raise ValueError("family index must be non-negative")
    permutation = np.random.Generator(
        np.random.PCG64(int(seed) + family_index)).permutation(int(n_trajectories))
    a, b = TRAIN_TRAJECTORIES, TRAIN_TRAJECTORIES + VALIDATION_TRAJECTORIES
    return permutation[:a], permutation[a:b], permutation[b:]


def index_digest(values):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(values, dtype=np.int64)).tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------
# access firewalls
# ----------------------------------------------------------------------------------------

def assert_role_openable(role, stage):
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    allowed = ROLE_PERMISSIONS.get(str(stage))
    if allowed is None:
        raise ValueError(f"unknown stage {stage!r}")
    if role not in allowed:
        raise PermissionError(f"stage {stage} may not open a {role} file")
    return True


def open_role_file(path, role, stage):
    """Refuses on the path, before any read."""
    assert_role_openable(role, stage)
    name = pathlib.Path(path).name
    marker = {"train": "_train_", "validation": "_train_", "unused": "_train_",
              "selection": "_valid_", "test": "_test_"}[role]
    if marker not in name:
        raise PermissionError(f"{name} is not a {role} file")
    return pathlib.Path(path)


class TrajectoryAccessLedger:
    """Records, per family, exactly which trajectory ids were requested and under which role."""

    def __init__(self):
        self._by_family = {}
        self._by_role = {role: 0 for role in ROLES}

    def record(self, family_index, indices, role):
        ids = [int(i) for i in indices]
        self._by_family.setdefault(int(family_index), []).extend(ids)
        self._by_role[role] += len(ids)

    def requested(self, family_index):
        return list(self._by_family.get(int(family_index), []))

    def summary(self):
        return dict(self._by_role)

    def as_record(self):
        return {"per_family": {str(k): sorted(v) for k, v in sorted(self._by_family.items())},
                "per_role": self.summary()}


def allowed_trajectories(family_index, role, stage, seed=SPLIT_SEED):
    """The only trajectory ids this stage may touch for this family and role."""
    assert_role_openable(role, stage)
    train, validation, unused = split_family(seed, family_index)
    if role == "train":
        # Stage A and D read TRAIN; nothing may reach VALIDATION or UNUSED through a train file
        # except Stage B and D, and then only under the `validation` role.
        return set(int(i) for i in train)
    if role == "validation":
        return set(int(i) for i in validation)
    if role == "unused":
        return set()
    raise ValueError(f"{role} is not indexed within a train file")


def request_trajectories(ledger, family_index, indices, role, stage, seed=SPLIT_SEED):
    """The single gate through which a train-file dataset may be indexed.

    A request is refused if it contains any id outside the sealed set for this role, which
    covers the whole-file read, the off-by-one and the quietly-widened slice alike.
    """
    allowed = allowed_trajectories(family_index, role, stage, seed=seed)
    ids = [int(i) for i in indices]
    if not ids:
        raise PermissionError("empty trajectory request")
    intruders = sorted(set(ids) - allowed)
    if intruders:
        raise PermissionError(
            f"family {family_index}: {len(intruders)} trajectory ids outside the sealed "
            f"{role} set for stage {stage}, first {intruders[0]}")
    if len(set(ids)) != len(ids):
        raise PermissionError(f"family {family_index}: duplicate trajectory ids requested")
    ledger.record(family_index, ids, role)
    return sorted(ids)


def empty_access_counters():
    return {"raw_bytes_hashed": {"train": 0, "selection": 0, "test": 0},
            "hdf5_field_reads": {role: 0 for role in ROLES}}


def validate_access_counters(counters, stage):
    """Raw-byte hashing is not a field read; only the latter is firewalled."""
    reads = counters["hdf5_field_reads"]
    allowed = ROLE_PERMISSIONS[str(stage)]
    for role, count in reads.items():
        if count and role not in allowed:
            raise PermissionError(f"stage {stage} recorded {count} {role} field reads")
    if reads.get("unused"):
        raise PermissionError("the UNUSED split was read")
    return True


# ----------------------------------------------------------------------------------------
# centring and rank
# ----------------------------------------------------------------------------------------

def center_train(y_train):
    """float64 accumulation, then the preregistered residual gate. See the pilot's finding."""
    y64 = np.asarray(y_train, dtype=np.float64)
    mean64 = np.mean(y64, axis=0, keepdims=True, dtype=np.float64)
    centered = y64 - mean64
    row_sum_l2 = float(np.linalg.norm(centered.sum(axis=0)))
    frobenius = float(np.sqrt(np.einsum("ij,ij->", centered, centered, optimize=False)))
    limit = CENTERING_RESIDUAL_MULTIPLIER * np.finfo(np.float64).eps
    relative = row_sum_l2 / frobenius if frobenius > 0 else 0.0
    if relative > limit:
        raise RuntimeError(f"centering residual {relative:.3e} above the limit {limit:.3e}")
    record = {"column_mean_max_abs": float(np.abs(centered.mean(axis=0)).max()),
              "row_sum_l2": row_sum_l2, "frobenius_norm": frobenius,
              "relative_row_sum": relative, "limit": limit,
              "mean_accumulation_dtype": "float64",
              "mean_input_dtype": str(np.asarray(y_train).dtype)}
    return np.ascontiguousarray(mean64[0], dtype=np.float32), centered, record


def assert_rank_admissible(rank, n_rows, largest_arm):
    if rank > n_rows - 1:
        raise ValueError(
            f"parent rank {rank} exceeds n_rows - 1 = {n_rows - 1}; centred data cannot have "
            "full row rank, so a spurious near-null direction was admitted")
    if rank < largest_arm:
        raise ValueError(f"parent rank {rank} below the largest arm {largest_arm}")
    return True


def basis_cap_record(train_rows, resolved_rank):
    """`basis_cap` is an API argument at the algebraic bound, never a selector truncation."""
    bound = int(train_rows) - 1
    if int(resolved_rank) != bound:
        raise ValueError(
            f"selector_rank {resolved_rank} differs from the algebraic bound {bound}; that is a "
            "truncation, and this protocol admits none")
    return {"basis_cap_argument": bound,
            "basis_cap_semantics": "algebraic_upper_bound_not_selector_truncation",
            "selector_rank": int(resolved_rank),
            "artificial_selector_cap": False}


# ----------------------------------------------------------------------------------------
# Source and run identity
# ----------------------------------------------------------------------------------------

def _repo_root():
    return HERE.parent


def sha256_file(path, block=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise StageBlocked(f"{pathlib.Path(path).name}: digest {actual} != expected {expected}")
    return actual


def require_previous_stage(directory, stage):
    candidate = pathlib.Path(directory) / f"STAGE_{stage}.json"
    if not candidate.is_file():
        raise StageBlocked(f"stage {stage} artifact missing at {candidate}")
    return candidate


def write_artifact(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def verify_protocol_and_seal(seal_path):
    protocol = _repo_root() / PROTOCOL_PATH
    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    if protocol_sha != PROTOCOL_SHA256:
        raise StageBlocked(f"protocol digest {protocol_sha} is not the frozen one")
    require_digest(seal_path, STAGE0_SEAL_SHA256)
    seal = json.loads(pathlib.Path(seal_path).read_text(encoding="utf-8"))
    if seal.get("status") != "DATA_SEAL_COMPLETE":
        raise StageBlocked(f"data seal status {seal.get('status')}")
    if seal.get("protocol_sha256") != PROTOCOL_SHA256:
        raise StageBlocked("the data seal was written against a different protocol")
    return protocol_sha, seal


def train_paths_from_seal(seal, train_dir):
    """Re-hash the eight train shards against the seal before any semantic read.

    Closes post-seal mutation. Selection and test are neither rehashed nor opened here.
    """
    rows = [r for r in seal["files"] if r["role"] == "train"]
    if len(rows) != len(FAMILIES):
        raise StageBlocked(f"seal holds {len(rows)} train files, expected {len(FAMILIES)}")
    ordered = []
    for family in FAMILIES:
        match = [r for r in rows if r["family"] == family]
        if len(match) != 1:
            raise StageBlocked(f"family {family} is not uniquely present in the seal")
        record = match[0]
        path = open_role_file(pathlib.Path(train_dir) / record["filename"], "train", STAGE)
        if not path.is_file():
            raise StageBlocked(f"{record['filename']} missing at {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise StageBlocked(
                f"{record['filename']} changed since the seal: {actual} != {record['sha256']}")
        ordered.append((family, path, actual))
    return ordered




# ----------------------------------------------------------------------------------------
# The protocol defines the stopping policy; this runner does not make it configurable.
# ----------------------------------------------------------------------------------------

PATIENCE = 20
MIN_DELTA = 1e-3
MAX_EPOCHS = 300

REFERENCE_TRACE_FIELDS = (
    "arm", "seed", "train_loss_per_epoch", "validation_true_field_mse_per_epoch",
    "raw_best_epoch", "raw_best_validation_mse", "raw_best_checkpoint_sha256",
    "selected_epoch", "selected_validation_mse", "selected_checkpoint_sha256",
    "stop_epoch", "stop_reason", "converged", "nonfinite_count", "runtime_seconds",
)

ARM_TRACE_FIELDS = REFERENCE_TRACE_FIELDS


def replay_stopping(curve, patience=PATIENCE, min_delta=MIN_DELTA):
    """The frozen rule, applied to a validation curve. Epochs are 1-indexed.

    `min_delta` is relative and governs the kept checkpoint as well as the patience counter, so
    `selected_epoch` is the last significant improvement. `raw_best_epoch` is the plain argmin
    over the epochs actually run and is diagnostic only: handing it downstream would apply a
    policy the pilot never validated.
    """
    values = [float(v) for v in curve]
    if not values:
        raise ValueError("empty curve")
    for value in values:
        if not math.isfinite(value):
            raise ValueError("non-finite value in validation curve")

    best, selected_epoch, counter = values[0], 1, 0
    stop_epoch, stop_reason = None, "CEILING_REACHED"
    for epoch, value in enumerate(values[1:], start=2):
        if best <= 0:
            raise ValueError("non-positive running best; the relative rule is undefined")
        if (best - value) / best >= min_delta:
            best, selected_epoch, counter = value, epoch, 0
        else:
            counter += 1
            if counter >= patience:
                stop_epoch, stop_reason = epoch, "PATIENCE"
                break

    ran = values[:stop_epoch] if stop_epoch else values
    raw_best = min(ran)
    return {"selected_epoch": selected_epoch, "selected_validation_mse": best,
            "raw_best_epoch": ran.index(raw_best) + 1, "raw_best_validation_mse": raw_best,
            "stop_epoch": stop_epoch if stop_epoch else len(values),
            "stop_reason": stop_reason, "converged": stop_reason == "PATIENCE",
            "epochs_run": len(ran)}


def validate_arm_trace(trace, fields=ARM_TRACE_FIELDS):
    missing = [f for f in fields if f not in trace]
    if missing:
        raise ValueError(f"{trace.get('arm')}: trace is missing {missing}")
    return True


def reference_checkpoint_for_stage_c(seal):
    """The only checkpoint Stage C may consume."""
    validate_reference_seal(seal)
    return seal["selected_checkpoint_sha256"]


def validate_reference_seal(seal):
    for field in ("selected_epoch", "selected_checkpoint_sha256",
                  "raw_best_epoch", "raw_best_checkpoint_sha256", "stop_reason"):
        if field not in seal:
            raise ValueError(f"reference seal missing {field}")
    for epoch_field in ("selected_epoch", "raw_best_epoch"):
        if int(seal[epoch_field]) > int(seal["stop_epoch"]):
            raise ValueError(
                f"{epoch_field}={seal[epoch_field]} is after stop_epoch={seal['stop_epoch']}; "
                "a checkpoint cannot come from an epoch the run never reached")
    if seal["raw_best_epoch"] != seal["selected_epoch"] and \
            seal["selected_checkpoint_sha256"] == seal["raw_best_checkpoint_sha256"]:
        raise ValueError(
            "the selected checkpoint equals the raw-argmin checkpoint while the epochs differ; "
            "the stopping policy's checkpoint was overwritten by the raw best")
    return True


def stage_b_status(seals):
    if any(s.get("stop_reason") == "CEILING_REACHED" for s in seals):
        return "STAGE_B_CEILING_REACHED"
    return "STAGE_B_COMPLETE"


def require_reference_convergence(seals):
    """A ceiling in any seed blocks downstream. It is not renegotiated by raising max_epochs."""
    stalled = [s for s in seals if s.get("stop_reason") == "CEILING_REACHED"]
    if stalled:
        raise StageBlocked(
            f"{len(stalled)} reference seed(s) reached max_epochs without the patience rule "
            "firing; convergence is not established and the constants are frozen")
    return True


def validation_true_field_mse(prediction, basis, mean, y_true):
    """MSE against the RAW field. `basis=None` means the prediction is already a field."""
    out = np.asarray(prediction, dtype=np.float64)
    if basis is not None:
        out = out @ np.asarray(basis, dtype=np.float64)
    if mean is not None:
        out = out + np.asarray(mean, dtype=np.float64)
    return float(((out - np.asarray(y_true, dtype=np.float64)) ** 2).mean())


# ----------------------------------------------------------------------------------------
# ladder and rank resolution
# ----------------------------------------------------------------------------------------

BASE_LADDER = (2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096)
TOLERANCE = 1.05


def effective_ladder(R):
    """Rungs strictly below the resolved parent rank.

    `R` itself is never a rung. The tail is a suffix sum with a terminal zero appended, so a rung
    placed at `R` is feasible for any data whatsoever; that is exactly how V8 produced an
    uninformative `K_pred = 256`.
    """
    R = int(R)
    ladder = [k for k in BASE_LADDER if k < R]
    if len(ladder) < 2:
        raise ValueError(f"effective ladder for R={R} has fewer than two rungs: {ladder}")
    if ladder and max(ladder) >= R:
        raise ValueError("ladder maximum must be strictly below R")
    return ladder


def assert_gain_width(width, R):
    """A gain vector narrower than the parent rank would truncate the ladder in silence."""
    if int(width) != int(R):
        raise ValueError(
            f"gain width {width} != parent rank {R}; a short gain vector silently caps the "
            "ladder, which is the V8 failure in a new place")
    return True


def resolve_ranks(selection, R):
    """`selection` is the canonical selector's return value. Fail closed, never degrade."""
    ladder = effective_ladder(R)
    reasons = list(selection.get("censor_reasons") or [])
    if reasons:
        raise StageBlocked(f"BLOCKED_SELECTOR_CENSORED: {reasons}")
    k_prop = int(selection["selected_k"])
    if k_prop not in ladder:
        raise StageBlocked(f"BLOCKED_RANK_OFF_LADDER: {k_prop} not in {ladder}")
    if k_prop == max(ladder):
        raise StageBlocked(
            f"BLOCKED_NO_K_NEXT: K_prop={k_prop} is the largest effective rung")
    return {"K_prop": k_prop, "K_next": int(ladder[ladder.index(k_prop) + 1]),
            "effective_ladder": ladder}


def arm_indices(stage_c, arm):
    """The exact basis rows an arm was built from, taken from the Stage C seal.

    Never a prefix. On the V6 cohort the gain order happened to be `[0, 1, 2, 3]`, and inheriting
    that coincidence would make the Stage E floor understate the true floor in the direction that
    flatters the method.
    """
    if arm not in ("K_prop", "K_next"):
        raise ValueError(f"unknown arm {arm!r}")
    indices = list(stage_c["index_sets"][arm])
    k = int(stage_c[arm])
    if len(indices) != k:
        raise ValueError(f"{arm}: index set has {len(indices)} entries, expected {k}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{arm}: duplicate indices")
    return indices


# ----------------------------------------------------------------------------------------
# Stage E statistics. Added only after the A-D role firewall was closed in code, so that no
# earlier stage could reach a TEST quantity through a shared helper.
# ----------------------------------------------------------------------------------------

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260811
STAGE_E_REQUIRED_FIELDS = ("point", "interval", "per_seed_ratios", "functional")


def floor_perp(basis, indices, y, mean):
    """`L_perp(S)` over the SELECTED rows, per row of `y`, averaged.

    Not a POD prefix. On this cohort `is_leading_prefix` is false for both arms, so substituting
    a prefix changes the number and therefore the verdict.
    """
    rows = np.asarray(basis, dtype=np.float64)[np.asarray(indices, dtype=np.int64)]
    residual = np.asarray(y, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    projected = residual @ rows.T @ rows
    return float(((residual - projected) ** 2).mean())


def floor_perp_per_trajectory(basis, indices, y, mean, groups):
    rows = np.asarray(basis, dtype=np.float64)[np.asarray(indices, dtype=np.int64)]
    residual = np.asarray(y, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    perp = residual - residual @ rows.T @ rows
    per_row = (perp ** 2).mean(axis=1)
    groups = np.asarray(groups, dtype=np.int64)
    return np.array([per_row[groups == g].mean() for g in np.unique(groups)])


def quality_point(arm_mse_by_seed, dense_mse_by_seed):
    """median over seeds of the per-seed ratio. Never a ratio of medians."""
    seeds = sorted(set(arm_mse_by_seed) & set(dense_mse_by_seed))
    if len(seeds) != len(arm_mse_by_seed):
        raise ValueError("seed sets differ between the arm and dense")
    return float(np.median([float(arm_mse_by_seed[s]) / float(dense_mse_by_seed[s])
                            for s in seeds]))


def floor_point(l_perp, dense_mse_by_seed):
    """Same shape as `quality_point`: median over seeds of the per-seed ratio.

    `l_perp` carries no seed index, and `x -> L/x` is monotone, so for an odd seed count this
    equals `L / median(dense)` exactly. That identity is documented, not exploited: the
    seed-paired form is kept so point and bootstrap share one functional.
    """
    return float(np.median([float(l_perp) / float(dense_mse_by_seed[s])
                            for s in sorted(dense_mse_by_seed)]))


def paired_bootstrap(per_trajectory, l_perp, replicates=BOOTSTRAP_REPLICATES,
                     seed=BOOTSTRAP_SEED, quantiles=(0.025, 0.975)):
    """One trajectory resample per replicate, shared by every arm, seed and floor quantity.

    `per_trajectory` maps (arm, seed) -> per-trajectory TEST losses; `l_perp` maps arm ->
    per-trajectory floor, or a scalar. Sharing the draw is what makes the intervals comparable
    and is the lesson of the V6 Stage F mismatch.
    """
    keys = sorted(per_trajectory)
    n = len(next(iter(per_trajectory.values())))
    for key in keys:
        if len(per_trajectory[key]) != n:
            raise ValueError(f"{key}: {len(per_trajectory[key])} trajectories, expected {n}")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    draws = generator.integers(0, n, size=(int(replicates), n), dtype=np.int64)
    digest = hashlib.sha256(np.ascontiguousarray(draws).tobytes()).hexdigest()

    arms = sorted({a for a, _ in keys if a != "dense"})
    seeds = sorted({s for a, s in keys if a == "dense"})
    out = {"shared_resample": True, "replicates": int(replicates), "seed": int(seed),
           "resample_digest": digest, "trajectories": int(n),
           "functional": "median_of_per_seed_ratios"}

    def mean_of(key, idx):
        return np.asarray(per_trajectory[key], dtype=np.float64)[idx].mean()

    for arm in arms:
        q_rep, f_rep = np.empty(replicates), np.empty(replicates)
        for r in range(int(replicates)):
            idx = draws[r]
            dense = {s: mean_of(("dense", s), idx) for s in seeds}
            q_rep[r] = quality_point({s: mean_of((arm, s), idx) for s in seeds}, dense)
            per = l_perp[arm]
            value = (float(np.asarray(per, dtype=np.float64)[idx].mean())
                     if np.ndim(per) else float(per))
            f_rep[r] = floor_point(value, dense)
        for label, series in (("quality", q_rep), ("floor", f_rep)):
            lo, hi = np.quantile(series, quantiles)
            out[f"{label}_{arm}"] = {"interval": (float(lo), float(hi)),
                                     "resample_digest": digest,
                                     "functional": "median_of_per_seed_ratios"}
    return out


def verdict(q, f, tolerance=TOLERANCE):
    """Frozen three-way branch, boundary inclusion exactly as declared."""
    if float(f) >= float(tolerance):
        return "REPRESENTATIONAL_SHORTFALL"
    if float(q) > float(tolerance):
        return "OPTIMISATION_OR_GENERALISATION_SHORTFALL"
    return "TOLERANCE_MET_UNDER_CONVERGENCE"


def validate_stage_e(block, fields=STAGE_E_REQUIRED_FIELDS):
    missing = [f for f in fields if f not in block]
    if missing:
        raise ValueError(f"Stage E block missing {missing}; a summary without its per-seed "
                         "ratios is not reportable under this protocol")
    return True


def assert_unchanged_after_test(sealed, current):
    """Nothing in the pre-TEST seal may move once TEST has been opened."""
    changed = [k for k in sealed if k not in current or current[k] != sealed[k]]
    if changed:
        raise ValueError(f"pre-TEST seal changed after TEST: {sorted(changed)}")
    return True


class OneShotTestEvaluator:
    """A create-only marker consumed *before* TEST is opened.

    Consuming it first is what closes the crash-and-retry path: if the process dies after the
    marker is written, the marker still exists and a second attempt is refused. A recovery route
    conditioned on `STAGE_E.json` being absent would reopen TEST, which the protocol forbids.
    """

    def __init__(self, marker_path, record=None):
        self.marker_path = pathlib.Path(marker_path)
        self.record = dict(record or {})
        self.opened = False

    def open(self):
        if self.opened:
            raise RuntimeError("this evaluator has already opened TEST")
        payload = dict(self.record)
        payload.update({"consumed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "class": "one_shot_test_access_marker"})
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.marker_path, "x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
        except FileExistsError as exc:
            raise RuntimeError(
                f"{self.marker_path} exists: TEST has already been opened in this namespace. "
                "Failures are preserved; they are not retried.") from exc
        self.opened = True
        return self.marker_path
