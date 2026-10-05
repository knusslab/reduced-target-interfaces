"""Zero-data contracts for DIFF_SORP_SECOND_CONTRACT_V1(+V1.1).

This module intentionally contains no HDF5 reader and no scientific production executor.
It defines only frozen splits, schema validation, numerical basis/selector contracts,
stopping replay, provenance seals, one-shot access marking, and paired bootstrap logic.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

PROTOCOL_ID = "DIFF_SORP_SECOND_CONTRACT_V1"
PROTOCOL_SHA256 = "9fa0f1cdd0d7cdda85e5b5f8f3e12ae587c5d563c24a51d43b028a18ec2a49f7"
ADDENDUM_SHA256 = "d1bbd1a8ccce29d08dff6ff4a656e73745c9190f699a2dfc1292617e181c2a21"
ADDENDUM_V1_2_SHA256 = "b4ca560b38d45caac4848107577882c78396aae5b904135402493a02a0ac162c"
ADDENDUM_V1_3_SHA256 = "3067650d5268ca56ee881aafd3cb6bef280cc256433269b7c6d7c4c343094894"
ADDENDUM_V1_4_SHA256 = "641409ab9ed4964dfd42b7408d5883869d7becb36ce165fb72da98eb765859bb"
OFFICIAL_SOURCE_BLOBS = {
    "data_registry": "3332b679127a6db3059243983a237332273483e7",
    "generator": "2f1462b81996d71df10ec8570420e951331fbf07",
    "generator_config": "2ad7e1f3725250b75ce876c16290ffa450984af9",
    "model_config": "7ed4371bafc4922d89e3f9245d1c4e7f82a2bf92",
    "dataset_loader": "d5bc69db79639ab2dc5e2c7fa3ca286eade8ea55",
}
OFFICIAL_SOURCE_BLOBS_V1_4 = {**OFFICIAL_SOURCE_BLOBS, "release_visualizer": "1c75218ea977fc408434c770acfb7b218417fa25"}
PINNED_PDEBENCH_COMMIT = "4ff3e3a4aa1561721b5571fa3a048a0a463e0568"
DATA_FILENAME = "1D_diff-sorp_NA_NA.h5"
PUBLISHER_MD5 = "9d466d1213065619d087319e16d9a938"
DATAFILE_ID = 133020

FIELD_DIM = 1024
TIME_FRAMES_USED = 101
PAIR_STRIDE = 5
SEEDS = (0, 1, 2)
LADDER = (2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024)
PRIMARY_TAU = 0.05
ROBUSTNESS_TAU = 0.10
DENSE_HIDDEN = 256
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
PATIENCE = 20
MIN_DELTA = 1e-3
MAX_EPOCHS = 300
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 4072430940531852417
BOOTSTRAP_QUANTILES = (0.025, 0.975)
QUALITY_TOLERANCE = 1.05

DEVELOPMENT_SPLIT_SEED = 2221047658708269114
TEST_SPLIT_SEED = 3220641587463752856
DEVELOPMENT_SPLIT_DERIVATION_SHA256 = "1ed2bf13f684f83ae85f5d82803669e1893318dc0e85b655b3e69b333f128898"
TEST_SPLIT_DERIVATION_SHA256 = "2cb20475c9a08c98b347f7e3692518745fd312dfac835afb0bbf5d12100c0914"
BOOTSTRAP_DERIVATION_SHA256 = "38842e5d1eb604810820e5651f0641426ad59377af419b8ba4adcc1a162fe872"

SPLIT_DIGESTS = {
    "train": "bdd5fe2abc0dc9706ddec253f702b9ebdd6751f69bbd5d9f5005d663cc60fcd2",
    "validation": "823b6f6a0ce08224ef14153babd0a52c54c1d2212492629fd6551d7c923e3ee6",
    "unused": "060451f6180354ffe574c8095da4bf0349a5032ea5cd8003102d77c9ac0242da",
    "selection": "d4ab1c03f1d2eb2a96baaeb1fea61ca42e73272a9f4ca5816df5198901290f44",
    "test": "38c307667103362555436ba5a02203df0dc6f60c69add785bb44e563d4db73a3",
}
ROWS_BY_ROLE = {"train": 9600, "validation": 3200, "unused": 3200,
                "selection": 4000, "test": 4000}

ROLES = {"train", "validation", "unused", "selection", "test"}
ROLE_PERMISSIONS = {
    "BASIS_TRAIN": {"train"},
    "DENSE_TRAIN": {"train", "validation"},
    "SELECT": {"selection"},
    "COMPRESSED_TRAIN": {"train", "validation"},
    "PRETEST_SEAL": set(),
    "TEST_EVAL": {"test"},
}

INTERPRETATION_MAP = {
    "reduced_and_pass": "second_PDE_same_MLP_fresh_retraining_support",
    "reduced_and_fail": "selector_did_not_transfer_to_fresh_retraining_on_this_PDE",
    "full_rank": "no_reduced_supervision_admitted",
    "seed12_selector": "stability_diagnostic_only",
    "ratio_below_one": "report_without_mechanism_attribution",
    "not_claimed": ["cross_backbone", "universal_K", "exact_minimum_K", "architecture_causality"],
}


class SchemaNoGo(RuntimeError):
    pass


class ProvenanceFail(RuntimeError):
    pass


class TrainingGateFail(RuntimeError):
    pass


class MLP(nn.Module):
    def __init__(self, din: int, dout: int, hidden: int = DENSE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(din), int(hidden)), nn.GELU(),
            nn.Linear(int(hidden), int(hidden)), nn.GELU(),
            nn.Linear(int(hidden), int(dout)),
        )

    def forward(self, x):
        return self.net(x)


def mlp_parameter_count(din: int, hidden: int, dout: int) -> int:
    din, hidden, dout = int(din), int(hidden), int(dout)
    return din * hidden + hidden + hidden * hidden + hidden + hidden * dout + dout


def sha256_file(path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def float32_array_digest(values) -> str:
    a = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def float64_array_digest(values) -> str:
    a = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def key_digest(keys) -> str:
    return hashlib.sha256("\n".join(str(k) for k in keys).encode("utf-8")).hexdigest()


def frozen_splits() -> dict[str, list[str]]:
    development = [f"{i:04d}" for i in range(9000)]
    order = np.random.Generator(np.random.PCG64(DEVELOPMENT_SPLIT_SEED)).permutation(9000)
    out = {
        "train": [development[i] for i in order[:480]],
        "validation": [development[i] for i in order[480:640]],
        "unused": [development[i] for i in order[640:800]],
        "selection": [development[i] for i in order[800:1000]],
    }
    official_test = [f"{i:04d}" for i in range(9000, 10000)]
    test_order = np.random.Generator(np.random.PCG64(TEST_SPLIT_SEED)).permutation(1000)
    out["test"] = [official_test[i] for i in test_order[:200]]
    for role, expected in SPLIT_DIGESTS.items():
        got = key_digest(out[role])
        if got != expected:
            raise ProvenanceFail(f"{role} frozen split digest {got} != {expected}")
    return out


def pair_time_indices() -> np.ndarray:
    return np.arange(0, 100, PAIR_STRIDE, dtype=np.int64)


def assert_role_openable(role: str, stage: str) -> bool:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    allowed = ROLE_PERMISSIONS.get(str(stage))
    if allowed is None:
        raise ValueError(f"unknown stage {stage!r}")
    if role not in allowed:
        raise PermissionError(f"{stage} may not open role {role}")
    return True


def validate_schema_record(record: Mapping) -> bool:
    expected_groups = [f"{i:04d}" for i in range(10000)]
    if list(record.get("group_names", [])) != expected_groups:
        raise SchemaNoGo("group names differ from exact 0000..9999 contract")
    expected_shapes = {"data_shape": [101, 1024, 1], "x_shape": [1024]}
    for key, expected in expected_shapes.items():
        if list(record.get(key, [])) != expected:
            raise SchemaNoGo(f"{key} {record.get(key)!r} != {expected!r}")
    t_shape = list(record.get("t_shape", []))
    if len(t_shape) != 1 or int(t_shape[0]) < 101:
        raise SchemaNoGo(f"t_shape {record.get('t_shape')!r} must be one-dimensional with length >=101")
    for key in ("data_dtype", "x_dtype", "t_dtype"):
        if str(record.get(key)) != "float32":
            raise SchemaNoGo(f"{key} must be exact float32 under V1.1")
    return True


def fit_train_basis(targets) -> dict:
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] < 1 or y.shape[1] < 1:
        raise ValueError("targets must be non-empty [N,D]")
    n, d = y.shape
    mean64 = y.mean(axis=0, dtype=np.float64)
    centred = y - mean64
    covariance = (centred.T @ centred) / float(n)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = np.ascontiguousarray(eigenvalues[order], dtype=np.float64)
    basis64 = np.ascontiguousarray(eigenvectors[:, order].T, dtype=np.float64)
    for row in basis64:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    basis32 = np.ascontiguousarray(basis64, dtype=np.float32)
    mean32 = np.ascontiguousarray(mean64, dtype=np.float32)
    b_t = torch.from_numpy(basis32)
    gram32 = np.ascontiguousarray((b_t @ b_t.T).cpu().numpy(), dtype=np.float32)
    clipped = np.clip(eigenvalues, 0.0, None)
    total = float(clipped.sum())
    if total <= 0:
        k_energy = d
    else:
        frac = np.cumsum(clipped) / total
        k_energy = int(np.searchsorted(frac, 0.999, side="left") + 1)
    residual = float(np.max(np.abs(gram32.astype(np.float64) - np.eye(d, dtype=np.float64))))
    return {
        "mean_float64": mean64,
        "mean_float32": mean32,
        "basis_float64": basis64,
        "basis_float32": basis32,
        "eigenvalues_float64": eigenvalues,
        "gram_float32": gram32,
        "mean_sha256": float32_array_digest(mean32),
        "basis_sha256": float32_array_digest(basis32),
        "mean64_sha256": float64_array_digest(mean64),
        "basis64_sha256": float64_array_digest(basis64),
        "eigenvalues_sha256": float64_array_digest(eigenvalues),
        "gram_sha256": float32_array_digest(gram32),
        "orthonormality_residual_max_abs": residual,
        "orthonormality_residual_float64_max_abs": float(np.max(np.abs(basis64 @ basis64.T - np.eye(d, dtype=np.float64)))),
        "k_energy": k_energy,
        "covariance_denominator": int(n),
        "field_dim": int(d),
    }


def projected_field_mse_from_coefficients(pred_coeff: torch.Tensor,
                                           target_coeff: torch.Tensor,
                                           basis_rows: torch.Tensor) -> torch.Tensor:
    if pred_coeff.shape != target_coeff.shape:
        raise ValueError("coefficient shapes differ")
    if basis_rows.ndim != 2 or basis_rows.shape[0] != pred_coeff.shape[1]:
        raise ValueError("basis shape incompatible with coefficients")
    d = int(basis_rows.shape[1])
    if d <= 0:
        raise ValueError("field dimension must be positive")
    err = pred_coeff - target_coeff
    gram = basis_rows @ basis_rows.T
    return torch.einsum("bi,ij,bj->", err, gram, err) / (err.shape[0] * d)


def select_budget(gain, *, ladder=LADDER, threshold: float) -> dict:
    g = np.asarray(gain, dtype=np.float64).reshape(-1)
    if g.size < 1 or not np.all(np.isfinite(g)):
        raise ValueError("invalid gain vector")
    positive = np.clip(g, 0.0, None)
    order = np.argsort(-positive, kind="stable")
    ladder_valid = [int(k) for k in ladder if 0 < int(k) <= g.size]
    if not ladder_valid or ladder_valid[-1] != g.size:
        raise ValueError("ladder must include the full dimension")
    tail_by_k = {}
    selected = None
    for k in ladder_valid:
        tail = float(positive[order[k:]].sum())
        if k == g.size:
            tail = 0.0
        tail_by_k[k] = tail
        if selected is None and tail <= float(threshold):
            selected = k
    if selected is None:
        selected = g.size
    indices = np.ascontiguousarray(order[:selected], dtype=np.int64)
    return {"selected_k": int(selected), "indices": indices, "order": order.astype(np.int64),
            "tail_by_k": tail_by_k, "threshold": float(threshold),
            "gain_sha256": float64_array_digest(g),
            "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "order_sha256": hashlib.sha256(np.ascontiguousarray(order, dtype=np.int64).tobytes()).hexdigest()}


def batch_permutation(n: int, *, seed: int, epoch: int) -> np.ndarray:
    generator = torch.Generator().manual_seed(int(seed) * 1000 + int(epoch) - 1)
    return torch.randperm(int(n), generator=generator).numpy()


def replay_stopping(curve, *, patience=PATIENCE, min_delta=MIN_DELTA) -> dict:
    values = [float(v) for v in curve]
    if not values or not all(np.isfinite(values)):
        raise ValueError("invalid validation curve")
    best = values[0]
    if best <= 0:
        raise ValueError("initial validation value must be positive")
    selected_epoch, counter = 1, 0
    stop_epoch, stop_reason = None, "CEILING_REACHED"
    for epoch, value in enumerate(values[1:], start=2):
        if (best - value) / best >= float(min_delta):
            best, selected_epoch, counter = value, epoch, 0
        else:
            counter += 1
            if counter >= int(patience):
                stop_epoch, stop_reason = epoch, "PATIENCE"
                break
    ran = values[:stop_epoch] if stop_epoch else values
    raw_best = min(ran)
    return {"selected_epoch": int(selected_epoch), "selected_validation_mse": float(best),
            "raw_best_epoch": int(ran.index(raw_best) + 1),
            "raw_best_validation_mse": float(raw_best),
            "stop_epoch": int(stop_epoch if stop_epoch else len(values)),
            "stop_reason": stop_reason, "converged": stop_reason == "PATIENCE",
            "epochs_run": len(ran)}


def _trace_manifest(traces) -> dict:
    traces = list(traces)
    expected = {(arm, seed) for arm in ("dense", "compressed") for seed in SEEDS}
    got = {(str(t.get("arm")), int(t.get("seed"))) for t in traces}
    if got != expected or len(traces) != 6:
        raise TrainingGateFail(f"expected six dense/compressed traces, got {sorted(got)}")
    out = {}
    for trace in traces:
        arm, seed = str(trace["arm"]), int(trace["seed"])
        if (not bool(trace.get("converged")) or trace.get("stop_reason") != "PATIENCE"
                or int(trace.get("nonfinite_count", -1)) != 0):
            raise TrainingGateFail(f"{arm} seed {seed} did not pass stopping gate")
        for field in ("checkpoint_file_sha256", "checkpoint_state_sha256", "batch_order_sha256"):
            value = trace.get(field)
            if not isinstance(value, str) or not value:
                raise TrainingGateFail(f"{arm} seed {seed}: missing {field}")
        out[f"{arm}_seed{seed}"] = {
            "checkpoint_file_sha256": trace["checkpoint_file_sha256"],
            "checkpoint_state_sha256": trace["checkpoint_state_sha256"],
            "selected_epoch": int(trace["selected_epoch"]),
            "batch_order_sha256": trace["batch_order_sha256"],
        }
    return dict(sorted(out.items()))


def _validate_selection(selection: Mapping) -> dict:
    required = ("proposal_seed", "K_prop", "indices_sha256", "gain_sha256", "selection_split_sha256")
    if any(key not in selection for key in required):
        raise ProvenanceFail("selection seal fields missing")
    if int(selection["proposal_seed"]) != 0:
        raise ProvenanceFail("proposal seed must remain seed 0")
    if int(selection["K_prop"]) not in LADDER:
        raise ProvenanceFail("K_prop outside frozen ladder")
    if selection["selection_split_sha256"] != SPLIT_DIGESTS["selection"]:
        raise ProvenanceFail("selection split digest changed")
    return {key: copy.deepcopy(selection[key]) for key in required}


def build_pretest_seal(traces, *, selection: Mapping, data_sha256: str,
                       basis_sha256: str, mean_sha256: str, basis64_sha256: str,
                       mean64_sha256: str, gram_sha256: str,
                       evaluator_sha256: str, bootstrap_sha256: str) -> dict:
    manifest = _trace_manifest(traces)
    sel = _validate_selection(selection)
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "addendum_sha256": ADDENDUM_SHA256,
        "addendum_v1_2_sha256": ADDENDUM_V1_2_SHA256,
        "addendum_v1_3_sha256": ADDENDUM_V1_3_SHA256,
        "addendum_v1_4_sha256": ADDENDUM_V1_4_SHA256,
        "official_source_blobs": copy.deepcopy(OFFICIAL_SOURCE_BLOBS_V1_4),
        "status": "PRETEST_SEAL_READY",
        "pinned_pdebench_commit": PINNED_PDEBENCH_COMMIT,
        "data_filename": DATA_FILENAME,
        "publisher_md5": PUBLISHER_MD5,
        "data_sha256": str(data_sha256),
        "basis_sha256": str(basis_sha256),
        "mean_sha256": str(mean_sha256),
        "basis64_sha256": str(basis64_sha256),
        "mean64_sha256": str(mean64_sha256),
        "gram_sha256": str(gram_sha256),
        "selector_basis": "float64_canonical",
        "deployment_basis": "float32_realised_gram",
        "split_digests": dict(SPLIT_DIGESTS),
        "selection": sel,
        "checkpoints": manifest,
        "architecture": {
            "dense": [FIELD_DIM, DENSE_HIDDEN, DENSE_HIDDEN, FIELD_DIM],
            "dense_parameters": mlp_parameter_count(FIELD_DIM, DENSE_HIDDEN, FIELD_DIM),
            "compressed": [FIELD_DIM, DENSE_HIDDEN, DENSE_HIDDEN, int(sel["K_prop"])],
            "compressed_parameters": mlp_parameter_count(FIELD_DIM, DENSE_HIDDEN, int(sel["K_prop"])),
        },
        "training_contract": {"seeds": list(SEEDS), "optimizer": "Adam", "learning_rate": LEARNING_RATE,
                              "batch_size": BATCH_SIZE, "patience": PATIENCE,
                              "min_delta_relative": MIN_DELTA, "max_epochs": MAX_EPOCHS,
                              "device": "cpu", "dtype": "float32", "torch_num_threads": 1},
        "access_counters": {"train_trajectories": 480, "validation_trajectories": 160,
                            "selection_trajectories": 200, "test": 0, "unused": 0},
        "evaluator_source_sha256": str(evaluator_sha256),
        "bootstrap_source_sha256": str(bootstrap_sha256),
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "quantiles": list(BOOTSTRAP_QUANTILES), "shared_trajectory_draw": True},
        "interpretation_map": copy.deepcopy(INTERPRETATION_MAP),
    }


def verify_pretest_seal(seal, traces, *, selection: Mapping, data_sha256: str,
                        basis_sha256: str, mean_sha256: str, basis64_sha256: str,
                        mean64_sha256: str, gram_sha256: str,
                        evaluator_sha256: str, bootstrap_sha256: str) -> bool:
    expected = build_pretest_seal(
        traces, selection=selection, data_sha256=data_sha256, basis_sha256=basis_sha256,
        mean_sha256=mean_sha256, basis64_sha256=basis64_sha256, mean64_sha256=mean64_sha256,
        gram_sha256=gram_sha256,
        evaluator_sha256=evaluator_sha256, bootstrap_sha256=bootstrap_sha256)
    if seal != expected:
        keys = sorted(set(seal) | set(expected))
        mismatch = next((key for key in keys if seal.get(key) != expected.get(key)), "unknown")
        raise ProvenanceFail(f"PRETEST seal mismatch: {mismatch}")
    return True


class OneShotMarker:
    def __init__(self, path, record=None):
        self.path = Path(path)
        self.record = dict(record or {})
        self.opened = False

    def open(self):
        if self.opened:
            raise RuntimeError("marker already consumed by this instance")
        payload = dict(self.record)
        payload.update({"protocol_id": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
                        "addendum_sha256": ADDENDUM_SHA256,
                        "addendum_v1_2_sha256": ADDENDUM_V1_2_SHA256,
                        "addendum_v1_3_sha256": ADDENDUM_V1_3_SHA256,
                        "addendum_v1_4_sha256": ADDENDUM_V1_4_SHA256,
                        "official_source_blobs": copy.deepcopy(OFFICIAL_SOURCE_BLOBS_V1_4),
                        "class": "one_shot_test_access_marker",
                        "consumed_at_utc": datetime.now(timezone.utc).isoformat()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.path, "x", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
        except FileExistsError as exc:
            raise RuntimeError(f"{self.path} exists: TEST has already been consumed") from exc
        self.opened = True
        return self.path


def _quality_point(arm_means: Mapping[int, float], dense_means: Mapping[int, float]) -> float:
    seeds = sorted(dense_means)
    if sorted(arm_means) != seeds or seeds != list(SEEDS):
        raise ValueError("seed sets differ from frozen contract")
    ratios = []
    for seed in seeds:
        denom = float(dense_means[seed])
        if denom <= 0 or not np.isfinite(denom):
            raise ValueError("dense risk must be finite and positive")
        ratios.append(float(arm_means[seed]) / denom)
    return float(np.median(ratios))


def paired_trajectory_bootstrap(per_trajectory_by_arm: Mapping[str, Mapping[int, np.ndarray]],
                                *, dense_arm="dense", replicates=BOOTSTRAP_REPLICATES,
                                seed=BOOTSTRAP_SEED, quantiles=BOOTSTRAP_QUANTILES) -> dict:
    if dense_arm not in per_trajectory_by_arm:
        raise ValueError("dense arm missing")
    arms = list(per_trajectory_by_arm)
    seeds = list(SEEDS)
    if sorted(per_trajectory_by_arm[dense_arm]) != seeds:
        raise ValueError("dense seed set differs")
    n = len(np.asarray(per_trajectory_by_arm[dense_arm][seeds[0]]))
    if n < 1:
        raise ValueError("no trajectories")
    arrays = {}
    for arm in arms:
        if sorted(per_trajectory_by_arm[arm]) != seeds:
            raise ValueError(f"{arm}: seed set differs")
        arrays[arm] = {}
        for s in seeds:
            arr = np.asarray(per_trajectory_by_arm[arm][s], dtype=np.float64)
            if arr.shape != (n,) or not np.all(np.isfinite(arr)):
                raise ValueError(f"{arm} seed {s}: invalid trajectory array")
            arrays[arm][s] = arr
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    draws = rng.integers(0, n, size=(int(replicates), n), dtype=np.int64)
    observed = {arm: {s: float(arrays[arm][s].mean()) for s in seeds} for arm in arms}
    dense_obs = observed[dense_arm]
    out = {"shared_resample": True, "replicates": int(replicates), "seed": int(seed),
           "trajectories": int(n), "resample_digest": hashlib.sha256(draws.tobytes()).hexdigest(),
           "functional": "median_of_per_seed_risk_ratios", "arms": {}}
    for arm in arms:
        if arm == dense_arm:
            continue
        vals = np.empty(int(replicates), dtype=np.float64)
        for r, idx in enumerate(draws):
            dense = {s: float(arrays[dense_arm][s][idx].mean()) for s in seeds}
            current = {s: float(arrays[arm][s][idx].mean()) for s in seeds}
            vals[r] = _quality_point(current, dense)
        lo, hi = np.quantile(vals, quantiles)
        out["arms"][arm] = {"point": _quality_point(observed[arm], dense_obs),
                            "interval": [float(lo), float(hi)]}
    return out
