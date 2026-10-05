"""Failing-first tests for QUALITY_CONVERGENCE_CONFIRM_V1, Stages A through E.

Written before `code/run_quality_convergence_confirm_v1.py` exists. Each test pins a clause of
the frozen protocol so that an implementation which drops one fails loudly.

Four of these are the confirmation's central falsifiers and are marked `@pytest.mark.falsifier`:

    the floor is over the selected index set, not a POD prefix
    the bootstrap resample is shared across arms, seeds and the floor
    the floor's denominator is per-seed dense, not a pooled dense median
    TEST is evaluated exactly once

The rest guard provenance, the role firewall, the ladder contract and trace symmetry.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

PROTOCOL = ROOT / "protocols" / "QUALITY_CONVERGENCE_CONFIRM_V1.md"
PROTOCOL_SHA256 = "4e3ee2da43720ec208b9e0a481823bbb26d3bdf5aa0f97c738f2b199b6dec6ca"
SEAL_SHA256 = "5bc35c2d840e31244ab9509cde1bde9db19e2509a4179ebc64fe73e658143045"

FAMILIES = (61749, 61991, 97473, 433143, 445636, 452245, 453445, 471651)
SPLIT_SEED = 776131
BASE_LADDER = (2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096)
PATIENCE, MIN_DELTA, MAX_EPOCHS = 20, 1e-3, 300
TOLERANCE = 1.05
ARMS = ("dense", "k_prop", "k_next")
SEEDS = (0, 1, 2)


def qcc():
    try:
        import run_quality_convergence_confirm_v1 as mod
    except ImportError as exc:  # pragma: no cover - the point of failing first
        pytest.fail(f"executor not implemented: {exc}")
    return mod


def orthonormal_basis(rng, rank, d):
    return np.linalg.qr(rng.normal(size=(d, rank)))[0].T.copy()


# ======================================================================================
# 1  Provenance and stage dependencies
# ======================================================================================

def test_protocol_digest_is_the_frozen_one():
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA256, (
        "the frozen protocol changed; if that was intended it is a V2, not an edit"
    )


def test_executor_binds_protocol_and_seal_digests():
    mod = qcc()
    assert mod.PROTOCOL_SHA256 == PROTOCOL_SHA256
    assert mod.STAGE0_SEAL_SHA256 == SEAL_SHA256


def test_stage_c_refuses_to_start_without_a_stage_b_seal(tmp_path):
    mod = qcc()
    with pytest.raises(mod.StageBlocked) as excinfo:
        mod.require_previous_stage(tmp_path, "B")
    assert "B" in str(excinfo.value)


def test_stage_refuses_a_mismatched_upstream_digest(tmp_path):
    mod = qcc()
    upstream = tmp_path / "STAGE_B.json"
    upstream.write_text(json.dumps({"stage": "B", "status": "STAGE_B_COMPLETE"}), "utf-8")
    with pytest.raises(mod.StageBlocked):
        mod.require_digest(upstream, "0" * 64)


def test_outputs_are_create_only(tmp_path):
    mod = qcc()
    target = tmp_path / "STAGE_A.json"
    mod.write_artifact(target, {"stage": "A"})
    with pytest.raises(FileExistsError):
        mod.write_artifact(target, {"stage": "A"})


# ======================================================================================
# 2  role-access firewall
# ======================================================================================

@pytest.mark.parametrize("stage,role,allowed", [
    ("A", "train", True), ("A", "selection", False), ("A", "test", False),
    ("B", "train", True), ("B", "validation", True), ("B", "selection", False),
    ("C", "selection", True), ("C", "test", False),
    ("D", "train", True), ("D", "validation", True), ("D", "test", False),
    ("E", "test", True),
])
def test_role_firewall_by_stage(stage, role, allowed):
    mod = qcc()
    if allowed:
        assert mod.assert_role_openable(role, stage)
    else:
        with pytest.raises(PermissionError):
            mod.assert_role_openable(role, stage)


def test_test_path_is_refused_before_it_is_opened(tmp_path):
    """The guard must fire on the path, not after a read."""
    mod = qcc()
    victim = tmp_path / "NavierStokes2D_test_61949_0.50000.h5"
    victim.write_bytes(b"")
    with pytest.raises(PermissionError):
        mod.open_role_file(victim, role="test", stage="D")


def test_unused_split_is_never_readable_at_any_stage():
    mod = qcc()
    for stage in ("A", "B", "C", "D", "E"):
        with pytest.raises(PermissionError):
            mod.assert_role_openable("unused", stage)


@pytest.mark.parametrize("stage", ["A", "B", "C", "D"])
def test_artifact_declares_zero_test_field_reads(stage):
    mod = qcc()
    counters = mod.empty_access_counters()
    assert counters["hdf5_field_reads"]["test"] == 0
    mod.validate_access_counters(counters, stage)


def test_raw_byte_hashing_does_not_count_as_a_field_read():
    """Stage 0 hashed all 24 files. That must not trip the Stage A-D test firewall."""
    mod = qcc()
    counters = mod.empty_access_counters()
    counters["raw_bytes_hashed"] = {"train": 8, "selection": 8, "test": 8}
    mod.validate_access_counters(counters, "D")          # must not raise
    counters["hdf5_field_reads"]["test"] = 1
    with pytest.raises(Exception):
        mod.validate_access_counters(counters, "D")


# ======================================================================================
# 3  split and basis
# ======================================================================================

def test_split_reproduces_the_frozen_rule():
    mod = qcc()
    for j in range(len(FAMILIES)):
        train, validation, unused = mod.split_family(SPLIT_SEED, j)
        expected = np.random.Generator(np.random.PCG64(SPLIT_SEED + j)).permutation(100)
        assert list(train) == list(expected[0:60])
        assert list(validation) == list(expected[60:80])
        assert list(unused) == list(expected[80:100])
        assert sorted(list(train) + list(validation) + list(unused)) == list(range(100))


def test_centering_is_float64_and_rank_is_bounded():
    mod = qcc()
    rng = np.random.Generator(np.random.PCG64(1))
    n, d = 400, 48
    y = (rng.normal(size=(n, d)) * 30.0 + 500.0).astype(np.float32)
    mean, centered, record = mod.center_train(y)
    assert centered.dtype == np.float64
    assert record["mean_accumulation_dtype"] == "float64"
    assert np.linalg.matrix_rank(centered) <= n - 1
    mod.assert_rank_admissible(n - 1, n, 4)
    with pytest.raises(ValueError):
        mod.assert_rank_admissible(n, n, 4)


def test_no_artificial_basis_cap_is_applied():
    mod = qcc()
    source = (CODE / "run_quality_convergence_confirm_v1.py").read_text(encoding="utf-8")
    assert "basis_cap = 256" not in source
    assert getattr(mod, "BASIS_CAP", None) is None, (
        "an artificial cap is the parameter behind three defects in this line"
    )


# ======================================================================================
# 4  selector and ladder
# ======================================================================================

@pytest.mark.parametrize("R,expected_max", [(6239, 4096), (300, 256), (100, 96), (5, 4)])
def test_effective_ladder_is_strictly_below_R(R, expected_max):
    mod = qcc()
    ladder = mod.effective_ladder(R)
    assert list(ladder) == [k for k in BASE_LADDER if k < R]
    assert max(ladder) == expected_max
    assert R not in ladder, "R is a boundary, never a candidate rung"


def test_effective_ladder_requires_two_rungs():
    mod = qcc()
    with pytest.raises(ValueError):
        mod.effective_ladder(3)          # only rung 2 survives


def test_selected_rank_uses_the_canonical_selector():
    """The selector lives where it is used, so check every executor, not one file."""
    qcc()
    modules = sorted(CODE.glob("run_quality_convergence_confirm_v1*.py"))
    assert modules, "no executor found"
    for path in modules:
        source = path.read_text(encoding="utf-8")
        assert "np.argsort(-gain" not in source, f"{path.name}: inline ordering reimplemented"
    stage_c = CODE / "run_quality_convergence_confirm_v1_stage_c.py"
    assert "canonical_predictive_gain_selector" in stage_c.read_text(encoding="utf-8"), (
        "Stage C must call the canonical selector, not reimplement it as V8 did"
    )


def test_censored_selection_blocks_before_any_arm_is_trained():
    mod = qcc()
    with pytest.raises(mod.StageBlocked) as excinfo:
        mod.resolve_ranks({"selected_k": 4096, "censor_reasons": ["ladder_cap"]}, R=6239)
    assert "BLOCKED_SELECTOR_CENSORED" in str(excinfo.value)


def test_top_rung_selection_blocks_for_lack_of_k_next():
    mod = qcc()
    with pytest.raises(mod.StageBlocked) as excinfo:
        mod.resolve_ranks({"selected_k": 4096, "censor_reasons": []}, R=6239)
    assert "BLOCKED_NO_K_NEXT" in str(excinfo.value)


@pytest.mark.parametrize("k_prop,k_next", [(4, 8), (8, 16), (16, 32), (2, 4), (2048, 4096)])
def test_k_next_is_the_following_rung(k_prop, k_next):
    mod = qcc()
    got = mod.resolve_ranks({"selected_k": k_prop, "censor_reasons": []}, R=6239)
    assert (got["K_prop"], got["K_next"]) == (k_prop, k_next)


# ======================================================================================
# 5  convergence policy and trace symmetry
# ======================================================================================

def test_convergence_constants_match_the_sealed_pilot():
    mod = qcc()
    assert mod.PATIENCE == PATIENCE
    assert mod.MIN_DELTA == pytest.approx(MIN_DELTA)
    assert mod.MAX_EPOCHS == MAX_EPOCHS


def test_checkpoint_updates_only_on_a_significant_improvement():
    """Option A. A 0.05 percent gain under a 0.1 percent min_delta must not move the checkpoint."""
    mod = qcc()
    curve = [1.0, 0.9, 0.89955, 0.89950, 0.89945]      # after epoch 2, all gains < min_delta
    state = mod.replay_stopping(curve, patience=PATIENCE, min_delta=MIN_DELTA)
    assert state["selected_epoch"] == 2, (
        "a raw argmin would pick epoch 5; the policy keeps the last significant improvement"
    )


def test_ceiling_is_not_convergence():
    mod = qcc()
    curve = [1.0 / (i + 1) for i in range(MAX_EPOCHS)]
    state = mod.replay_stopping(curve, patience=PATIENCE, min_delta=MIN_DELTA)
    assert state["stop_reason"] == "CEILING_REACHED"
    assert state.get("converged") is False


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_including_dense_records_the_full_trace(arm):
    mod = qcc()
    required = {"train_loss_per_epoch", "validation_true_field_mse_per_epoch",
                "raw_best_epoch", "selected_epoch", "stop_epoch", "stop_reason",
                "selected_checkpoint_sha256"}
    assert required <= set(mod.ARM_TRACE_FIELDS)
    assert arm not in getattr(mod, "ARMS_WITHOUT_TRACE", ())
    complete = {f: 1 for f in mod.ARM_TRACE_FIELDS}
    complete.update({"arm": arm, "seed": 0})
    mod.validate_arm_trace(complete)
    for missing in sorted(required):
        broken = dict(complete)
        broken.pop(missing)
        with pytest.raises(Exception):
            mod.validate_arm_trace(broken)


def test_validation_metric_is_the_raw_field():
    mod = qcc()
    rng = np.random.Generator(np.random.PCG64(2))
    d, k, n = 40, 5, 24
    basis = orthonormal_basis(rng, k, d)
    mean = np.zeros(d)
    y = rng.normal(size=(n, d))
    perfect = y @ basis.T                       # best possible in-subspace predictor
    got = mod.validation_true_field_mse(perfect, basis, mean, y)
    assert got > 1e-6, "a perfect in-subspace predictor must still carry the off-subspace floor"
    assert got == pytest.approx(float((((perfect @ basis) - y) ** 2).mean()))


# ======================================================================================
# 6  the floor is over the SELECTED index set                       *** falsifier ***
# ======================================================================================

@pytest.mark.falsifier
def test_floor_uses_the_selected_indices_not_a_prefix():
    mod = qcc()
    rng = np.random.Generator(np.random.PCG64(7))
    d, rank, n = 32, 12, 200
    basis = orthonormal_basis(rng, rank, d)
    mean = rng.normal(size=d)
    # coefficients are (n, rank); an earlier revision multiplied (n, d) by a (rank, d) basis
    coefficients = rng.normal(size=(n, rank)) * np.linspace(3.0, 0.2, rank)
    y = mean + coefficients @ basis + 0.3 * rng.normal(size=(n, d))

    selected = [0, 3, 7, 9]                     # deliberately not the leading prefix
    prefix = [0, 1, 2, 3]
    assert selected != prefix

    got = mod.floor_perp(basis, selected, y, mean)
    want = mod.reference_floor(basis, selected, y, mean) if hasattr(mod, "reference_floor") else None

    residual = y - mean
    proj = residual @ basis[selected].T @ basis[selected]
    expected = float(((residual - proj) ** 2).mean())
    expected_prefix = float(
        ((residual - residual @ basis[prefix].T @ basis[prefix]) ** 2).mean())

    assert expected != pytest.approx(expected_prefix), "fixture failed to separate the two"
    assert got == pytest.approx(expected, rel=1e-10)
    assert got != pytest.approx(expected_prefix, rel=1e-6), (
        "the floor was computed over a POD prefix; the V6 coincidence gain_order[:4] == [0,1,2,3] "
        "must not be inherited"
    )
    if want is not None:
        assert want == pytest.approx(expected, rel=1e-10)


@pytest.mark.falsifier
def test_floor_is_not_used_to_gate_arm_execution():
    qcc()
    # stronger than scanning one file by position: no stage before E may mention the floor at
    # all, so it cannot gate an arm no matter how the executors are laid out
    for stage in ("a", "b", "c", "d"):
        path = CODE / f"run_quality_convergence_confirm_v1_stage_{stage}.py"
        if not path.exists():
            continue
        assert "floor_perp" not in path.read_text(encoding="utf-8"), (
            f"stage {stage.upper()} references the floor; it is a Stage E endpoint and using it "
            "earlier would select an arm on a TEST-derived quantity"
        )
    stage_e = CODE / "run_quality_convergence_confirm_v1_stage_e.py"
    assert stage_e.exists(), "Stage E executor missing"
    assert "floor_perp" in stage_e.read_text(encoding="utf-8")


# ======================================================================================
# 7  Stage E statistics                                             *** falsifier ***
# ======================================================================================

@pytest.mark.falsifier
def test_quality_point_is_the_median_of_per_seed_ratios():
    """The fixture must actually separate the two functionals, which most do not.

    With three seeds, `median_s(a_s / d_s)` and `median_s(a_s) / median_s(d_s)` coincide unless
    the seed that is central in the ratios differs from the seeds that are central in the two
    marginals. Here the ratios rank 2, 0, 1 while the marginals rank differently, so they split.
    """
    mod = qcc()
    dense = {0: 1.0, 1: 2.0, 2: 100.0}
    arm = {0: 1.1, 1: 100.0, 2: 3.0}
    median_of_ratios = float(np.median([1.1, 50.0, 0.03]))                      # 1.1
    ratio_of_medians = float(np.median([1.1, 100.0, 3.0]) / np.median([1.0, 2.0, 100.0]))  # 1.5
    assert median_of_ratios != pytest.approx(ratio_of_medians), (
        "fixture failed to separate the two functionals; it would assert nothing"
    )
    assert mod.quality_point(arm, dense) == pytest.approx(median_of_ratios), (
        "ratio-of-medians is the V6 Stage F mismatch and is not this protocol's functional"
    )


def test_floor_point_is_seed_paired_and_the_two_orders_coincide():
    """Documents an identity rather than pretending to falsify one.

    `L_perp(S)` is a property of the basis and the TEST field, so it carries no seed index.
    Since `x -> L/x` is monotone, the median commutes with it and for any odd seed count

        median_s( L / dense_s )  ==  L / median_s( dense_s )

    exactly. An earlier revision of this file asserted these two differ and tried to build a
    fixture separating them; no fixture can. That claim is withdrawn.

    The floor's real exposure is not aggregation order but whether it shares the trajectory
    resample with the arms, which `test_bootstrap_shares_one_resample_across_arms_seeds_and_floor`
    covers. The seed-paired form is kept here because it is the same shape as the quality
    functional, not because the two orders could disagree.
    """
    mod = qcc()
    l_perp = 1.0
    for dense in ({0: 1.0, 1: 2.0, 2: 100.0},
                  {0: 3.0, 1: 0.5, 2: 7.0},
                  {0: 1e-3, 1: 1.0, 2: 1e3}):
        per_seed = float(np.median([l_perp / dense[s] for s in dense]))
        pooled = l_perp / float(np.median(list(dense.values())))
        assert per_seed == pytest.approx(pooled), "the identity should hold for any odd count"
        assert mod.floor_point(l_perp, dense) == pytest.approx(per_seed)


@pytest.mark.falsifier
def test_floor_is_not_divided_by_a_single_arbitrary_seed():
    """What the floor denominator can still get wrong: picking one seed instead of the median."""
    mod = qcc()
    l_perp = 1.0
    dense = {0: 1.0, 1: 2.0, 2: 100.0}
    correct = float(np.median([1.0, 0.5, 0.01]))            # 0.5
    for wrong in (l_perp / dense[0], l_perp / dense[2], l_perp / float(np.mean(
            list(dense.values())))):
        assert correct != pytest.approx(wrong)
    assert mod.floor_point(l_perp, dense) == pytest.approx(correct)


@pytest.mark.falsifier
def test_bootstrap_shares_one_resample_across_arms_seeds_and_floor():
    mod = qcc()
    per_trajectory = {
        ("dense", s): np.full(40, 1.0 + 0.01 * s) for s in SEEDS
    }
    per_trajectory.update({("k_prop", s): np.full(40, 1.2 + 0.01 * s) for s in SEEDS})
    per_trajectory.update({("k_next", s): np.full(40, 1.1 + 0.01 * s) for s in SEEDS})

    out = mod.paired_bootstrap(per_trajectory, l_perp={"k_prop": 0.5, "k_next": 0.3},
                               replicates=64, seed=20260811)
    assert out["shared_resample"] is True
    assert out["replicates"] == 64
    # every reported interval must come from the same draw sequence
    assert out["resample_digest"], "the resample must be recorded so a reader can recompute it"
    for key in ("quality_k_prop", "quality_k_next", "floor_k_prop", "floor_k_next"):
        assert out[key]["resample_digest"] == out["resample_digest"], (
            f"{key} used a different resample; intervals would not be comparable"
        )


def test_bootstrap_functional_matches_the_point_functional():
    mod = qcc()
    per_trajectory = {("dense", s): np.full(30, 1.0) for s in SEEDS}
    per_trajectory.update({("k_prop", s): np.full(30, 1.04) for s in SEEDS})
    per_trajectory.update({("k_next", s): np.full(30, 1.02) for s in SEEDS})
    out = mod.paired_bootstrap(per_trajectory, l_perp={"k_prop": 0.0, "k_next": 0.0},
                               replicates=64, seed=1)
    arm = {s: 1.04 for s in SEEDS}
    dense = {s: 1.0 for s in SEEDS}
    point = mod.quality_point(arm, dense)
    low, high = out["quality_k_prop"]["interval"]
    # a degenerate fixture puts the point exactly on the boundary; compare at float tolerance
    # rather than asserting strict containment of a value equal to the endpoint
    assert low - 1e-12 <= point <= high + 1e-12, (
        "the interval must belong to the reported point"
    )
    assert out["quality_k_prop"]["functional"] == "median_of_per_seed_ratios"


# ======================================================================================
# 8  one-shot TEST and the verdict                                  *** falsifier ***
# ======================================================================================

@pytest.mark.falsifier
def test_test_evaluator_runs_exactly_once(tmp_path):
    mod = qcc()
    evaluator = mod.OneShotTestEvaluator(tmp_path / "TEST_ACCESS_ONCE.json")
    evaluator.open()
    with pytest.raises(Exception):
        evaluator.open()


def test_post_test_changes_are_refused(tmp_path):
    mod = qcc()
    sealed = {"K_prop": 4, "K_next": 8, "tolerance": TOLERANCE,
              "ladder": list(BASE_LADDER), "patience": PATIENCE,
              "basis_sha256": "a" * 64}
    for field, value in [("K_prop", 8), ("tolerance", 1.10), ("patience", 30),
                         ("basis_sha256", "b" * 64)]:
        changed = dict(sealed)
        changed[field] = value
        with pytest.raises(Exception):
            mod.assert_unchanged_after_test(sealed, changed)
    assert mod.assert_unchanged_after_test(sealed, dict(sealed))


@pytest.mark.parametrize("q,f,expected", [
    (1.30, 1.20, "REPRESENTATIONAL_SHORTFALL"),
    (1.08, 1.02, "OPTIMISATION_OR_GENERALISATION_SHORTFALL"),
    (1.02, 1.00, "TOLERANCE_MET_UNDER_CONVERGENCE"),
    (1.05, 1.00, "TOLERANCE_MET_UNDER_CONVERGENCE"),      # inclusive at the threshold
    (1.06, 1.05, "REPRESENTATIONAL_SHORTFALL"),           # floor at the threshold
])
def test_verdict_branches_are_exclusive_and_complete(q, f, expected):
    mod = qcc()
    assert mod.verdict(q, f, tolerance=TOLERANCE) == expected


def test_verdict_covers_every_pair():
    """No (q, f) may fall through, and no pair may match two branches."""
    mod = qcc()
    rng = np.random.Generator(np.random.PCG64(9))
    for _ in range(500):
        f = float(rng.uniform(0.8, 1.4))
        q = float(rng.uniform(f, 1.6))       # q >= f always, since R = L_perp + R_parallel
        out = mod.verdict(q, f, tolerance=TOLERANCE)
        assert out in ("REPRESENTATIONAL_SHORTFALL",
                       "OPTIMISATION_OR_GENERALISATION_SHORTFALL",
                       "TOLERANCE_MET_UNDER_CONVERGENCE")


def test_per_seed_ratios_are_always_reported_beside_the_summary():
    mod = qcc()
    assert "per_seed_ratios" in mod.STAGE_E_REQUIRED_FIELDS
    assert "interval" in mod.STAGE_E_REQUIRED_FIELDS
    assert "point" in mod.STAGE_E_REQUIRED_FIELDS
    with pytest.raises(Exception):
        mod.validate_stage_e({"point": 1.0, "interval": (0.9, 1.1)})


# ======================================================================================
# 2b  trajectory-index firewall inside one physical file          *** falsifier ***
#
# TRAIN, VALIDATION and UNUSED share a single train HDF5. "Stage A opened only train files"
# is therefore not evidence that Stage A read only TRAIN. The guard has to work at the
# trajectory index, not at the file.
# ======================================================================================

@pytest.mark.falsifier
def test_stage_a_requests_exactly_the_sealed_train_indices():
    mod = qcc()
    ledger = mod.TrajectoryAccessLedger()
    for j in range(len(FAMILIES)):
        train, validation, unused = mod.split_family(SPLIT_SEED, j)
        mod.request_trajectories(ledger, family_index=j, indices=train,
                                 role="train", stage="A")
        recorded = ledger.requested(j)
        assert sorted(recorded) == sorted(train)
        assert not set(recorded) & set(validation), "VALIDATION outcome read during Stage A"
        assert not set(recorded) & set(unused), "UNUSED outcome read during Stage A"
        assert len(recorded) == 60


@pytest.mark.falsifier
@pytest.mark.parametrize("bad_role", ["validation", "unused"])
def test_trajectory_guard_rejects_non_train_indices_at_stage_a(bad_role):
    mod = qcc()
    ledger = mod.TrajectoryAccessLedger()
    train, validation, unused = mod.split_family(SPLIT_SEED, 0)
    forbidden = validation if bad_role == "validation" else unused
    with pytest.raises(PermissionError):
        mod.request_trajectories(ledger, family_index=0, indices=forbidden,
                                 role="train", stage="A")
    # and a request that merely *includes* one forbidden index must also fail
    mixed = list(train[:59]) + [int(forbidden[0])]
    with pytest.raises(PermissionError):
        mod.request_trajectories(ledger, family_index=0, indices=mixed,
                                 role="train", stage="A")


@pytest.mark.falsifier
def test_reading_all_100_then_slicing_is_refused():
    """The specific shortcut that would silently defeat the split."""
    mod = qcc()
    ledger = mod.TrajectoryAccessLedger()
    with pytest.raises(PermissionError):
        mod.request_trajectories(ledger, family_index=0, indices=list(range(100)),
                                 role="train", stage="A")


def test_stage_a_source_has_no_whole_dataset_read():
    mod = qcc()
    path = CODE / "run_quality_convergence_confirm_v1_stage_a.py"
    assert path.exists(), "Stage A executor missing"
    source = path.read_text(encoding="utf-8")
    for pattern in ('["u"][:]', "['u'][:]", '["u"][()]', "['u'][()]"):
        assert pattern not in source, (
            f"{pattern} reads all 100 trajectories; the sealed TRAIN indices must be "
            "gathered directly from the dataset"
        )


def test_stage_a_access_ledger_reports_zero_for_every_other_role():
    mod = qcc()
    ledger = mod.TrajectoryAccessLedger()
    for j in range(len(FAMILIES)):
        train, _, _ = mod.split_family(SPLIT_SEED, j)
        mod.request_trajectories(ledger, family_index=j, indices=train,
                                 role="train", stage="A")
    summary = ledger.summary()
    assert summary["train"] == 8 * 60
    for role in ("validation", "unused", "selection", "test"):
        assert summary[role] == 0, f"{role} trajectories were read during Stage A"


def test_basis_cap_argument_is_labelled_as_an_upper_bound():
    """`basis_cap` is an API parameter here, not a selector truncation."""
    mod = qcc()
    record = mod.basis_cap_record(train_rows=6240, resolved_rank=6239)
    assert record["basis_cap_argument"] == 6239
    assert record["basis_cap_semantics"] == "algebraic_upper_bound_not_selector_truncation"
    assert record["selector_rank"] == 6239
    assert record["artificial_selector_cap"] is False
    with pytest.raises(ValueError):
        mod.basis_cap_record(train_rows=6240, resolved_rank=256)   # a real truncation


# ======================================================================================
# 5b  Stage B: the reference checkpoints                            *** falsifier ***
# ======================================================================================

def test_stage_b_separates_raw_best_from_the_selected_checkpoint():
    """Two different epochs, two different fields, and only one of them is downstream."""
    mod = qcc()
    # Improvements after epoch 2 must stay sub-min_delta *against the running best*, which does
    # not move. A constant per-epoch decrement fails that: the gap accumulates and clears
    # min_delta every ninth epoch, so the rule never fires. A geometrically shrinking decrement
    # keeps the total remaining improvement inside the threshold forever.
    curve = [1.0, 0.90] + [0.90 - 1e-5 * (1.0 - 2.0 ** -i) for i in range(1, 60)]
    state = mod.replay_stopping(curve, patience=PATIENCE, min_delta=MIN_DELTA)
    assert state["selected_epoch"] == 2
    assert state["raw_best_epoch"] > state["selected_epoch"], (
        "the fixture must separate the two, otherwise the test asserts nothing"
    )
    assert state["stop_reason"] == "PATIENCE"
    assert state["stop_epoch"] == 2 + PATIENCE


@pytest.mark.falsifier
def test_stage_c_receives_the_selected_checkpoint_not_the_raw_best():
    mod = qcc()
    seal = {"arm": "dense", "seed": 0,
            "selected_epoch": 2, "selected_checkpoint_sha256": "a" * 64,
            "raw_best_epoch": 22, "raw_best_checkpoint_sha256": "b" * 64,
            "stop_epoch": 22, "stop_reason": "PATIENCE"}
    assert mod.reference_checkpoint_for_stage_c(seal) == "a" * 64, (
        "handing Stage C the raw argmin checkpoint would silently use a policy the pilot "
        "never validated"
    )
    # a seal cannot select a checkpoint from an epoch the run never reached
    swapped = dict(seal, selected_checkpoint_sha256="b" * 64, selected_epoch=41)
    with pytest.raises(Exception):
        mod.validate_reference_seal(swapped)
    # nor may the selected digest silently become the raw-best digest at a different epoch
    aliased = dict(seal, selected_checkpoint_sha256="b" * 64)
    with pytest.raises(Exception):
        mod.validate_reference_seal(aliased)


@pytest.mark.falsifier
def test_a_validation_index_inside_a_train_batch_is_refused():
    mod = qcc()
    ledger = mod.TrajectoryAccessLedger()
    train, validation, _ = mod.split_family(SPLIT_SEED, 0)
    contaminated = list(train[:59]) + [int(validation[0])]
    with pytest.raises(PermissionError):
        mod.request_trajectories(ledger, family_index=0, indices=contaminated,
                                 role="train", stage="B")
    assert ledger.summary()["train"] == 0, "the request must be refused before it is recorded"


def test_stage_b_may_read_validation_but_only_its_own_indices():
    mod = qcc()
    ledger = mod.TrajectoryAccessLedger()
    train, validation, unused = mod.split_family(SPLIT_SEED, 0)
    assert mod.request_trajectories(ledger, 0, validation, role="validation", stage="B")
    for intruder in (train[0], unused[0]):
        with pytest.raises(PermissionError):
            mod.request_trajectories(ledger, 0, [int(intruder)],
                                     role="validation", stage="B")


@pytest.mark.parametrize("role", ["selection", "test"])
def test_stage_b_refuses_selection_and_test(role, tmp_path):
    mod = qcc()
    with pytest.raises(PermissionError):
        mod.assert_role_openable(role, "B")
    name = {"selection": "NavierStokes2D_valid_61849_0.50000.h5",
            "test": "NavierStokes2D_test_61949_0.50000.h5"}[role]
    victim = tmp_path / name
    victim.write_bytes(b"")
    with pytest.raises(PermissionError):
        mod.open_role_file(victim, role=role, stage="B")


def test_ceiling_in_any_seed_blocks_downstream():
    mod = qcc()
    seals = [{"arm": "dense", "seed": s, "stop_reason": "PATIENCE", "selected_epoch": 30,
              "selected_checkpoint_sha256": "a" * 64} for s in SEEDS]
    assert mod.stage_b_status(seals) == "STAGE_B_COMPLETE"
    seals[2]["stop_reason"] = "CEILING_REACHED"
    assert mod.stage_b_status(seals) == "STAGE_B_CEILING_REACHED"
    with pytest.raises(mod.StageBlocked):
        mod.require_reference_convergence(seals)


def test_stage_b_trace_schema_is_identical_for_every_seed():
    mod = qcc()
    required = {"train_loss_per_epoch", "validation_true_field_mse_per_epoch",
                "raw_best_epoch", "selected_epoch", "stop_epoch", "stop_reason",
                "selected_checkpoint_sha256"}
    assert required <= set(mod.REFERENCE_TRACE_FIELDS)


# ======================================================================================
# 4b  Stage C: the selector, and where it may get its reference from
# ======================================================================================

def test_stage_c_source_does_not_scan_for_checkpoints():
    """The Stage B seal is the only route to a reference checkpoint."""
    path = CODE / "run_quality_convergence_confirm_v1_stage_c.py"
    assert path.exists(), "Stage C executor missing"
    source = path.read_text(encoding="utf-8")
    for pattern in ('glob("*.pt")', "glob('*.pt')", 'rglob("*.pt")', "iterdir()"):
        assert pattern not in source, (
            f"{pattern}: Stage C must take checkpoints from STAGE_B.json, never from the "
            "directory, and must not re-select a best checkpoint"
        )
    assert "reference_checkpoints" in source


def test_stage_c_refuses_a_stage_b_that_hit_the_ceiling():
    mod = qcc()
    seals = [{"arm": "dense", "seed": s, "stop_reason": "PATIENCE", "selected_epoch": 30,
              "selected_checkpoint_sha256": "a" * 64} for s in SEEDS]
    seals[1]["stop_reason"] = "CEILING_REACHED"
    with pytest.raises(mod.StageBlocked):
        mod.require_reference_convergence(seals)


def test_stage_c_selection_access_is_the_official_valid_file_only():
    mod = qcc()
    assert mod.assert_role_openable("selection", "C")
    for role in ("test", "unused"):
        with pytest.raises(PermissionError):
            mod.assert_role_openable(role, "C")


def test_gain_width_matches_the_parent_basis_rank():
    """A gain vector shorter than R would silently truncate the ladder."""
    mod = qcc()
    with pytest.raises(ValueError):
        mod.assert_gain_width(256, 6239)
    assert mod.assert_gain_width(6239, 6239)


# ======================================================================================
# 6b  Stage D: the arms                                            *** falsifier ***
# ======================================================================================

@pytest.mark.falsifier
def test_stage_d_uses_the_sealed_index_sets_not_a_prefix():
    mod = qcc()
    # K must equal the index-set length; an earlier revision of this fixture declared 32 with
    # four indices and failed on its own inconsistency rather than on the behaviour under test.
    stage_c = {"K_prop": 4, "K_next": 6,
               "index_sets": {"K_prop": [0, 3, 7, 9], "K_next": [0, 3, 7, 9, 11, 12]}}
    got = mod.arm_indices(stage_c, "K_prop")
    assert list(got) == [0, 3, 7, 9]
    assert list(got) != list(range(4)), "a prefix was substituted for the selected set"
    with pytest.raises(Exception):
        mod.arm_indices({"K_prop": 4, "index_sets": {"K_prop": [0, 1, 2]}}, "K_prop")


def test_stage_d_reuses_the_stage_b_dense_checkpoints():
    path = CODE / "run_quality_convergence_confirm_v1_stage_d.py"
    assert path.exists(), "Stage D executor missing"
    source = path.read_text(encoding="utf-8")
    assert "reference_checkpoints" in source, "dense must be reused from the Stage B seal"
    assert "reference_dense" not in source.split("def train_arm")[-1], (
        "Stage D must not retrain dense; the denominator is already sealed"
    )


@pytest.mark.parametrize("role", ["test", "unused"])
def test_stage_d_refuses_test_and_unused(role):
    mod = qcc()
    with pytest.raises(PermissionError):
        mod.assert_role_openable(role, "D")
