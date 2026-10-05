# QUALITY_CONVERGENCE_CONFIRM_V1 — is the rank boundary a rank result or a budget result?

    STATUS   FROZEN BEFORE EXECUTION, 2026-08-11
             Every rule is fixed here. The only thing still outstanding is the SHA-256 of the
             cohort files, which is bound in the Stage 0 data seal after the freeze and by
             design: naming the digests first would let the rules be chosen to suit the data.
             Nothing below may change after this point. If something must, it becomes V2.
    LINEAGE  BIGDATA_QUALITY_2D_CONFIRM_V1…V6 exist and are unchanged. This does not supersede
             them, does not reinterpret V6, and shares no cohort with any of them.

## The single question

> When the model is trained to a validation-defined stopping rule instead of a fixed seven-epoch
> budget, does supervision at the proposed rank meet the declared tolerance, and if not, is the
> shortfall in the representation or in the optimisation?

V6 answered a narrower question than it is being read as answering. Its arms ran for seven optimizer
epochs with no early stopping, no schedule and no best-checkpoint selection, and every arm's training
loss was still falling when the budget ended. See `TRAINING_BUDGET_CONVERGENCE_AUDIT_V1.md`. The
sealed `SELECTED_QUALITY = REFUTED` at `1.050389` is a fixed-budget result and remains one.

This protocol does not revisit V6. It runs the question again on a cohort V6 never touched.

## What it cannot answer

It cannot show that the proposed rank is sufficient in general, on other systems, other backbones or
other tolerances. It cannot establish a rank-selection method as novel; that question is closed
against this line in `NOVELTY_RESWEEP_V2.md`. It measures one thing on one fresh cohort.

## Cohort

Eight PDEArena NS2D families, from the rule recorded in `FRESH_COHORT_INVENTORY_V1.md` before any
outcome was accessed:

    61749  61991  97473  433143  445636  452245  453445  471651

Derivation: exclude the eight V13 families, exclude the eight V6 families, keep only families whose
train, valid and test files are all published upstream, sort ascending, take the first eight. All 52
families satisfy the publication condition, so the rule reduces to the first eight of the 36
eligible. The historical V6 skip of `61749`, `61991`, `97473` and `296423` is **not** inherited; no
record explains it.

Source: Hugging Face `datasets/pdearena/NavierStokes-2D`, revision
`cd99556a883a20acb9102c1f4bfdfae66ae33495`, pinned here rather than at download time.

## Four roles, not three

V6 used TRAIN, SELECTION and TEST. A convergence rule needs a fourth, and it must not be the same
data that chooses the rank, or the reference checkpoint and the rank it produces would be selected on
one split.

    TRAIN        60 trajectories per family     model fitting, POD basis, channel statistics
    VALIDATION   20 trajectories per family     checkpoint selection and stopping, nothing else
    UNUSED       20 trajectories per family     reserved, never read
    SELECTION    official valid file, 25/family reference residual, gains, K_prop
    TEST         official test file,  25/family opened exactly once, at Stage E

Each official train file holds 100 trajectories and each trajectory yields 13 next-step pairs, so the
geometry is fixed before any file is opened:

    TRAIN        8 x 60 x 13 = 6240 rows
    VALIDATION   8 x 20 x 13 = 2080 rows
    UNUSED       8 x 20 x 13 = 2080 rows
    SELECTION    8 x 25 x 13 = 2600 rows
    TEST         8 x 25 x 13 = 2600 rows

### The split rule, fixed here

    SPLIT_SEED = 776131

    perm_j      = Generator(PCG64(SPLIT_SEED + j)).permutation(100)   j = family index, 0..7
    TRAIN       = perm_j[0:60]
    VALIDATION  = perm_j[60:80]
    UNUSED      = perm_j[80:100]

Family index `j` is the position of the family in the ascending cohort list above, so `61749` is
`j = 0`. `SPLIT_SEED` is an arbitrary fixed integer with no meaning; what matters is that it is
written here before any outcome from this cohort has been accessed. It is deliberately not a date,
because date-shaped seeds have twice cost this project an audit to establish that they were not
added after the fact.

The V6 seed `PCG64(20260809 + family_index)` is not reused.

Sealed in Stage 0, per family: the full permutation digest and the TRAIN, VALIDATION and UNUSED index
digests.

VALIDATION never contributes a gradient. SELECTION is opened only after the reference checkpoint has
been chosen on VALIDATION, so no checkpoint is selected using the data that determines the rank.

Enforced, not merely intended. Every artifact carries an access ledger, and two counters are checked
at the point they matter rather than asserted afterwards:

    Stage A, B artifacts     selection_files_opened == 0    else BLOCKED_SELECTION_TOO_EARLY
    Stage A..D artifacts     test_files_opened == 0         else BLOCKED_TEST_TOO_EARLY

Stage C refuses to start unless the Stage B seal exists and names the chosen checkpoint digests. A
prose ordering that no counter enforces is the kind of clause this project has twice found to have
been silently skipped.

## Basis, and the removal of the artificial cap

The POD parent basis is computed from the **TRAIN** matrix, not from SELECTION. SELECTION supplies
the rows on which gains are estimated; it does not bound the basis. V6's Stage A records the chain
exactly:

    train_rows            6240
    r_gram_resolved       6239        = 6240 - 1, centred
    basis_cap              256        an argument, not a property of the data
    selector_basis_rank    256        = min(basis_cap, r_gram_resolved)

This protocol removes the cap.

    R := r_gram_resolved(TRAIN), from the canonical Gram contract
    selector_basis_rank := R

`R` is determined at Stage A from this cohort's own TRAIN matrix and is not written here as a number.
The new geometry gives `R <= 6239`, and V6 resolved exactly 6239 on the same shape, but assuming the
value in advance is the mistake this protocol exists to avoid.

One parameter, `basis_cap = 256`, has two realised defects and one unrealised vulnerability behind it.
In V8 it made `K_pred = 256` a vacuous admission and it silently dropped the energy arm, since
`K_energy = 1487 > 256`; both are defects and both are recorded in
`V8_ARM_COMPLETENESS_AUDIT_V1.md`. In V6 it truncated `R = 6239` to `256` so that the ladder maximum
coincided with the basis rank, which made a vacuous top-rung admission *possible*.

**V6's own result did not go that way.** `K = 4` is the second of ten rungs, nine rungs were feasible,
`censor_reasons` was empty, and the full-range diagnostic held the rank at 4 out to 6239. Nothing here
implies the V6 numbers are wrong. Removing the cap removes the vulnerability, not a result.

## Ladder and ranks

    BASE_LADDER      2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096
    effective_ladder { k in BASE_LADDER : k < R }

    required         len(effective_ladder) >= 2
                     max(effective_ladder) < R

    K_prop           canonical selector's selected_k on SELECTION, primary reference seed
    K_next           the next rung above K_prop in effective_ladder

    censor_reasons non-empty      -> BLOCKED_SELECTOR_CENSORED, no arm is trained
    K_prop is the largest rung    -> BLOCKED_NO_K_NEXT, no arm is trained

`K_prop` and `K_next` are defined by rule, not by value. Nothing hardcodes `4` or `8`. If this cohort
proposes `16`, the arms become `16` and `32` and no line of this protocol changes.

**`R` is a boundary, never a candidate.** The tail is a suffix sum over the retained basis with a
terminal zero appended, so `tail[R] = 0` and a rung placed at `R` would be feasible for any data
whatsoever. That is exactly how V8 arrived at an uninformative `K_pred = 256`. `BASE_LADDER` therefore
stops at a fixed `4096` rather than extending to `R`, and `effective_ladder` takes only rungs
strictly below `R`.

If the selector returns the largest effective rung, the run stops. A rung selected at the ladder
ceiling is a censored selection, not a demonstration that the rank suffices, and the canonical
selector flags it as `ladder_cap`.

The canonical selector `bigdata_predictive_gain_selector_v1.canonical_predictive_gain_selector` is
called directly. It is not reimplemented inline. Its `censor_reasons` is recorded, and a non-empty
value terminates the run as `BLOCKED_SELECTOR_CENSORED` before any comparison arm is trained.
`used_max_rung_fallback == False` is not accepted as evidence of an uncensored selection.

## Arms

    dense      full-resolution supervision, the reference and the comparator
    K_prop     supervision stored in the leading K_prop gain-ordered directions
    K_next     the same at the next rung

Three model seeds each, nine trained models. The dense reference trained in Stage B is reused as the
dense comparator; it is not trained twice.

Held identical across arms: cohort, splits, basis, channel statistics, backbone, optimizer, learning
rate, batch size, batch order per seed, convergence rule, evaluator. Only the supervision
representation differs.

## Convergence rule

Every arm trains to the same validation-defined stopping rule and is evaluated at its best
validation checkpoint, not at its terminal epoch.

    optimizer            as in the V6 lineage, unchanged
    validation cadence   every epoch, on VALIDATION only
    checkpoint           best validation loss

    patience             20
    min_delta            1e-3, relative: an epoch improves when (b - v) / b >= min_delta
                         against the running best b
    max_epochs           300

An arm that reaches `max_epochs` without the patience counter firing is recorded as
`CEILING_REACHED` and is not called converged.

### `min_delta` governs the checkpoint as well as the counter

This is stated explicitly because the two are separable and the choice changes which weights are
evaluated on TEST.

    on an epoch with (b - v) / b >= min_delta:  update the running best, the kept checkpoint,
                                                and reset the counter
    otherwise:                                  increment the counter, keep the old checkpoint

So the evaluated checkpoint is the **last significant improvement**, not the raw `argmin` of the
validation curve up to the stopping epoch. A raw `argmin` would credit the run with weights the
stopping policy never saved, and would require checkpointing on every downward flicker.

`e_best`, used only for the pilot's `max_epochs` derivation, is a different quantity and remains the
raw `argmin` over the full curve.

### Provenance of the three constants

Fixed by `CONVERGENCE_POLICY_PILOT_V1` on the eight already-spent V6 families, twelve cells to a
200-epoch ceiling with no early stopping, opening no valid or test file of any cohort. Six of the
nine candidate rules survived; the three with `patience = 10` were rejected because dense seeds 0
and 1 lost 1.9 and 1.4 percent to early truncation, above the declared `gap <= 1.01`. The winner is
the cheapest survivor at `max stop_epoch = 103` and `max gap = 1.0044`. `max_epochs` follows the
declared derivation from the largest full-curve best epoch over all pilot cells, 193, giving
`ceil(1.5 x 193) = 290` rounded up to 300.

    pilot DECISION.json      7322732657e3ecad87ceafe8025b8822b440a369051609180ab09d213cef7318
    pilot RULE_REPLAY.json   a01d160da858cc93cca82843e659417ec9383c017707e080d2976db0f32a13b8
    pilot MANIFEST.json      6cbcc93e18228f6d5bacea4c133df0a2f06903fcf1da4ad1348225f23f06a2e4

Two facts from that pilot bear on this protocol and are recorded because they were not anticipated.
The arm that needed the most patience was **dense**, not the large-head stress arm, so a pilot run
only on compressed arms would have chosen a rule that under-trains the denominator of every ratio
here. And the ceiling was set by the large-head arm, whose best checkpoint arrived at epoch 193 out
of 200. Both inclusions in the pilot earned their place, for different reasons.

No pilot number is a result of this line, and none appears in the manuscript.

### Where the constants come from

`max_epochs`, `patience` and `min_delta` were calibrated on a **development-only pilot over the
already-consumed V6 families**, using their TRAIN and VALIDATION rows and opening no TEST file of any
cohort. The values below are frozen and applied unchanged to the confirmation cohort.

They are deliberately **not** tuned on the new cohort's VALIDATION. Doing so would leave TEST
untouched but would still develop the training policy against the confirmation data, which weakens
the outcome-blind claim for no benefit. Those families are already spent; spending them again on a
policy question costs nothing.

The permitted sentence afterwards:

> the convergence rule was calibrated on previously consumed development families and frozen before
> any outcome from the confirmation cohort was inspected.

### Traces, for every arm including dense

Recorded per arm and seed, in the same format, and required for the artifact to be valid:

    train loss per epoch          validation loss per epoch
    best_epoch                    stop_epoch
    stop_reason                   in {patience, max_epochs}
    gradients finite throughout   checkpoint digest

**Dense is not exempt.** In V6 the dense reference was trained in Stage B, which recorded no loss
series at all, while the compressed arms trained in Stage C recorded theirs. The consequence is that
the denominator of the headline ratio has no convergence trace anywhere in that run and its
convergence cannot be assessed even in principle. Here both halves of every ratio carry identical
instrumentation, and an arm missing its curves fails the artifact.

An arm that terminates at `max_epochs` has not satisfied the stopping rule, and the artifact records
that as `CEILING_REACHED` for that arm. It is reported, not silently treated as converged.

This is operational convergence against a declared rule. It is not a claim of global optimality, and
the manuscript must not upgrade it into one.

## The floor is an endpoint, not a gate

For each rank the evaluated TEST error decomposes as

    R_S  =  L_perp(S)  +  R_parallel(S)

where `S` is the **exact index set the arm was trained on**, `S_prop = gain_order[:K_prop]` and
`S_next = gain_order[:K_next]`, and

    L_perp(S) = mean over TEST rows of || (I - P_{U_S}) (y - mu) ||^2 / D

with `U_S` the rows of the parent basis at those indices. It involves no model and no training, and
no training budget can reduce it.

**It is not the floor of a POD prefix.** On the V6 cohort the gain order happened to select
`[0, 1, 2, 3]`, which coincides with the leading prefix, and that coincidence must not be inherited.
A fresh cohort may return a gain subset that skips leading directions, in which case a prefix floor
would understate the true floor and the whole decomposition would be wrong in the direction that
flatters the method. The floor is computed from the recorded index set and its digest, the same one
the arm's basis was built from.

`L_perp` is computed **at Stage E, together with everything else**, and is never used to decide
whether an arm runs. Deciding arm execution from a TEST-derived quantity would be selecting an
ablation on TEST, which is forbidden. All nine models are trained and sealed before TEST opens.

## Statistic, fixed now

Point and interval use the same functional. The Stage F mismatch in V6, where a ratio of seed-median
errors was reported beside a bootstrap of median seed-ratios, is the reason this is stated here
rather than left to the evaluator. See `STAGE_F_AGGREGATION_SENSITIVITY_V1.md`.

    quality      r_s(K) = MSE(K, seed s) / MSE(dense, seed s)
                 point  = median over seeds of r_s(K)

    floor        f_s(K) = L_perp(K) / MSE(dense, seed s)
                 point  = median over seeds of f_s(K)

    interval     trajectory-clustered bootstrap of the same median, 10000 replicates, PCG64,
                 one resample per replicate shared by every arm, seed and floor quantity

    reported     all three per-seed ratios alongside every summary, never the summary alone

`L_perp(K)` has no seed dependence, so `f_s(K)` varies only through the denominator. That is
intentional: it puts floor and quality on the same denominator and the same functional, so the two
are directly comparable.

Tolerance: `1.05`, unchanged from the V6 lineage, declared before any run.

## Outcomes, decided now

Per rank, with `q` the quality point and `f` the floor point:

    f >= 1.05
        REPRESENTATIONAL_SHORTFALL
        the retained basis cannot meet the tolerance on this cohort at this rank under any
        predictor. Optimisation is not the explanation and no training budget changes it.

    f < 1.05 and q > 1.05
        OPTIMISATION_OR_GENERALISATION_SHORTFALL
        the representation admits a passing model and the trained one did not reach it.

    q <= 1.05
        TOLERANCE_MET_UNDER_CONVERGENCE
        the fixed-budget refutation in V6 does not survive a validation-defined budget on a
        fresh cohort. V6's verdict is not amended; the two are reported side by side as what
        they are, different budgets on different cohorts.

Interval placement relative to `1.05` is reported for each, and an interval containing the threshold
is reported as containing it. No outcome is a success or a failure of the protocol. All three are
publishable and the protocol is not run to obtain any particular one.

`K_prop` and `K_next` are reported separately and never combined into one scalar.

## Stages

    0   bind cohort, revision, file digests, splits, ladder, seeds, convergence constants,
        statistic. No data content is read. Create-only seal.
    A   TRAIN only. POD basis, channel statistics, L_perp precomputation inputs.
    B   TRAIN and VALIDATION. Train the three dense references to the convergence rule.
    C   SELECTION opens. Reference residual, gains, canonical selector, K_prop and K_next.
        censor_reasons must be empty.
    D   TRAIN and VALIDATION. Train the K_prop and K_next arms to the same rule. Seal all nine
        checkpoints, the basis, both ranks, every curve and every digest.
    E   TEST opens, exactly once. Compute MSE for all three arms, L_perp for both ranks, the
        paired statistic and the bootstrap. Write the verdict.

TEST files are not downloaded until Stage D is sealed, and `test_files_opened` is recorded as zero in
every artifact before Stage E.

## Forbidden

- opening TEST more than once, or before Stage D is sealed;
- choosing a rank, an arm, a seed or a stopping constant after seeing any TEST quantity;
- using `L_perp` to decide whether an arm runs;
- reusing any V6 or V13 family;
- reimplementing the selector inline;
- proceeding when `censor_reasons` is non-empty;
- amending, re-running or reinterpreting V6, or merging its numbers into this table;
- reading a storage, system or ordering conclusion out of this protocol.

## Left to bind before this is FROZEN

Closed since the first draft, and now stated above rather than deferred:

    basis rule       R = r_gram_resolved(TRAIN), no artificial cap
    ladder rule      BASE_LADDER to 4096, effective rungs strictly below R
    split rule       SPLIT_SEED = 776131, perm_j[0:60] / [60:80] / [80:100]

    convergence rule patience 20, min_delta 1e-3, max_epochs 300, from the sealed pilot

    source revision  Hugging Face `pdearena/NavierStokes-2D`, commit
                     cd99556a883a20acb9102c1f4bfdfae66ae33495, resolved 2026-08-11 from the
                     dataset API. Pinned here rather than at download time: a revision is
                     metadata, not an outcome, so fixing it before the freeze is strictly
                     stronger. Any other revision invalidates the run.

One item remains.

1. **The SHA-256 of the 24 cohort files.** Computed after download and before Stage A, and bound in
   the Stage 0 seal. TEST files are downloaded but not opened; their access count stays at zero
   until Stage E.

   Counts, which are not the same number: **24 files are sealed, 16 are downloaded.** The eight
   confirmation TRAIN shards are already on `prbig4060` among the 52, verified in
   `FRESH_COHORT_INVENTORY_V1.md`. Only the eight `valid` and eight `test` shards are new, roughly
   1.1 GB each group. They go to the external volume, never to the root volume.

Digests must not precede the freeze: a protocol that names its data digests before its rules is a
protocol whose rules could have been chosen to suit the data.

Also carried forward from the pilot, and binding here: **centre in float64.** A float32 mean
accumulated over 6240 rows leaves a residual large enough that the Gram-resolved rank returns 6240
where a centred 6240-row matrix can have rank at most 6239, admitting one spurious near-null
direction into the parent basis. Measured on this geometry. The canonical Stage A centring and its
preregistered residual gate apply unchanged, and `R` is asserted to satisfy `R <= n_rows - 1`.

## Amendment record

Everything below was corrected before this protocol was frozen and before any confirmation-cohort
file was opened. It is recorded rather than quietly folded in.

**The checkpoint semantics were ambiguous and are now fixed.** An earlier draft defined the kept
checkpoint two incompatible ways: `min_delta` gates the checkpoint update, and separately
`e_sel = argmin over 1..stop_epoch`. Those differ whenever an epoch improves on the running best by
less than `min_delta`. The pilot executor implemented the first. The protocol now states the first
explicitly, because it is what an early-stopping implementation can actually deliver.

The pilot's decision was recomputed from `CURVES.jsonl` under both semantics before the text was
changed:

    semantics A, min_delta gates the checkpoint   6/9 survive, patience 20, min_delta 1e-3, maxstop 103
    semantics B, raw argmin up to stop_epoch      6/9 survive, patience 20, min_delta 1e-3, maxstop 103

Identical. The constants above do not depend on the resolution, and the pilot is not re-run.

**The floor is over the selected index set, not a prefix.** Corrected above. The V6 coincidence
`gain_order[:4] == [0, 1, 2, 3]` is not inherited.

**The download count was wrong.** An earlier draft implied 24 files are fetched. Sixteen are.

**Centring precision was added** after the pilot's timing smoke returned a parent rank of 6240 where
6239 is the algebraic maximum.

## Execution

Create-only outputs under a fresh namespace. Nothing under any existing results directory is written
to. Failures are recorded as failures and no arm is substituted.

    results/quality_convergence_confirm_v1/stage_<s>_<attempt>/*.json

Large intermediates and downloads stay on the external volume; the root volume is not used as
scratch.
