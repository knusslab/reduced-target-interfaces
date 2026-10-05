# Scientific protocol summary

- Dataset: released PDEBench 1D diffusion-sorption.
- Fixed target view: orthogonal K=8, indices `[0,1,2,3,4,5,6,8]`; no reselection in this experiment.
- Roles: TRAIN 480 trajectories, VALIDATION 160, previously unopened evaluation role 160; original selection and test values are not reused by this successor.
- One-step pairs: 20 per admitted trajectory, input frames `0,5,...,95`, target frame `input+1`.
- Learners: MLP dense/compact and FNO dense/compact.
- New seeds: `10,11,12,13,14` for both learner families.
- MLP: hidden width 256, Adam 1e-3, batch 64, max 300 epochs, patience 20.
- FNO: modes 16, width 32, 74,209 trainable parameters, Adam 1e-3, batch 32, max 120 epochs, patience 12.
- Dense models use full-field target loss. Compact arms receive only the eight target coefficients during fitting and decode through the fixed K=8 basis for dense-field scoring.
- Dense models must beat one-step persistence on validation before held-out evaluation.
- Primary statistic: family medians of compact/dense held-out MSE ratios; `Q_port` is the larger family median; pass limit 1.05.
- All ten seed ratios are reported. A 10,000-draw paired 160-trajectory cluster bootstrap is descriptive uncertainty, not an additional acceptance rule.
- The held-out role is evaluated once after training/checkpoint selection is complete.

The original frozen protocol is bound by SHA-256 `52275a1302092801b8bd16d5818cf24ce5212f3bee4df3428e30bee1d2e66dba`. This reviewer summary removes author-side operational bookkeeping while preserving the scientific contract used by the released source.
