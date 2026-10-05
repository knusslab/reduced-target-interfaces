# Reduced Target Interfaces

Code, experiment protocols, and result artifacts for:

**Qualifying Reduced Scientific-Data Views as Training Interfaces for Scientific AI**

Repository: https://github.com/knusslab/reduced-target-interfaces

This repository supports the experiments reported in the paper. It is intentionally narrower than the private research workspace: it contains the code and result artifacts needed to understand and reproduce the reported experimental scope, while excluding manuscript source, private research infrastructure, external datasets, and large checkpoints.

## What is included

| Study | Public contents | Role in the paper |
| --- | --- | --- |
| NS2D / PDEArena | experiment code, frozen protocol, tests, portable control, reference results | direct training after an inconclusive projection-error screen |
| Diffusion-sorption / PDEBench | main experiment code, reuse study, matched-rank control, tests, reference results | aggregation sensitivity and learner-family reuse |
| Broadband 3D | recovered spectral solver, later protocol-faithful reimplementation, result summary | a representation rejected by the projection-error screen |
| CFDBench | frozen protocol and author-generated reference results | reuse of one fixed representation across learner families |

The recovered CFDBench executable model definitions are not redistributed because their upstream source-code redistribution terms were not established.

## Quick start

The public runtime environment previously used for the packaged code is recorded in `repro/requirements-runtime-tested.txt`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r repro/requirements-runtime-tested.txt
python -m pytest -q
```

The previously verified public code completed **150 tests with 5 intentional skips**. The skipped checks require archived terminal arrays that are not redistributed here. The final release keeps the experiment Python bytes unchanged from that verified package.

## Paper-to-code map

### NS2D

- code: `experiments/ns2d/code/`
- protocol: `experiments/ns2d/protocols/QUALITY_CONVERGENCE_CONFIRM_V1.md`
- tests: `experiments/ns2d/tests/`
- portable control: `experiments/ns2d/controls/v033/`
- reference results: `experiments/ns2d/reference_results/`

### Diffusion-sorption

- primary experiment: `experiments/diffusion_sorption/main/`
- learner-family reuse: `experiments/diffusion_sorption/reuse/`
- matched-rank control: `experiments/diffusion_sorption/matched_control/`
- reference results: `experiments/diffusion_sorption/reference_results/`

### Broadband 3D

- recovered historical solver: `experiments/broadband3d/solver/ns3d_spectral.py`
- later protocol-faithful reimplementation: `experiments/broadband3d/reimplementation/`
- source-recovery boundary: `experiments/broadband3d/provenance/ORIGINAL_EXECUTOR_RECOVERY_STATUS.md`
- paper-facing result summary: `experiments/broadband3d/reference_results/REFERENCE_RESULT_SUMMARY.json`

The reimplementation is not presented as the recovered historical orchestration, training launcher, or TEST evaluator.

### CFDBench

- frozen protocol: `experiments/cfdbench/CFDBENCH_TUBE_PB_CROSS_LEARNER_PORTABILITY_V1_3_GPU_PROTOCOL.md`
- reference results: `experiments/cfdbench/reference_results/`
- source-availability boundary: `experiments/cfdbench/SOURCE_AVAILABILITY.md`

## Reported values

A compact reader-facing index is in [RESULTS.md](RESULTS.md). The same values are available in machine-readable form at `results/reported_results.json`.

The experiment-specific JSON files under `experiments/*/reference_results/` remain the closest public records to the preserved study outputs.

## External data

No external scientific dataset or trained checkpoint is redistributed in this repository.

See [DATASETS.md](DATASETS.md) for pinned dataset identifiers, checksums, and acquisition instructions.

## What is intentionally not included

- the paper's LaTeX/Overleaf source;
- private paths, hostnames, credentials, or research-governance files;
- external datasets and trained checkpoints;
- duplicate historical source trees;
- CFDBench executable model definitions with unresolved redistribution terms;
- archived terminal arrays that are not needed to inspect the reported paper-level results.

## Third-party material

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The repository-level MIT license applies to author-written material distributed under that license; third-party resources remain governed by their own licenses and notices.

## License

Author-written software in this repository is released under the [MIT License](LICENSE), subject to the third-party boundaries documented above.

## Citation

If you use this repository, please cite:

> Jaegeun Jang and Young-Woo Kwon, *Qualifying Reduced Scientific-Data Views as Training Interfaces for Scientific AI*, ICDM 2026 ADSD Workshop.

Final proceedings metadata can be added after publication.
