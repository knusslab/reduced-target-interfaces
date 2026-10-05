# Reduced Target Interfaces

Code, experiment protocols, checkpoints, and result artifacts for:

**Qualifying Reduced Scientific-Data Views as Training Interfaces for Scientific AI**

This repository supports the experiments reported in the paper. The central question is whether a fixed reduced representation can be used as a training target while preserving performance at the original full-field endpoint.

## Quick verification

The tested runtime dependencies are recorded in `repro/requirements-runtime-tested.txt`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r repro/requirements-runtime-tested.txt
python -m pytest -q
```

## Studies

| Study | Public material | Paper role |
| --- | --- | --- |
| NS2D / PDEArena | experiment code, frozen protocol, tests, portable control, reference results | direct training after an inconclusive projection-error screen |
| Diffusion-sorption / PDEBench | main experiment code, learner-family reuse, matched-rank control, checkpoints, reference results | direct training and aggregation sensitivity |
| Broadband 3D | exact recovered spectral solver, protocol reference implementation, reference results | decisive projection-error rejection |
| CFDBench | frozen protocol and author-generated reference results | reuse of one fixed representation across learner families |

## Datasets

The external datasets are obtained from their original publishers:

- **NS2D / PDEArena:** [pdearena/NavierStokes-2D](https://huggingface.co/datasets/pdearena/NavierStokes-2D)
- **Diffusion-sorption / PDEBench:** [PDEBench dataset on DaRUS](https://doi.org/10.18419/darus-2986)
- **CFDBench:** [chen-yingfa/CFDBench](https://huggingface.co/datasets/chen-yingfa/CFDBench)
- **Broadband 3D:** synthetic data generated from the study's numerical-solver/protocol lineage

Pinned revisions, file identifiers, checksums, and example download commands are in [DATASETS.md](DATASETS.md).

## Checkpoints

The exact dense-target and reduced-target checkpoints for the reported **diffusion-sorption K = 8** study are included under:

```text
experiments/diffusion_sorption/main/checkpoints/
```

The corresponding basis, training mean, selected coefficient indices, and SHA-256 manifest are stored alongside them so that the reduced predictions can be decoded in the same output space used by the paper.

## Paper-to-code map

### NS2D

- code: `experiments/ns2d/code/`
- POD basis implementation: `experiments/ns2d/code/pod_basis.py`
- predictive-gain selector: `experiments/ns2d/code/predictive_gain_selector.py`
- protocol: `experiments/ns2d/protocols/QUALITY_CONVERGENCE_CONFIRM_V1.md`
- tests: `experiments/ns2d/tests/`
- portable control: `experiments/ns2d/controls/v033/`
- reference results: `experiments/ns2d/reference_results/`

### Diffusion-sorption

- primary experiment: `experiments/diffusion_sorption/main/`
- checkpoints and decode artifacts: `experiments/diffusion_sorption/main/checkpoints/`
- learner-family reuse: `experiments/diffusion_sorption/reuse/`
- matched-rank control: `experiments/diffusion_sorption/matched_control/`
- reference results: `experiments/diffusion_sorption/reference_results/`

### Broadband 3D

- recovered historical solver: `experiments/broadband3d/solver/ns3d_spectral.py`
- protocol reference implementation: `experiments/broadband3d/reference_implementation/`
- reference results: `experiments/broadband3d/reference_results/`

The historical solver is preserved exactly. The protocol reference implementation is provided to make the frozen Broadband contract executable and testable; it is not presented as the byte-identical historical training/evaluation executor.

### CFDBench

- frozen protocol: `experiments/cfdbench/CFDBENCH_TUBE_PB_CROSS_LEARNER_PORTABILITY_V1_3_GPU_PROTOCOL.md`
- reference results: `experiments/cfdbench/reference_results/`

See `experiments/cfdbench/SOURCE_AVAILABILITY.md` for the source-redistribution boundary of the recovered upstream model definitions.

## Reported results

A compact index of the paper's main values is in [RESULTS.md](RESULTS.md), with machine-readable values in `results/reported_results.json`.

The experiment-specific JSON files under `experiments/*/reference_results/` are the closest public records to the reported study outputs.

## Third-party material

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dataset and upstream-source licensing notes.

## License

Author-written software in this repository is released under the [MIT License](LICENSE), subject to the third-party boundaries documented above.

## Citation

If you use this repository, please cite:

> Jaegeun Jang and Young-Woo Kwon, *Qualifying Reduced Scientific-Data Views as Training Interfaces for Scientific AI*, ICDM 2026 ADSD Workshop.

Final proceedings metadata can be added after publication.
