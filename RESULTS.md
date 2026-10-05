# Reported results

This page indexes the main quantitative results used in the paper. Machine-readable values are in `results/reported_results.json` and the experiment-specific reference-result files.

The paper compares reduced-target and dense-target models by held-out full-field MSE. The prespecified ratio criterion is **1.05**.

| Study | Setting | Reported value | Interpretation | Public source |
| --- | --- | ---: | --- | --- |
| NS2D | K = 32 direct training | median ratio **0.9747** | direct training meets the criterion | `experiments/ns2d/reference_results/REFERENCE_RESULT_SUMMARY.json` |
| NS2D | K = 32 projection-error screen | lower bound **0.6997** | below 1.05, so the screen is inconclusive | `experiments/ns2d/reference_results/STAGE_E.json` |
| CFDBench | K = 16 cross-family reuse | larger family median **0.5335** | the fixed representation remains within the criterion for both learner families | `experiments/cfdbench/reference_results/REFERENCE_RESULT_SUMMARY.json` |
| Diffusion-sorption | K = 8 main study | median ratio **0.7076** | median criterion is met, but one of three runs exceeds 1.05 | `experiments/diffusion_sorption/reference_results/REFERENCE_RESULT_SUMMARY.json` |
| Diffusion-sorption | K = 8 reuse | larger family median **0.9239** | median criterion is met, while 3 of 10 runs exceed 1.05 | `experiments/diffusion_sorption/reuse/RESULT_BINDING.json` |
| Broadband 3D | K = 80 screen | lower bound **10.69345** | representation error alone rejects the candidate | `experiments/broadband3d/reference_results/REFERENCE_RESULT_SUMMARY.json` |
| Broadband 3D | complete training span K = 95 | lower bound **10.5994** | even the complete rank-95 training span is above 1.05 | `experiments/broadband3d/reference_results/REFERENCE_RESULT_SUMMARY.json` |

## NS2D

K = 32 seedwise reduced/dense ratios:

```text
0.9746798011
0.9181681057
0.9910934153
```

Median: **0.9746798011**.

The normalized projection-error lower bound is **0.6997430028**, so the screen alone does not decide the case.

## CFDBench

For the fixed K = 16 representation:

- FNO family median: **0.3626839444**
- U-Net family median: **0.5334728566**
- cross-family decision statistic: **0.5334728566**

All six reported seedwise ratios are below 1.05.

## Diffusion-sorption

Main K = 8 seedwise ratios:

```text
0.6180599887
1.3140997436
0.7076493522
```

Median: **0.7076493522**.

The reuse study contains 10 seedwise ratios across MLP and FNO families. The larger family median is **0.9239**, while **3/10** individual runs exceed 1.05.

## Broadband 3D

- K = 80 normalized projection-error lower bound: **10.69345**
- K = 80 preserved direct-test median ratio: **10.72040**
- complete training span K = 95 lower bound: **10.5994**

The K = 95 value comes from a separate fresh screening set and does not require another reduced-target training run.

## NS2D implementation-level measurements

The paper also reports preserved system-level measurements for the K = 32 implementation:

| Quantity | Dense | Reduced-target pipeline | Reduced / dense |
| --- | ---: | ---: | ---: |
| complete training-split object size | 780.0 MiB | 392.824 MiB | 0.504× |
| fixed-step epoch time | 0.3235 s | 0.1887 s | 0.583× |

These are complete-implementation measurements from the reported experiment environment; they are not presented as a causal estimate of target-byte savings alone.
