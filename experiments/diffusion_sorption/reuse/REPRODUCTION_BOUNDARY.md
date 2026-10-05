# Reproduction boundary

## Included exactly

- frozen runner SHA-256 `5a156b4f44d096630434f91c21b4d37e76ddc9ae1208772a6711fc64488a5712`;
- frozen independent auditor SHA-256 `55cbdc4a63560bbc9a3b5051bdc0d69dfb55a68a44a49759bc8b3b0cebc9df3c`;
- frozen failure-first tests SHA-256 `c93dd1b693cc9680d5d80a0beddf383cb4e1764babb7230f9b446daf9d64145a`;
- exact K=8 basis, training mean, and selected-index arrays bound in `SOURCE_BINDING.json`.

## Public external input

The runner expects the published PDEBench `1D_diff-sorp_NA_NA.h5` bytes whose URL, size, SHA-256, and publisher MD5 are recorded in `SOURCE_BINDING.json`. Those 4.2 GB benchmark bytes are not redistributed here.

## Not included

The review archive does not redistribute trained checkpoints or the final per-trajectory held-out error object. It therefore does not support a no-compute replay of the complete training trajectory from this directory alone. The manuscript's scalar decisions and seed ratios are separately included and checked by the top-level reviewer archive.

The original execution requested deterministic CUDA algorithms and disabled TF32, but exact production dependency versions were not recovered into a portable lockfile. Consequently, this snapshot supports source inspection and frozen failure-first verification; it does not claim bitwise-identical training on arbitrary hardware.
