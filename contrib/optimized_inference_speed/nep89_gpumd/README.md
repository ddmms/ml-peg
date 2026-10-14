# nep89_gpumd (placeholder)

This directory is a **placeholder** for the NEP89 container that runs MD with
[GPUMD](https://github.com/brucefan1983/GPUMD). The container is developed in a
separate repository; once it exists, replace `container.def` with its definition
at a pinned commit and fill in the repository URL and commit below.

- Repository: `<TBD>`
- Commit: `<TBD>`

Building the placeholder fails on purpose.

## Expected labels

| Label | Value |
| --- | --- |
| `mlpeg.model` | `nep89` (the ML-PEG model name) |
| `mlpeg.precisions` | `fp32` (whatever the GPUMD build honestly supports) |
| `mlpeg.weights_sha256` | `75168ece02e840e4a32644f982b78d43cba697f5b64b4c8134ab66c7a8c28be1` |
| `mlpeg.contract_version` | `1.0` |

The checksum is that of `nep89_20250409.txt` from GPUMD v4.0, the file ML-PEG
downloads for the `nep89` model. The container must bake in exactly this file:
static validation compares the container's energies and forces against calorine's
`CPUNEP` in float64 using the same file, and the checksum is checked against
`ml_peg/calcs/molecular_dynamics/optimized_inference_speed/models.yml`.
