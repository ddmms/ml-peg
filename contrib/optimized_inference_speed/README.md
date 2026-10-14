# Optimised inference speed containers

Containers for the ML-PEG optimised inference speed benchmark
(`molecular_dynamics/optimized_inference_speed`). Each directory holds one
container: a `container.def` and a short README. ML-PEG owns the workload, runs the
container, measures it from outside, validates its results and plots them.

## Contributing a container

1. Add `contrib/optimized_inference_speed/<name>/container.def` following the
   contract below, and a short `README.md`.
2. Add `<name>` to
   `ml_peg/calcs/molecular_dynamics/optimized_inference_speed/models.yml`, with the
   ML-PEG model it implements and the sha256 of its weights. The ML-PEG model's
   calculator is used as the float64 reference for static validation.
3. Check it builds and runs:

   ```bash
   ml_peg inference-speed stage --base-dir base_cells --out campaign
   ml_peg inference-speed build --model <name>
   ml_peg inference-speed run --campaign campaign --models <name>
   ml_peg inference-speed analyse --campaign campaign
   ```

`_mock/` is a minimal, standard-library-only example.

## Contract v1.0

### Invocation

One process per run:

```bash
apptainer run --nv --containall --cleanenv --net --network none \
  --bind $CAMPAIGN/in:/mlpeg/in:ro --bind $RUN_DIR:/mlpeg/out \
  --env CUDA_VISIBLE_DEVICES=0,OMP_NUM_THREADS=8 \
  <model>.sif --spec /mlpeg/in/spec.json --run-id <material>-n<n_atoms>-r<repeat> \
  --precision <fp64|fp32|mixed> --out /mlpeg/out
```

- `%runscript` accepts exactly these four arguments and exits non-zero on anything
  else, including a precision mode not listed in `mlpeg.precisions`.
- The container reads only `/mlpeg/in`, writes only `/mlpeg/out`, and needs no
  network.
- Exit code 0 means `inference.json` was written, including for the statuses `oom`,
  `unsupported_elements` and `skipped`.

### `.def` requirements

- Base image pinned by digest (`From: <image>@sha256:<digest>`; Apptainer does not
  accept a tag and a digest together); code from a public git repo at a pinned
  commit; weights baked in at build time.
- Builds without a GPU.
- `%labels` include `mlpeg.model`, `mlpeg.precisions` (comma-separated),
  `mlpeg.weights_sha256` and `mlpeg.contract_version 1.0`.

### Inputs

- `spec.json` holds the campaign workload: MD settings, precision modes, execution
  settings and, for each material, its elements, `contains_hydrogen`, and sizes
  with `n_atoms`, `supercell`, `file`, `sha256`, `validate` and `reference`.
- Structure paths in `spec.json` are relative to `/mlpeg/in`.
- extxyz files carry positions, cell, PBC, atomic numbers, masses and momenta.
- Units follow ASE: Å, eV, fs, amu.

### MD

- **Integrator:** NVE, velocity Verlet, starting from the momenta in the file, never
  re-thermalized.
- **Timestep:** 1.0 fs, or 0.5 fs if `contains_hydrogen`.
- **Steps:** 100 warm-up steps, then 1,000 measured steps. The measured phase may
  stop after 600 s, but runs at least 20 steps.
- **Every step:** energy and forces for every atom. No stress, no trajectory output.
- **Logging:** total energy may be logged every 100 steps.
- **Neighbour lists:** any scheme, provided it is exact within the cutoff.

### Precision modes

- `fp64`: double precision throughout.
- `fp32`: single precision with TF32 disabled.
- `mixed`: any reduced precision.

### `/mlpeg/out/inference.json`

```json
{
  "schema_version": "1.0",
  "run_id": "sio2_quartz-n1125-r1",
  "status": "ok | oom | unsupported_elements | skipped | error",
  "error": null,
  "model": {"name": "<ML-PEG model name>", "weights_sha256": "<hex>", "precision": "fp32"},
  "implementation": {"engine": "GPUMD", "engine_version": "<v>", "repo": "<url>", "commit": "<sha>", "settings": {}},
  "system": {"material": "sio2_quartz", "n_atoms": 1125},
  "timestamps_unix": {"start": 0.0, "setup_end": 0.0, "warmup_end": 0.0, "measure_end": 0.0},
  "steps": {"warmup": 100, "measured": 1000},
  "energy": {"total_start_eV": 0.0, "total_end_eV": 0.0},
  "self_reported": {}
}
```

- `start` is the moment the runscript is entered.
- `setup_end` is when the model is loaded and the system built.
- `warmup_end` and `measure_end` bound the measured phase.
- Energies are total (kinetic plus potential) at the start and end of the measured
  phase.
- For statuses other than `ok`, `steps`, `energy` and the later timestamps may be
  `null`.
- `self_reported` is free-form and informational only. ML-PEG never plots it.

### `/mlpeg/out/validation.extxyz`

Written only for sizes with `validate: true`. A single frame: the input positions
with the potential energy and per-atom forces computed at those positions.

## What ML-PEG measures and checks

ML-PEG writes `mlpeg_record.json` next to `inference.json`, with the wall time, exit
code, peak GPU and host memory (over the whole run and the measured phase) and a
hardware record. A run counts only if `inference.json` matches the schema, the
weights checksum matches the label and ML-PEG, static energies and forces match
ML-PEG's reference (for `validate` sizes), NVE drift is at most 1 meV/atom, the
timestamps are consistent with at least 20 measured steps, and an
`unsupported_elements` status agrees with the model's elements.
