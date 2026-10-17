# GPUMD/NEP89 reference container for ML-PEG inference timing (POC)

An Apptainer container that implements ML-PEG's inference-timing **contract v1.0** with
[GPUMD](https://github.com/brucefan1983/GPUMD) and the NEP89 foundation model. One
invocation runs one NVE simulation (100 warm-up + 1000 measured steps) as fast as GPUMD
allows and writes `/mlpeg/out/inference.json` (and `validation.extxyz` when requested).

```
container.def            Apptainer definition (multi-stage, builds without a GPU)
runner/mlpeg_gpumd.py    entry point: strict CLI, run orchestration, inference.json
runner/gpumd_io.py       GPUMD file I/O via calorine (model.xyz, run.in, thermo.out,
                         dump_xyz, nep.txt) + GPUMD stdout parsing
tests/                   pytest, no GPU (a fake gpumd replays v5.9.1-format output)
template_spec.json       example campaign spec from ML-PEG
```

## Pins

| What | Pin |
|---|---|
| GPUMD | tag `v5.9.1`, commit `11ba6fbdb4d71b2cdb351b39972fa057521585b3` (checked at build) |
| Build image | `nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04`, amd64 digest `sha256:b4db213759eb86d55a7271909bdad891fab300ec5700fb4f4656463b2f51980f` |
| Runtime image | `nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04`, amd64 digest `sha256:5380b8155c710531212b65503b39fd7fef0715fef8aa4de791f67ba4faad4d7f` |
| NEP89 | `potentials/nep/nep89_20250409/nep89_20250409.txt` at GPUMD commit `ca78f110d7aed5b7f7ee8e87b571eac2ce5fedce`, sha256 `75168ece02e840e4a32644f982b78d43cba697f5b64b4c8134ab66c7a8c28be1` (build fails on mismatch) |
| Python | Ubuntu 24.04 `python3` (3.12) venv at `/opt/venv`: `calorine==4.0` (wheel built in the devel stage; PyPI has only an sdist with a C++ extension), `ase==3.27.0`, `numpy==2.3.5`, `scipy==1.16.3`, `pandas==2.3.3`, `scikit-learn==1.7.2`, `matplotlib==3.10.7` |
| CUDA archs | build arg `CUDA_ARCHS`, default `70 80 86 89 90 120`: SASS for each, PTX for the last |

The `From:` lines use digest-only references (`nvidia/cuda@sha256:...`) because Apptainer
refuses references carrying both a tag and a digest.

Labels: `mlpeg.contract_version=1.0`, `mlpeg.model=nep89`, `mlpeg.precisions=mixed`,
`mlpeg.weights_sha256=75168ece…`, `gpumd.version=5.9.1`, `gpumd.commit=11ba6fbd…`, plus
`mlpeg.runner_repo` / `mlpeg.runner_commit` from build args.

## Build, test, self-test

```bash
cd gpumd-poc

# Tests (no GPU), using ml-peg's dev environment
../.venv/bin/python -m pytest tests

# Build (no GPU needed; ~5 min on a 28-thread workstation, SIF ~3.3 GB)
apptainer build --build-arg RUNNER_COMMIT=$(git rev-parse HEAD) mlpeg-gpumd.sif container.def
apptainer inspect --labels mlpeg-gpumd.sif

# Self-test on a GPU: 768-atom amorphous SiO2, 300 K momenta (seed 0), full path
apptainer run --nv --containall --cleanenv --env CUDA_VISIBLE_DEVICES=0 \
    mlpeg-gpumd.sif --self-test
```

The self-test prints `inference.json`, the NVE drift, the step-0 energy and
`SELF-TEST PASSED` when the status is `ok` and the drift is below 1 meV/atom. Its exit
code is non-zero otherwise.

A contract invocation:

```bash
apptainer run --nv --containall --cleanenv --net --network none \
  --bind $CAMPAIGN/in:/mlpeg/in:ro --bind $RUN_DIR:/mlpeg/out \
  --env CUDA_VISIBLE_DEVICES=0,OMP_NUM_THREADS=8 \
  mlpeg-gpumd.sif --spec /mlpeg/in/spec.json --run-id sio2_quartz-n1125-r1 \
  --precision mixed --out /mlpeg/out
```

## How a run works

1. `start = time.time()` is the first statement of the runner, before any heavy import.
2. Strict CLI: exactly `--spec --run-id --precision --out` (no abbreviations), and
   `--precision` must be in `mlpeg.precisions`; otherwise exit 2. `--self-test` alone is
   the only other accepted invocation.
3. The run ID selects the system and size; the structure (path relative to the spec's
   directory, i.e. `/mlpeg/in`) is read with `ase.io.read`. If the spec carries a real
   sha256 for it, the file is verified.
4. Elements not among the model's species (`calorine.nep.read_model(...).types`) →
   `unsupported_elements`, exit 0.
5. A scratch dir (`/tmp`, or `/mlpeg/out/.gpumd-scratch/` if `/tmp` is too small; see
   limitations) gets a `nep.txt` symlink, `model.xyz` written with
   `calorine.gpumd.write_xyz` (positions, cell, PBC, masses, velocities from momenta;
   the input's free-text `info` is dropped because GPUMD tokenises the comment line on
   spaces), and `run.in` written with `calorine.gpumd.write_runfile` and checked to read
   back identically with `read_runfile`:

   ```
   potential nep.txt
   time_step 1.0        (0.5 if contains_hydrogen)
   ensemble nve
   dump_thermo 100
   run 100              (warm-up)
   ensemble nve
   dump_thermo 100
   run 1000             (measured)
   ```

6. `stdbuf -oL gpumd` runs with stdout streamed line by line. `measure_end` is the wall
   clock when the second run's `Time used for this run` line is read; `warmup_end =
   measure_end − t₂` and `setup_end = warmup_end − t₁`, with t₁, t₂ GPUMD's own per-run
   timers. The wall clock at which run 1's line was read must lie within 1 s of
   `warmup_end`, else `self_reported.warning` is set.
7. `total_start_eV` / `total_end_eV` are KE + PE of the last `thermo.out` row of the
   warm-up and measured blocks, read with `calorine.gpumd.read_thermo` (which lays the
   per-run blocks end to end; the row counts are checked against the run lengths).
8. Non-zero GPUMD exit: `oom` if stdout/stderr contains a CUDA out-of-memory message,
   otherwise `error` with the last 50 lines of stderr. Both exit 0.
9. For `validate: true` sizes, an untimed single point (below) writes
   `validation.extxyz`: input positions, `info['energy']`, `arrays['forces']`. The
   GPUMD dump is read with `calorine.gpumd.read_xyz`.

## GPUMD v5.9.1 facts this relies on (verified in the source)

Paths are relative to the GPUMD repository at the pinned commit.

### 1. Starting velocities

- `model.xyz` header: `lattice=` is required, `pbc=` defaults to periodic, and
  `properties=` must contain `species` and `pos`; `mass` and `vel` are optional. Keys are
  case-insensitive (`src/model/read_xyz.cu:173-301`). Columns are located by name and
  offsets are summed over *all* listed columns, so the extra `masses`/`momenta`
  columns that calorine's ASE-based writer emits are skipped harmlessly. calorine
  writes `Lattice="…" Properties=species:S:1:pos:R:3:masses:R:1:momenta:R:3:mass:R:1:vel:R:3 pbc="T T T"`.
- calorine's `write_xyz` omits `vel` when every velocity is below 1e-6 (ASE units).
  GPUMD would then draw random 300 K velocities, so the runner checks the written header
  and refuses (status `error`) instead.
- When `vel` is present, velocities are taken verbatim. `Velocity::initialize_cpu` only
  draws random velocities, zeroes linear/angular momentum and rescales when the file has
  *no* velocities (`src/main_gpumd/velocity.cu:322-344`, `src/main_gpumd/run.cu:149-156`).
  GPUMD prints `Initialized velocities with data in model.xyz.`
- A `velocity <T>` keyword is silently ignored when the file has velocities
  (`run.cu:418-427`); we omit it anyway. Momentum removal only happens with an explicit
  `correct_velocity` (`run.cu:430-468`), which we never write.
- `ensemble nve` is plain velocity Verlet with no thermostat or rescaling
  (`src/integrate/ensemble_nve.cu`).

### 2. Velocity units

`model.xyz` velocities are in Å/fs and are multiplied by `TIME_UNIT_CONVERSION =
10.18051` into GPUMD's natural units (`read_xyz.cu:382-388`,
`src/utilities/common.cuh:26`). GPUMD's natural units are eV, Å and amu, the same as
ASE's. So `v[Å/fs] = (momenta / masses) * ase.units.fs`.

The constants differ slightly: 10.18051 against ASE's 1/fs = 10.1805057, a relative
difference of 4e-7, and k_B differs by 1.5e-6. `tests/test_units.py` proves the
conversion. It writes `model.xyz` for the 300 K amorphous SiO₂ through calorine,
re-reads it the way GPUMD does (column offsets from `Properties`, ×10.18051), and checks
that ½Σmv² equals ASE's kinetic energy to within 1e-5 relative. That tolerance covers
the `%16.8f` extxyz number format (velocities to 1e-8 Å/fs, about six significant
digits) as well as the constant mismatch. It also checks the calorine
`write_xyz`/`read_xyz` velocity round trip.

### 3. NVE and `thermo.out`

- Keywords: `time_step <fs>` (persists across runs), `ensemble nve`,
  `dump_thermo <interval>` and `run <N>` with N > 0.
- Ensemble and all output keywords are cleared after every `run`
  (`src/integrate/integrate.cu` `Integrate::finalize`,
  `src/measure/measure.cu:378-398`), so each block repeats them.
- `thermo.out` is opened in append mode. Each run block first writes a 5-line header
  (`src/measure/dump_thermo.cu:60-71`):

  ```
  # dump_thermo 100
  # format_version 1
  # num_atoms N
  # dt_output 1.0000000000e+02 fs
  # columns T KE PE sxx syy szz syz sxz sxy ax ay az bx by bz cx cy cz
  ```

- Rows are 18 × `%20.10e` and are written when `(step+1) % interval == 0`, i.e. at steps
  100, 200, …. There is never a row at step 0.
- KE is `1.5·N·k_B·T`, which equals ½Σmv² (`src/integrate/thermo.cu`). Total energy is
  `KE + PE`, columns 2 + 3.
- The bundled `examples/` `thermo.out` uses an older headerless format, and the parser
  rejects that format.

### 4. Stdout lines and errors

From `src/main_gpumd/run.cu:233-310, 508`:

- Each run block starts with `Use NVE ensemble for this run.`, `Dump thermo every 100
  steps.` and `Run N steps.`, then `    k steps completed.` every N/10 steps.
- The end of each run block is printed between dashed lines:
  ```
  Time used for this run = <s> second.
  Speed of this run = <atoms*steps/s> atom*step/second.
  ```
- The per-run timer starts **after** the run's initial force evaluation and stops after
  the last step. It includes per-step dump I/O.
- Whole program: `Started running GPUMD.` … `Time used = <s> s.` … `Finished running
  GPUMD.`
- Device info is printed first: `number of GPUs = n`, `Device name:`, `Compute
  capability:`.
- CUDA errors, including OOM, go through `CHECK()` (`src/utilities/error.cuh:23-34`).
  They print to **stderr** as below, then `exit(1)`:
  ```
  CUDA Error:
      File: … Line: …
      Error code: 2
      Error text: out of memory
  ```
  Input errors print an `Input Error:` block on stderr and also exit 1.

### 5. Initial-frame energy and forces

- `run 0` is rejected ("number of steps should be positive.").
- `dump_force`/`dump_exyz` no longer exist (`src/measure/measure.cu:87-127`).
- Every dump, including `dump_xyz`, `dump_thermo` and `compute`, first writes at step 1
  (`(step+1) % interval`).
- The untimed single point therefore uses a **zero time step**:

  ```
  potential nep.txt
  time_step 0
  ensemble nve
  dump_xyz 1 step0.xyz precision double force potential
  run 1
  ```

- `time_step` has no positivity check (`run.cu:470-490`). With dt = 0, velocity Verlet
  leaves positions and velocities unchanged, so the frame dumped "at step 1" is a fresh
  force evaluation at the input positions.
- `dump_xyz` writes `energy=` (total PE) in the comment line and `forces:R:3` per atom
  (`src/measure/dump_xyz.cu:205-290`).
- The runner checks that the dumped species order and minimum-image positions match
  the input (to within 1e-6 Å) before writing `validation.extxyz`.

### 6. NEP precision

- **fp32:** pair vectors, descriptors, the ANN, `Fp`, the partial forces and per-atom
  force/virial accumulators are `float` (`src/force/nep.cu:372-645`,
  `src/force/nep.cuh:26-36`, `src/utilities/nep_utilities.cuh`).
- **fp64:** the results are added into `double` global per-atom arrays. Positions,
  velocities, forces, per-atom energies, velocity Verlet
  (`src/integrate/ensemble.cu:123-135`) and thermo reductions are double.
- There is no compile flag that switches NEP to double (no `USE_FP64` or similar in
  `src/`). The MD path uses no cuBLAS, so no TF32.
- This is neither fp64 throughout nor fp32 throughout, so the container declares
  **`mlpeg.precisions=mixed`** only. `--precision fp32|fp64` exits non-zero.

### Neighbour list

- A global cell list is built at `rc + 1.0 Å` skin (`src/force/neighbor.cuh:170`). It is
  rebuilt when any atom has moved more than 0.5 Å since the last build
  (`src/force/neighbor.cu:754-792`).
- Every force call then filters it with the exact per-type-pair radial (6 Å) and angular
  (5 Å) cutoffs (`nep.cu:372-421`), so the list is exact within the cutoff.
- Boxes thinner than 2.5·(rc+1) along a periodic direction use the small-box path.
- `MN_radial`/`MN_angular` capacities come from `nep.txt` × 1.25 and are reported in
  `implementation.settings`.

## Known limitations

- **No 600 s cap.** GPUMD cannot stop a `run` early and has no wall-time keyword, so
  the measured phase always runs all 1000 steps. See *Measured-phase cap* below.
- **`mixed` only.** No honest `fp32` or `fp64` mode exists without patching NEP.
- **One GPU.** If more than one GPU is visible, `potential nep.txt` switches to the
  multi-GPU NEP (`src/force/force.cu:118-145`). The runner reports status `error` when
  GPUMD reports `number of GPUs` ≠ 1. The contract's `CUDA_VISIBLE_DEVICES=0` avoids
  this.
- **Unavoidable extra work.**
  - NEP always computes per-atom virials (used for the thermo stress columns).
  - Every 1000 force calls GPUMD appends one line to `neighbor.out`.
  - Both happen inside the measured window, and neither can be switched off without
    patching GPUMD.
- **Timer boundaries.** "Time used for this run" excludes the run's initial force call
  and `pre_run`. The gap before run 2 (one force call plus file opening) therefore lies
  between the seen end of run 1 and `warmup_end`. `setup_end = warmup_end − t₁` also
  excludes it.
- **Step-0 trick.** The single point relies on `time_step 0`, which GPUMD accepts but
  does not document. The self-test exercises it and checks the frame against the input
  positions.
- **Fixtures.**
  - `tests/data/gpumd_stdout_v5.9.1.txt`, `thermo_v5.9.1.out` and `step0_v5.9.1.xyz` are
    real captures from the built container: GPUMD v5.9.1 on an RTX 5080, 768-atom SiO₂.
  - The OOM stderr fixture is built from GPUMD's `CHECK` format. A real OOM was not
    triggered, because 1.02M atoms of SiO₂ fit on a 16 GB card.
- **Scratch space.**
  - Under `--containall`, `/tmp` is Apptainer's session tmpfs. It is capped by the
    site's `sessiondir max size` (64 MB by default, as on the test host), which is too
    small for `model.xyz` beyond about 250k atoms (calorine's `model.xyz` is about
    210 B/atom, 11 columns of `%16.8f`).
  - The runner therefore uses `/tmp` when it has room, and otherwise a hidden
    `/mlpeg/out/.gpumd-scratch/` directory. That directory is always removed.
  - `self_reported.scratch_dir` records which was used.
  - Writing the inputs happens before `setup_end`, so timing is unaffected.
- **Pinned dependencies.** Transitive dependencies of matplotlib, pandas and
  scikit-learn (pillow, joblib, …) are not pinned. ase needs scipy at import; calorine
  needs pandas and scikit-learn.
- **Input precision.** calorine writes `model.xyz` through ASE's extxyz writer with
  `%16.8f`: positions are rounded to 1e-8 Å and velocities to 1e-8 Å/fs. The
  validation frame carries the original input positions; the difference is far below
  the validation tolerances (verified against `CPUNEP`).
- **Runtime JIT for newer GPUs.** Newer GPUs that are not in `CUDA_ARCHS` JIT-compile the
  PTX at first launch. With `--containall` the JIT cache does not persist, so the JIT
  happens before `setup_end` on every run.

## Measured-phase cap

GPUMD has no early stop, so all 1,000 measured steps always run.

Measured NEP89 speeds below are contract invocations of the built container (calorine
I/O) on an otherwise idle **RTX 5080** (16 GB), on amorphous SiO₂ supercells at 1 fs.

| n_atoms | setup | warm-up (100) | measured (1000) | speed (atom·step/s) | NVE drift (meV/atom) |
|---:|---:|---:|---:|---:|---:|
| 768 | 2.0 s | 0.09 s | 0.85 s | 9.0e5 | +0.0034 |
| 20,736 | 2.1 s | 0.15 s | 1.5 s | 1.4e7 | −0.0006 |
| 165,888 | 3.2 s | 0.99 s | 9.9 s | 1.7e7 | −0.0005 |
| 559,872 | 5.7 s | 3.1 s | 31.3 s | 1.8e7 | −0.0006 |
| 1,022,208 | 8.8 s | 5.9 s | 58.9 s | 1.7e7 | −0.0008 |

1M atoms × 1000 steps took 59 s, about 10× below the 600 s cap.

- An A100 or H100 should be at least as fast. SiO₂ at about 2.2 g/cm³ has a neighbour
  count typical of the campaign materials, so **NEP89 at about 1M atoms is not expected
  to approach 600 s**.
- Liquid water at 0.5 fs costs the same per step; only the simulated time halves.
- The cap would matter only on much slower GPUs (e.g. V100) or far denser systems.

If it ever does matter, the simplest fix is an untimed probe run during setup, e.g.
`run 20`. It gives GPUMD's own speed estimate, from which the runner sets
`measured = clamp(floor(0.8 · 600 s · speed / n_atoms), 20, 1000)` before writing the
real `run.in`. This is not implemented in the POC.

## Verified on a GPU

These checks were run on an RTX 5080, using CUDA 12.9 SASS for sm_120.

- **Self-test:** `--self-test` gives status `ok`. NVE drift is +0.0034 meV/atom over
  1000 steps at 1 fs. Run 1's end was seen within 0.3 ms of the derived `warmup_end`.
- **Validation frame vs calorine `CPUNEP`:** compared on the same `nep.txt` (768 atoms).
  - |ΔE| = 0.0006 meV/atom
  - force RMSE = 0.006 meV/Å
  - max |ΔF| = 0.03 meV/Å

  The tolerances are 1 meV/atom and 10 meV/Å.
- **Contract invocations:** the exact contract flags (`--containall --cleanenv --net
  --network none`, read-only `/mlpeg/in`) give status `ok` for every size up to
  1,022,208 atoms (table above).
