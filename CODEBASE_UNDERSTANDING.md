# ML-PEG — Codebase Understanding

_A personal orientation note. Not tracked by git (see `.gitignore`)._

Reference: https://github.com/ddmms/ml-peg · Live guide: https://ml-peg.stfc.ac.uk

---

## What it is

**ML-PEG** ("ML Performance and Extrapolation Guide") is a benchmarking suite and
interactive [Dash](https://dash.plotly.com/) web app for evaluating **machine-learned
interatomic potentials (MLIPs / foundation models)** — MACE, Orb, PET-MAD, UMA
(fairchem), MatterSim, GRACE, CHGNet, etc. — against reference (usually DFT, sometimes
CCSD(T)/DMC/experimental) data across many physical/chemical benchmarks.

The end product is a scored, normalised leaderboard: each model gets a unitless score
per benchmark, rolled up into category scores and one overall score, all explorable in
the app (https://ml-peg.stfc.ac.uk).

Maintained by DDMMS (STFC / Cambridge). GPLv3. Python ≥3.10, packaged as `ml-peg` on
PyPI. Uses `uv` for dependency management, `janus-core` + `mlipx` for MLIP plumbing,
`ase` for atomistic calculations, and `dvc` + an S3 bucket (`ml-peg-data` on STFC Echo)
for data storage.

## The three-stage pipeline

Everything is organised around a **calc → analyse → app** flow, and around a
`category / benchmark` directory convention that is mirrored across three top-level
package folders. Both `calc` and `analyse` steps are just **pytest test functions** so
they can be discovered and run uniformly.

```
ml_peg/
├── calcs/     <category>/<benchmark>/calc_<benchmark>.py     # run the MLIPs → outputs/
├── analysis/  <category>/<benchmark>/analyse_<benchmark>.py  # outputs → metrics + plots (JSON)
│                                     metrics.yml             # good/bad thresholds, weights, tooltips
├── app/       <category>/<benchmark>/app_<benchmark>.py      # assemble Dash tab from JSON
│              <category>/<category>.yml                      # category title/description
├── models/    models.yml + models.py + get_models.py         # MLIP registry & loaders
└── cli/       cli.py                                          # `ml_peg` typer CLI
```

**1. Calculations** (`ml_peg/calcs/...`): `calc_<name>.py` runs the actual MLIP
calculations for each model, typically via
`@pytest.mark.parametrize("mlip", MODELS.items())`. `MODELS = load_models(current_models)`
loads calculators from `models.yml`. Results are written to a local `outputs/<model_name>/`
dir. Input/output data is generally not committed — it is pulled from S3 via
`download_s3_data(...)` / `ml_peg download`, or tracked with DVC (note the many
`.dvc/`, `dvc.yaml`, `.dvcignore` files). Some benchmarks alternatively define an
`mlipx`/`zntrack` Node run through `dvc repro`.

**2. Analysis** (`ml_peg/analysis/...`): `analyse_<name>.py` reads the `outputs/`,
computes metrics, and emits JSON (tables + Plotly figures) into
`ml_peg/app/data/<category>/<benchmark>/`. Heavy use of **decorators** from
`analysis/utils/decorators.py` that wrap a function returning a dict and, as a side
effect, serialise a figure/table:
- `@build_table` — turns `{metric: {model: value}}` into a scored DataTable JSON
  (applies normalisation via thresholds, weights, tooltips).
- `@plot_parity`, `@plot_scatter`, `@plot_density_scatter`, `@plot_periodic_table`,
  `@cell_to_scatter` — Plotly figure generators.
- benchmark-local `decorators.py` add bespoke ones (e.g. `@plot_hist`, `@cell_to_bar`).
Functions are chained as `@pytest.fixture`s feeding a final `metrics` fixture and a
`test_<name>` entry point. `metrics.yml` per benchmark supplies `good`/`bad` thresholds,
`unit`, `weight`, `tooltip`, `level_of_theory`, loaded via `load_metrics_config`.

**3. App** (`ml_peg/app/...`): `app_<name>.py` subclasses `BaseApp` (`base_app.py`),
loading the JSON produced by analysis to rebuild the table (`rebuild_table`) and layout,
and registering Dash callbacks. `build_app.py` discovers every `app_*.py` via glob,
groups tabs by category, builds per-category summary tables and one overall summary
table, and wires benchmark→category→overall score callbacks. `run_app.py` launches it.

## Scoring & normalisation (key concept)

5-level hierarchy: **raw metric → normalised metric score (0–1) → benchmark score →
category score → overall score**, each level a weighted average of the one below.

Normalisation uses per-metric **`good`** and **`bad`** thresholds from `metrics.yml`:
linear between them, clamped to [0,1] outside (1 = as good as anyone needs, 0 = avoid).
Users can override thresholds and weights live in the app. Advanced users can add custom
`normalizer` functions in `ml_peg/analysis/utils/utils.py`. Details:
`docs/source/developer_guide/scoring_and_normalisation.rst`.

## Models registry

`ml_peg/models/models.yml` is the model registry (many entries commented out). Each entry
names a `module`/`class_name`, `device`, `default_dtype`, `level_of_theory`, `kwargs`,
and D3-dispersion info. `models.py` defines dataclass wrappers around
`mlipx.GenericASECalculator` (`GenericASECalc`, `PetMadCalc`, `OrbCalc`, `FairChemCalc`),
all inheriting `SumCalc` which can bolt on a **TorchDFTD3** D3-dispersion correction
(`add_d3_calculator`, skipped when `trained_on_d3=True`). `get_models.py` provides
`load_models` (→ calculator objects), `get_model_names` (→ names for analysis), and
`get_subset` / `load_model_configs`. `current_models` is a module-global set by the CLI
to restrict the run to a subset (`--models a,b,c`).

## CLI (`ml_peg` command, Typer)

- `ml_peg calc`     — run calculations (globs `calc_*` files, invokes pytest; `--models`, `--category`, `--test`, `--run-slow/--run-very-slow`).
- `ml_peg analyse`  — run analysis (globs `analyse_*`, invokes pytest).
- `ml_peg app`      — launch the Dash app (`--models`, `--category`, `--port`, `--debug`).
- `ml_peg download` / `ml_peg upload` — S3 (`ml-peg-data` bucket, STFC Echo endpoint) data transfer.
- `--version`.

## Benchmark categories (each a folder under calcs/analysis/app)

- **aqueous_solutions** — bulk_water, ice
- **bulk_crystal** — elasticity, lattice_constants, phonons
- **conformers** — 37Conf8, DipCONFS, Glucose205, Maltose222, MPCONF196, solvMPCONF196, OpenFF_Tors, UpU46
- **molecular** — GMTKN55, Wiggle150
- **molecular_crystal** — X23, DMC_ICE13
- **molecular_reactions** — BH2O_36
- **nebs** — li_diffusion
- **physicality** — diatomics, extensivity, locality
- **supramolecular** — LNCI16, PLF547, S30L
- **surfaces** — OC157, S24, elemental_slab_oxygen_adsorption, **copper_water_interface**

## Current branch: `cu-h2o-rdf` (copper–water interface)

This branch develops the **copper_water_interface** surface benchmark (the open file is
`calcs/surfaces/copper_water_interface/calc_copper_water_interface.py`).

- **Calc** runs **Langevin MD** (deuterated H, 330 K, fixed bottom slab layers via
  `fix_indices`) on a Cu/water interface, writing `md-pos.xyz` (coords), `md-velc.xyz`
  (velocities) and `md.thermo` (incl. per-area z-dipole). Input pulled from S3
  (`copper_water_interface.zip`).
- **Analysis** compares trajectory-derived observables against PBE-D3 reference:
  - **RDF** (radial distribution functions) per element pair — via `aml.py` + `mdtraj`.
  - **VDOS** (vibrational density of states) from velocities.
  - **VACF** (velocity autocorrelation function).
  - **Dipole-moment z distribution** — std-dev deviation + histogram.
  Each yields a percentage-based score (`error_score_percentage`); thresholds in
  `metrics.yml` (`good`/`bad` = 100/80 for the correlation scores). Curve data is
  pickled into `app/data/.../{rdf,vdos,vacf}_curves/` for interactive plots.
- Recent commits: "add core functionality", "added bulk water and ice", "fixing liqice".

## Practical notes

- `.gitignore` is aggressive: most data/plot/structure formats (`*.json`, `*.xyz`,
  `*.png`, `*.csv`, `*.pkl`-via-`*.dat`, etc.) are ignored, plus per-benchmark `data/`
  dirs. Committed outputs (e.g. X23/S24 `.xyz`) are deliberate exceptions.
- Dev setup: `uv sync --extra mace --extra orb --extra d3 …`; `pre-commit install`
  (ruff + numpydoc, numpy docstring convention enforced); `pytest -v`.
- Docker/compose available in `containers/`; app served on port 8050.
- Local venvs per the global aliases (`maceenv`, `scfenv`, etc.) — `scfenv` usually fine.
