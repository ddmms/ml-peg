# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**ML-PEG** ("ML Performance and Extrapolation Guide") is a benchmarking suite and interactive [Dash](https://dash.plotly.com/) web app for evaluating machine-learned interatomic potentials (MLIPs / foundation models) — MACE, Orb, PET-MAD, UMA, MatterSim, GRACE, CHGNet, etc. — against reference data (usually DFT, sometimes CCSD(T)/DMC/experiment) across many physical/chemical benchmarks. The end product is a normalised, weighted leaderboard: per-benchmark scores roll up into category scores and one overall score, explored in the app (https://ml-peg.stfc.ac.uk).

`CODEBASE_UNDERSTANDING.md` (untracked, personal notes) contains a deeper orientation and is worth reading for detail beyond this file.

## Environment & common commands

Use `uv` for dependency management. There is a `.venv` in the repo; the user's global aliases also provide venvs (`scfenv` is usually fine). Always sync/activate the environment before running tests.

```shell
uv sync --extra mace --extra orb --extra d3   # add extras per the model(s) you need
source .venv/bin/activate
pre-commit install
```

- Run all tests: `pytest -v`
- Run a single test file: `pytest -v tests/test_mlip_testing.py`
- Run one test: `pytest -v path/to/test_file.py::test_name`
- Lint/format (also enforced via pre-commit + CI): `ruff check --fix` and `ruff format`; docstrings validated by `numpydoc` (numpy convention).

Model extras conflict — you cannot install `uma`+`mace`, `mattersim`+`mace`, `uma`+`grace`, `mattersim`+`grace` together (see `[tool.uv].conflicts` in `pyproject.toml`).

### The `ml_peg` CLI (Typer, entry point `ml_peg.cli.cli:app`)

```shell
ml_peg calc     --test X23 --models mace-mp-0a,orb-v3-consv-inf-omat   # run MLIP calculations
ml_peg analyse  --test X23                                             # outputs -> metrics + plots (JSON)
ml_peg app      --port 8050                                            # launch Dash app
ml_peg download --key app/data/data.tar.gz --filename data.tar.gz      # S3 (STFC Echo, bucket ml-peg-data)
ml_peg upload   ...
ml_peg list     calcs | analysis | app | models
```

`calc`/`analyse`/`app` accept `--models`, `--category`, `--test`; `calc` also takes `--run-slow`/`--run-very-slow`. Add `--help` to any subcommand for options.

## Architecture: the calc → analyse → app pipeline

Everything follows a **calc → analyse → app** flow, mirrored across three top-level package folders under a shared `<category>/<benchmark>/` directory convention. Both `calc` and `analyse` steps are **pytest test functions**, so the CLI drives them by globbing files and invoking pytest.

```
ml_peg/
├── calcs/     <category>/<benchmark>/calc_<benchmark>.py     # run MLIPs -> local outputs/<model>/
├── analysis/  <category>/<benchmark>/analyse_<benchmark>.py  # outputs -> metrics + Plotly JSON
│                                     metrics.yml             # good/bad thresholds, weights, units, tooltips
├── app/       <category>/<benchmark>/app_<benchmark>.py      # assemble Dash tab from the JSON
│              <category>/<category>.yml                      # category title/description
├── models/    models.yml + models.py + get_models.py         # MLIP registry & loaders
└── cli/       cli.py                                          # Typer CLI
```

1. **Calc** (`calcs/`): `calc_<name>.py` runs each MLIP, typically via `@pytest.mark.parametrize("mlip", MODELS.items())` where `MODELS = load_models(current_models)`. Writes to a local `outputs/<model_name>/`. Input/output data is generally **not committed** — pulled from S3 via `download_s3_data(...)` or tracked with DVC (`dvc.yaml`, `.dvc/`). Some benchmarks run an `mlipx`/`zntrack` Node via `dvc repro`.

2. **Analysis** (`analysis/`): `analyse_<name>.py` reads `outputs/`, computes metrics, and emits JSON (tables + figures) into `ml_peg/app/data/<category>/<benchmark>/`. Heavy use of decorators in `analysis/utils/decorators.py` that wrap a dict-returning function and serialise a figure/table as a side effect — e.g. `@build_table` (scored DataTable with normalisation), `@plot_parity`, `@plot_scatter`, `@plot_periodic_table`. Functions are chained as `@pytest.fixture`s feeding a final `metrics` fixture and a `test_<name>` entry point. Per-benchmark `metrics.yml` supplies `good`/`bad`, `unit`, `weight`, `tooltip`, `level_of_theory` (loaded via `load_metrics_config`).

3. **App** (`app/`): `app_<name>.py` subclasses `BaseApp` (`base_app.py`), loads the analysis JSON to rebuild the table (`rebuild_table`) and layout, and registers Dash callbacks. `build_app.py` discovers every `app_*.py` by glob, groups tabs by category, builds per-category and overall summary tables, and wires benchmark→category→overall score callbacks. `run_app.py` launches it (`run_app.py` at repo root is the thin launcher).

### Benchmark reference data

Reference (DFT/AIMD) data is shipped per benchmark as a zip in the S3 `ml-peg-data`
bucket, pulled by `download_s3_data(...)` and cached under `~/.cache/ml_peg/` (if the
zip is already present there, S3 is never contacted). For **local testing**, drop the
benchmark zip in `~/.cache/ml_peg/` directly.

The raw simulation trajectories are usually far too large to ship, so each benchmark's
reference *observables* (RDF/VDOS/VACF curves, dipole arrays, density profiles, …) are
precomputed offline and only those small artifacts go in the zip. For
`copper_water_interface` this generation lives **outside the repo** at:

`/share/ijp30/projects/estatics-archer/07-04-2025-estatics/aimd/cu111/production-run/long-run-pbe-dzvp/`
— the raw PBE-D3 AIMD run (`pbe-d3-md-pos.xyz` positions, `pbed3-cu-h2o-vel-1-units-corr.xyz`
velocities), with an `ml-peg/` subfolder holding:
- `make_reference.py` — reads the raw trajectories, reuses the repo's analysis primitives
  to compute each reference artifact (must be run in the repo venv so `ml_peg` imports).
- `stage_local.py` — zips the artifacts listed in `REFERENCE_FILES` into
  `copper_water_interface.zip` and copies it into `~/.cache/ml_peg/`.

So to add/regenerate a reference observable: add its generation to `make_reference.py`,
add the filename to `stage_local.py`'s `REFERENCE_FILES`, then run `make_reference.py`
followed by `stage_local.py`. (Uploading the new zip to S3 for the live app is a
separate `ml_peg upload` step.)

### Scoring & normalisation (key concept)

5-level hierarchy: **raw metric → normalised metric score (0–1) → benchmark score → category score → overall score**, each a weighted average of the level below. Normalisation is linear between per-metric `good` and `bad` thresholds from `metrics.yml`, clamped to [0,1] (1 = as good as needed, 0 = avoid). Users can override thresholds/weights live in the app; advanced custom `normalizer` functions live in `analysis/utils/utils.py`. See `docs/source/developer_guide/scoring_and_normalisation.rst`.

### Models registry

`ml_peg/models/models.yml` is the registry (many entries commented out); each entry names `module`/`class_name`, `device`, `default_dtype`, `level_of_theory`, `kwargs`, and D3-dispersion info. `models.py` defines dataclass wrappers around `mlipx.GenericASECalculator` (`GenericASECalc`, `PetMadCalc`, `OrbCalc`, `FairChemCalc`), all inheriting `SumCalc`, which can add a TorchDFTD3 D3-dispersion correction (`add_d3_calculator`, skipped when `trained_on_d3=True`). `get_models.py` provides `load_models`, `get_model_names`, `get_subset`, `load_model_configs`. The module-global `models.current_models` restricts a run to a subset and is set from the CLI `--models` flag via `conftest.py`.

## Testing notes

- Custom pytest markers `slow` / `very_slow` are skipped unless `--run-slow` / `--run-very-slow` is passed (see root `conftest.py`).
- A **mock model** is available for fast testing: `--run-mock` (add mock alongside real models) or `--mock-only` (mock only), configured in `ml_peg/calcs/conftest.py`; see `models/mock.py`, `mock.yml`.
- `pytest.ini_options` sets `pythonpath = ["."]` and collects `test_*.py`.

## Conventions

- Ruff enforces PEP 8, isort (`force-sort-within-sections`, `required-imports = ["from __future__ import annotations"]`), and numpydoc docstrings. Every module starts with `from __future__ import annotations`.
- `.gitignore` is aggressive: most data/plot/structure formats (`*.json`, `*.xyz`, `*.png`, `*.csv`, etc.) and per-benchmark `data/` dirs are ignored. Some committed outputs (e.g. X23/S24 `.xyz`) are deliberate exceptions — don't assume a data file is meant to be committed.
- Adding a new benchmark: create matching `calcs/`, `analysis/`, `app/` files under `<category>/<benchmark>/` following the `calc_`/`analyse_`/`app_` naming; see the tutorial `docs/source/tutorials/python/adding_benchmark.ipynb` and the developer guide.
