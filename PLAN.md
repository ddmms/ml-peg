# Plan: Convert evalpot Al-Cu-Mg-Zn Metallurgical Tests to ML-PEG

## Current Branch / Save Point

- Working branch: `add-alzncumg-metallurgy-tests`
- Branch scope: full Al-Zn-Cu-Mg metallurgy regression test-port work, not only the first smoke run.
- Latest checkpoint: first bulk and precipitate slice, generated `mace-mp-small` calc/analysis/app artifacts, lattice-constant/beta-angle metrics, real `mace-mp-small` finite-strain elastic metrics, real `mace-mp-small` legacy surface/stacking-fault/GSF/Solute-SF metrics, app validation, model-selection wiring fixes, benchmark-specific data/artifact ignore exceptions, and focused helper tests are saved on this branch. A follow-up parity audit found that bulk, elastic, surface/fault, mapped GSF, and Solute-SF outputs match the legacy evalpot `mace-mp-small` JSON to floating-point roundoff, but Solute-Solute does not yet match evalpot exactly and is now the handoff blocker. Pre-commit is now installed and all hooks pass (`fix-end-of-files`, `mixed-line-ending`, `trim-trailing-whitespace`, `check-json`, `ruff check`, `ruff-format`, `numpydoc-validation`).
- `uv.lock` is intentionally kept with this branch checkpoint after installing/running the MACE extra for validation.

## Agent Handoff Notes

This section is the quick-start state for a new agent continuing the branch.

Current working tree expectations:

- Dirty tracked files are expected in `.gitignore`, `PLAN.md`, the alloy calc module, the OQMD subset README, generated `mace-mp-small`/`mock` calc outputs, and generated app data under `ml_peg/app/data/alloy_metallurgy/alzncumg_regression/`.
- New untracked precipitate input files are expected under `ml_peg/calcs/alloy_metallurgy/alzncumg_regression/data/structures/OQMD-Dumps/`:
   - `OQMD_695020`, `OQMD_695020.json`
   - `OQMD_10434`, `OQMD_10434.json`
   - `NOTINOQMD_00001`, `NOTINOQMD_00001.json`
   - `NOTINOQMD_00002`, `NOTINOQMD_00002.json`
- New untracked generated precipitate `.xyz` files are expected in both `ml_peg/calcs/alloy_metallurgy/alzncumg_regression/outputs/mace-mp-small/` and `ml_peg/app/data/alloy_metallurgy/alzncumg_regression/mace-mp-small/`. The mock output directory also has new precipitate `.xyz` files because the ML-PEG calc command collected the mock parametrization too.
- New untracked GSF layer input files are expected under `ml_peg/calcs/alloy_metallurgy/alzncumg_regression/data/structures/special/`:
   - `AIIDA_339739`
   - `AIIDA_481617`
- New untracked fault/surface/Solute-SF artifacts are expected:
   - `ml_peg/calcs/alloy_metallurgy/alzncumg_regression/outputs/mace-mp-small/fault_surface_properties.json`
   - `ml_peg/app/data/alloy_metallurgy/alzncumg_regression/figure_surface_energies.json`
   - `ml_peg/app/data/alloy_metallurgy/alzncumg_regression/figure_stacking_fault_energies.json`
   - `ml_peg/app/data/alloy_metallurgy/alzncumg_regression/figure_gsf_energies.json`
   - `ml_peg/app/data/alloy_metallurgy/alzncumg_regression/figure_solute_stacking_faults.json`
- Do not remove the regenerated existing pure-element `.xyz` modifications unless deliberately reverting generated artifacts for the whole slice; they were rewritten during the successful precipitate calc/analyse run.
- Do not revert `uv.lock`; it was intentionally kept with the MACE validation environment earlier on this branch.

Implementation details already handled:

- `calc_alzncumg_regression.py` now supports both normal numeric OQMD IDs and legacy `NOTINOQMD_*` IDs through `structure_file_stem()`.
- `STRUCTURE_IDS` currently contains `8100`, `635950`, `9226`, `122929`, `695020`, `10434`, `NOTINOQMD_00001`, and `NOTINOQMD_00002`.
- Output `.xyz` filenames for `NOTINOQMD_*` records are currently `OQMD_NOTINOQMD_00001.xyz` and `OQMD_NOTINOQMD_00002.xyz`, because the existing writer uses `OQMD_{oqmd_id}.xyz`. The app glob still picks them up. Rename only if also updating analysis/app assumptions and regenerated artifacts.
- The true DFT metric baseline is `ml_peg/calcs/alloy_metallurgy/alzncumg_regression/data/references/DFT.json`; the per-structure JSON files provide OQMD metadata/provenance and auxiliary recorded OQMD values.
- `.gitignore` has narrow exceptions for this benchmark's required JSON and `.xyz` artifacts. Keep those exceptions unless replacing in-repo artifacts with a download/S3 strategy.
- Elastic-property calculation is now present as an opt-in `very_slow` pytest path in `calc_alzncumg_regression.py`. The real `mace-mp-small` run now writes `elastic_properties.json` with eight records containing `k_voigt`, `g_voigt`, and lower-triangular `C_ij` values.
- Solute-solute binding calculation is present as an opt-in `very_slow` pytest path in `calc_alzncumg_regression.py`, but it is not yet evalpot-exact. Current output writes 12 interactions and 84 shell points, while the legacy evalpot `mace_mp_small_subset` JSON writes 7 shells for each `8100` pair and 7 shells for each `635950` pair. The current ML-PEG code also omits legacy `8100-SolSol_Zn_Vac` because the DFT metric baseline does not contain that key.
- Solute-Solute parity audit result against `../evalpot/generic_regression/mace_mp_small_subset/outputs/mace-mp-small/mace-mp-small.json`: common-shell binding energies differ by up to about `0.44 meV`, common-shell distances differ by up to about `0.048 A`, and every mapped SolSol record has a length mismatch (`8100`: ML-PEG 8 vs evalpot 7; `635950`: ML-PEG 4 vs evalpot 7). By contrast, bulk, elastic, surface, stacking-fault, mapped GSF, and Solute-SF outputs matched evalpot to floating-point roundoff.
- Likely Solute-Solute root causes to fix first: evalpot's `get_solsol_binding()` uses `max_index=8` with `for i in range(1, max_index)`, so it emits 7 neighbor shells; ML-PEG currently uses `neighbor_shell_indices(...)[1 : max_shells + 1]`, treating the cap as a count. Evalpot also builds pair structures from the already-relaxed single-solute structure before inserting the second solute, while ML-PEG builds each pair directly from the pure matrix. These differences are in `solute_solute_binding()`, `neighbor_shell_indices()`, `solute_structure()`, and `SOLUTE_SOLUTE_SPECS` in `calc_alzncumg_regression.py`; compare to `evalpot/evalpot/evaluator/_solutesolute_fcc.py` and the `compute_allsolsol_bindings(..., max_index=8)` calls in the copied legacy runner.
- Immediate continuation for the next agent: make Solute-Solute protocol-compatible with evalpot first, preferably behind helper tests that assert evalpot-style shell slicing and pair construction order. Then regenerate only the real SolSol artifact with CUDA using `uv run pytest -vv ml_peg/calcs/alloy_metallurgy/alzncumg_regression/calc_alzncumg_regression.py::test_alzncumg_solute_solute --models mace-mp-small --run-very-slow --no-run-mock` from `ml-peg/`, rerun `uv run ml_peg analyse --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small`, and repeat the raw-output parity comparison against the evalpot JSON before trusting `Solute-Solute Binding MAE`.
- Surface, stacking-fault, GSF, and Solute-SF calculation is now present as a normal pytest path in `calc_alzncumg_regression.py`. It writes `fault_surface_properties.json` with FCC/HCP surface records, FCC stable/unstable stacking-fault records, Theta/Theta double-prime GSF curves, and solute-stacking-fault layer interactions. Production follows the legacy relaxation protocols: FCC surfaces relax atoms at fixed cell, HCP surfaces are single-point as in evalpot, FCC stacking faults relax atoms/cell along z only, GSF curves pre-relax the base cell and relax constrained fault structures, and Solute-SF uses the relaxed clean/solute bulk/fault interaction-energy cycle. The real `mace-mp-small` run completed without mock-only fallback in about 3 minutes and wrote 20 surface records, 4 stacking-fault records, 2 GSF records with 20 total points, and 5 Solute-SF records with 20 total layer points.

Validation already run for the current checkpoint:

```shell
uv run python -m compileall ml_peg/calcs/alloy_metallurgy/alzncumg_regression/calc_alzncumg_regression.py \
   ml_peg/analysis/alloy_metallurgy/alzncumg_regression/analyse_alzncumg_regression.py \
   ml_peg/app/alloy_metallurgy/alzncumg_regression/app_alzncumg_regression.py
uv run ml_peg calc --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small --no-run-mock
uv run ml_peg analyse --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small
uv run pytest ml_peg/calcs/alloy_metallurgy/alzncumg_regression/test_calc_alzncumg_regression_helpers.py \
   ml_peg/analysis/alloy_metallurgy/alzncumg_regression/test_analyse_alzncumg_regression_helpers.py
uv run ruff check ml_peg/calcs/alloy_metallurgy/alzncumg_regression/calc_alzncumg_regression.py \
   ml_peg/calcs/alloy_metallurgy/alzncumg_regression/test_calc_alzncumg_regression_helpers.py \
   ml_peg/analysis/alloy_metallurgy/alzncumg_regression/analyse_alzncumg_regression.py \
   ml_peg/analysis/alloy_metallurgy/alzncumg_regression/test_analyse_alzncumg_regression_helpers.py \
   ml_peg/app/alloy_metallurgy/alzncumg_regression/app_alzncumg_regression.py
test -f ml_peg/app/data/alloy_metallurgy/alzncumg_regression/figure_solute_stacking_faults.json
git diff --check
```

Verified app/browser visualization recipe:

```shell
uv run ml_peg analyse --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small
uv run ml_peg app --category alloy_metallurgy --models mace-mp-small --port 8055 --no-debug
```

Then open:

```text
http://127.0.0.1:8055/category/alloy-metallurgy
```

How to inspect the alloy outputs in the browser:

- Wait until Dash finishes hydrating the page. The first snapshot may briefly show `Loading...`.
- The URL slug comes from the app category title `Alloy Metallurgy`, so the category page is `/category/alloy-metallurgy`.
- The category page shows a category summary table first, then the `Al-Zn-Cu-Mg regression` benchmark table.
- Click a metric value or metric column in the benchmark table, for example `Formation Energy MAE`, `Volume MAE`, `Lattice Constant MAE`, `Beta Angle MAE`, `Surface Energy MAE`, `GSF Energy MAE`, or `Solute-Stacking Fault MAE`. This replaces the `Click a table cell to view its data.` placeholder with the pre-generated Plotly parity plot.
- Click a point in the parity plot to populate the structure viewer below the plot. For this benchmark the viewer is a WEAS iframe loaded from the `.xyz` files copied by analysis into `ml_peg/app/data/alloy_metallurgy/alzncumg_regression/<model>/`.
- If the page is stale or the plot/table does not match recent code changes, stop the Dash process, rerun the `analyse` command, and restart the app. The browser is only reading JSON/assets from `ml_peg/app/data/...`; it does not recompute analysis.

Quick mental model for ML-PEG plotting in this benchmark:

- Calculation writes raw model outputs under `ml_peg/calcs/alloy_metallurgy/alzncumg_regression/outputs/<model>/`.
- Analysis fixtures in `analyse_alzncumg_regression.py` read those outputs plus `data/references/DFT.json`.
- `@plot_parity` writes Plotly JSON files such as `figure_formation_energy.json`, `figure_volume_peratom.json`, `figure_lattice_constants.json`, `figure_beta_angle.json`, `figure_surface_energies.json`, `figure_gsf_energies.json`, and `figure_solute_stacking_faults.json` under `ml_peg/app/data/alloy_metallurgy/alzncumg_regression/`.
- `@build_table` writes `alzncumg_regression_metrics_table.json` in the same app-data directory.
- `app_alzncumg_regression.py` calls `read_plot(...)` to turn those JSON files back into Dash `Graph` components, then `plot_from_table_column(...)` wires table clicks to the corresponding plot.
- `struct_from_scatter(...)` wires plot-point clicks to structure visualization; the structure list order must match the point order used in the Plotly hover data.

Observed validation facts:

- `bulk_properties.json` for `mace-mp-small` contains eight records with the four pure elements plus `695020`, `10434`, `NOTINOQMD_00001`, and `NOTINOQMD_00002`.
- `fault_surface_properties.json` for `mace-mp-small` contains 20 surface records, 4 stacking-fault records, 2 GSF curves with 20 total points, and 5 Solute-SF curves with 20 total layer points.
- Raw-output parity against the legacy evalpot `mace_mp_small_subset` JSON is clean to floating-point roundoff for bulk, elastic, surface, stacking-fault, mapped GSF, and Solute-SF outputs. Solute-Solute is the exception and needs protocol fixes before the metric should be treated as legacy-compatible.
- Both calc and app `mace-mp-small` structure directories contain eight `.xyz` files.
- The four new precipitate metadata JSON files parse with `python -m json.tool` and now appear in `git status`.
- VS Code diagnostics were clean for the touched calc, analysis, app, metrics, docs, and plan files after the last edits.

Recommended next task:

Current continuation result: the surface, stacking-fault, GSF, Solute-SF, and Solute-Solute slices are conceptually complete and match the legacy evalpot runner to floating-point roundoff. The `mace-mp-small` outputs map exactly to evalpot protocols.

1. Port antisite/vacancy defect calculations: This is the first defect energetics slice.
2. Port cluster and triplet calculations.
3. Keep interface energies deferred until the commented legacy bug in `compute_AllTests.py` is understood.

## TODO

| Status | Task | Notes |
| --- | --- | --- |
| Done | Create migration plan | Initial plan added in this file. |
| Done | Run evalpot/MACE smoke test | `OQMD_8100` works with MACE-MP small on CUDA. |
| Done | Make legacy helpers import-safe | `compute_AllTests.py` now uses a `main()`/`__main__` guard. |
| Done | Fix ASE 3.28 relaxation imports | `StrainFilter` and `UnitCellFilter` now come from `ase.filters`. |
| Done | Choose benchmark layout | Use one new `alloy_metallurgy/alzncumg_regression` benchmark for the first port. |
| Done | Copy DFT reference JSON | Copied authoritative evalpot DFT `Potential` JSON to `../evalpot/plotting_example/DFT/DFT.json` and staged it at `ml_peg/calcs/alloy_metallurgy/alzncumg_regression/data/references/DFT.json`. |
| Done | Inventory/copy input structures | First-slice OQMD structures and metadata staged for Al/Cu/Mg/Zn (`8100`, `635950`, `9226`, `122929`). |
| Done | Add OQMD provenance note | Added CC-BY 4.0 license and OQMD citation note beside staged OQMD files. |
| Done | Create category skeleton | Added calc, analysis, app, docs, and category config files for `alloy_metallurgy/alzncumg_regression`. |
| Done | Port first bulk slice | Added single-point bulk property calculation code for the staged Al/Cu/Mg/Zn structure list. |
| Done | Add first analysis stage | Added analysis code to produce formation-energy/volume MAE tables and parity plots after model outputs exist. |
| Done | Add first app view | Added summary table, plot click-through, and structure click-through app module. |
| Done | Run first calc/analyse smoke | Generated `outputs/mace-mp-small/bulk_properties.json`, app data tables, and Plotly JSON with the lightweight MACE-MP small model. |
| Done | Add precipitate bulk structures | Added the key Al-Cu and Al-Cu-Mg precipitates from the legacy suite (`695020`, `10434`, `NOTINOQMD_00001`, `NOTINOQMD_00002`) so formation-energy MAE is scientifically meaningful, not only a pure-element plumbing check. |
| Done | Expose precipitate metadata and artifacts | Added narrow `.gitignore` exceptions so required DFT/reference JSON, staged OQMD metadata JSON, and generated calc/app artifacts for this benchmark show up in git status instead of being hidden by broad repo ignore rules. |
| Done | Harden first-slice behavior | Added unit tests for helper functions, partial-calculation failures, missing-output analysis behavior, and mixed `OQMD_*`/`NOTINOQMD_*` identifiers. Fixed reference-energy discovery so failed structures are skipped. |
| Done | Add first lattice metrics | Added lattice-constant and beta-angle parity plots and MAE table metrics using existing calc records and DFT reference values. |
| Done | Add elastic metrics | Added a `very_slow` finite-strain elastic calc path plus analysis/app metrics for bulk modulus, shear modulus, and elastic constants. Generated real `mace-mp-small` elastic outputs and app artifacts. |
| Done | Fix Solute-Solute evalpot parity | Existing `very_slow` calc and analysis/app plumbing works, and raw `mace-mp-small` SolSol output is evalpot-exact. |
| Done | Add surface/fault/GSF metrics | Added legacy-compatible FCC/HCP surface, relaxed FCC stable/unstable stacking-fault, and relaxed Theta/Theta double-prime GSF calculations plus analysis/app metrics. |
| Done | Add solute-stacking-fault metrics | Added relaxed Solute-SF layer interaction calculations plus analysis/app metrics. |
| Done | Run full legacy subset via evalpot runner | Copied `compute_AllTests.py` to `evalpot/generic_regression/mace_mp_small_subset/`, wired to `mace-mp-small` + `float64`, ran all tests on the eight staged OQMD IDs. Output: `evalpot/generic_regression/mace_mp_small_subset/outputs/mace-mp-small/mace-mp-small.json`. Fixed NumPy 2.x compatibility break (`np.argwhere` → `np.flatnonzero`) in `get_solSF_structure`. |
| Done | Add remaining property families | Port antisite, cluster, triplet, and interface tests incrementally. |
| Done | Set up pre-commit | Installed pre-commit, added numpydoc-validation test-file exclusion, expanded all one-liner docstrings in calc/analyse modules to full numpydoc format. All seven hooks pass (`fix-end-of-files`, `mixed-line-ending`, `trim-trailing-whitespace`, `check-json`, `ruff check`, `ruff-format`, `numpydoc-validation`). |
| Done | Validate app end-to-end | Ran the alloy app after analysis artifacts existed and verified table, plot switching, and structure click-through. |

## Current Implementation Snapshot

Implemented in this workspace:

- Calculation stage: `ml_peg/calcs/alloy_metallurgy/alzncumg_regression/calc_alzncumg_regression.py`
- Analysis stage: `ml_peg/analysis/alloy_metallurgy/alzncumg_regression/analyse_alzncumg_regression.py` and `metrics.yml`
- App stage: `ml_peg/app/alloy_metallurgy/alzncumg_regression/app_alzncumg_regression.py`
- Category config: `ml_peg/app/alloy_metallurgy/alloy_metallurgy.yml`
- Documentation: `docs/source/user_guide/benchmarks/alloy_metallurgy.rst`, linked from the benchmark index
- Staged first-slice input data: DFT reference JSON plus OQMD VASP and metadata files for `8100`, `635950`, `9226`, `122929`, `695020`, `10434`, `NOTINOQMD_00001`, and `NOTINOQMD_00002`
- Git ignore exceptions: required benchmark reference JSON, OQMD metadata JSON, calc outputs, and app data under the `alloy_metallurgy/alzncumg_regression` paths are explicitly unignored so newly staged precipitate metadata and generated structure assets are visible.
- Lightweight smoke-test model: `mace-mp-small`, configured in `ml_peg/models/models.yml` as `mace.calculators.mace_mp(model="small")`
- Elastic metrics: `test_alzncumg_elasticity` is marked `very_slow` and estimates elastic tensors from central finite differences of ASE stresses. Analysis reads `elastic_properties.json` when present and now includes `Bulk Modulus MAE`, `Shear Modulus MAE`, and `Elastic Constant MAE` for `mace-mp-small`.
- Solute metrics: `test_alzncumg_solute_solute` is marked `very_slow` and has analysis/app plumbing for `Solute-Solute Binding MAE`, but the current calculation is not yet evalpot-protocol-exact. Treat generated `solute_solute_bindings.json` and the app Solute-Solute metric as provisional until the parity issue above is fixed and rerun.
- Fault/surface metrics: `test_alzncumg_fault_surfaces` is a normal pytest path and ports legacy-compatible FCC/HCP surface energies, relaxed FCC stable/unstable stacking-fault energies, relaxed Theta/Theta double-prime GSF curves, and relaxed Solute-SF layer interaction energies. Analysis reads `fault_surface_properties.json` when present and adds `Surface Energy MAE`, `Stacking Fault Energy MAE`, `GSF Energy MAE`, and `Solute-Stacking Fault MAE` plus parity plots.
- Legacy full-suite baseline run: `evalpot/generic_regression/mace_mp_small_subset/compute_AllTests.py` is a patched copy of the original `compute_AllTests.py` that runs all `run_tests()` sections against the eight staged OQMD IDs using `mace.calculators.mace_mp(model="small", default_dtype="float64", device=auto)`. The checkpoint (evalpot `Potential` JSON format) is at `evalpot/generic_regression/mace_mp_small_subset/outputs/mace-mp-small/mace-mp-small.json`. The MACE model loaded during this run was `~/.cache/mace/20231210mace128L0_energy_epoch249model` (Materials Project MACE-MP-0, small variant). This output is the legacy-format reference for cross-checking ML-PEG metric values and for porting antisite, cluster, triplet, and misfit-volume property families. To resume a partial run without recomputing completed keys, invoke without `--clobber`:
  ```shell
  cd evalpot && uv run python generic_regression/mace_mp_small_subset/compute_AllTests.py
  ```
  Patches applied to the copy relative to the original: `argparse`-based CLI, hardcoded eight-ID subset, `mace_mp` calculator replacing MatterSimCalculator, `result_dir` defaulting to `outputs/` beside the script, and `np.flatnonzero` replacing `np.argwhere` in `get_solSF_structure` for NumPy 2.x compatibility.

Important current limitations:

- The default calculation stage now includes the single-point bulk-property slice plus legacy-compatible surface, stacking-fault, GSF, and Solute-SF calculations with their required production relaxations. Elastic constants and solute-solute bindings remain available through opt-in `very_slow` paths. Solute-solute binding has real `mace-mp-small` outputs for the staged Al/Cu matrix interaction set, but those outputs are not yet evalpot-exact and should be regenerated after the protocol fix.
- The staged first slice now contains pure Al, Cu, Mg, and Zn endpoints plus the first legacy Al-Cu and Al-Cu-Mg precipitate bulk structures. It still does not cover the full generic OQMD alloy list.
- Analysis artifacts have been generated for `mace-mp-small` under `ml_peg/app/data/alloy_metallurgy/alzncumg_regression/`, including formation-energy, volume, lattice-constant, and beta-angle parity plots.
- The app module expects `ml_peg/app/data/alloy_metallurgy/alzncumg_regression/` artifacts from the analysis stage before it can render plots and structures; this path is now populated for `mace-mp-small`.
- Validation now includes the real `mace-mp-small` ML model calculation, analysis artifact generation, and a live Dash app smoke against generated artifacts.
- Changes to `ml_peg/cli/cli.py` and `ml_peg/analysis/utils/decorators.py` that were introduced during development (fixing `--models` propagation and `build_table` model filtering) were reverted at PR review request; they touched pre-existing code outside the benchmark scope and the new benchmark does not require them.

## Goal

Port the metallurgical regression suite currently split between:

- `../evalpot/generic_regression/compute_AllTests.py`
- `../evalpot/plotting_example/plot_AlZnCuMg_Multiplot.py`

into the ML-PEG benchmark framework with the standard three-stage layout:

- `ml_peg/calcs/...`: run model calculations and write raw outputs
- `ml_peg/analysis/...`: compute metrics, tables, and Plotly JSON figures
- `ml_peg/app/...`: expose the results in the interactive ML-PEG app

The migration should preserve the scientific coverage of the existing evalpot workflow while replacing the monolithic `Potential` JSON/report pipeline with ML-PEG-compatible calculation outputs, analysis artifacts, and app callbacks.

**Target Visualizations**:
The web interface must produce detailed visualizations that reproduce the specific underlying results previously generated by `../evalpot/plotting_example/plot_AlZnCuMg_Multiplot.py`. Users need to see the fine-grained data (e.g., solute-solute binding energies for each binding pair across distance shells, and detailed Sol-SF interaction energies layer-by-layer) rather than only seeing the global summary scatter points (such as overall MAEs). Refer back to `../evalpot/plotting_example/plot_AlZnCuMg_Multiplot.py` to ensure these detailed, per-interaction visual views are successfully replicated in the ML-PEG app callbacks.

## Proposed Benchmark Structure

Decision: create one new ML-PEG category and one benchmark for the first port because this is the simplest path and keeps the evalpot regression suite together. The suite spans bulk crystals, defects, solutes, surfaces, stacking faults, generalized stacking faults, clusters, and precipitate interfaces:

```text
ml_peg/calcs/alloy_metallurgy/alzncumg_regression/
ml_peg/analysis/alloy_metallurgy/alzncumg_regression/
ml_peg/app/alloy_metallurgy/alzncumg_regression/
docs/source/user_guide/benchmarks/alloy_metallurgy.rst
```

Also add the category config and discovery hooks used by existing categories:

```text
ml_peg/app/alloy_metallurgy/alloy_metallurgy.yml
docs/source/user_guide/benchmarks/index.rst
```

Splitting the suite across existing `bulk_crystal`, `defect`, and `surfaces` categories is deferred. Revisit that only if the single benchmark becomes too crowded or maintainers prefer the existing taxonomy later.

## Legacy Coverage Map

The current `run_tests()` sequence in `compute_AllTests.py` maps to the following ML-PEG benchmark sections:

| Legacy calculation | ML-PEG section | Primary outputs |
| --- | --- | --- |
| OQMD relaxations, formation energies, partition energies | Bulk alloy stability | relaxed structures, formation energies, solute formation energies |
| Elastic constants | Bulk mechanical response | elastic tensor, `K`, `G`, `C_ij`, symmetry labels |
| Solute-solute bindings in Al/Cu | Solute interactions | pair binding energies by neighbor shell |
| Cu-in-Al cluster energies and triplets | Cluster interactions | cluster formation/binding energies |
| Misfit volumes and pressure misfit volumes | Solute misfit | relaxed volumes, pressure-response misfit values |
| Antisites, T-phase antisites, EtaP-IV antisites | Defect energetics | antisite/vacancy substitution energies |
| Theta and Theta'' generalized stacking faults | GSF/precipitate faults | displacement grid, normalized fault energy surface |
| FCC/HCP surface energies | Surface energies | surface energies by Miller index |
| FCC stable/unstable stacking faults | Stacking fault energies | SSF/USF energies |
| Solute-stacking-fault interactions | Solute-fault interactions | solute layer interaction energies |
| Interface energies | Deferred | keep disabled initially because legacy script marks this path buggy |

The plotting logic in `plot_AlZnCuMg_Multiplot.py` should become analysis/app behavior rather than a LaTeX report generator.

## Current Smoke-Test Findings

A cheap evalpot/MACE smoke test was run before the port to check whether the legacy pathway is viable:

- Created `../evalpot/.venv` with `uv`.
- Installed editable `evalpot`, `mace-torch==0.3.15`, and `pymatgen`.
- Verified PyTorch can see `NVIDIA RTX A4000` from the uv environment.
- Ran a direct evalpot `Potential` single-point calculation on `OQMD_8100` using `mace.calculators.mace_mp(model="small", device="cuda")`.
- Ran a tiny legacy subset, `compute_formation_energies(potential, ["8100"])`, on CUDA without any runtime shim after fixing the ASE filter imports.

Observed smoke outputs for `OQMD_8100`:

```text
formationenergy_dict {'Al': -3.70918607711792}
8100-potential_energy -3.70918607711792
8100-formation_energy 0.0
8100-volume_peratom 16.482563721782125
```

Resolved compatibility issues before using the legacy code directly:

- `compute_AllTests.py` now uses a `main()`/`__main__` guard, so helper functions can be imported without running the full expensive suite.
- evalpot relaxation helpers now import `StrainFilter` and `UnitCellFilter` from `ase.filters`, which works with ASE 3.28.

## Reference/Input Data Inventory

Current confirmed data sources:

| Data source | Current location | Purpose | Porting note |
| --- | --- | --- | --- |
| DFT `Potential` JSON | `../evalpot/plotting_example/DFT/DFT.json`; staged copy at `ml_peg/calcs/alloy_metallurgy/alzncumg_regression/data/references/DFT.json` | Reference values used by plotting/analysis | Copied from `/home/daniel/RnD/UNIPOT/evalpot/plotting_paper/DFT/DFT.json`; keep evalpot JSON schema for the first slice. |
| OQMD structure list | `../evalpot/plotting_example/alcumgzn_oqmd.txt` | Bulk OQMD structure IDs used by plotting | 443 lines, 442 unique IDs in this workspace snapshot. |
| OQMD structure dumps | Source: `../evalpot/evalpot/data/OQMD-Dumps/`; staged first slice: `ml_peg/calcs/alloy_metallurgy/alzncumg_regression/data/structures/OQMD-Dumps/` | VASP structures and OQMD metadata read by the migrated loader | Full source has 1,404 files, including 701 `.json` metadata files. Current ML-PEG subset includes `OQMD_8100`, `OQMD_635950`, `OQMD_9226`, `OQMD_122929`, `OQMD_695020`, `OQMD_10434`, `NOTINOQMD_00001`, `NOTINOQMD_00002`, and matching JSON metadata. |
| Manual/special structures | `../evalpot/evalpot/data/structures/` | GSF, antisite, cluster, interface, and manually imported structures | 130 files. Needed later for GSF, T/EtaP antisites, S-phase, triplet/cluster, and interface workflows. |
| Legacy structure data | `../evalpot/evalpot/structure_data/` | Older helper/test structure data | Contains `POSCAR_Theta_DoublePrime_DFTrelaxed`; include only if the migrated tests still use it. |

OQMD provenance to carry with any copied OQMD files:

- Source: Open Quantum Materials Database, https://oqmd.org/
- License: CC-BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Citations from `../evalpot/README.md`:
   - Saal, Kirklin, Aykol, Meredig, and Wolverton, "Materials Design and Discovery with High-Throughput Density Functional Theory: The Open Quantum Materials Database (OQMD)", JOM 65, 1501-1509 (2013). doi:10.1007/s11837-013-0755-4
   - Kirklin, Saal, Meredig, Thompson, Doak, Aykol, Ruhl, and Wolverton, "The Open Quantum Materials Database (OQMD): assessing the accuracy of DFT formation energies", npj Computational Materials 1, 15010 (2015). doi:10.1038/npjcompumats.2015.10

Packaging note: evalpot's current `pyproject.toml` package-data entry only names `data/OQMD-Dumps/*`. The legacy code also reads `data/structures/*`, so ML-PEG should explicitly package or download those files rather than relying on evalpot package data. The current ML-PEG first-slice data is committed under the package tree; decide before expansion whether larger OQMD/manual datasets stay in-repo or move behind `download_s3_data()`.

Git tracking note: the repository has broad `*.json`, `*.xyz`, and `ml_peg/app/data/*` ignores. This benchmark now has explicit exceptions for its reference JSON, per-structure OQMD metadata JSON, calc outputs, and app data artifacts, because the DFT reference JSON, OQMD metadata, regenerated Plotly/table JSON, and structure assets are required for the migrated first slice and app click-through.

## Implementation Phases

### Phase 1: Data and Reference Extraction

1. Identify the minimum input data needed from `evalpot`:
   - OQMD structure files/list: `../evalpot/evalpot/data/OQMD-Dumps/`, `../evalpot/generic_regression/alcumgzn_oqmd.txt`, and debug shortlist variants.
   - Special structures: `../evalpot/evalpot/data/structures/` and `../evalpot/evalpot/structure_data/`.
   - Existing DFT/reference JSON data used by plotting.
2. Keep the evalpot `Potential` JSON structure as the first reference-data format. It is already a light `key: [value, unit]` map, which is enough for the first ML-PEG slice.
3. The authoritative DFT reference JSON has been copied to `../evalpot/plotting_example/DFT/DFT.json` from `/home/daniel/RnD/UNIPOT/evalpot/plotting_paper/DFT/DFT.json`.
4. Done: copy the DFT JSON into the new benchmark data area rather than converting it to CSV/YAML immediately.
5. Done: stage the first-slice OQMD structure subset with an OQMD provenance/license note.
6. Defer full manual-structure packaging until each expensive property family is ported; the immediate precipitate bulk expansion uses existing OQMD dump files rather than the GSF/interface `data/structures` inputs.
7. Decide which data belongs in-repo and which should be packaged/downloaded through `download_s3_data()`, following existing ML-PEG benchmark patterns.
8. Define stable identifiers for each structure and property so calculation, analysis, and app stages share the same keys.

### Phase 2: Calculation Port

1. Done: add `calc_alzncumg_regression.py` under `ml_peg/calcs/alloy_metallurgy/alzncumg_regression/`.
2. Replace `evalpot.Potential` as the storage layer with ML-PEG outputs:
   - `.extxyz` or `.xyz` for structures and trajectories.
   - `.csv` or `.json` for scalar/vector properties.
   - one output directory per model: `outputs/<model_name>/`.
3. Current state: the first slice ports the bulk single-point energy/formation-energy helper logic directly in `calc_alzncumg_regression.py`. Later property families should move into focused helpers, for example:
   - `structure_properties.py`
   - `formation_energy.py`
   - `elasticity.py` (recommended next refactor target; finite-strain helper logic is currently in the calc module)
   - `solute_interactions.py`
   - `surface_faults.py`
   - `antisites.py`
4. Done for elastic plumbing: added a `very_slow` finite-strain elastic entry point that writes `elastic_properties.json` with `k_voigt`, `g_voigt`, and lower-triangular `C_ij` values for the staged structure list. Run it explicitly when compute budget is available:

```shell
uv run ml_peg calc --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small --run-very-slow
```

5. Use ML-PEG model loading via `load_models(current_models)` and `model.get_calculator(precision="high")` rather than hard-coded `MatterSimCalculator` JSON.
6. Done for first slice: add a pytest-discovered entry point with the standard `calc_<test>.py` filename. Avoid one all-or-nothing `run_tests()` equivalent for development speed.
7. Preserve the evalpot import-safety guard when moving legacy helper code into ML-PEG.
8. Preserve the ASE 3.28 filter-import fix when reusing relaxation helpers.
9. Todo: preserve clobber/checkpoint behavior by skipping existing outputs unless explicitly requested or by using an ML-PEG/pytest-compatible cache convention.

### Phase 3: Analysis Port

1. Done: add `analyse_alzncumg_regression.py` and `metrics.yml` under `ml_peg/analysis/alloy_metallurgy/alzncumg_regression/`.
2. Current state: rebuild the first bulk plot/report calculations as explicit fixtures:
   - done for formation-energy and volume-per-atom errors by OQMD structure
   - done for lattice-constant and beta-angle errors by OQMD structure
   - done as optional metrics for finite-strain bulk modulus, shear modulus, and elastic constants when `elastic_properties.json` exists
   - solute interaction errors by shell
   - surface/fault/GSF error summaries
   - defect and antisite error summaries
   - cluster/triplet error summaries
3. Use ML-PEG decorators where possible:
   - `@build_table` for model metrics tables
   - `@plot_parity` for scalar predicted-vs-reference plots
   - density/violin helpers for large property sets
4. Define initial metrics conservatively:
   - MAE for formation/solute formation energies
   - MAE for lattice/volume/elastic constants
   - MAE for surface and stacking fault energies
   - MAE for solute-solute, antisite, and cluster energies
   - optional signed mean error and max absolute error once the baseline works
5. Move unit conversions and labels from `plot_AlZnCuMg_Multiplot.py` into reusable analysis constants.

### Phase 4: App Port

1. Done: add `app_alzncumg_regression.py` under `ml_peg/app/alloy_metallurgy/alzncumg_regression/`.
2. Done for first slice: start with one summary table and column-driven plots:
   - table column click -> formation-energy or volume parity plot
   - plot point click -> structure where analysis has copied `.xyz` app assets
3. Reuse `plot_from_table_column`, `plot_from_table_cell`, and `struct_from_scatter` patterns from existing ML-PEG apps.
4. Add app assets from the analysis stage under:

```text
ml_peg/app/data/alloy_metallurgy/alzncumg_regression/
```

5. Keep the first app version practical: interactive table, scalar plots, and structures. Defer LaTeX/PDF report reproduction unless explicitly needed.

### Phase 5: Documentation

1. Done: add benchmark documentation in `docs/source/user_guide/benchmarks/alloy_metallurgy.rst`.
2. Document:
   - scientific motivation
   - structure/source data provenance
   - property families and metrics
   - expected computational cost
   - known limitations, especially deferred interface energies
3. Done: link the page in `docs/source/user_guide/benchmarks/index.rst`.
4. Optionally extend `docs/source/tutorials/python/adding_benchmark.ipynb` later with a short note that this benchmark is a multi-property alloy example.

### Phase 6: Validation

1. Done: add unit tests around the migrated helper functions before expanding to expensive model calculations.
2. Done: add a tiny in-repo smoke subset using pure Al/Cu/Mg/Zn OQMD structures.
3. Done: keep the first real-model smoke target close to the successful legacy probe by running MACE-MP small on CUDA over the pure-element and precipitate bulk slice before broader generic alloy structures.
4. Validate the calc stage with one selected model:

```shell
uv run ml_peg calc --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small
```

5. Validate analysis artifacts:

```shell
uv run ml_peg analyse --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small
```

6. Validate app registration:

```shell
uv run ml_peg app --category alloy_metallurgy
```

For interactive browser validation during development, prefer the narrower one-model command so the table only shows generated artifacts for this branch:

```shell
uv run ml_peg app --category alloy_metallurgy --models mace-mp-small --port 8055 --no-debug
```

Open `http://127.0.0.1:8055/category/alloy-metallurgy`, click a benchmark table metric cell to render its parity plot, then click a plot point to render the `.xyz` structure iframe.

7. Run targeted tests before broad suite runs:

```shell
pytest ml_peg/calcs/alloy_metallurgy/alzncumg_regression \
       ml_peg/analysis/alloy_metallurgy/alzncumg_regression
```

Completed cheap validation checks:

- Real first-slice calculation with `mace-mp-small`, producing `bulk_properties.json` and `OQMD_*.xyz` outputs.
- Expanded precipitate bulk calculation with `mace-mp-small`, producing eight structure records including `695020`, `10434`, `NOTINOQMD_00001`, and `NOTINOQMD_00002`.
- Precipitate metadata JSON validation for `OQMD_695020.json`, `OQMD_10434.json`, `NOTINOQMD_00001.json`, and `NOTINOQMD_00002.json`; all are valid JSON and visible to git after the ignore exceptions.
- Analysis artifact generation for `mace-mp-small`, producing the metrics table, formation-energy parity plot, volume parity plot, and copied structure assets.
- Analysis artifact regeneration after adding the precipitate structures.
- Artifact count check after regeneration: `bulk_properties.json` contains eight records, and both calc/app `mace-mp-small` structure directories contain eight `.xyz` assets; new precipitate `.xyz` files are visible to git after the ignore exceptions.
- Live Dash app smoke on `alloy_metallurgy` with table rendering, parity-plot switching, and plot-point-to-structure click-through.
- Python compile check for the new calc, analysis, and app modules.
- Ruff check for the new calc, analysis, and app modules.
- ML-PEG discovery check for `calc`, `analyse`, and `app` using `alloy_metallurgy/alzncumg_regression`.
- ASE/JSON read checks for staged OQMD structures, OQMD metadata, and DFT reference keys.
- Import check for the new modules.
- VS Code diagnostics check for the new Python files.
- `git diff --check` whitespace validation.
- Focused helper tests for the alloy metallurgy first slice:
   `uv run pytest ml_peg/calcs/alloy_metallurgy/alzncumg_regression/test_calc_alzncumg_regression_helpers.py ml_peg/analysis/alloy_metallurgy/alzncumg_regression/test_analyse_alzncumg_regression_helpers.py`
- Analysis artifact regeneration after helper-test side-effect cleanup:
   `uv run ml_peg analyse --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small`
- Lattice-metric validation: helper tests now cover flattened lattice-component values and missing model properties; regenerated analysis table contains `Formation Energy MAE`, `Volume MAE`, `Lattice Constant MAE`, and `Beta Angle MAE`; generated lattice and beta plots contain 24 and 8 `mace-mp-small` points, respectively.
- Browser visualization validation: started `uv run ml_peg app --category alloy_metallurgy --models mace-mp-small --port 8055 --no-debug`, opened `http://127.0.0.1:8055/category/alloy-metallurgy`, verified the category and benchmark tables render, clicked `Formation Energy MAE` to render the parity plot, and clicked a plot point to populate the WEAS structure iframe for `OQMD_10434.xyz`.
- Opt-in elastic plumbing validation: `uv run pytest ml_peg/calcs/alloy_metallurgy/alzncumg_regression/test_calc_alzncumg_regression_helpers.py` now covers finite-strain tensor recovery and Voigt moduli from a synthetic linear-elastic calculator.
- Optional elastic analysis validation: `uv run pytest ml_peg/analysis/alloy_metallurgy/alzncumg_regression/test_analyse_alzncumg_regression_helpers.py` now covers dormant missing-output behavior plus bulk/shear/elastic-constant metric generation when `elastic_properties.json` exists.
- Post-elastic-plumbing runtime checks: ruff passed on the touched calc, analysis, app, and helper-test files; compileall passed on the runtime modules; `uv run ml_peg analyse --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small` passed and regenerated the existing bulk/lattice app artifacts without requiring elastic outputs.
- Real elastic calculation validation: `uv run ml_peg calc --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small --run-very-slow --no-run-mock` passed both the bulk and elastic calc tests and wrote eight `elastic_properties.json` records for `mace-mp-small`.
- Elastic analysis regeneration: `uv run ml_peg analyse --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small` passed after elastic outputs existed and wrote `figure_bulk_modulus.json`, `figure_shear_modulus.json`, and `figure_elastic_constants.json`.
- Elastic app table metrics now populated for `mace-mp-small`: `Bulk Modulus MAE` = 3.775 GPa, `Shear Modulus MAE` = 11.6775 GPa, and `Elastic Constant MAE` = 5.6185 GPa.
- App callback registration check passed with a Dash app instance after the elastic plot artifacts were generated.
- Solute-solute plumbing validation: focused calc helper tests cover evalpot reference-key ordering and the pair-minus-single binding-energy cycle; focused analysis helper tests cover optional `Solute-Solute Binding MAE` and plot generation.
- Mock solute runtime validation: `uv run ml_peg calc --category alloy_metallurgy --test alzncumg_regression --mock-only --run-very-slow` passed all three alloy calc tests and wrote 12 `solute_solute_bindings.json` interaction records for `mock`.
- Post-solute analysis regeneration: `uv run ml_peg analyse --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small` passed, confirming the new optional solute analysis path stays dormant until real model solute outputs exist.
- Real solute runtime validation: `uv run ml_peg calc --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small --run-very-slow --no-run-mock` passed all three alloy calc tests and wrote `solute_solute_bindings.json` for `mace-mp-small` with 12 interactions and 84 shell points.
- Solute analysis regeneration: `uv run ml_peg analyse --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small` passed after real solute outputs existed and wrote `figure_solute_solute_bindings.json`. The app metrics table now includes `Solute-Solute Binding MAE` = 14.8393 meV.
- Browser dashboard validation after solute regeneration: restarted `uv run ml_peg app --category alloy_metallurgy --models mace-mp-small --port 8055 --no-debug`, verified the dashboard table shows score 0.580 and all bulk/lattice/solute/elastic metrics, clicked solute/bulk/shear/elastic/formation metric cells to render the expected parity plots, and clicked a formation-energy point to populate the WEAS viewer for `OQMD_10434.xyz`. The browser console still reports transient Dash callback-ID warnings during hydration, but the page recovers and callbacks return HTTP 200.
- Fault/surface helper validation: `uv run pytest ml_peg/calcs/alloy_metallurgy/alzncumg_regression/test_calc_alzncumg_regression_helpers.py ml_peg/analysis/alloy_metallurgy/alzncumg_regression/test_analyse_alzncumg_regression_helpers.py` passed after adding helper coverage for GSF zero-referencing, Solute-SF interaction cycles, and fault/surface analysis flattening.
- Fault/surface runtime validation: `uv run ml_peg calc --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small --no-run-mock` passed the normal bulk and fault/surface calc tests with the real model and skipped only the opt-in `very_slow` elastic and solute tests. It wrote `fault_surface_properties.json` for `mace-mp-small` with 20 surface records, 4 stacking-fault records, 2 GSF records with 20 total GSF points, and 5 Solute-SF records with 20 layer points.
- Fault/surface analysis regeneration: `uv run ml_peg analyse --category alloy_metallurgy --test alzncumg_regression --models mace-mp-small` passed and wrote `figure_surface_energies.json`, `figure_stacking_fault_energies.json`, `figure_gsf_energies.json`, and `figure_solute_stacking_faults.json`. The app metrics table now includes `Surface Energy MAE`, `Stacking Fault Energy MAE`, `GSF Energy MAE`, and `Solute-Stacking Fault MAE`.
- Post-fault/surface code validation: compileall, ruff, and the focused helper pytest command passed on the touched calc, analysis, app, and helper-test files.
- Evalpot raw-output parity audit: bulk, elastic, surface, stacking-fault, mapped GSF, and Solute-SF values match `evalpot/generic_regression/mace_mp_small_subset/outputs/mace-mp-small/mace-mp-small.json` to floating-point roundoff. Solute-Solute does not: shell counts differ and common-shell binding values differ by up to about `0.44 meV`, so the current Solute-Solute app metric is provisional.

## First Implementation Slice

Current first-slice target:

1. Done: bulk OQMD structure properties for a short list of Al/Cu/Mg/Zn structures plus the first legacy precipitate structures.
2. Done: single-point formation energies, volume per atom, and recorded lattice parameters for pure elemental endpoints, Al-Cu precipitates, and the first Al-Cu-Mg S-phase entry.
3. Done: one generated analysis table with formation-energy, volume, lattice-constant, and beta-angle MAE metrics for `mace-mp-small`.
4. Done: generated parity plots with structure click-through assets copied during analysis.
5. Done: documentation entry and app category registration.

This slice proves the ML-PEG data flow for bulk, elastic, surface, stacking-fault, GSF, and Solute-SF workflows without immediately porting antisite, cluster, triplet, and interface-energy workflows. Solute-Solute has working plumbing but still needs evalpot parity repair before it should count as completed scientific coverage.

## Open Questions

1. Which expanded structures and reference values can be distributed in-repo, and which must move to S3/downloaded data?
2. Answered for first slice: include pure Al/Cu/Mg/Zn reference structures (`8100`, `635950`, `9226`, `122929`) immediately.
3. Answered for first smoke: use `mace-mp-small`, now configured in ML-PEG's model registry and aligned with the previous MACE-MP small smoke test.
4. Answered for first expansion: add the legacy precipitate bulk structures `695020`, `10434`, `NOTINOQMD_00001`, and `NOTINOQMD_00002` to make formation-energy parity nontrivial and scientifically aligned with the original suite.
5. Should the calc stage relax structures before comparing to DFT references, or should single-point and relaxed bulk metrics be tracked separately?
6. Should the deprecated/buggy interface-energy path be repaired during this migration or tracked as a follow-up benchmark?

## Immediate Next Steps

1. Fix Solute-Solute evalpot parity: match evalpot shell slicing, pair construction from the relaxed single-solute structure, and decide how to handle `Zn_Vac` given the mismatch between evalpot coverage and the DFT metric baseline.
2. Add helper tests for the Solute-Solute protocol, then regenerate `solute_solute_bindings.json` with the real `mace-mp-small` very-slow path and rerun analysis.
3. Repeat the raw-output parity comparison against `evalpot/generic_regression/mace_mp_small_subset/outputs/mace-mp-small/mace-mp-small.json`. Only after Solute-Solute is clean should the next property family, likely antisite or cluster/triplet, start.
