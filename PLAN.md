# Plan: Convert evalpot Al-Cu-Mg-Zn Metallurgical Tests to ML-PEG

## Current Branch / Save Point

- Working branch: `add-alzncumg-metallurgy-tests`
- Branch scope: full Al-Zn-Cu-Mg metallurgy regression test-port work, not only the first smoke run.
- Latest checkpoint: first bulk slice, generated `mace-mp-small` calc/analysis/app artifacts, app validation, and model-selection wiring fixes are ready to save on this branch.
- `uv.lock` is intentionally kept with this branch checkpoint after installing/running the MACE extra for validation.

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
| Todo | Harden first-slice behavior | Add unit tests for helper functions, partial-calculation failures, and missing-output analysis behavior. |
| Todo | Add first alloy structure | Add at least one binary or multicomponent OQMD structure so formation-energy MAE is scientifically meaningful, not only a pure-element plumbing check. |
| Todo | Add remaining property families | Port solute, surface, stacking fault, GSF, antisite, cluster, and triplet tests incrementally. |
| Done | Validate app end-to-end | Ran the alloy app after analysis artifacts existed and verified table, plot switching, and structure click-through. |

## Current Implementation Snapshot

Implemented in this workspace:

- Calculation stage: `ml_peg/calcs/alloy_metallurgy/alzncumg_regression/calc_alzncumg_regression.py`
- Analysis stage: `ml_peg/analysis/alloy_metallurgy/alzncumg_regression/analyse_alzncumg_regression.py` and `metrics.yml`
- App stage: `ml_peg/app/alloy_metallurgy/alzncumg_regression/app_alzncumg_regression.py`
- Category config: `ml_peg/app/alloy_metallurgy/alloy_metallurgy.yml`
- Documentation: `docs/source/user_guide/benchmarks/alloy_metallurgy.rst`, linked from the benchmark index
- Staged first-slice input data: DFT reference JSON plus OQMD VASP and metadata files for `8100`, `635950`, `9226`, and `122929`
- Lightweight smoke-test model: `mace-mp-small`, configured in `ml_peg/models/models.yml` as `mace.calculators.mace_mp(model="small")`

Important current limitations:

- The first calculation stage is a single-point bulk-property slice. It does not yet relax structures, compute elastic constants, or run partition/solute/defect/fault/surface workflows.
- The staged first slice currently contains only pure Al, Cu, Mg, and Zn endpoints. This is useful for plumbing elemental references and app flow, but it does not yet exercise nonzero alloy formation energies.
- Analysis artifacts have been generated for `mace-mp-small` under `ml_peg/app/data/alloy_metallurgy/alzncumg_regression/`.
- The app module expects `ml_peg/app/data/alloy_metallurgy/alzncumg_regression/` artifacts from the analysis stage before it can render plots and structures; this path is now populated for `mace-mp-small`.
- Validation now includes the real `mace-mp-small` ML model calculation, analysis artifact generation, and a live Dash app smoke against generated artifacts.
- The ML-PEG app CLI and shared analysis table builder were fixed so model selection is applied consistently for app and table generation.

## Goal

Port the metallurgical regression suite currently split between:

- `../evalpot/generic_regression/compute_AllTests.py`
- `../evalpot/plotting_example/plot_AlZnCuMg_Multiplot.py`

into the ML-PEG benchmark framework with the standard three-stage layout:

- `ml_peg/calcs/...`: run model calculations and write raw outputs
- `ml_peg/analysis/...`: compute metrics, tables, and Plotly JSON figures
- `ml_peg/app/...`: expose the results in the interactive ML-PEG app

The migration should preserve the scientific coverage of the existing evalpot workflow while replacing the monolithic `Potential` JSON/report pipeline with ML-PEG-compatible calculation outputs, analysis artifacts, and app callbacks.

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
| OQMD structure dumps | Source: `../evalpot/evalpot/data/OQMD-Dumps/`; staged first slice: `ml_peg/calcs/alloy_metallurgy/alzncumg_regression/data/structures/OQMD-Dumps/` | VASP structures and OQMD metadata read by the migrated loader | Full source has 1,404 files, including 701 `.json` metadata files. Current ML-PEG subset includes `OQMD_8100`, `OQMD_635950`, `OQMD_9226`, `OQMD_122929`, and matching JSON metadata. |
| Manual/special structures | `../evalpot/evalpot/data/structures/` | GSF, antisite, cluster, interface, and manually imported structures | 130 files. Needed later for GSF, T/EtaP antisites, S-phase, triplet/cluster, and interface workflows. |
| Legacy structure data | `../evalpot/evalpot/structure_data/` | Older helper/test structure data | Contains `POSCAR_Theta_DoublePrime_DFTrelaxed`; include only if the migrated tests still use it. |

OQMD provenance to carry with any copied OQMD files:

- Source: Open Quantum Materials Database, https://oqmd.org/
- License: CC-BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Citations from `../evalpot/README.md`:
   - Saal, Kirklin, Aykol, Meredig, and Wolverton, "Materials Design and Discovery with High-Throughput Density Functional Theory: The Open Quantum Materials Database (OQMD)", JOM 65, 1501-1509 (2013). doi:10.1007/s11837-013-0755-4
   - Kirklin, Saal, Meredig, Thompson, Doak, Aykol, Ruhl, and Wolverton, "The Open Quantum Materials Database (OQMD): assessing the accuracy of DFT formation energies", npj Computational Materials 1, 15010 (2015). doi:10.1038/npjcompumats.2015.10

Packaging note: evalpot's current `pyproject.toml` package-data entry only names `data/OQMD-Dumps/*`. The legacy code also reads `data/structures/*`, so ML-PEG should explicitly package or download those files rather than relying on evalpot package data. The current ML-PEG first-slice data is committed under the package tree; decide before expansion whether larger OQMD/manual datasets stay in-repo or move behind `download_s3_data()`.

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
6. Defer full manual-structure packaging until each expensive property family is ported.
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
   - `elasticity.py`
   - `solute_interactions.py`
   - `surface_faults.py`
   - `antisites.py`
4. Use ML-PEG model loading via `load_models(current_models)` and `model.get_calculator(precision="high")` rather than hard-coded `MatterSimCalculator` JSON.
5. Done for first slice: add a pytest-discovered entry point with the standard `calc_<test>.py` filename. Avoid one all-or-nothing `run_tests()` equivalent for development speed.
6. Preserve the evalpot import-safety guard when moving legacy helper code into ML-PEG.
7. Preserve the ASE 3.28 filter-import fix when reusing relaxation helpers.
8. Todo: preserve clobber/checkpoint behavior by skipping existing outputs unless explicitly requested or by using an ML-PEG/pytest-compatible cache convention.

### Phase 3: Analysis Port

1. Done: add `analyse_alzncumg_regression.py` and `metrics.yml` under `ml_peg/analysis/alloy_metallurgy/alzncumg_regression/`.
2. Current state: rebuild the first bulk plot/report calculations as explicit fixtures:
   - done for formation-energy and volume-per-atom errors by OQMD structure
   - todo for lattice constants/angles as metrics rather than recorded-only scalar outputs
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

1. Todo: add unit tests around the migrated helper functions before expanding to expensive model calculations.
2. Done: add a tiny in-repo smoke subset using pure Al/Cu/Mg/Zn OQMD structures.
3. Keep the first real-model smoke target close to the successful legacy probe: MACE-MP small on CUDA, starting with the staged `OQMD_8100` path but running the full four-structure first slice. Add one non-elemental OQMD structure immediately after the plumbing smoke succeeds.
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

7. Run targeted tests before broad suite runs:

```shell
pytest ml_peg/calcs/alloy_metallurgy/alzncumg_regression \
       ml_peg/analysis/alloy_metallurgy/alzncumg_regression
```

Completed cheap validation checks:

- Real first-slice calculation with `mace-mp-small`, producing `bulk_properties.json` and `OQMD_*.xyz` outputs.
- Analysis artifact generation for `mace-mp-small`, producing the metrics table, formation-energy parity plot, volume parity plot, and copied structure assets.
- Live Dash app smoke on `alloy_metallurgy` with table rendering, parity-plot switching, and plot-point-to-structure click-through.
- Python compile check for the new calc, analysis, and app modules.
- Ruff check for the new calc, analysis, and app modules.
- ML-PEG discovery check for `calc`, `analyse`, and `app` using `alloy_metallurgy/alzncumg_regression`.
- ASE/JSON read checks for staged OQMD structures, OQMD metadata, and DFT reference keys.
- Import check for the new modules.
- VS Code diagnostics check for the new Python files.
- `git diff --check` whitespace validation.

## First Implementation Slice

Current first-slice target:

1. Done: bulk OQMD structure properties for a short list of Al/Cu/Mg/Zn structures.
2. Done: single-point formation energies, volume per atom, and recorded lattice parameters for pure elemental endpoints.
3. Done: one generated analysis table with formation-energy and volume MAE metrics for `mace-mp-small`.
4. Done: generated parity plots with structure click-through assets copied during analysis.
5. Done: documentation entry and app category registration.

This slice proves the ML-PEG data flow without immediately porting expensive solute, GSF, antisite, and surface workflows. Once it works, add the remaining property families incrementally.

## Open Questions

1. Which expanded structures and reference values can be distributed in-repo, and which must move to S3/downloaded data?
2. Answered for first slice: include pure Al/Cu/Mg/Zn reference structures (`8100`, `635950`, `9226`, `122929`) immediately.
3. Answered for first smoke: use `mace-mp-small`, now configured in ML-PEG's model registry and aligned with the previous MACE-MP small smoke test.
4. Which first binary or multicomponent OQMD structure should be added to make formation-energy parity nontrivial?
5. Should the calc stage relax structures before comparing to DFT references, or should single-point and relaxed bulk metrics be tracked separately?
6. Should the deprecated/buggy interface-energy path be repaired during this migration or tracked as a follow-up benchmark?

## Immediate Next Steps

1. Add one binary or multicomponent OQMD structure from the existing evalpot list, with metadata and DFT reference coverage confirmed.
2. Add focused unit tests for the first-slice helper functions before porting additional property families.
3. Decide whether the next port should be bulk relaxation/lattice metrics, elastic constants, or solute-solute interactions.