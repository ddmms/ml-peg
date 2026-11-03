# Binary liquid mixtures benchmark

Enthalpy of mixing (ΔHmix), mixture density and excess volume (V^E) for 10
binary liquid mixtures at three compositions each, plus the pure-liquid
densities not already scored by `liquid_densities` (50 boxes in total). Each
box is run once per model: 0.2 ns NPT equilibration and 1 ns production.

## Inputs

Starting boxes and reference data are pulled from S3
(`inputs/molecular_dynamics/binary_mixtures/binary_mixtures.zip`).
They are generated in a separate repository,
[ml-peg-liquid-mixtures-input-generation](https://github.com/fjclark/ml-peg-liquid-mixtures-input-generation)
(Python package `liquid_mixtures_inputs`). That repository covers:

- system selection from the OpenFF Sage physical-property sets and NIST ThermoML;
- building the boxes with Sage;
- the Sage (openff-2.3.0) reference runs;
- the selection, Sage and power-analysis reports, including the metric
  thresholds in `metrics.yml`.

The OpenFF stack it needs is conda-only, so it is kept out of ml-peg.

That repository's `summarise_sage` imports
`ml_peg.analysis.molecular_dynamics.binary_mixtures.mixing`, so the Sage
reference and the MLIPs share one estimator. Treat the public names in
`mixing.py` as an API.

## Running and sharing results

The runs are too long for users to repeat, so results are shared as app data
rather than as calc outputs. Run each box as a separate job (`--box-id` from 0
to 49; a rerun continues from the last saved frame):

```bash
pytest -v ml_peg/calcs/molecular_dynamics/binary_mixtures/calc_binary_mixtures.py \
    --run-multi-day --models <model> --box-id <n>
```

This writes `outputs/<model>/<box_id>.csv` (scalars every 100 fs) and
`<box_id>.traj` (frames every 10 ps). When all 50 boxes are complete, add the
model to the existing app data with:

```bash
ml_peg analyse --category molecular_dynamics --test binary_mixtures \
    --models <model> --update
```

All of the app data for a model, including the final structures, is written
to `ml_peg/app/data/molecular_dynamics/binary_mixtures/<model>/` (about 6 MB),
so the app needs none of the calc outputs. The updated app data is then
published in `app/data/data.tar.gz` on S3 (see the developer guide's Data
page), which users download to view the results.

The MD is the expensive stage, so also archive the calc outputs, in case the
analysis or `metrics.yml` changes. Zip the model's output directory (about
0.7 GB) and upload it:

```bash
cd ml_peg/calcs/molecular_dynamics/binary_mixtures/outputs
zip -r <model>.zip <model>
ml_peg upload --key outputs/molecular_dynamics/binary_mixtures/<model>.zip \
    --filename <model>.zip --credentials credentials.json
```

To re-analyse a model, download its zip and unzip it in `outputs/`:

```bash
ml_peg download --key outputs/molecular_dynamics/binary_mixtures/<model>.zip \
    --filename <model>.zip
unzip <model>.zip
```

## Data contract

The zip contains `binary_mixtures/` with `boxes/<box_id>.xyz` (extxyz,
`exp_temperature`, SMILES and molecule counts in `atoms.info`) and the three
tables below. Columns read by the calc and analysis:

- `systems.csv` (one row per mixture point): `system_id`, `a`, `b` (SMILES),
  `temperature`, `x_box`, `box_id`, `hmix_exp`, `hmix_sigma`, `density_exp`,
  `density_sigma` (both empty where there is no mixture density), `vex_exp`,
  `vex_sigma`, and the Sage reference columns `hmix_sage`, `density_sage`,
  `vex_sage`, `sigma_ref_hmix`, `sigma_ref_density`, `sigma_ref_vex`.
- `boxes.csv` (one row per box): `box_id`, `kind` (`mixture` or `pure`),
  `smiles`, `counts`, `temperature`, `x`, and the Sage like-neighbour reference
  `like_fraction_sage`, `like_fraction_sage_sd` (mixture boxes only).
- `pure.csv`: `smiles`, `temperature`, `density`, `sigma`, `scored` (true for
  pure liquids not already scored by `liquid_densities`), and the Sage
  reference columns `density_sage`, `sigma_ref_density`.

The Sage reference columns are written by `summarise_sage` in the input
repository, only once every Sage replicate there is complete; ML-PEG reads
them as fixed values. When any column changes, rename the zip (and `DATA_ZIP`
in `calc_binary_mixtures.py`) so that cached copies are not reused.
