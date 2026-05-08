# Polymer-density benchmark

Predicts room-temperature amorphous densities for **130 polymers** with an
ML interatomic potential and compares against the experimental reference in
[`resources/data.csv`](resources/data.csv). **Metric:** MAE (g/cm³).

## Protocol

24-stage Polymatic-style equilibration ported from
[SimPoly](https://github.com/microsoft/simpoly)'s LAMMPS
`build_21steps_protocol` to ASE. 0.5 fs timestep, **2.56 ns total** per
(model, polymer) at the default `--time-prefactor 1.0`:

- minimization + Maxwell-Boltzmann velocity init
- 100 ps NVT preheats
- 600 ps "upward shaking" (3 NPT/NVT/NVT cycles, 0.02 → 0.6 → 1.0 × P_max)
- 60 ps "downward shaking" (3 cycles, 0.5 → 0.1 → 0.01 × P_max)
- 800 ps NPT equilibration
- 1000 ps NPT **production** (density averaged here)

`P_max = 49 346.2 atm`. LAMMPS `npt aniso` → `MaskedMTKNPT(mask=(True, True,
True))`. The original SimPoly protocol has 25 stages; the LAMMPS
`nve_preheat` is omitted here (no ASE analogue, unnecessary for ML PESs).

## Running

Starting structures are pulled from S3
(`inputs/molecular_dynamics/polymers/polymers.zip`); see
[`generation/README.md`](generation/README.md) to rebuild them.

```bash
# Full protocol (~2.56 ns; hours on a GPU)
uv run pytest -v -s \
    ml_peg/calcs/molecular_dynamics/polymers/calc_polymers.py \
    --poly-id PS --models mace-mp-0a

# Smoke test (~13 ps; minutes on a GPU)
uv run pytest -v -s \
    ml_peg/calcs/molecular_dynamics/polymers/calc_polymers.py \
    --poly-id PS --models mace-mp-0a --time-prefactor 0.005
```

Outputs land in `outputs/<model>/<poly_id>/` (one trajectory per stage,
plus a `state.json` for resumability). Failures are caught and
`pytest.skip`'d so the rest of the matrix still runs.

Analysis (averages density over the production stage, writes parity plot
and MAE table for the dashboard):

```bash
uv run pytest -v -s ml_peg/analysis/molecular_dynamics/polymers/analyse_polymers.py
```

## Citation

```bibtex
@online{Simm2025SimPoly,
  title = {{{SimPoly}}: {{Simulation}} of {{Polymers}} with {{Machine Learning Force Fields Derived}} from {{First Principles}}},
  author = {Simm, Gregor N. C. and Hélie, Jean and Schulz, Hannes and Chen, Yicheng and Simeon, Guillem and Kuzina, Anna and Martinez-Baez, Ernesto and Gasparotto, Piero and Tocci, Gabriele and Chen, Chi and Li, Yatao and Cheng, Lixue and Wang, Zun and Nguyen, Bichlien H. and Smith, Jake A. and Sun, Lixin},
  date = {2025-10-15},
  eprint = {2510.13696},
  eprinttype = {arXiv},
  eprintclass = {physics},
  doi = {10.48550/arXiv.2510.13696},
  url = {http://arxiv.org/abs/2510.13696}
}
```
