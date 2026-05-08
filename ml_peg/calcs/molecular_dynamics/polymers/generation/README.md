# Polymer cell generation (optional)

This sub-package generates polymer starting cells from SMILES via the
[EMC](http://montecarlo.sourceforge.net/emc/) packer. The main `polymers`
benchmark in `ml-peg` does **not** require this code — it consumes
pre-built starting structures pulled from S3. This module exists so users
can regenerate (or extend) those structures themselves.

## Requirements

You need one extra package that is **not** part of the default ml-peg
install:

- [`emc-pypi`](https://pypi.org/project/emc-pypi/) (`>=2025.8.21`) — bundles
  the EMC binary and setup script. Imported in Python as `pyemc`.

Install it into your existing ml-peg environment:

```bash
pip install "emc-pypi>=2025.8.21"
```

Or with `uv`:

```bash
uv pip install "emc-pypi>=2025.8.21"
```

EMC ships a Linux x86_64 binary inside the `emc-pypi` wheel; macOS and other
platforms are not supported by the upstream package.

## Usage

Run the CLI as a module from the repo root, providing one or more polymer
ids (or none, to build all 130) and an output directory:

```bash
python -m ml_peg.calcs.molecular_dynamics.polymers.generation.build \
    --poly-id PS --output-dir ./cells --temp-k 300
```

This writes `./cells/PS.xyz` — an extxyz starting cell with `charge=0`,
`spin=1`, `n_ru_per_chain`, `seed`, and `build_date` stored in
`atoms.info`. EMC's intermediate files (`.esh`, `.data`, `.in`) are written
to a temporary directory and removed automatically.

Other knobs (see `--help`):

| Flag                  | Default | Notes                                  |
| --------------------- | ------- | -------------------------------------- |
| `--n-total`           | 10000   | Total number of atoms in the cell      |
| `--n-ru-per-chain`    | 10      | Number of repeat units per chain       |
| `--initial-density`   | 0.5     | EMC packing density in g/cm³           |
| `--seed`              | 42      | EMC random seed                        |

EMC's multi-threaded mode is unreliable in practice and the CLI always runs
single-threaded (`n_threads=1` in `wrapper.prepare`). If you want to
experiment with EMC threading, call `wrapper.prepare` directly with
`n_threads=...`.

## Generating the full S3 archive

The benchmark expects a zip archive at the S3 key
`inputs/molecular_dynamics/polymers/polymers.zip` whose top-level directory
is `polymers/` and which contains one `<poly_id>.xyz` per row in
`data.csv`. The CLI's "all polymers" mode builds them straight into a
single directory:

```bash
python -m ml_peg.calcs.molecular_dynamics.polymers.generation.build \
    --output-dir polymers --temp-k 300
zip -r polymers.zip polymers
```

## Provenance

The science (EMC config, polymer stats, and LAMMPS-data → ASE conversion)
was ported from
[`microsoft/simpoly`](https://github.com/microsoft/simpoly) /
[`simpoly.poly_arena.generation`](https://github.com/microsoft/simpoly/tree/main/src/simpoly/poly_arena/generation)
under the MIT license.
