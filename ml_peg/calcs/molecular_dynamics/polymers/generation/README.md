# Polymer cell generation (optional)

This sub-package builds the starting structures the
[polymer benchmark](../README.md) consumes from S3. The benchmark itself
does **not** require this code. Use it to regenerate or extend the
`polymers.zip` archive.

## Install

Linux x86_64 only — EMC ships as a prebuilt binary.

```bash
uv pip install "emc-pypi>=2025.8.21"   # imported as `pyemc`
```

## Usage

```bash
python -m ml_peg.calcs.molecular_dynamics.polymers.generation.build --help
```

The CLI takes one or more `--poly-id` (omit to build all 130 from
[`../resources/data.csv`](../resources/data.csv)) and writes
`<output_dir>/<poly_id>.xyz`. EMC's intermediate files go to a tempdir and
are removed automatically. `atoms.info` carries `charge=0`, `spin=1`,
`n_ru_per_chain`, `seed`, and `build_date`.

To rebuild the S3 archive:

```bash
python -m ml_peg.calcs.molecular_dynamics.polymers.generation.build \
    --output-dir polymers --temp-k 300
zip -r polymers.zip polymers
```

The benchmark expects the archive at S3 key
`inputs/molecular_dynamics/polymers/polymers.zip` with `polymers/` as the
top-level directory and one `<poly_id>.xyz` per row of `data.csv`.

## Provenance & citation

Ported from
[`microsoft/simpoly`](https://github.com/microsoft/simpoly) (MIT). For the
BibTeX entry see [the parent README](../README.md#citation).
