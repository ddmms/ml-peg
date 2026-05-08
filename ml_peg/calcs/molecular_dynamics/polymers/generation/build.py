r"""CLI: build polymer starting cells with EMC and write extxyz files.

Run as a module from the repo root, e.g.

    # Build a single polymer
    python -m ml_peg.calcs.molecular_dynamics.polymers.generation.build \
        --poly-id PS --temp-k 300 --output-dir ./cells

    # Build several polymers
    python -m ml_peg.calcs.molecular_dynamics.polymers.generation.build \
        --poly-id PS --poly-id PE --output-dir ./cells

    # Build all polymers in data.csv (omit --poly-id)
    python -m ml_peg.calcs.molecular_dynamics.polymers.generation.build \
        --output-dir ./cells

The script looks up the SMILES + end groups for each ``--poly-id`` in the
bundled ``data.csv``, runs EMC to pack a polymer cell, and writes
``<output_dir>/<poly_id>.xyz`` (extxyz, with ``charge=0``, ``spin=1``,
``n_ru_per_chain``, ``seed``, and ``build_date`` set in ``atoms.info``).
EMC's intermediate files are written to a temporary directory that is
cleaned up automatically.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import pathlib
import tempfile

import ase.io
import pandas as pd

from ml_peg.calcs.molecular_dynamics.polymers.generation import tools, wrapper

LOG = logging.getLogger(__name__)

DATA_CSV = pathlib.Path(__file__).resolve().parents[1] / "resources" / "data.csv"


def _load_polymer_table() -> pd.DataFrame:
    """Load the polymer table indexed by polymer id."""
    df = pd.read_csv(DATA_CSV, na_values=["NaN"], encoding="utf-8", comment="%")
    return df.set_index("id").sort_index()


def build_polymer_cell(
    *,
    poly_id: str,
    output_dir: pathlib.Path,
    temp_k: float,
    n_total: int = 2_000,
    n_ru_per_chain: int = 10,
    initial_density_g_cm3: float = 0.5,
    seed: int = 42,
) -> pathlib.Path:
    """
    Build one polymer starting cell with EMC and write ``<poly_id>.xyz``.

    EMC's intermediate ``.esh`` / ``.data`` / ``.in`` files are written into a
    fresh per-call temporary directory that is removed on exit. Only the
    final extxyz lands in ``output_dir``.

    Parameters
    ----------
    poly_id
        Polymer identifier (must be present in ``data.csv``).
    output_dir
        Directory the final ``<poly_id>.xyz`` is written to (created if
        missing). EMC scratch files do **not** end up here.
    temp_k
        Target temperature passed to EMC (in K).
    n_total
        Total number of atoms in the cell. Default: 2 000.
    n_ru_per_chain
        Number of repeat units per chain. Default: 10.
    initial_density_g_cm3
        Initial packing density passed to EMC (in g/cm³). Default: 0.5.
    seed
        Random seed for EMC. Default: 42.

    Returns
    -------
    pathlib.Path
        Path to the produced extxyz file.
    """
    table = _load_polymer_table()
    if poly_id not in table.index:
        raise KeyError(f"poly_id '{poly_id}' not found in {DATA_CSV}")
    row = table.loc[poly_id]

    output_dir.mkdir(parents=True, exist_ok=True)

    config = wrapper.Config(
        ru_smiles=str(row["smiles"]),
        first_cap=str(row["end_group_0"]),
        second_cap=str(row["end_group_1"]),
        n_ru_per_chain=n_ru_per_chain,
        n_total=n_total,
        density=initial_density_g_cm3,
        temperature=temp_k,
        seed=seed,
    )

    with tempfile.TemporaryDirectory(prefix=f"emc-{poly_id}-") as tmp_str:
        work_dir = pathlib.Path(tmp_str)
        LOG.info(f"Building cell for {poly_id} in {work_dir}")
        # EMC's multi-threading is unreliable in practice; always run single-threaded.
        lammps_data_path = wrapper.prepare(config, work_dir, n_threads=1)
        atoms = tools.lammps_data_to_atoms(lammps_data_path)

    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.info["n_ru_per_chain"] = int(n_ru_per_chain)
    atoms.info["seed"] = int(seed)
    atoms.info["date"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    extxyz_path = output_dir / f"{poly_id}.xyz"
    ase.io.write(str(extxyz_path), atoms, format="extxyz")
    LOG.info(f"Wrote {extxyz_path}")
    return extxyz_path


def main() -> None:
    """CLI entry point — see module docstring for usage."""
    table = _load_polymer_table()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--poly-id",
        action="append",
        choices=table.index.tolist(),
        metavar="POLY_ID",
        help=(
            f"Polymer id (one of {len(table)} entries in {DATA_CSV.name}; "
            "e.g., PS, PE, CR). Repeat the flag to build several polymers; "
            "omit it to build all polymers in the table."
        ),
    )
    ap.add_argument("--output-dir", required=True, type=pathlib.Path)
    ap.add_argument("--temp-k", type=float, default=300.0)
    ap.add_argument("--n-total", type=int, default=2_000)
    ap.add_argument("--n-ru-per-chain", type=int, default=10)
    ap.add_argument("--initial-density", type=float, default=0.5, help="g/cm^3")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    poly_ids: list[str] = list(args.poly_id) if args.poly_id else table.index.tolist()
    LOG.info(f"Building {len(poly_ids)} polymer cell(s)")

    for poly_id in poly_ids:
        build_polymer_cell(
            poly_id=poly_id,
            output_dir=args.output_dir,
            temp_k=args.temp_k,
            n_total=args.n_total,
            n_ru_per_chain=args.n_ru_per_chain,
            initial_density_g_cm3=args.initial_density,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
