"""
Probe which elements each MLIP supports via homonuclear dimer energies.

For every element, build a dimer and try to compute its energy with the
model. If it raises (or returns a non-finite energy), the element is treated as
unsupported and the error is logged. Models that don't import in the current
environment are skipped, so run this once per ``uv sync --extra <x>`` and the
results accumulate.

Run once per environment, for example::

    uv sync --extra mace
    python ml_peg/models/element_coverage/find_supported_elements.py \
        --models mace-mp-0a,mace-mp-0b3
    # or, probe every model that imports in this env (omit --models):
    python ml_peg/models/element_coverage/find_supported_elements.py

Outputs (merged across runs) live next to this script:
- ``model_supported_elements.json``        {model: [sorted supported symbols]}
- ``model_supported_elements_errors.jsonl`` one record per unsupported element
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
import numpy as np

from ml_peg.analysis.utils.periodic_table import PERIODIC_TABLE_SYMBOLS
from ml_peg.models.get_models import get_model_names, load_models

OUT_PATH = Path(__file__).parent / "model_supported_elements.json"
ERR_PATH = Path(__file__).parent / "model_supported_elements_errors.jsonl"

_BOX = 20.0


def make_dimer(symbol: str) -> Atoms:
    """
    Build a homonuclear dimer for ``symbol`` in a large vacuum cell.

    Parameters
    ----------
    symbol
        Element symbol used for both atoms of the dimer.

    Returns
    -------
    Atoms
        Two-atom dimer in a vacuum cell with neutral charge and spin set.
    """
    z = atomic_numbers[symbol]
    d = 2.0 * covalent_radii[z]
    if not np.isfinite(d) or d <= 0.5:
        d = 2.0
    atoms = Atoms(
        [symbol, symbol],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, d]],
        cell=[_BOX, _BOX, _BOX],
        pbc=True,
    )
    atoms.center()
    # some models require a charge/spin
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return atoms


def probe_model(model_name: str, device: str) -> list[str] | None:
    """
    Return the elements ``model_name`` supports, or ``None`` if it won't load.

    Parameters
    ----------
    model_name
        Model key from ``models.yml``.
    device
        Device to force on the calculator (this probe uses tiny dimers, so CPU
        is fine and portable).

    Returns
    -------
    list[str] | None
        Sorted supported element symbols, or ``None`` if the model could not be
        loaded in the current environment.
    """
    try:
        model = load_models(model_name)[model_name]
        if hasattr(model, "device"):
            model.device = device
        calc = model.get_calculator()
    except Exception as exc:  # noqa: BLE001 - env-dependent import/load failures
        print(
            f"  ! {model_name}: could not load in this env "
            f"({type(exc).__name__}: {exc})"
        )
        return None

    supported: list[str] = []
    errors: list[dict] = []
    for symbol in PERIODIC_TABLE_SYMBOLS:
        atoms = make_dimer(symbol)
        atoms.calc = calc
        try:
            energy = atoms.get_potential_energy()
            if not np.isfinite(energy):
                raise ValueError("non-finite energy")
            supported.append(symbol)
        except Exception as exc:  # noqa: BLE001 - unsupported element surfaces here
            errors.append(
                {
                    "model": model_name,
                    "element": symbol,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                }
            )

    if errors:
        with open(ERR_PATH, "a") as handle:
            for record in errors:
                handle.write(json.dumps(record) + "\n")
    total = len(PERIODIC_TABLE_SYMBOLS)
    print(f"  {model_name}: {len(supported)}/{total} elements supported")
    return supported


def main() -> None:
    """Probe the requested models and merge results into the output JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names. Default: every model in models.yml.",
    )
    parser.add_argument("--device", default="cpu", help="Device for the calculator.")
    args = parser.parse_args()

    names = get_model_names(args.models) if args.models else get_model_names(None)

    results: dict[str, list[str]] = {}
    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text())

    for name in names:
        supported = probe_model(name, args.device)
        if supported is not None:
            results[name] = supported
            OUT_PATH.write_text(json.dumps(dict(sorted(results.items())), indent=2))

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
