"""Tests for the Alexandria convex-hull benchmark."""

from __future__ import annotations

from ase import Atoms
import numpy as np
import pytest

from ml_peg.analysis.bulk_crystal.convex_hull_alexandria.analyse_convex_hull_alexandria import (  # noqa: E501
    classification_f1,
    compare_model,
    element_values,
    hull_energies,
    hull_rows,
    model_metrics,
    signed_errors,
)
from ml_peg.calcs.bulk_crystal.convex_hull_alexandria.prepare_inputs import (
    is_plus_u,
    select_entries,
)


def make_structure(symbols: str, mat_id: str, dft_energy: float) -> Atoms:
    """
    Make a structure with reference data in ``info``.

    Parameters
    ----------
    symbols
        Chemical symbols, such as ``"Cu3Zn"``.
    mat_id
        Identifier.
    dft_energy
        Reference total energy (eV).

    Returns
    -------
    Atoms
        Structure, with positions irrelevant to the hull.
    """
    atoms = Atoms(symbols, cell=[3, 3, 3], pbc=True)
    atoms.info = {"mat_id": mat_id, "dft_energy": dft_energy}
    return atoms


# Cu-Zn system: elements at -1 and -2 eV/atom; formation energies -0.5 (CuZn, on the
# hull), -0.4 (CuZn polymorph, 0.1 above) and -0.2 eV/atom (Cu3Zn, 0.05 above the
# hull at -0.25 eV/atom between Cu and CuZn).
STRUCTURES = [
    make_structure("Cu", "cu", -1.0),
    make_structure("Zn", "zn", -2.0),
    make_structure("CuZn", "cuzn", -4.0),
    make_structure("CuZn", "cuzn-2", -3.8),
    make_structure("Cu3Zn", "cu3zn", -5.8),
]
REFERENCE = {atoms.info["mat_id"]: atoms.info["dft_energy"] for atoms in STRUCTURES}


def test_hull_energies():
    """Test formation energies and energies above the hull of a binary system."""
    results = hull_energies(STRUCTURES, REFERENCE)
    assert set(results) == {"cuzn", "cuzn-2", "cu3zn"}
    assert results["cuzn"]["system"] == "Cu-Zn"
    for mat_id, e_form, e_hull in (
        ("cuzn", -0.5, 0.0),
        ("cuzn-2", -0.4, 0.1),
        ("cu3zn", -0.2, 0.05),
    ):
        assert results[mat_id]["e_form"] == pytest.approx(e_form)
        assert results[mat_id]["e_hull"] == pytest.approx(e_hull, abs=1e-12)


def test_hull_energies_failed():
    """Test that a failed or missing calculation fails its whole system."""
    for energies in (REFERENCE | {"zn": np.nan}, REFERENCE | {"cu3zn": np.nan}):
        results = hull_energies(STRUCTURES, energies)
        assert all(
            r["e_hull"] is None and r["e_form"] is None for r in results.values()
        )
        assert results["cuzn"]["system"] == "Cu-Zn"

    missing = {k: v for k, v in REFERENCE.items() if k != "cuzn"}
    assert hull_energies(STRUCTURES, missing)["cu3zn"]["e_hull"] is None


def test_compare_model_shifted_elements():
    """Test that shifting the energy of each element leaves the hull unchanged."""
    shift = {"Cu": 0.3, "Zn": -0.7}
    energies = {
        atoms.info["mat_id"]: atoms.info["dft_energy"]
        + sum(shift[symbol] for symbol in atoms.get_chemical_symbols())
        for atoms in STRUCTURES
    }
    metrics = model_metrics(compare_model(STRUCTURES, energies))
    assert metrics["Energy above hull MAE"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["Formation energy MAE"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["Stability F1"] == 1.0
    assert metrics["Failed systems"] == 0.0


def test_compare_model_errors():
    """Test the metrics when the model destabilises CuZn by 0.15 eV/atom."""
    energies = REFERENCE | {"cuzn": -4.0 + 2 * 0.15}
    comparison = compare_model(STRUCTURES, energies)
    # The model hull runs from Cu to the CuZn polymorph (-0.4 eV/atom)
    assert comparison["cuzn"]["e_hull_pred"] == pytest.approx(0.05)
    assert comparison["cuzn-2"]["e_hull_pred"] == pytest.approx(0.0, abs=1e-12)
    assert comparison["cu3zn"]["e_hull_pred"] == pytest.approx(0.0, abs=1e-12)

    metrics = model_metrics(comparison)
    # E_hull errors 50, 100 and 50 meV/atom; formation errors 150, 0 and 0 meV/atom
    assert metrics["Energy above hull MAE"] == pytest.approx(200 / 3)
    assert metrics["Formation energy MAE"] == pytest.approx(50.0)
    assert metrics["Formation energy RMSE"] == pytest.approx(np.sqrt(150**2 / 3))
    # Reference stable: CuZn. Predicted stable: CuZn polymorph and Cu3Zn.
    assert metrics["Stability F1"] == 0.0

    assert element_values(comparison, "e_hull") == {
        "Cu": pytest.approx(200 / 3),
        "Zn": pytest.approx(200 / 3),
    }


def test_signed_errors_and_hull_rows():
    """Test the signed errors and the hull data when CuZn is destabilised."""
    comparison = compare_model(STRUCTURES, REFERENCE | {"cuzn": -4.0 + 2 * 0.15})
    assert signed_errors(comparison, "e_form") == pytest.approx([150.0, 0.0, 0.0])
    assert signed_errors(comparison, "e_hull") == pytest.approx([50.0, -100.0, -50.0])

    rows = {row["mat_id"]: row for row in hull_rows(STRUCTURES, comparison)["Cu-Zn"]}
    assert set(rows) == {"cuzn", "cuzn-2", "cu3zn", "Cu", "Zn"}
    assert rows["cu3zn"]["x"] == 0.25
    assert rows["cu3zn"]["formula"] == "Cu3Zn"
    assert (rows["Cu"]["x"], rows["Zn"]["x"]) == (0, 1)
    assert rows["cuzn"]["e_form_pred"] == pytest.approx(-350.0)
    assert (rows["cuzn"]["stable_ref"], rows["cuzn"]["stable_pred"]) == (True, False)
    assert (rows["cu3zn"]["stable_ref"], rows["cu3zn"]["stable_pred"]) == (False, True)

    failed = compare_model(STRUCTURES, REFERENCE | {"zn": np.nan})
    assert signed_errors(failed, "e_hull") == []
    assert hull_rows(STRUCTURES, failed) == {}


def test_model_metrics_failed():
    """Test the metrics when a model was not run or all its systems failed."""
    assert set(model_metrics(None).values()) == {None}

    metrics = model_metrics(compare_model(STRUCTURES, REFERENCE | {"zn": np.nan}))
    assert metrics["Failed systems"] == 100.0
    assert metrics["Energy above hull MAE"] is None


def test_classification_f1():
    """Test the F1 score of the stability classification."""
    e_hull_ref = np.array([0.0, 0.0, 0.1, 0.2])
    # One true positive, one false negative and one false positive
    assert classification_f1(
        e_hull_ref, np.array([0.0, 0.05, 0.0, 0.2])
    ) == pytest.approx(0.5)
    assert classification_f1(e_hull_ref, e_hull_ref) == 1.0
    assert classification_f1(e_hull_ref, np.array([0.1, 0.1, 0.0, 0.0])) == 0.0


def test_is_plus_u():
    """Test which chemical systems are treated with PBE+U."""
    assert is_plus_u({"Fe", "O"})
    assert is_plus_u({"F", "W"})
    assert not is_plus_u({"Fe", "S"})
    assert not is_plus_u({"Mg", "O"})
    assert not is_plus_u({"Fe", "Ni"})


def make_entry(composition: dict[str, int], mat_id: str, energy: float) -> dict:
    """
    Make an Alexandria-like entry.

    Parameters
    ----------
    composition
        Number of atoms of each element.
    mat_id
        Identifier.
    energy
        Total energy (eV).

    Returns
    -------
    dict
        Entry with the fields used by ``select_entries``.
    """
    return {
        "composition": composition,
        "data": {"mat_id": mat_id, "energy_total": energy},
    }


def test_select_entries():
    """Test the selection of ground states, stable and sampled unstable binaries."""
    entries = [
        make_entry({"Cu": 1}, "cu-1", -1.0),
        make_entry({"Cu": 2}, "cu-2", -2.2),  # Cu ground state, -1.1 eV/atom
        make_entry({"Zn": 1}, "zn-1", -2.0),
        make_entry({"Ag": 1}, "ag-1", -3.0),  # Element of no selected binary
        make_entry({"Cu": 1, "Zn": 1}, "stable", -4.0),
    ]
    e_hull = {"stable": 0.0}
    for i, value in enumerate((0.01, 0.02, 0.03, 0.07, 0.15, 0.5, 0.6, 0.7)):
        entries.append(make_entry({"Cu": 1, "Zn": 1}, f"unstable-{i}", -3.0))
        e_hull[f"unstable-{i}"] = value

    selected = {
        entry["data"]["mat_id"]: entry for entry in select_entries(entries, e_hull)
    }
    assert {"stable", "cu-2", "zn-1"} <= set(selected)
    assert not {"cu-1", "ag-1"} & set(selected)
    unstable = [mat_id for mat_id in selected if mat_id.startswith("unstable")]
    assert len(unstable) == 5
    # One entry from each energy-above-hull bin
    bins = [(0, 0.025), (0.025, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, np.inf)]
    values = [selected[mat_id]["data"]["e_above_hull"] for mat_id in unstable]
    for low, high in bins:
        assert sum(low < value <= high for value in values) == 1
    assert select_entries(entries, e_hull) == select_entries(entries, e_hull)
