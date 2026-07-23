"""Tests for optional crystal-symmetry and structure-comparison helpers."""

# ruff: noqa: E402

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.framework("matbench-discovery")

pytest.importorskip("ase")
pytest.importorskip("moyopy")
pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure

from ml_peg.analysis.bulk_crystal.geo_opt.analyse_geo_opt import (
    analyze_geo_opt_dataframes,
)
from ml_peg.analysis.bulk_crystal.geo_opt.metrics import (
    N_STRUCTURES,
    SYMMETRY_MATCH,
    calc_geo_opt_metrics,
)
from ml_peg.analysis.bulk_crystal.geo_opt.schema import (
    ANGLE_TOLERANCE,
    CONVERGED,
    ENERGY,
    HALL_SYMBOL,
    INTERNATIONAL_SPG_NAME,
    MATERIAL_ID,
    MAX_PAIR_DIST,
    N_STEPS,
    N_SYM_OPS,
    N_SYM_OPS_DIFF,
    SITE_SYMMETRY_SYMBOLS,
    SPG_NUM,
    SPG_NUM_DIFF,
    STRUCTURE,
    STRUCTURE_RMSD_VS_DFT,
    SYMPREC,
)
from ml_peg.analysis.bulk_crystal.geo_opt.symmetry import (
    get_sym_info_from_structs,
    pred_vs_ref_struct_symmetry,
)


@pytest.fixture
def cubic_structure() -> Structure:
    """Create a two-site cubic CsCl structure."""
    return Structure(
        Lattice.cubic(4.0),
        ["Cs", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _compare_structures(
    predicted_structures: dict[str, Structure],
    reference_structures: dict[str, Structure],
) -> pd.DataFrame:
    """Compare structure mappings after deriving their symmetry tables."""
    return pred_vs_ref_struct_symmetry(
        get_sym_info_from_structs(predicted_structures, pbar=False),
        get_sym_info_from_structs(reference_structures, pbar=False),
        predicted_structures,
        reference_structures,
        pbar=False,
    )


@pytest.fixture
def geo_opt_dataframes(
    cubic_structure: Structure,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create matching prediction and reference tables."""
    structure_dict = cubic_structure.as_dict()
    predictions = pd.DataFrame(
        [
            {
                MATERIAL_ID: "sample",
                STRUCTURE: structure_dict,
                ENERGY: -1.0,
                CONVERGED: True,
                N_STEPS: 5,
            }
        ]
    )
    references = pd.DataFrame([{MATERIAL_ID: "sample", STRUCTURE: structure_dict}])
    return predictions, references


@pytest.mark.parametrize("use_ase_atoms", [False, True])
def test_get_sym_info_from_structs_supports_input_types(
    cubic_structure: Structure, use_ase_atoms: bool
) -> None:
    """Analyze equivalent pymatgen Structure and ASE Atoms inputs."""
    structure = cubic_structure.to_ase_atoms() if use_ase_atoms else cubic_structure

    symmetry = get_sym_info_from_structs(
        {"sample": structure},
        pbar=False,
        symprec=1e-2,
        angle_tolerance=0.05,
    )

    symmetry_row = symmetry.loc["sample"]
    assert (symmetry.index.name, symmetry.index.tolist()) == (MATERIAL_ID, ["sample"])
    assert symmetry_row[
        [
            SPG_NUM,
            INTERNATIONAL_SPG_NAME,
            SITE_SYMMETRY_SYMBOLS,
            HALL_SYMBOL,
            N_SYM_OPS,
        ]
    ].tolist() == [221, "P m -3 m", ["m-3m", "m-3m"], "-P 4 2 3", 48]
    assert symmetry_row[[SYMPREC, ANGLE_TOLERANCE]].tolist() == pytest.approx(
        [1e-2, 0.05]
    )


@pytest.mark.parametrize(
    ("symprec", "angle_tolerance", "message"),
    [
        (0.0, None, "symprec must be positive"),
        (float("nan"), None, "symprec must be positive"),
        (1e-2, float("inf"), "angle_tolerance must be"),
        (1e-2, -0.1, "angle_tolerance must be"),
    ],
)
def test_get_sym_info_from_structs_rejects_invalid_tolerances(
    cubic_structure: Structure,
    symprec: float,
    angle_tolerance: float | None,
    message: str,
) -> None:
    """Reject invalid moyopy tolerance values before analysis."""
    with pytest.raises(ValueError, match=message):
        get_sym_info_from_structs(
            {"sample": cubic_structure},
            pbar=False,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )


def test_get_sym_info_from_structs_rejects_invalid_structure() -> None:
    """Reject values that are neither pymatgen structures nor ASE atoms."""
    with pytest.raises(TypeError, match="Expected pymatgen Structure or ASE Atoms"):
        get_sym_info_from_structs({"sample": {}}, pbar=False)


@pytest.mark.parametrize(
    ("comparison_kind", "expectation"),
    [
        ("perfect", "zero"),
        ("distorted", "positive"),
        ("unmatched", "missing"),
    ],
)
def test_pred_vs_ref_struct_symmetry_matching_outcomes(
    cubic_structure: Structure,
    comparison_kind: str,
    expectation: str,
) -> None:
    """Return normalized distances for perfect, distorted, and unmatched pairs."""
    predicted_structure = cubic_structure.copy()
    if comparison_kind == "distorted":
        predicted_structure.translate_sites(
            [1],
            [0.05, 0.0, 0.0],
            frac_coords=True,
            to_unit_cell=True,
        )
    elif comparison_kind == "unmatched":
        predicted_structure = Structure(
            cubic_structure.lattice,
            ["Cs"],
            [[0.0, 0.0, 0.0]],
        )

    compared = _compare_structures(
        {"sample": predicted_structure},
        {"sample": cubic_structure},
    )

    assert {SPG_NUM_DIFF, N_SYM_OPS_DIFF} <= set(compared)
    distances = compared.loc["sample", [STRUCTURE_RMSD_VS_DFT, MAX_PAIR_DIST]].to_numpy(
        dtype=float
    )
    if expectation == "zero":
        assert distances == pytest.approx([0.0, 0.0], abs=1e-12)
    elif expectation == "positive":
        assert np.all(distances > 0)
    else:
        assert np.all(pd.isna(distances))


@pytest.mark.parametrize(
    ("failure_kind", "message"),
    [
        ("invalid-index", "index.name"),
        ("no-shared-ids", "No shared material IDs"),
        ("structure-symmetry-parity", "must contain identical IDs"),
    ],
)
def test_pred_vs_ref_struct_symmetry_rejects_invalid_inputs(
    cubic_structure: Structure,
    failure_kind: str,
    message: str,
) -> None:
    """Reject invalid indexes, disjoint IDs, and structure/symmetry ID mismatch."""
    predicted_structures = {"sample": cubic_structure}
    reference_structures = {"sample": cubic_structure}
    symmetry_structures = predicted_structures
    if failure_kind == "no-shared-ids":
        predicted_structures = {"predicted": cubic_structure}
        reference_structures = {"reference": cubic_structure}
        symmetry_structures = predicted_structures
    elif failure_kind == "structure-symmetry-parity":
        symmetry_structures = predicted_structures | {"ghost": cubic_structure}

    predicted_symmetry = get_sym_info_from_structs(symmetry_structures, pbar=False)
    reference_symmetry = get_sym_info_from_structs(reference_structures, pbar=False)
    if failure_kind == "invalid-index":
        reference_symmetry = reference_symmetry.rename_axis("wrong_id")

    with pytest.raises(ValueError, match=message):
        pred_vs_ref_struct_symmetry(
            predicted_symmetry,
            reference_symmetry,
            predicted_structures,
            reference_structures,
            pbar=False,
        )


def test_pred_vs_ref_struct_symmetry_retains_predicted_ids_without_references(
    cubic_structure: Structure,
) -> None:
    """Predicted-only structures retain the source RMSD penalty semantics."""
    predicted_structures = {
        "sample": cubic_structure,
        "predicted-only": cubic_structure,
    }
    reference_structures = {"sample": cubic_structure}
    compared = _compare_structures(predicted_structures, reference_structures)

    assert compared.index.tolist() == ["sample", "predicted-only"]
    assert pd.isna(compared.loc["predicted-only", STRUCTURE_RMSD_VS_DFT])
    assert calc_geo_opt_metrics(compared)[STRUCTURE_RMSD_VS_DFT] == pytest.approx(0.5)


def test_analyze_geo_opt_dataframes_returns_json_safe_results(
    geo_opt_dataframes: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Analyze both default symprecs and return JSON-serializable output."""
    predictions, references = geo_opt_dataframes
    extra_reference = references.copy()
    extra_reference[MATERIAL_ID] = "unused-reference"
    references = pd.concat([references, extra_reference], ignore_index=True)

    result = analyze_geo_opt_dataframes(
        predictions,
        references,
        angle_tolerance=np.float32(0.05),
        include_analysis=True,
    )

    assert set(result["symprecs"]) == {"symprec=1e-2", "symprec=1e-5"}
    for symprec_result in result["symprecs"].values():
        metrics = symprec_result["metrics"]
        assert (
            metrics[STRUCTURE_RMSD_VS_DFT],
            metrics[SYMMETRY_MATCH],
            metrics[N_STRUCTURES],
        ) == pytest.approx(
            (0.0, 1.0, 1),
            abs=1e-12,
        )
        assert symprec_result["analysis"][0][MATERIAL_ID] == "sample"
        assert isinstance(symprec_result["angle_tolerance"], float)
    assert result["versions"]["moyopy"]
    assert result["n_references"] == 1
    json.dumps(result, allow_nan=False)

    metrics_only = analyze_geo_opt_dataframes(predictions, references, symprecs=(1e-2,))
    assert "analysis" not in metrics_only["symprecs"]["symprec=1e-2"]


def test_analyze_geo_opt_dataframes_rejects_missing_references(
    geo_opt_dataframes: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Every predicted material must have a supplied reference structure."""
    predictions, references = geo_opt_dataframes
    references[MATERIAL_ID] = "other"

    with pytest.raises(ValueError, match="missing predicted material IDs"):
        analyze_geo_opt_dataframes(predictions, references)
