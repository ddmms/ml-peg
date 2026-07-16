"""Test helpers for the Al-Cu-Mg-Zn metallurgy regression analysis."""

from __future__ import annotations

import json

import pytest

from ml_peg.analysis.alloy_metallurgy.alzncumg_regression import (
    analyse_alzncumg_regression as analyse,
)


def write_bulk_properties(output_dir, records: list[dict[str, object]]) -> None:
    """Write a minimal calc output file for one model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bulk_properties.json").write_text(
        json.dumps({"elemental_reference_energies": {}, "structures": records})
    )


def write_elastic_properties(output_dir, records: list[dict[str, object]]) -> None:
    """Write a minimal elastic output file for one model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "elastic_properties.json").write_text(
        json.dumps({"structures": records})
    )


def write_solute_solute_properties(
    output_dir, records: list[dict[str, object]]
) -> None:
    """Write a minimal solute-solute output file for one model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "solute_solute_bindings.json").write_text(
        json.dumps({"interactions": records})
    )


def write_fault_surface_properties(
    output_dir,
    *,
    surfaces: list[dict[str, object]] | None = None,
    stacking_faults: list[dict[str, object]] | None = None,
    gsf: list[dict[str, object]] | None = None,
    solute_stacking_faults: list[dict[str, object]] | None = None,
) -> None:
    """Write a minimal fault/surface output file for one model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fault_surface_properties.json").write_text(
        json.dumps(
            {
                "surfaces": surfaces or [],
                "stacking_faults": stacking_faults or [],
                "gsf": gsf or [],
                "solute_stacking_faults": solute_stacking_faults or [],
            }
        )
    )


def test_reference_value_returns_numeric_values_only() -> None:
    """Reference extraction handles missing and nonnumeric evalpot entries."""
    references = {
        "8100-formation_energy": ["-0.25", "eV/atom"],
        "635950-formation_energy": [None, "eV/atom"],
        "9226-formation_energy": ["not-a-number", "eV/atom"],
    }

    value = analyse.reference_value(references, "8100", "formation_energy")

    assert value == pytest.approx(-0.25)
    assert analyse.reference_value(references, "635950", "formation_energy") is None
    assert analyse.reference_value(references, "9226", "formation_energy") is None
    assert analyse.reference_value(references, "122929", "formation_energy") is None


def test_load_model_records_skips_missing_model_outputs(tmp_path, monkeypatch) -> None:
    """Analysis only loads models with existing bulk-property outputs."""
    monkeypatch.setattr(analyse, "CALC_PATH", tmp_path)
    monkeypatch.setattr(analyse, "MODELS", ["complete-model", "missing-model"])
    write_bulk_properties(
        tmp_path / "complete-model",
        [
            {
                "oqmd_id": "8100",
                "formation_energy": 0.0,
                "volume_peratom": 16.0,
            }
        ],
    )

    records_by_model = analyse.load_model_records()

    assert set(records_by_model) == {"complete-model"}
    volume = records_by_model["complete-model"]["8100"]["volume_peratom"]

    assert volume == pytest.approx(16.0)


def test_common_structure_ids_intersects_models_and_references(monkeypatch) -> None:
    """Common IDs exclude missing model records and missing reference values."""
    monkeypatch.setattr(
        analyse,
        "load_references",
        lambda: {
            "8100-formation_energy": ["0.0", "eV/atom"],
            "695020-formation_energy": ["-0.2", "eV/atom"],
        },
    )
    records_by_model = {
        "model-a": {
            "8100": {"formation_energy": 0.01},
            "695020": {"formation_energy": -0.1},
            "10434": {"formation_energy": -0.3},
        },
        "model-b": {
            "8100": {"formation_energy": 0.02},
            "10434": {"formation_energy": -0.4},
        },
    }

    common_ids = analyse.common_structure_ids(records_by_model, "formation_energy")

    assert common_ids == ["8100"]


def test_common_structure_ids_excludes_missing_model_properties(monkeypatch) -> None:
    """Common IDs require each model record to contain the requested property."""
    monkeypatch.setattr(
        analyse,
        "load_references",
        lambda: {
            "8100-lattice_a": ["2.8", "Angstrom"],
            "695020-lattice_a": ["4.1", "Angstrom"],
        },
    )
    records_by_model = {
        "model-a": {
            "8100": {"lattice_a": 2.9},
            "695020": {"formation_energy": -0.2},
        }
    }

    common_ids = analyse.common_structure_ids(records_by_model, "lattice_a")

    assert common_ids == ["8100"]


def test_multi_property_values_flattens_lattice_components(monkeypatch) -> None:
    """Multi-property values are flattened in property then sorted-ID order."""
    monkeypatch.setattr(
        analyse,
        "load_references",
        lambda: {
            "8100-lattice_a": ["2.8", "Angstrom"],
            "695020-lattice_a": ["4.1", "Angstrom"],
            "8100-lattice_b": ["2.8", "Angstrom"],
            "695020-lattice_b": ["4.1", "Angstrom"],
        },
    )
    monkeypatch.setattr(
        analyse,
        "load_model_records",
        lambda: {
            "model-a": {
                "8100": {"lattice_a": 2.9, "lattice_b": 3.0},
                "695020": {"lattice_a": 4.0, "lattice_b": 4.2},
            }
        },
    )

    values = analyse.multi_property_values(("lattice_a", "lattice_b"))

    assert values == {
        "ref": [4.1, 2.8, 4.1, 2.8],
        "model-a": [4.0, 2.9, 4.2, 3.0],
    }


def test_formation_and_volume_series_ignore_missing_model_outputs(
    tmp_path, monkeypatch
) -> None:
    """Parity data is built from available model outputs and reference values."""
    references_path = tmp_path / "DFT.json"
    references_path.write_text(
        json.dumps(
            {
                "8100-formation_energy": ["0.0", "eV/atom"],
                "695020-formation_energy": ["-0.2", "eV/atom"],
                "8100-volume_peratom": ["16.0", "Angstrom^3/atom"],
                "695020-volume_peratom": ["14.5", "Angstrom^3/atom"],
            }
        )
    )
    monkeypatch.setattr(analyse, "REFERENCE_PATH", references_path)
    monkeypatch.setattr(analyse, "CALC_PATH", tmp_path / "outputs")
    monkeypatch.setattr(analyse, "MODELS", ["complete-model", "missing-model"])
    write_bulk_properties(
        tmp_path / "outputs" / "complete-model",
        [
            {
                "oqmd_id": "8100",
                "formation_energy": 0.01,
                "volume_peratom": 16.2,
            },
            {
                "oqmd_id": "695020",
                "formation_energy": -0.25,
                "volume_peratom": 14.0,
            },
        ],
    )

    assert analyse.formation_energies.__wrapped__.__wrapped__() == {
        "ref": [-0.2, 0.0],
        "complete-model": [-0.25, 0.01],
    }
    assert analyse.volumes_per_atom.__wrapped__.__wrapped__() == {
        "ref": [14.5, 16.0],
        "complete-model": [14.0, 16.2],
    }


def test_elastic_metrics_are_optional_and_write_available_plots(
    tmp_path, monkeypatch
) -> None:
    """Elastic metrics are added only when elastic calc outputs are present."""
    monkeypatch.setattr(analyse, "CALC_PATH", tmp_path / "outputs")
    monkeypatch.setattr(analyse, "MODELS", ["elastic-model", "missing-model"])
    monkeypatch.setattr(
        analyse,
        "load_references",
        lambda: {
            "8100-k_voigt": [80.0, "GPa"],
            "8100-g_voigt": [30.0, "GPa"],
            "8100-C_11": [100.0, "GPa"],
            "8100-C_21": [60.0, "GPa"],
        },
    )
    write_elastic_properties(
        tmp_path / "outputs" / "elastic-model",
        [
            {
                "oqmd_id": "8100",
                "k_voigt": 85.0,
                "g_voigt": 32.0,
                "C_11": 110.0,
                "C_21": 55.0,
            }
        ],
    )
    written_plots = []

    def collect_plot(values, *, filename, title, x_label, y_label, hoverdata):
        written_plots.append(filename.name)

    monkeypatch.setattr(analyse, "write_parity_plot", collect_plot)

    metrics = analyse.elastic_metrics()

    assert metrics == {
        "Bulk Modulus MAE": {"elastic-model": pytest.approx(5.0)},
        "Shear Modulus MAE": {"elastic-model": pytest.approx(2.0)},
        "Elastic Constant MAE": {"elastic-model": pytest.approx(7.5)},
    }
    assert set(written_plots) == {
        "figure_bulk_modulus.json",
        "figure_shear_modulus.json",
        "figure_elastic_constants.json",
    }


def test_solute_solute_metrics_are_optional_and_write_available_plot(
    tmp_path, monkeypatch
) -> None:
    """Solute-solute metrics are added only when calc outputs are present."""
    monkeypatch.setattr(analyse, "CALC_PATH", tmp_path / "outputs")
    monkeypatch.setattr(analyse, "MODELS", ["solute-model", "missing-model"])
    monkeypatch.setattr(
        analyse,
        "load_references",
        lambda: {
            "8100-SolSol_Cu_Cu_BindingEnergy": [[10.0, 20.0], "meV"],
        },
    )
    write_solute_solute_properties(
        tmp_path / "outputs" / "solute-model",
        [
            {
                "reference_key": "8100-SolSol_Cu_Cu",
                "binding_energies": [11.0, 18.0],
            }
        ],
    )
    written_plots = []

    def collect_plot(values, *, filename, title, x_label, y_label, hoverdata):
        written_plots.append((filename.name, hoverdata["Interaction"]))

    monkeypatch.setattr(analyse, "write_parity_plot", collect_plot)

    metrics = analyse.solute_solute_metrics()

    assert metrics == {
        "Solute-Solute Binding MAE": {"solute-model": pytest.approx(1.5)},
    }
    assert written_plots == [
        (
            "figure_solute_solute_bindings.json",
            ["8100-SolSol_Cu_Cu shell 1", "8100-SolSol_Cu_Cu shell 2"],
        )
    ]


def test_solute_solute_metrics_accept_sorted_dft_pair_keys(
    tmp_path, monkeypatch
) -> None:
    """Solute-solute analysis handles evalpot output order and sorted DFT keys."""
    monkeypatch.setattr(analyse, "CALC_PATH", tmp_path / "outputs")
    monkeypatch.setattr(analyse, "MODELS", ["solute-model"])
    monkeypatch.setattr(
        analyse,
        "load_references",
        lambda: {
            "8100-SolSol_Cu_Zn_BindingEnergy": [[10.0, 20.0], "meV"],
        },
    )
    write_solute_solute_properties(
        tmp_path / "outputs" / "solute-model",
        [
            {
                "reference_key": "8100-SolSol_Zn_Cu",
                "binding_energies": [11.0, 18.0],
            }
        ],
    )
    written_plots = []

    def collect_plot(values, *, filename, title, x_label, y_label, hoverdata):
        written_plots.append((filename.name, hoverdata["Interaction"]))

    monkeypatch.setattr(analyse, "write_parity_plot", collect_plot)

    metrics = analyse.solute_solute_metrics()

    assert metrics == {
        "Solute-Solute Binding MAE": {"solute-model": pytest.approx(1.5)},
    }
    assert written_plots == [
        (
            "figure_solute_solute_bindings.json",
            ["8100-SolSol_Zn_Cu shell 1", "8100-SolSol_Zn_Cu shell 2"],
        )
    ]


def test_fault_surface_metrics_are_optional_and_write_available_plots(
    tmp_path, monkeypatch
) -> None:
    """Surface, stacking-fault, and GSF metrics are added when outputs exist."""
    monkeypatch.setattr(analyse, "CALC_PATH", tmp_path / "outputs")
    monkeypatch.setattr(analyse, "MODELS", ["fault-model", "missing-model"])
    monkeypatch.setattr(
        analyse,
        "load_references",
        lambda: {
            "8100-SurfaceEnergy_111": [1000.0, "mJ/m^2"],
            "8100-StableSF": [150.0, "mJ/m^2"],
            "NOTINOQMD_00001-GSF_0m11_normE": [[0.0, 0.04], "eV/A^2"],
            "8100-SolSF_Cu": [[0.1, 0.2], "eV"],
        },
    )
    write_fault_surface_properties(
        tmp_path / "outputs" / "fault-model",
        surfaces=[
            {"reference_key": "8100-SurfaceEnergy_111", "surface_energy": 1100.0}
        ],
        stacking_faults=[
            {"reference_key": "8100-StableSF", "stacking_fault_energy": 180.0}
        ],
        gsf=[
            {
                "reference_key": "NOTINOQMD_00001-GSF_0m11",
                "norm_energies": [0.0, 0.05],
            }
        ],
        solute_stacking_faults=[
            {
                "reference_key": "8100-SolSF_Cu",
                "interaction_energies": [0.15, 0.18],
            }
        ],
    )
    written_plots = []

    def collect_plot(values, *, filename, title, x_label, y_label, hoverdata):
        written_plots.append((filename.name, next(iter(hoverdata.values()))))

    monkeypatch.setattr(analyse, "write_parity_plot", collect_plot)

    metrics = analyse.fault_surface_metrics()

    assert metrics == {
        "Surface Energy MAE": {"fault-model": pytest.approx(100.0)},
        "Stacking Fault Energy MAE": {"fault-model": pytest.approx(30.0)},
        "GSF Energy MAE": {"fault-model": pytest.approx(0.005)},
        "Solute-Stacking Fault MAE": {"fault-model": pytest.approx(0.035)},
    }
    assert written_plots == [
        ("figure_surface_energies.json", ["8100-SurfaceEnergy_111"]),
        ("figure_stacking_fault_energies.json", ["8100-StableSF"]),
        (
            "figure_gsf_energies.json",
            [
                "NOTINOQMD_00001-GSF_0m11 point 1",
                "NOTINOQMD_00001-GSF_0m11 point 2",
            ],
        ),
        (
            "figure_solute_stacking_faults.json",
            ["8100-SolSF_Cu layer 0", "8100-SolSF_Cu layer 1"],
        ),
    ]
