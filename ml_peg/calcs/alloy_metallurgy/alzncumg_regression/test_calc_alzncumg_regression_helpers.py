"""Test helpers for the Al-Cu-Mg-Zn metallurgy regression calculation."""

from __future__ import annotations

import json

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write
from ase.units import GPa
import numpy as np
import pytest

from ml_peg.calcs.alloy_metallurgy.alzncumg_regression import (
    calc_alzncumg_regression as calc,
)


class EnergyByIdCalculator(Calculator):
    """Return configured energies and fail for selected structure IDs."""

    implemented_properties = ["energy"]

    def __init__(self, energies: dict[str, float], failing_ids: set[str] | None = None):
        super().__init__()
        self.energies = energies
        self.failing_ids = failing_ids or set()

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Calculate a fixed energy for the current structure."""
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("Atoms are required")

        oqmd_id = atoms.info["oqmd_id"]
        if oqmd_id in self.failing_ids:
            raise RuntimeError(f"configured failure for {oqmd_id}")

        self.results["energy"] = self.energies[oqmd_id]


class FixedEnergyModel:
    """Small model stub exposing the ML-PEG calculator interface."""

    def __init__(self, calculator: Calculator):
        self.calculator = calculator

    def get_calculator(self, precision: str) -> Calculator:
        """Return the configured calculator."""
        assert precision == "high"
        return self.calculator


class LinearElasticCalculator(Calculator):
    """Return stresses from a configured elastic tensor."""

    implemented_properties = ["energy", "stress"]

    def __init__(self, reference_cell: np.ndarray, elastic_tensor_gpa: np.ndarray):
        super().__init__()
        self.reference_cell = reference_cell
        self.elastic_tensor = elastic_tensor_gpa * GPa

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Calculate linear-elastic stress for the current cell."""
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("Atoms are required")

        deformation = atoms.cell.array @ np.linalg.inv(self.reference_cell)
        strain = (deformation + deformation.T) / 2.0 - np.eye(3)
        strain_voigt = np.array(
            [
                strain[0, 0],
                strain[1, 1],
                strain[2, 2],
                2.0 * strain[1, 2],
                2.0 * strain[0, 2],
                2.0 * strain[0, 1],
            ]
        )

        self.results["energy"] = 0.0
        self.results["stress"] = self.elastic_tensor @ strain_voigt


class PairDistanceCalculator(Calculator):
    """Return a pair-distance binding contribution for two Cu solutes."""

    implemented_properties = ["energy"]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Calculate an energy whose binding term is the Cu-Cu distance."""
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("Atoms are required")

        symbols = atoms.get_chemical_symbols()
        cu_indices = [index for index, symbol in enumerate(symbols) if symbol == "Cu"]
        energy = 0.2 * len(cu_indices)
        if len(cu_indices) == 2:
            distance = atoms.get_distance(cu_indices[0], cu_indices[1], mic=False)
            energy += distance / 1000.0
        self.results["energy"] = energy


@pytest.mark.parametrize(
    ("oqmd_id", "expected_stem"),
    [
        ("8100", "OQMD_8100"),
        ("NOTINOQMD_00001", "NOTINOQMD_00001"),
    ],
)
def test_structure_file_stem_handles_numeric_and_legacy_ids(
    oqmd_id: str, expected_stem: str
) -> None:
    """Structure filenames preserve legacy non-OQMD identifiers."""
    assert calc.structure_file_stem(oqmd_id) == expected_stem


def test_solute_pair_reference_key_sorts_legacy_pairs() -> None:
    """Solute reference keys match the ordering used in the DFT JSON."""
    assert calc.solute_pair_reference_key("8100", "Zn", "Cu") == (
        "8100-SolSol_Cu_Zn"
    )
    assert calc.solute_pair_reference_key("635950", "Vac", "Al") == (
        "635950-SolSol_Al_Vac"
    )


@pytest.mark.parametrize(
    ("oqmd_id", "filename", "expected_name"),
    [
        ("8100", "OQMD_8100", "OQMD_8100"),
        ("NOTINOQMD_00001", "NOTINOQMD_00001", "NOTINOQMD_00001"),
    ],
)
def test_load_oqmd_structure_reads_numeric_and_legacy_metadata(
    tmp_path, monkeypatch, oqmd_id: str, filename: str, expected_name: str
) -> None:
    """OQMD and legacy structure IDs load matching VASP and metadata files."""
    atoms = Atoms("Al", cell=[4.0, 4.0, 4.0], pbc=True)
    structure_path = tmp_path / filename
    write(structure_path, atoms, format="vasp")
    structure_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "OQMD_Composition": "Al1",
                "OQMD_FormationEnergy": "-0.125",
                "OQMD_Volumepa": "16.0",
            }
        )
    )
    monkeypatch.setattr(calc, "OQMD_PATH", tmp_path)

    loaded_atoms = calc.load_oqmd_structure(oqmd_id)

    assert loaded_atoms.info["oqmd_id"] == oqmd_id
    assert loaded_atoms.info["name"] == expected_name
    assert loaded_atoms.info["oqmd_composition"] == "Al1"
    assert loaded_atoms.info["oqmd_formation_energy"] == pytest.approx(-0.125)
    assert loaded_atoms.info["oqmd_volume_per_atom"] == pytest.approx(16.0)


def test_formation_energy_per_atom_uses_element_counts() -> None:
    """Formation energy is normalized by atom count after elemental subtraction."""
    atoms = Atoms("Al2Cu", cell=[4.0, 4.0, 4.0], pbc=True)

    formation_energy = calc.formation_energy_per_atom(
        atoms,
        total_energy=-11.5,
        reference_energies={"Al": -3.0, "Cu": -4.0},
    )

    assert formation_energy == pytest.approx(-0.5)


def test_finite_strain_elastic_tensor_recovers_linear_response() -> None:
    """Elastic finite differences recover a synthetic linear stress response."""
    atoms = Atoms("Al", cell=[4.0, 4.0, 4.0], pbc=True)
    atoms.info["oqmd_id"] = "8100"
    elastic_tensor = np.diag([120.0, 130.0, 140.0, 40.0, 50.0, 60.0])
    elastic_tensor[1, 0] = elastic_tensor[0, 1] = 55.0
    elastic_tensor[2, 0] = elastic_tensor[0, 2] = 65.0
    elastic_tensor[2, 1] = elastic_tensor[1, 2] = 75.0
    calculator = LinearElasticCalculator(atoms.cell.array, elastic_tensor)

    calculated_tensor = calc.finite_strain_elastic_tensor(atoms, calculator)
    properties = calc.elastic_properties(atoms, calculated_tensor)

    assert calculated_tensor == pytest.approx(elastic_tensor)
    assert properties["k_voigt"] == pytest.approx(86.67)
    assert properties["g_voigt"] == pytest.approx(43.0)
    assert properties["C_21"] == pytest.approx(55.0)


def test_solute_solute_binding_uses_neighbor_shells_and_energy_cycle() -> None:
    """Solute-solute binding uses the legacy pair-minus-single energy cycle."""
    pure_structure = bulk("Al", "fcc", a=4.0, cubic=True).repeat((2, 2, 2))

    distances, binding_energies = calc.solute_solute_binding(
        pure_structure,
        PairDistanceCalculator(),
        "Cu",
        "Cu",
        max_shells=2,
        relax_steps=0,
    )

    assert len(distances) == 2
    assert binding_energies == pytest.approx(distances)


def test_calculation_writes_successful_records_after_partial_failure(
    tmp_path, monkeypatch
) -> None:
    """A failed structure calculation warns but does not discard successful outputs."""
    structures = {
        "Al": Atoms("Al", cell=[4.0, 4.0, 4.0], pbc=True),
        "Cu": Atoms("Cu", cell=[3.5, 3.5, 3.5], pbc=True),
        "AlCu": Atoms("AlCu", cell=[4.0, 4.0, 4.0], pbc=True),
        "broken": Atoms("Al", cell=[4.0, 4.0, 4.0], pbc=True),
    }
    for oqmd_id, atoms in structures.items():
        atoms.info["oqmd_id"] = oqmd_id

    calculator = EnergyByIdCalculator(
        {"Al": -3.0, "Cu": -4.0, "AlCu": -7.5}, failing_ids={"broken"}
    )
    model = FixedEnergyModel(calculator)
    monkeypatch.setattr(calc, "STRUCTURE_IDS", tuple(structures))
    monkeypatch.setattr(calc, "OUT_PATH", tmp_path)
    monkeypatch.setattr(calc, "load_oqmd_structure", structures.__getitem__)

    with pytest.warns(UserWarning, match="Error calculating OQMD_broken"):
        calc.test_alzncumg_regression(("stub-model", model))

    output_path = tmp_path / "stub-model" / "bulk_properties.json"
    output_data = json.loads(output_path.read_text())
    records = {record["oqmd_id"]: record for record in output_data["structures"]}

    assert set(records) == {"Al", "Cu", "AlCu"}
    assert output_data["elemental_reference_energies"] == {"Al": -3.0, "Cu": -4.0}
    assert records["AlCu"]["formation_energy"] == pytest.approx(-0.25)
    assert (tmp_path / "stub-model" / "OQMD_Al.xyz").is_file()
    assert (tmp_path / "stub-model" / "OQMD_Cu.xyz").is_file()
    assert (tmp_path / "stub-model" / "OQMD_AlCu.xyz").is_file()
    assert not (tmp_path / "stub-model" / "OQMD_broken.xyz").exists()
