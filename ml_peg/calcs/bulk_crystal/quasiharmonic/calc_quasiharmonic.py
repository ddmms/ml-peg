"""
Run calculations for quasiharmonic benchmark using atomate2.

This module implements a quasiharmonic approximation (QHA) workflow using
atomate2's ForceFieldQhaMaker. It supports all force fields in ml-peg through
atomate2's ASE calculator integration.

The workflow computes temperature-dependent thermodynamic properties including:
- Equilibrium volume vs temperature
- Lattice constants vs temperature
- Thermal expansion coefficient
- Heat capacity and free energy
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
import uuid

from jobflow import JobStore, run_locally
from maggma.stores import MemoryStore
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor
import pytest

from ml_peg.models.get_models import get_model_names, load_model_configs
from ml_peg.models.models import current_models

if TYPE_CHECKING:
    from pymatgen.core import Structure

MODELS = get_model_names(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# Load reference data
REFERENCE_FILE = DATA_PATH / "quasiharmonic_reference.json"


def get_atomate2_config(model_name: str) -> dict[str, Any]:
    """
    Get atomate2-compatible configuration for a ml-peg model.

    Parameters
    ----------
    model_name
        Name of the model in ml-peg's models.yml.

    Returns
    -------
    dict[str, Any]
        Configuration dict with 'force_field_name' and 'calculator_kwargs'.

    Raises
    ------
    ValueError
        If the model is not supported by atomate2.
    """
    from atomate2.forcefields import MLFF

    configs, _ = load_model_configs((model_name,))
    cfg = configs.get(model_name, {})
    if not cfg:
        raise ValueError(f"Model '{model_name}' not found in models.yml")
    class_name = cfg.get("class_name", "")
    kwargs = cfg.get("kwargs", {})

    # Map ml-peg models to atomate2 MLFF and calculator_kwargs
    if class_name == "mace_mp":
        model_id = kwargs.get("model", "medium")

        # Map to atomate2 MLFF names
        if model_id == "medium":
            return {
                "force_field_name": MLFF.MACE_MP_0,
                "calculator_kwargs": {"model": "medium"},
            }
        if model_id == "medium-0b3":
            return {
                "force_field_name": MLFF.MACE_MP_0B3,
                "calculator_kwargs": {},
            }
        if model_id in ("medium-mpa-0", "mpa-0"):
            return {
                "force_field_name": MLFF.MACE_MPA_0,
                "calculator_kwargs": {},
            }
        if model_id in ("medium-omat-0", "omat-0"):
            # Use MACE generic with specific model
            return {
                "force_field_name": MLFF.MACE,
                "calculator_kwargs": {"model": model_id},
            }
        if "matpes" in model_id.lower() or "r2scan" in model_id.lower():
            return {
                "force_field_name": MLFF.MATPES_R2SCAN,
                "calculator_kwargs": kwargs,
            }
        return {
            "force_field_name": MLFF.MACE,
            "calculator_kwargs": {"model": model_id},
        }

    if class_name == "OrbCalc":
        # Orb models - use monty dict format for external calculator
        orb_name = kwargs.get("name", "orb_v3_conservative_inf_omat")
        return {
            "force_field_name": {
                "@module": "orb_models.forcefield.calculator",
                "@callable": "ORBCalculator",
            },
            "calculator_kwargs": {
                "model_name": orb_name,
                "device": cfg.get("device", "cpu"),
            },
        }

    if class_name == "PETMADCalculator":
        # PET-MAD - use monty dict format
        return {
            "force_field_name": {
                "@module": "pet_mad.calculator",
                "@callable": "PETMADCalculator",
            },
            "calculator_kwargs": kwargs,
        }

    if class_name == "FAIRChemCalculator":
        # FAIRChem/UMA calculators
        return {
            "force_field_name": {
                "@module": "fairchem.core",
                "@callable": "FAIRChemCalculator",
            },
            "calculator_kwargs": {
                "model_name": kwargs.get("model_name"),
                "task_name": kwargs.get("task_name", "omat"),
            },
        }

    if class_name == "CHGNetCalculator":
        return {
            "force_field_name": MLFF.CHGNet,
            "calculator_kwargs": kwargs,
        }

    if class_name == "M3GNetCalculator":
        return {
            "force_field_name": MLFF.M3GNet,
            "calculator_kwargs": kwargs,
        }

    # Try generic ASE calculator via monty dict
    module = cfg.get("module", "")
    if not module:
        raise ValueError(
            f"Model '{model_name}' with class '{class_name}' "
            "is not supported by atomate2"
        )
    return {
        "force_field_name": {
            "@module": module,
            "@callable": class_name,
        },
        "calculator_kwargs": kwargs,
    }


def load_reference_data() -> dict[str, Any]:
    """
    Load reference data for the quasiharmonic benchmark.

    Returns
    -------
    dict[str, Any]
        Reference data containing materials, conditions, and settings.
    """
    with REFERENCE_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def load_structure(material: dict[str, Any]) -> Structure:
    """
    Load a structure from the data directory.

    Parameters
    ----------
    material
        Material metadata with 'cif_file' or bulk parameters.

    Returns
    -------
    Structure
        Pymatgen Structure object.
    """
    from pymatgen.core import Structure

    if "cif_file" in material:
        cif_path = DATA_PATH / material["cif_file"]
        if cif_path.exists():
            return Structure.from_file(str(cif_path))

    # Build from bulk parameters using ASE and convert
    from ase.build import bulk

    atoms = bulk(
        material["symbols"],
        material["lattice_type"],
        a=material["a0"],
    )
    return AseAtomsAdaptor.get_structure(atoms)


def run_qha_workflow(
    structure: Structure,
    model_name: str,
    settings: dict[str, Any],
    pressure_gpa: float = 0.0,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run atomate2 QHA workflow for a given structure and model.

    Parameters
    ----------
    structure
        Pymatgen Structure to run QHA on.
    model_name
        Name of the ml-peg model to use.
    settings
        QHA workflow settings.
    pressure_gpa
        External pressure in GPa.
    work_dir
        Working directory for the workflow. If None, uses current directory.

    Returns
    -------
    dict[str, Any]
        QHA workflow results.
    """
    from atomate2.forcefields.flows.phonons import PhononMaker
    from atomate2.forcefields.flows.qha import ForceFieldQhaMaker
    from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker

    # Get atomate2 configuration for this model
    atomate2_cfg = get_atomate2_config(model_name)
    force_field_name = atomate2_cfg["force_field_name"]
    calculator_kwargs = atomate2_cfg["calculator_kwargs"]

    # Extract settings
    volume_scale = settings.get("volume_scale_range", [-0.05, 0.05])
    n_volumes = settings.get("n_volumes", 10)
    eos_type = settings.get("eos_type", "vinet")
    fmax = settings.get("relax_fmax", 0.001)
    temp_range = settings.get("temperature_range", [0, 500])
    temp_step = settings.get("temperature_step", 50)
    min_length = settings.get("min_length", 10.0)

    # Create makers with the force field configuration
    initial_relax_maker = ForceFieldRelaxMaker(
        force_field_name=force_field_name,
        relax_cell=True,
        relax_kwargs={"fmax": fmax},
        calculator_kwargs=calculator_kwargs,
    )

    eos_relax_maker = ForceFieldRelaxMaker(
        force_field_name=force_field_name,
        relax_cell=False,  # Fixed cell for EOS
        relax_kwargs={"fmax": fmax},
        calculator_kwargs=calculator_kwargs,
    )

    phonon_static_maker = ForceFieldStaticMaker(
        force_field_name=force_field_name,
        calculator_kwargs=calculator_kwargs,
    )
    phonon_static_maker.name = f"{model_name} phonon static"

    phonon_maker = PhononMaker(
        generate_frequencies_eigenvectors_kwargs={
            "tmin": temp_range[0],
            "tmax": temp_range[1],
            "tstep": temp_step,
        },
        bulk_relax_maker=None,  # Already relaxed
        born_maker=None,  # Skip Born charges for ML potentials
        static_energy_maker=phonon_static_maker,
        phonon_displacement_maker=phonon_static_maker,
        min_length=min_length,
    )

    # Create QHA flow
    qha_maker = ForceFieldQhaMaker(
        initial_relax_maker=initial_relax_maker,
        eos_relax_maker=eos_relax_maker,
        phonon_maker=phonon_maker,
        linear_strain=tuple(volume_scale),
        number_of_frames=n_volumes,
        pressure=pressure_gpa if pressure_gpa != 0.0 else None,
        t_max=temp_range[1],
        ignore_imaginary_modes=False,
        skip_analysis=False,
        eos_type=eos_type,
        min_length=min_length,
    )

    # Create flow
    flow = qha_maker.make(structure=structure)

    # Set up job store
    job_store = JobStore(
        MemoryStore(),
        additional_stores={"data": MemoryStore()},
    )

    # Run locally with specified working directory
    run_kwargs = {"ensure_success": True}
    if work_dir is not None:
        run_kwargs["root_dir"] = str(work_dir)

    responses = run_locally(flow, store=job_store, **run_kwargs)

    # Extract the PhononQHADoc from the last job's output
    # The last job is the analyze_free_energy job that returns the QHA document
    last_uuid = list(responses.keys())[-1]
    last_response = responses[last_uuid]
    # The response is a list, index 1 contains the output
    qha_doc = last_response[1].output

    return {"qha_doc": qha_doc}


def _interpolate_at_temperature(
    temps: list[float],
    values: list[float] | None,
    target_temp: float,
) -> float | None:
    """
    Get value at target temperature using nearest neighbor interpolation.

    Parameters
    ----------
    temps
        Temperature values.
    values
        Property values (can be None).
    target_temp
        Target temperature.

    Returns
    -------
    float | None
        Interpolated value or None if no data.
    """
    if not temps or not values:
        return None
    idx = min(range(len(temps)), key=lambda i: abs(temps[i] - target_temp))
    return values[idx]


def extract_qha_properties(
    qha_doc: Any,
    temperature: float,
    n_atoms_conventional: int = 8,
) -> dict[str, float | None]:
    """
    Extract QHA properties at a specific temperature from PhononQHADoc.

    Parameters
    ----------
    qha_doc
        PhononQHADoc from atomate2 workflow.
    temperature
        Target temperature in K.
    n_atoms_conventional
        Number of atoms in the conventional cell (for volume conversion).
        Default is 8 for diamond structure.

    Returns
    -------
    dict[str, float | None]
        Extracted properties at target temperature.
    """
    properties: dict[str, float | None] = {
        "volume_per_atom": None,
        "lattice_a": None,
        "thermal_expansion": None,
        "bulk_modulus": None,
        "heat_capacity": None,
    }

    if qha_doc is None:
        return properties

    temps = getattr(qha_doc, "temperatures", None) or []
    if not temps:
        return properties

    # Volume per atom (atomate2 outputs volume per atom directly)
    vol_t = getattr(qha_doc, "volume_temperature", None)
    vol_per_atom = _interpolate_at_temperature(temps, vol_t, temperature)
    if vol_per_atom is not None:
        properties["volume_per_atom"] = vol_per_atom
        # Lattice constant from conventional cell volume
        properties["lattice_a"] = (vol_per_atom * n_atoms_conventional) ** (1 / 3)

    # Thermal expansion (K^-1, convert to 10^-6 K^-1)
    te = getattr(qha_doc, "thermal_expansion", None)
    te_value = _interpolate_at_temperature(temps, te, temperature)
    if te_value is not None:
        properties["thermal_expansion"] = te_value * 1e6

    # Bulk modulus (GPa)
    bm = getattr(qha_doc, "bulk_modulus_temperature", None)
    bm_value = _interpolate_at_temperature(temps, bm, temperature)
    if bm_value is not None:
        properties["bulk_modulus"] = bm_value

    # Heat capacity (J/mol·K)
    cp = getattr(qha_doc, "heat_capacity_p_numerical", None)
    cp_value = _interpolate_at_temperature(temps, cp, temperature)
    if cp_value is not None:
        properties["heat_capacity"] = cp_value

    return properties


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_quasiharmonic(model_name: str) -> None:
    """
    Run quasiharmonic benchmark for a given MLIP using atomate2.

    Parameters
    ----------
    model_name
        Name of the model from ml-peg models registry.
    """
    # Check if model is supported
    try:
        get_atomate2_config(model_name)
    except ValueError as err:
        pytest.skip(str(err))

    # Load reference data
    ref_data = load_reference_data()
    materials = {mat["name"]: mat for mat in ref_data["materials"]}
    conditions = ref_data["conditions"]
    settings = ref_data["qha_settings"]

    # Create output directory
    model_out_dir = OUT_PATH / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for intermediate workflow files
    temp_base_dir = model_out_dir / "tmp"
    temp_base_dir.mkdir(parents=True, exist_ok=True)

    results_list = []

    for condition in conditions:
        material_name = condition["material"]
        material = materials[material_name]
        temperature = condition["temperature_K"]
        pressure = condition["pressure_GPa"]

        # Load structure
        structure = load_structure(material)

        # Create unique temporary directory for this calculation
        calc_id = f"{material_name}_T{temperature}_P{pressure}_{uuid.uuid4().hex[:8]}"
        calc_work_dir = temp_base_dir / calc_id
        calc_work_dir.mkdir(parents=True, exist_ok=True)

        # Run workflow
        try:
            workflow_result = run_qha_workflow(
                structure,
                model_name,
                settings,
                pressure_gpa=pressure,
                work_dir=calc_work_dir,
            )

            # Extract predictions at target temperature
            qha_doc = workflow_result.get("qha_doc")
            properties = extract_qha_properties(qha_doc, temperature)

            results_list.append(
                {
                    "material": material_name,
                    "temperature_K": temperature,
                    "pressure_GPa": pressure,
                    "ref_lattice_a": condition["ref_lattice_a"],
                    "pred_lattice_a": properties["lattice_a"],
                    "ref_volume_per_atom": condition.get("ref_volume_per_atom"),
                    "pred_volume_per_atom": properties["volume_per_atom"],
                    "ref_thermal_expansion_1e6_K": condition.get(
                        "ref_thermal_expansion_1e6_K"
                    ),
                    "pred_thermal_expansion_1e6_K": properties["thermal_expansion"],
                    "ref_bulk_modulus_GPa": condition.get("ref_bulk_modulus_GPa"),
                    "pred_bulk_modulus_GPa": properties["bulk_modulus"],
                    "ref_heat_capacity_J_mol_K": condition.get(
                        "ref_heat_capacity_J_mol_K"
                    ),
                    "pred_heat_capacity_J_mol_K": properties["heat_capacity"],
                    "status": "success",
                }
            )

            # Save full workflow results
            results_file = (
                model_out_dir / f"{material_name}_T{temperature}_P{pressure}_qha.json"
            )
            with results_file.open("w") as f:
                # Convert to JSON-serializable format
                json.dump(
                    {"properties": properties, "condition": condition},
                    f,
                    indent=2,
                    default=str,
                )

        except Exception as e:
            results_list.append(
                {
                    "material": material_name,
                    "temperature_K": temperature,
                    "pressure_GPa": pressure,
                    "ref_lattice_a": condition["ref_lattice_a"],
                    "pred_lattice_a": None,
                    "status": "failed",
                    "error": str(e),
                }
            )

    # Save results summary
    df = pd.DataFrame(results_list)
    df.to_csv(model_out_dir / "results.csv", index=False)

    # Also save as JSON for more detailed data
    with (model_out_dir / "results.json").open("w") as f:
        json.dump(results_list, f, indent=2)
