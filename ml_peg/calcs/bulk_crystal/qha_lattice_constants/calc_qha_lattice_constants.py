"""Run calculations for QHA lattice constants benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.build import bulk
from ase.units import GPa
from janus_core.calculations.geom_opt import GeomOpt
import numpy as np
import pandas as pd
import pytest

from ml_peg.models.get_models import load_model_configs
from ml_peg.models.models import current_models

MODEL_CONFIGS, _MODEL_LEVELS = load_model_configs(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"
DATA_FILE = DATA_PATH / "qha_lattice_constants_fake.json"

# QHA workflow controls
EOS_VOLUME_SCALE = 0.04
EOS_N_VOLUMES = 5
EOS_TYPE = "vinet"
EOS_MINIMIZE_FMAX = 0.0002
EOS_MINIMIZE_STEPS = 500
RELAX_FMAX = 0.0002
RELAX_STEPS = 500

PHONON_SUPERCELL = (2, 2, 2)
PHONON_DISPLACEMENT = 0.01
PHONON_MESH = (8, 8, 8)
PHONON_MINIMIZE_FMAX = 0.0002
PHONON_MINIMIZE_STEPS = 500

KJ_MOL_TO_EV = 1.0 / 96.4853321233


def _load_dataset() -> dict[str, Any]:
    """
    Load fake QHA lattice-constant dataset.

    Returns
    -------
    dict[str, Any]
        Dataset containing materials and condition grid.
    """
    with DATA_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


def _build_structure(material: dict[str, Any]) -> Atoms:
    """
    Build a bulk structure from material metadata.

    Parameters
    ----------
    material
        Material metadata with ``symbols``, ``lattice_type``, and ``a0``.

    Returns
    -------
    Atoms
        ASE Atoms instance for the material.
    """
    atoms = bulk(material["symbols"], material["lattice_type"], a=material["a0"])
    atoms.info["material"] = material["name"]
    atoms.info["lattice_type"] = material["lattice_type"]
    atoms.info["a0"] = material["a0"]
    return atoms


def _build_janus_core_config(model_name: str) -> dict[str, Any]:
    """
    Map ml-peg model config onto janus-core calculation settings.

    Parameters
    ----------
    model_name
        Model identifier from models.yml.

    Returns
    -------
    dict[str, Any]
        Mapping containing ``arch``, ``model``, ``device``, and ``calc_kwargs``.
    """
    cfg = MODEL_CONFIGS.get(model_name) or {}
    class_name = cfg.get("class_name", "")

    arch_map = {
        "mace_mp": "mace_mp",
        "mace_off": "mace_off",
        "OrbCalc": "orb",
        "FAIRChemCalculator": "uma",
        "PETMADCalculator": "pet_mad",
        "MatterSimCalculator": "mattersim",
        "TPCalculator": "grace",
    }
    arch = arch_map.get(class_name)
    if arch is None:
        raise ValueError(f"Unsupported model class '{class_name}' for {model_name}")

    model_id = None
    kwargs = cfg.get("kwargs", {})
    for key in ("model", "name", "model_name", "load_path", "checkpoint_path"):
        if key in kwargs:
            model_id = kwargs[key]
            break
    if model_id is None:
        raise ValueError(f"Missing model identifier for {model_name}")

    calc_kwargs: dict[str, Any] = {}
    if cfg.get("default_dtype") is not None:
        calc_kwargs["default_dtype"] = cfg["default_dtype"]

    return {
        "arch": arch,
        "model": model_id,
        "device": cfg.get("device", "cpu"),
        "calc_kwargs": calc_kwargs,
    }


def _relax_structure_pressure(
    atoms: Atoms,
    janus_cfg: dict[str, Any],
    pressure_gpa: float,
) -> Atoms:
    """
    Relax a structure at external pressure using janus-core GeomOpt.

    Parameters
    ----------
    atoms
        Structure to relax.
    janus_cfg
        Janus-core configuration mapping.
    pressure_gpa
        External pressure in GPa.

    Returns
    -------
    Atoms
        Relaxed structure.
    """
    geom_opt = GeomOpt(
        struct=atoms.copy(),
        arch=janus_cfg["arch"],
        model=janus_cfg["model"],
        device=janus_cfg["device"],
        calc_kwargs=janus_cfg["calc_kwargs"],
        fmax=RELAX_FMAX,
        steps=RELAX_STEPS,
        filter_kwargs={"scalar_pressure": pressure_gpa},
    )
    geom_opt.run()
    return geom_opt.struct


def _run_eos(
    atoms: Atoms,
    janus_cfg: dict[str, Any],
    pressure_gpa: float,
) -> tuple[list[float], list[float], list[float]]:
    """
    Run janus-core EOS calculation to sample volumes and energies.

    Parameters
    ----------
    atoms
        Reference structure.
    janus_cfg
        Janus-core configuration mapping.
    pressure_gpa
        External pressure in GPa.

    Returns
    -------
    tuple[list[float], list[float], list[float]]
        Volumes, energies, and lattice scale factors.
    """
    from janus_core.calculations.eos import EoS

    eos = EoS(
        struct=atoms,
        arch=janus_cfg["arch"],
        model=janus_cfg["model"],
        device=janus_cfg["device"],
        calc_kwargs=janus_cfg["calc_kwargs"],
        min_volume=1 - EOS_VOLUME_SCALE,
        max_volume=1 + EOS_VOLUME_SCALE,
        n_volumes=EOS_N_VOLUMES,
        eos_type=EOS_TYPE,
        minimize=True,
        minimize_kwargs={
            "fmax": EOS_MINIMIZE_FMAX,
            "steps": EOS_MINIMIZE_STEPS,
            "filter_kwargs": {"scalar_pressure": pressure_gpa},
        },
    )
    eos.run()

    volumes = list(eos.volumes)
    energies = list(eos.energies)
    lattice_scalars = list(getattr(eos, "lattice_scalars", []))
    if not lattice_scalars:
        ref_volume = volumes[0]
        lattice_scalars = [(vol / ref_volume) ** (1 / 3) for vol in volumes]
    return volumes, energies, lattice_scalars


def _thermal_free_energy(
    atoms: Atoms,
    janus_cfg: dict[str, Any],
    temperatures: list[float],
    pressure_gpa: float,
    file_prefix: Path,
) -> list[float]:
    """
    Compute phonon free energies for a structure using janus-core phonons.

    Parameters
    ----------
    atoms
        Structure at a fixed volume.
    janus_cfg
        Janus-core configuration mapping.
    temperatures
        Temperature grid.
    pressure_gpa
        External pressure in GPa.
    file_prefix
        File prefix for phonon outputs.

    Returns
    -------
    list[float]
        Free energies (eV) at requested temperatures.
    """
    from janus_core.calculations.phonons import Phonons

    temp_min = min(temperatures)
    temp_max = max(temperatures)
    temp_step = min(
        {abs(b - a) for a, b in zip(temperatures, temperatures[1:], strict=False)}
        or {100.0}
    )

    phonons = Phonons(
        struct=atoms,
        arch=janus_cfg["arch"],
        model=janus_cfg["model"],
        device=janus_cfg["device"],
        calc_kwargs=janus_cfg["calc_kwargs"],
        supercell=PHONON_SUPERCELL,
        displacement=PHONON_DISPLACEMENT,
        mesh=PHONON_MESH,
        temp_min=temp_min,
        temp_max=temp_max,
        temp_step=temp_step,
        minimize=True,
        minimize_kwargs={
            "fmax": PHONON_MINIMIZE_FMAX,
            "steps": PHONON_MINIMIZE_STEPS,
            "filter_kwargs": {"scalar_pressure": pressure_gpa},
        },
        file_prefix=str(file_prefix),
    )

    phonons.calc_force_constants()
    phonons.calc_thermal_props(write_thermal=True)

    thermal_path = file_prefix.with_name(f"{file_prefix.name}-thermal.dat")
    thermal_data = np.loadtxt(thermal_path)
    temps = thermal_data[:, 0]
    free_kj_mol = thermal_data[:, 1]
    free_ev = free_kj_mol * KJ_MOL_TO_EV
    return np.interp(temperatures, temps, free_ev).tolist()


def _run_phonopy_qha(
    volumes: list[float],
    energies: list[float],
    temperatures: list[float],
    free_energy: list[list[float]],
) -> list[float]:
    """
    Run phonopy QHA to obtain equilibrium volumes.

    Parameters
    ----------
    volumes
        Sampled volumes.
    energies
        Static energies for each volume.
    temperatures
        Temperature grid.
    free_energy
        Free energy grid (n_temperatures, n_volumes).

    Returns
    -------
    list[float]
        Equilibrium volumes for each temperature.
    """
    try:
        from phonopy.qha import QHA
    except ImportError as exc:
        raise ImportError("phonopy is required for QHA.") from exc

    qha = QHA(
        volumes,
        energies,
        temperatures,
        free_energy=free_energy,
        eos=EOS_TYPE,
    )
    qha.run()
    volume_temperature = getattr(qha, "volume_temperature", None)
    if volume_temperature is None:
        volume_temperature = qha.get_volume_temperature()
    return list(volume_temperature)


def _volume_temperature_curve(
    atoms: Atoms,
    janus_cfg: dict[str, Any],
    temperatures: list[float],
    pressure_gpa: float,
    out_dir: Path,
) -> dict[float, float]:
    """
    Build temperature -> equilibrium volume mapping.

    Parameters
    ----------
    atoms
        Reference structure.
    janus_cfg
        Janus-core configuration mapping.
    temperatures
        Temperature grid.
    pressure_gpa
        External pressure in GPa.
    out_dir
        Directory to store intermediate phonon outputs.

    Returns
    -------
    dict[float, float]
        Mapping from temperature to equilibrium volume.
    """
    base = _relax_structure_pressure(atoms, janus_cfg, pressure_gpa)
    volumes, energies, lattice_scalars = _run_eos(base, janus_cfg, pressure_gpa)

    free_energy: list[list[float]] = []
    # for temp in temperatures:
    #     free_energy.append([])

    for idx, scale in enumerate(lattice_scalars):
        scaled = base.copy()
        scaled.set_cell(scaled.cell * scale, scale_atoms=True)
        pressure_tag = f"{pressure_gpa:.2f}".replace(".", "p")
        file_prefix = out_dir / f"{atoms.info['material']}_p{pressure_tag}_v{idx}"
        free_at_t = _thermal_free_energy(
            scaled,
            janus_cfg,
            temperatures,
            pressure_gpa,
            file_prefix,
        )
        for temp_idx, free_val in enumerate(free_at_t):
            free_energy[temp_idx].append(free_val)

    energies = [
        energy + pressure_gpa * GPa * volume
        for energy, volume in zip(energies, volumes, strict=True)
    ]
    volume_temperature = _run_phonopy_qha(
        volumes,
        energies,
        temperatures,
        free_energy,
    )
    return dict(zip(temperatures, volume_temperature, strict=True))


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODEL_CONFIGS.keys())
def test_qha_lattice_constants(model_name: str) -> None:
    """
    Run QHA lattice constants benchmark.

    Parameters
    ----------
    model_name
        Model name from registry.
    """
    try:
        janus_cfg = _build_janus_core_config(model_name)
    except ValueError as err:
        pytest.skip(str(err))

    dataset = _load_dataset()
    materials = {mat["name"]: mat for mat in dataset["materials"]}
    conditions = dataset["conditions"]

    temperatures = sorted({row["temperature_K"] for row in conditions})
    pressures = sorted({row["pressure_GPa"] for row in conditions})

    model_out_dir = OUT_PATH / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    volume_curves: dict[str, dict[float, dict[float, float]]] = {}
    for material in materials.values():
        atoms = _build_structure(material)
        volume_curves[material["name"]] = {}
        for pressure in pressures:
            volume_curves[material["name"]][pressure] = _volume_temperature_curve(
                atoms,
                janus_cfg,
                temperatures,
                pressure,
                model_out_dir,
            )

    results = []
    for row in conditions:
        material = materials[row["material"]]
        volume_t = volume_curves[row["material"]][row["pressure_GPa"]][
            row["temperature_K"]
        ]
        lattice_a = volume_t ** (1.0 / 3.0)

        results.append(
            {
                "material": row["material"],
                "temperature_K": row["temperature_K"],
                "pressure_GPa": row["pressure_GPa"],
                "ref_lattice_a": row["ref_lattice_a"],
                "pred_lattice_a": lattice_a,
                "lattice_type": material["lattice_type"],
                "a0": material["a0"],
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(model_out_dir / "results.csv", index=False)
