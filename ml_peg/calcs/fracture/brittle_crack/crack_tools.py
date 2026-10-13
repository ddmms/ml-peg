"""Tools for the brittle_crack calculation: cases, constants, clusters and traces."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.units import GPa, J, m
import numpy as np
import yaml

CASES_PATH = Path(__file__).with_name("cases.yml")
TRACE = {
    "ftol": 2e-3,
    "dr_cap": 0.05,
    "max_points": 600,
    "k_window": (0.3, 3.0),
    "stop_after_folds": 2,
    "extra_after_last_fold": 1.15,
    "max_calls_per_point": 60,
    "start_max_iter": 300,
}


def load_cases(path: Path = CASES_PATH) -> dict[str, dict]:
    """
    Load the benchmark cases.

    Parameters
    ----------
    path
        Path to cases.yml.

    Returns
    -------
    dict[str, dict]
        Case settings by case name.
    """
    return yaml.safe_load(Path(path).read_text())["cases"]


def material_constants(calc, symbol: str = "Si") -> dict:
    """
    Compute the lattice constant and relaxed-ion cubic elastic constants of a model.

    Parameters
    ----------
    calc
        ASE calculator.
    symbol
        Element of the diamond lattice.

    Returns
    -------
    dict
        Lattice constant ``a0`` (A) and ``C11``, ``C12``, ``C44`` (eV/A^3).
    """
    from ase.build import bulk
    from ase.optimize import LBFGS
    from matscipy.dislocation import get_elastic_constants
    from matscipy.elasticity import fit_elastic_constants

    a0, *_ = get_elastic_constants(calculator=calc, symbol=symbol, verbose=False)
    cell = bulk(symbol, "diamond", a=a0, cubic=True)
    cell.calc = calc
    c, _ = fit_elastic_constants(
        cell,
        symmetry="cubic",
        delta=1e-3,
        N_steps=5,
        optimizer=LBFGS,
        fmax=1e-6,
        verbose=False,
    )
    return {
        "a0": float(a0),
        "C11": float(c[0, 0]),
        "C12": float(c[0, 1]),
        "C44": float(c[3, 3]),
    }


def cleavage_energy(calc, a0: float, surface, front, symbol: str = "Si") -> float:
    """
    Compute the relaxed energy of the surface the crack opens.

    The slab is built in the crack's own orientation and cut at the widest interlayer
    gap nearest the cell middle (the shuffle plane for (111)).

    Parameters
    ----------
    calc
        ASE calculator.
    a0
        Lattice constant (A).
    surface
        Crack plane normal (Miller indices).
    front
        Crack front direction.
    symbol
        Element.

    Returns
    -------
    float
        Surface energy (eV/A^2).
    """
    from ase.optimize import LBFGS
    from matscipy.fracture_mechanics.clusters import diamond

    slab = diamond(symbol, a0, [2, 6, 1], list(surface), list(front))
    slab.set_pbc(True)
    slab.wrap()
    slab.calc = calc
    e_bulk = slab.get_potential_energy()
    y = np.sort(np.unique(np.round(slab.positions[:, 1], 4)))
    gaps, centres = np.diff(y), 0.5 * (y[1:] + y[:-1])
    widest = gaps > gaps.max() - 1e-3
    mid = slab.cell.lengths()[1] / 2
    y_cut = centres[widest][np.argmin(np.abs(centres[widest] - mid))]
    slab.translate([0, -y_cut, 0])
    slab.wrap()
    cell = slab.cell.copy()
    cell[1, 1] += 15.0
    slab.set_cell(cell)
    LBFGS(slab, logfile=None).run(fmax=1e-5, steps=2000)
    area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[2]))
    return float((slab.get_potential_energy() - e_bulk) / (2 * area))


def crack(case: dict, consts: dict):
    """
    Build the anisotropic crack of a case for given elastic constants.

    Parameters
    ----------
    case
        Case settings.
    consts
        Elastic constants ``C11``, ``C12``, ``C44`` (eV/A^3).

    Returns
    -------
    CubicCrystalCrack
        The crack.
    """
    from matscipy.fracture_mechanics.crack import CubicCrystalCrack

    return CubicCrystalCrack(
        case["crack_surface"],
        case["crack_front"],
        C11=consts["C11"],
        C12=consts["C12"],
        C44=consts["C44"],
    )


def topology_constants(case: dict) -> dict:
    """
    Return the reference (DFT) constants of a case in eV/A units.

    Parameters
    ----------
    case
        Case settings.

    Returns
    -------
    dict
        ``a0`` (A), ``C11``, ``C12``, ``C44`` (eV/A^3) and ``gamma`` (eV/A^2).
    """
    t = case["topology"]
    return {
        "a0": t["a0"],
        "C11": t["C11"] * GPa,
        "C12": t["C12"] * GPa,
        "C44": t["C44"] * GPa,
        "gamma": t["surface_energy"] * J / m**2,
    }


def reference_cluster(case: dict) -> Atoms:
    """
    Build the DFT-topology cluster of a case.

    Parameters
    ----------
    case
        Case settings.

    Returns
    -------
    Atoms
        The reference cluster (``clamped_cluster`` with the topology constants).
    """
    from matscipy.fracture_mechanics.clusters import clamped_cluster

    t = topology_constants(case)
    crk = crack(case, t)
    k_g = crk.k1g(t["gamma"])
    return clamped_cluster(
        crk,
        case["topology"].get("k", 1.0) * k_g,
        tuple(case["n"]),
        tuple(case["skin"]),
        a0=t["a0"],
        crack_surface=case["crack_surface"],
        crack_front=case["crack_front"],
        symbol=case["symbol"],
        tip_offset=case["tip_offset"],
        stagger_caps=case["stagger_caps"],
        passivate=case["passivate"],
    )


def model_problem(case: dict, calc, consts: dict):
    """
    Set up a model's clamped problem on the case's reference topology.

    Parameters
    ----------
    case
        Case settings.
    calc
        ASE calculator of the model.
    consts
        Model constants: ``a0``, ``C11``, ``C12``, ``C44`` (eV/A^3), ``gamma`` (eV/A^2).

    Returns
    -------
    tuple
        The ``ClampedProblem``, the model cluster and K_G (eV/A^2.5).
    """
    from matscipy.fracture_mechanics.clusters import cluster_like
    from matscipy.fracture_mechanics.continuation import ClampedProblem
    from matscipy.fracture_mechanics.crack import SinclairCrack, regions_from_groups

    ref = reference_cluster(case)
    crk = crack(case, consts)
    k_g = crk.k1g(consts["gamma"])
    k0 = case["start_k"]
    atoms = cluster_like(ref, crk, k0 * k_g, consts["a0"])
    cryst = atoms.copy()
    cryst.positions = atoms.arrays["reference"]
    regions_from_groups(cryst)
    sinclair = SinclairCrack(
        crk,
        cryst,
        calc,
        k0 * k_g,
        alpha=atoms.info["tip_x"] - cryst.cell[0, 0] / 2,
        vacuum=0.0,
        variable_alpha=False,
        variable_k=True,
        energy="total",
    )
    return ClampedProblem(sinclair, k_g, atoms=atoms), atoms, k_g


def precon_calculator(name: str):
    """
    Build the many-body potential of the corrector's surrogate Jacobian.

    Parameters
    ----------
    name
        Potential name from cases.yml.

    Returns
    -------
    Calculator
        A matscipy Manybody calculator with a Hessian.
    """
    from matscipy.calculators.manybody import Manybody

    if name == "stillinger_weber":
        from matscipy.calculators.manybody.explicit_forms.stillinger_weber import (
            Stillinger_Weber_PRB_31_5262_Si,
            StillingerWeber,
        )

        return Manybody(**StillingerWeber(Stillinger_Weber_PRB_31_5262_Si))
    raise ValueError(f"unknown preconditioner potential {name!r}")


def trace_case(case: dict, calc, consts: dict, **settings) -> tuple[list[Atoms], str]:
    """
    Trace a model's S-curve on a case.

    Parameters
    ----------
    case
        Case settings.
    calc
        ASE calculator of the model.
    consts
        Model constants (``a0``, ``C11``, ``C12``, ``C44``, ``gamma``).
    **settings
        Overrides of the trace settings (``TRACE``).

    Returns
    -------
    tuple[list[Atoms], str]
        Traced frames (with info ``REF_energy``) and the trace status;
        ``"failed: <message>"`` if the tracer raised.
    """
    from matscipy.fracture_mechanics import continuation as cont

    problem, _, _ = model_problem(case, calc, consts)
    surrogate = cont.HessianSurrogate(problem, precon_calculator(case["precon"]))
    trace_settings = cont.TraceSettings(**{**TRACE, **settings})
    try:
        frames, status = cont.trace_s_curve(
            problem, surrogate, case["start_k"], settings=trace_settings
        )
    except RuntimeError as exc:
        return [], f"failed: {exc}"
    for f in frames:
        f.info["REF_energy"] = f.info["energy"]
    return frames, status
