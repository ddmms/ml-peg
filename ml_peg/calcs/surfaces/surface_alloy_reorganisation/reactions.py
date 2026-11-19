"""Reactions in the Surface Alloy Reorganisation dataset, for calc and analysis."""

from __future__ import annotations

from itertools import product
from typing import NamedTuple

HOSTS = ["Ag", "Cu"]
MECHANISMS = [
    "diffusion_over_step",
    "step_lift_in",
    "terrace_pop_in",
    "terrace_slide_in",
    "vacancy_mediated_diffusion",
]
STATES = ["initial_state", "final_state", "transition_state"]
ALL_ELEMENTS = ["Ag", "Cu", "Mo", "Nb", "Pd", "Rh", "Ru", "Tc", "Y", "Zr"]
# Host/mechanism pairs for which only a subset of elements is available
ELEMENTS_SUBSETS = {
    ("Cu", "diffusion_over_step"): ["Ag", "Cu", "Nb", "Pd", "Rh", "Ru", "Tc", "Zr"],
    ("Ag", "diffusion_over_step"): [
        "Ag",
        "Cu",
        "Mo",
        "Nb",
        "Pd",
        "Rh",
        "Ru",
        "Tc",
        "Zr",
    ],
    ("Cu", "step_lift_in"): ["Cu", "Mo", "Nb", "Pd", "Rh", "Ru", "Tc", "Y", "Zr"],
}


# Mechanisms whose initial and final states are identical up to symmetry when the
# dopant is the host element itself
SELF_DOPANT_SYMMETRIC_MECHANISMS = ("terrace_pop_in", "terrace_slide_in")
# Mechanisms whose initial and final states are always identical up to symmetry
SYMMETRIC_MECHANISMS = ("vacancy_mediated_diffusion",)


class Reaction(NamedTuple):
    """Reaction in the dataset, and whether it is evaluated for the energy MAE."""

    host: str
    mechanism: str
    element: str
    energy_evaluated: bool


def is_energy_evaluated(host: str, mechanism: str, element: str) -> bool:
    """
    Get whether a reaction is evaluated for the reaction energy MAE.

    Reactions whose initial and final states are identical up to symmetry have a
    reaction energy of zero by construction, so are excluded from the energy metric.

    Parameters
    ----------
    host
        Host metal.
    mechanism
        Reorganisation mechanism.
    element
        Dopant element.

    Returns
    -------
    bool
        Whether the reaction is evaluated for the reaction energy MAE.
    """
    if mechanism in SYMMETRIC_MECHANISMS:
        return False
    return not (mechanism in SELF_DOPANT_SYMMETRIC_MECHANISMS and element == host)


def get_reactions() -> list[Reaction]:
    """
    Get all reactions in the dataset.

    Returns
    -------
    list[Reaction]
        Host, mechanism, dopant element, and whether the reaction is evaluated for
        the reaction energy MAE, for every reaction.
    """
    return [
        Reaction(
            host, mechanism, element, is_energy_evaluated(host, mechanism, element)
        )
        for host, mechanism in product(HOSTS, MECHANISMS)
        for element in ELEMENTS_SUBSETS.get((host, mechanism), ALL_ELEMENTS)
    ]


def get_systems() -> list[tuple[str, str, str, str]]:
    """
    Get all (host, mechanism, state, element) combinations in the dataset.

    Returns
    -------
    list[tuple[str, str, str, str]]
        Host, mechanism, state and element for every structure to evaluate.
    """
    return [
        (host, mechanism, state, element)
        for host, mechanism, state in product(HOSTS, MECHANISMS, STATES)
        for element in ELEMENTS_SUBSETS.get((host, mechanism), ALL_ELEMENTS)
    ]
