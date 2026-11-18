"""Reactions in the Surface Alloy Reorganisation dataset, for calc and analysis."""

from __future__ import annotations

from itertools import product

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


def get_reactions() -> list[tuple[str, str, str]]:
    """
    Get all (host, mechanism, element) reactions in the dataset.

    Returns
    -------
    list[tuple[str, str, str]]
        Host, mechanism and dopant element for every reaction.
    """
    return [
        (host, mechanism, element)
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
