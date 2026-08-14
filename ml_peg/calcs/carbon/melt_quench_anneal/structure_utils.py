"""Clean and relax structures from carbon melt-quench-anneal trajectories."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.optimize import LBFGS
import numpy as np

# Isolated fragments smaller than MIN_CLUSTER_SIZE are dropped before relaxing,
# so gas-phase molecules ejected during the melt do not contribute to the final
# amorphous structure.
MIN_CLUSTER_SIZE = 20
BOND_SCALE = 1.10
RELAX_FMAX = 0.01
RELAX_STEPS = 5000


def _bond_graph(atoms: Atoms, bond_scale: float = BOND_SCALE) -> list[list[int]]:
    """
    Build the bonded-neighbour adjacency list of a structure.

    Parameters
    ----------
    atoms
        Structure to analyse.
    bond_scale
        Multiplier applied to the natural covalent cutoffs.

    Returns
    -------
    list[list[int]]
        Neighbour indices for each atom.
    """
    cutoffs = natural_cutoffs(atoms, mult=bond_scale)
    neighbor_list = NeighborList(
        cutoffs, skin=0.0, self_interaction=False, bothways=True
    )
    neighbor_list.update(atoms)
    return [
        neighbor_list.get_neighbors(index)[0].tolist() for index in range(len(atoms))
    ]


def _components(graph: list[list[int]]) -> list[list[int]]:
    """
    Find the connected components of a bond graph.

    Parameters
    ----------
    graph
        Neighbour indices for each atom.

    Returns
    -------
    list[list[int]]
        Atom indices making up each connected component.
    """
    seen = np.zeros(len(graph), dtype=bool)
    components = []
    for start in range(len(graph)):
        if seen[start]:
            continue
        queue = deque([start])
        seen[start] = True
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in graph[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        components.append(component)
    return components


def remove_small_clusters(
    atoms: Atoms,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    bond_scale: float = BOND_SCALE,
) -> tuple[Atoms, dict[str, Any]]:
    """
    Remove isolated fragments smaller than a minimum size.

    Parameters
    ----------
    atoms
        Structure to clean.
    min_cluster_size
        Smallest connected component retained.
    bond_scale
        Multiplier applied to the natural covalent cutoffs.

    Returns
    -------
    tuple[ase.Atoms, dict[str, Any]]
        Cleaned structure and a summary of what was removed.
    """
    components = _components(_bond_graph(atoms, bond_scale=bond_scale))
    sizes = np.array([len(component) for component in components], dtype=int)
    small = sizes < min_cluster_size
    removed = sorted(
        {
            index
            for component_index in np.where(small)[0]
            for index in components[component_index]
        }
    )

    mask = np.ones(len(atoms), dtype=bool)
    if removed:
        mask[removed] = False
    cleaned = atoms[mask]

    info = {
        "n_atoms_in": len(atoms),
        "n_atoms_out": len(cleaned),
        "n_clusters": len(components),
        "n_small_clusters": int(small.sum()),
        "small_cluster_sizes": sorted(sizes[small].tolist()),
        "n_removed_atoms": len(removed),
        "min_cluster_size": min_cluster_size,
        "bond_scale": bond_scale,
    }
    return cleaned, info


def clean_and_relax(
    structure_path: Path,
    calc: Any,
    file_prefix: Path,
) -> dict[str, Any]:
    """
    Remove small clusters from an MD structure and relax what remains.

    Parameters
    ----------
    structure_path
        Final structure written by an MD stage.
    calc
        Calculator used for the relaxation.
    file_prefix
        Prefix for the cleaned, relaxed, log and trajectory outputs.

    Returns
    -------
    dict[str, Any]
        Cluster removal summary and relaxation outcome.
    """
    atoms = read(structure_path)
    cleaned, info = remove_small_clusters(atoms)
    # Velocities from the MD stage are meaningless after relaxation.
    cleaned.arrays.pop("momenta", None)
    write(file_prefix.with_name(f"{file_prefix.name}-cleaned.extxyz"), cleaned)

    # A low-density cell can fragment entirely into small clusters, leaving
    # nothing to relax.
    if not len(cleaned):
        info.update(
            {"converged": False, "relax_steps": 0, "empty_after_cleaning": True}
        )
        return info

    cleaned.calc = calc
    optimizer = LBFGS(
        cleaned,
        logfile=str(file_prefix.with_name(f"{file_prefix.name}-relax.log")),
        trajectory=str(file_prefix.with_name(f"{file_prefix.name}-relax.traj")),
    )
    converged = optimizer.run(fmax=RELAX_FMAX, steps=RELAX_STEPS)
    write(file_prefix.with_name(f"{file_prefix.name}-relaxed.extxyz"), cleaned)

    info.update(
        {
            "converged": bool(converged),
            "relax_steps": int(optimizer.get_number_of_steps()),
            "max_relax_steps": RELAX_STEPS,
            "fmax": RELAX_FMAX,
            "max_force": float(np.linalg.norm(cleaned.get_forces(), axis=1).max()),
            "energy": float(cleaned.get_potential_energy()),
        }
    )
    return info
