"""Test the binary mixtures analysis calculations."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.molecular_dynamics.binary_mixtures.mixing import (
    combine,
    component_a_molecules,
    drift_test,
    find_molecules,
    frame_checks,
    holm,
    mixing_matrix,
    molecule_centres,
    read_production,
    same_molecules,
)
from ml_peg.analysis.utils.timeseries import (
    TimeSeriesEstimate,
    bootstrap_metric,
)


def ar1(n_samples: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate an AR(1) series with unit noise.

    Parameters
    ----------
    n_samples
        Number of samples.
    phi
        Autoregressive coefficient.
    rng
        Random generator.

    Returns
    -------
    np.ndarray
        Time series.
    """
    noise = rng.standard_normal(n_samples)
    series = np.zeros(n_samples)
    for i in range(1, n_samples):
        series[i] = phi * series[i - 1] + noise[i]
    return series


@pytest.fixture
def toy_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Make a one-system set with two compositions.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, list[str]]
        ``systems.csv``, ``boxes.csv`` and the box order.
    """
    systems = pd.DataFrame(
        {
            "system_id": ["a_b", "a_b"],
            "a": ["A", "A"],
            "b": ["B", "B"],
            "x_box": [0.25, 0.5],
        }
    )
    boxes = pd.DataFrame(
        {
            "box_id": ["a_b_x250", "a_b_x500", "pure_a", "pure_b"],
            "kind": ["mixture", "mixture", "pure", "pure"],
            "smiles": ["A;B", "A;B", "A", "B"],
            "temperature": [298.15] * 4,
            "x": [0.24, 0.5, 1.0, 0.0],
        }
    )
    return systems, boxes, ["pure_b", "a_b_x500", "pure_a", "a_b_x250"]


def test_mixing_matrix(toy_tables):
    """ΔHmix from the matrix equals h_mix - x h_a - (1 - x) h_b."""
    systems, boxes, order = toy_tables
    matrix = mixing_matrix(systems, boxes, order)
    h = {"pure_b": -10.0, "a_b_x500": -12.0, "pure_a": -15.0, "a_b_x250": -11.5}
    values = matrix @ np.array([h[b] for b in order])
    # The exact box composition (0.24), not the nominal x_box, is used
    expected_250 = -11.5 - 0.24 * -15.0 - 0.76 * -10.0
    expected_500 = -12.0 - 0.5 * -15.0 - 0.5 * -10.0
    np.testing.assert_allclose(values, [expected_250, expected_500])


def test_combine():
    """Standard errors of a linear combination add in quadrature."""
    estimates = [
        TimeSeriesEstimate(mean=1.0, sem=0.3, n_eff=200, ci_95=0.6),
        TimeSeriesEstimate(mean=2.0, sem=0.4, n_eff=100, ci_95=0.8),
        TimeSeriesEstimate(mean=5.0, sem=9.0, n_eff=10, ci_95=18.0),
    ]
    combined = combine([1.0, -0.5, 0.0], estimates)
    assert combined.mean == pytest.approx(1.0 - 0.5 * 2.0)
    assert combined.sem == pytest.approx(np.sqrt((1.0 * 0.3) ** 2 + (0.5 * 0.4) ** 2))
    assert combined.ci_95 == pytest.approx(1.96 * combined.sem)
    # The zero-weight term does not count
    assert combined.n_eff == 100


def test_box_level_bootstrap_includes_shared_pure_boxes(toy_tables):
    """Resampling boxes gives a wider CI than treating points as independent."""
    systems, boxes, order = toy_tables
    matrix = mixing_matrix(systems, boxes, order)
    sem = {"pure_b": 0.2, "a_b_x500": 0.01, "pure_a": 0.2, "a_b_x250": 0.01}
    boxes_est = [
        TimeSeriesEstimate(mean=0.0, sem=sem[b], n_eff=500, ci_95=1.96 * sem[b])
        for b in order
    ]
    ref = np.zeros(len(systems))

    def mean_error(reference, h):
        return float(np.mean(matrix @ h - reference))

    box_level = bootstrap_metric(mean_error, ref, boxes_est)
    points = [combine(row, boxes_est) for row in matrix]
    naive = bootstrap_metric(lambda r, p: float(np.mean(p - r)), ref, points)
    # 1.2 is a fairly loose threshold
    assert box_level.ci_high - box_level.ci_low > 1.2 * (naive.ci_high - naive.ci_low)


def test_maze_uses_fixed_sigma(toy_tables):
    """MAZE does not change when the model's own uncertainties change."""
    systems, boxes, order = toy_tables
    matrix = mixing_matrix(systems, boxes, order)
    sigma_ref = np.array([0.1, 0.2])
    ref = np.array([-0.5, -0.8])

    def maze(reference, h):
        return float(np.mean(np.abs(matrix @ h - reference) / sigma_ref))

    means = [-10.0, -12.0, -15.0, -11.5]
    values = [
        bootstrap_metric(
            maze,
            ref,
            [TimeSeriesEstimate(mean=m, sem=s, n_eff=100, ci_95=2 * s) for m in means],
        ).value
        for s in (0.01, 1.0)
    ]
    assert values[0] == values[1]


def test_drift_test():
    """A drifting series is flagged; a stationary AR(1) series is not."""
    rng = np.random.default_rng(0)
    t = np.arange(20_000) * 0.1
    stationary = ar1(len(t), 0.9, rng)
    assert drift_test(t, stationary) > 0.01
    assert drift_test(t, stationary + 0.01 * t) < 1e-4


def test_holm():
    """Holm-adjusted p-values for a worked example, in the input order."""
    # Sorted: 0.005×4 = 0.02, 0.01×3 = 0.03, 0.03×2 = 0.06, 0.04×1 = 0.04 → 0.06
    # (the adjusted values can only increase with rank)
    np.testing.assert_allclose(
        holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02]
    )


def test_read_production(tmp_path: Path):
    """Truncated runs are rejected and equilibration is discarded."""
    time_ps = np.arange(0, 2200.01, 0.1)
    data = pd.DataFrame(
        {
            "step": np.arange(len(time_ps)),
            "time_ps": time_ps,
            "epot_eV": 0.0,
            "ekin_eV": 0.0,
            "temperature_K": 298.15,
            "volume_A3": 1.0,
            "density_g_cm3": 1.0,
        }
    )
    full, truncated = tmp_path / "full.csv", tmp_path / "truncated.csv"
    data.to_csv(full, index=False)
    data[data.time_ps < 1500].to_csv(truncated, index=False)
    prod = read_production(full, 200.0, 2200.0)
    assert prod.time_ps.min() > 200.0
    assert len(prod) == 20_000
    with pytest.raises(ValueError, match="incomplete"):
        read_production(truncated, 200.0, 2200.0)
    with pytest.raises(FileNotFoundError):
        read_production(tmp_path / "missing.csv", 200.0, 2200.0)


@pytest.fixture
def mixture_box() -> Atoms:
    """
    Make a water and a hydrogen sulfide molecule, the water split by the boundary.

    Returns
    -------
    Atoms
        Periodic box with the mixture metadata (water is component a).
    """
    atoms = Atoms(
        "OH2SH2",
        positions=[
            [0.2, 5.0, 5.0],
            [9.6, 5.0, 5.4],  # bonded to the O at x = 0.2 through the boundary
            [0.5, 5.9, 5.0],
            [5.0, 5.0, 5.0],
            [5.96, 5.0, 5.0],
            [4.7, 5.9, 5.0],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.info.update({"smiles_a": "O", "smiles_b": "S", "n_a": 1, "n_b": 1})
    return atoms


def test_find_molecules_and_reaction_flag(mixture_box):
    """Molecules are found across boundaries, and a moved proton is detected."""
    atoms = mixture_box
    labels = find_molecules(atoms)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1])

    moved = atoms.copy()
    moved.positions[4] = [1.1, 5.0, 5.0]  # H of the H2S onto the water
    assert not same_molecules(labels, find_molecules(moved))
    reacted, fraction = frame_checks([atoms, moved], atoms, mixture=True)
    assert reacted
    assert 0.0 <= fraction <= 1.0
    reacted, _ = frame_checks([atoms], atoms, mixture=False)
    assert not reacted


def test_molecule_centres_across_boundary(mixture_box):
    """A molecule split by the boundary gets the centre of mass of the whole one."""
    atoms = mixture_box
    whole = atoms.copy()
    whole.positions[1, 0] -= 10.0  # the H at x = 9.6, moved next to its O
    centres = molecule_centres(atoms, find_molecules(atoms))
    np.testing.assert_allclose(centres[0], whole[:3].get_center_of_mass())
    np.testing.assert_allclose(centres[1], atoms[3:].get_center_of_mass())


def test_component_a_molecules(mixture_box):
    """Components are found by formula, whatever their order in the box."""
    atoms = mixture_box
    np.testing.assert_array_equal(
        component_a_molecules(atoms, find_molecules(atoms)), [True, False]
    )
    swapped = atoms[[3, 4, 5, 0, 1, 2]]  # H2S first
    np.testing.assert_array_equal(
        component_a_molecules(swapped, find_molecules(swapped)), [False, True]
    )
    atoms.info["n_a"] = 2
    with pytest.raises(ValueError, match="expected"):
        component_a_molecules(atoms, find_molecules(atoms))
    atoms.info.update({"n_a": 1, "smiles_b": "N"})
    with pytest.raises(ValueError, match="neither"):
        component_a_molecules(atoms, find_molecules(atoms))
