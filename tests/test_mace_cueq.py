"""Opt-in CUDA parity checks for the MACE cuEq extra."""

from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ML_PEG_TEST_MACE_CUEQ") != "1",
    reason="Set ML_PEG_TEST_MACE_CUEQ=1 with the mace-cueq extra and a CUDA GPU",
)


@pytest.mark.parametrize("precision", ["high", "low"])
def test_mace_cueq_matches_e3nn(precision):
    """Compare real CUDA energies, forces and stress through the ML-PEG wrapper."""
    from ase.build import bulk
    import torch

    from ml_peg.models.get_models import load_models

    assert torch.cuda.is_available()
    model = load_models("mace-mp-0a", run_mock=False)["mace-mp-0a"]
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.positions[0] += [0.03, -0.02, 0.01]
    results = []
    for enabled in (False, True):
        configured = replace(model, kwargs={**model.kwargs, "enable_cueq": enabled})
        calc = configured.get_calculator(precision=precision)
        assert calc._enable_cueq is enabled
        dtype = torch.float64 if precision == "high" else torch.float32
        assert next(calc.models[0].parameters()).dtype == dtype
        if enabled:
            assert any(
                type(module).__module__.startswith("cuequivariance_torch")
                for module in calc.models[0].modules()
            )
            # Linear layers intentionally use ordinary torch operations; the
            # costly tensor products and symmetric contractions must use CUDA.
            accelerated_ops = [
                interaction.conv_tp.conv_tp
                for interaction in calc.models[0].interactions
            ] + [
                product.symmetric_contractions.f for product in calc.models[0].products
            ]
            assert accelerated_ops
            assert all(
                operation.method in {"uniform_1d", "fused_tp"}
                for operation in accelerated_ops
            )
            assert all(
                type(operation.m).__name__ != "SegmentedPolynomialNaive"
                for operation in accelerated_ops
            )
        atoms.calc = calc
        values = (atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())
        assert all(np.isfinite(value).all() for value in values)
        results.append(values)
    atol = 1e-8 if precision == "high" else 2e-4
    for reference, accelerated in zip(*results, strict=True):
        np.testing.assert_allclose(accelerated, reference, rtol=1e-4, atol=atol)


@pytest.mark.parametrize("precision", ["high", "low"])
def test_polar_mace_cueq(precision):
    """Exercise the graph-longrange dependency with PolarMACE and cuEq."""
    from ase.build import molecule
    import torch

    from ml_peg.models.get_models import load_models

    assert torch.cuda.is_available()
    model = load_models("mace-polar-1-s", run_mock=False)["mace-polar-1-s"]
    configured = replace(
        model, device="cuda", kwargs={**model.kwargs, "enable_cueq": True}
    )
    calc = configured.get_calculator(precision=precision)
    assert calc._enable_cueq
    dtype = torch.float64 if precision == "high" else torch.float32
    assert next(calc.models[0].parameters()).dtype == dtype
    atoms = molecule("H2O")
    atoms.info.update(charge=0, spin=1)
    atoms.calc = calc
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.isfinite(forces).all()
