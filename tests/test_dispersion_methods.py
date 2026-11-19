"""Check separate dispersion metadata and calculation provenance."""

from __future__ import annotations

import pytest

from ml_peg.app.utils.utils import build_level_of_theory_warnings
from ml_peg.models.get_models import validate_config
from ml_peg.models.methods import (
    effective_method,
    method_metadata,
)


def test_flat_metadata_and_legacy_labels():
    """Read flat fields and legacy labels without conflating unknown and none."""
    assert method_metadata("PBE", "d3") == method_metadata("PBE+D3")
    assert method_metadata("PBE")["dispersion"] is None
    assert method_metadata("PBE", "none")["dispersion"] == "none"
    with pytest.raises(ValueError):
        method_metadata("PBE+D3", "D4")


@pytest.mark.parametrize("trained,scheme", [(False, "D3"), (True, "none")])
def test_reject_inconsistent_guard(trained, scheme):
    """Reject declarations that conflict with the double-counting guard."""
    with pytest.raises(ValueError, match="inconsistent"):
        validate_config(
            "example",
            {
                "module": "example",
                "class_name": "Example",
                "trained_on_dispersion": trained,
                "dispersion_method": scheme,
            },
        )


@pytest.mark.parametrize("scheme,warning", [("D3", False), ("none", True)])
def test_app_compares_effective_dispersion(scheme, warning):
    """Compare configured dispersion independently from the base functional."""
    config = {"level_of_theory": "PBE", "dispersion_method": scheme}
    styles, tooltips = build_level_of_theory_warnings(
        [{"MLIP": "example"}],
        {"example": "PBE"},
        {"Metric": {"level_of_theory": "PBE", "dispersion_method": "D3"}},
        {"example": config},
    )
    assert bool(styles) is warning
    assert ("Dispersion mismatch" in tooltips[0]["MLIP"]["value"]) is warning


def test_app_does_not_infer_runtime_from_suffix():
    """A cosmetic D3 suffix must not establish the effective calculation method."""
    styles, tooltips = build_level_of_theory_warnings(
        [{"MLIP": "example-D3"}],
        {"example-D3": "PBE"},
        {"Metric": {"level_of_theory": "PBE", "dispersion_method": "D3"}},
        {"example-D3": {"dispersion_method": "none"}},
    )
    assert styles
    assert "Dispersion mismatch" in tooltips[0]["MLIP"]["value"]


def test_metric_yaml_keeps_dispersion_separate(tmp_path):
    """Carry the flat metric declaration into analysis thresholds and tooltips."""
    from ml_peg.analysis.utils.utils import load_metrics_config

    path = tmp_path / "metrics.yml"
    path.write_text(
        "metrics:\n  MAE:\n    good: 0\n    bad: 1\n    unit: eV\n"
        "    level_of_theory: PBE\n    dispersion_method: D3\n    tooltip: Error\n"
    )
    thresholds, tooltips, _ = load_metrics_config(path)
    assert thresholds["MAE"]["level_of_theory"] == "PBE"
    assert thresholds["MAE"]["dispersion_method"] == "D3"
    assert "Dispersion: D3" in tooltips["MAE"]["value"]


@pytest.mark.parametrize("trained,expected", [(False, "none"), (True, None)])
def test_omitted_model_dispersion_defaults_from_training_flag(trained, expected):
    """Default native dispersion from the flag without requiring a scheme."""
    config = {
        "module": "example",
        "class_name": "Example",
        "level_of_theory": "PBE",
        "trained_on_dispersion": trained,
    }
    validate_config("example", config)
    assert effective_method(config)["dispersion"] == expected


def test_default_none_still_allows_runtime_d3():
    """Use benchmark configuration to compare existing D3 results."""
    method = effective_method(
        {
            "level_of_theory": "PBE",
            "trained_on_dispersion": False,
            "runtime_dispersion_method": "D3",
        }
    )
    assert method == {"level_of_theory": "PBE", "dispersion": "D3"}


def test_native_dispersion_is_not_overridden():
    """Preserve the native method when the shared helper would skip D3."""
    method = effective_method(
        {
            "level_of_theory": "PBE",
            "trained_on_dispersion": True,
            "dispersion_method": "VV10",
            "runtime_dispersion_method": "D3",
        }
    )
    assert method["dispersion"] == "VV10"
