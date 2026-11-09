"""Tests that every model in the registry is configured correctly."""

from __future__ import annotations

from pathlib import Path

import pytest
from yaml import safe_load

from ml_peg.models.get_models import load_models, validate_config
from ml_peg.models.models import (
    AIMNetCalc,
    DpaCalc,
    FairChemCalc,
    GenericASECalc,
    GraceCalc,
    MatterSimCalc,
    OrbCalc,
    SevenNetCalc,
    TaceCalc,
    UPETCalc,
    VivaceCalc,
    accepted_kwargs,
    check_kwargs,
    import_calculator,
)

MODELS_FILE = Path(__file__).parents[1] / "ml_peg" / "models" / "models.yml"
REGISTRY = safe_load(MODELS_FILE.read_text(encoding="utf8"))

#: Wrapper each `class_name` in the registry is loaded through.
WRAPPERS = {
    "AIMNet2ASE": AIMNetCalc,
    "DP": DpaCalc,
    "FAIRChemCalculator": FairChemCalc,
    "ORBCalculator": OrbCalc,
    "MatterSimCalculator": MatterSimCalc,
    "UPETCalculator": UPETCalc,
    "SevenNetCalculator": SevenNetCalc,
    "TACEAseCalc": TaceCalc,
    "MLFFCalculator": VivaceCalc,
    "grace_fm": GraceCalc,
    "mace_mp": GenericASECalc,
    "mace_off": GenericASECalc,
    "mace_omol": GenericASECalc,
    "mace_polar": GenericASECalc,
}

BASE_CONFIG = {
    "module": "ase.calculators.emt",
    "class_name": "EMT",
    "trained_on_dispersion": True,
}


@pytest.mark.parametrize("name", REGISTRY)
def test_config_is_valid(name: str):
    """Every entry only sets keys that are used, including models not run."""
    validate_config(name, REGISTRY[name])


@pytest.mark.parametrize(
    "coverage",
    (
        pytest.param({"datasets": ["MPTrj"]}, id="miscased dataset"),
        pytest.param({"datasets": "MPtrj"}, id="dataset not a list"),
        pytest.param({"additional_supported_elements": ["Cuu"]}, id="not an element"),
        # YAML reads an unquoted "No" as False
        pytest.param(
            {"additional_supported_elements": ["Md", False]}, id="unquoted No"
        ),
    ),
)
def test_unresolvable_coverage_rejected(coverage: dict):
    """Element coverage that would silently resolve to nothing is rejected."""
    with pytest.raises(ValueError):
        validate_config("example", BASE_CONFIG | coverage)


def test_resolvable_coverage_accepted():
    """Recognised datasets and element symbols are accepted."""
    validate_config(
        "example",
        BASE_CONFIG
        | {"datasets": ["MPtrj"], "additional_supported_elements": ["No", "Lr"]},
    )


@pytest.mark.parametrize("name", REGISTRY)
def test_kwargs_are_accepted(name: str):
    """Every entry's kwargs are accepted by its calculator, where installed."""
    config = REGISTRY[name]
    try:
        target = import_calculator(config["module"], config["class_name"])
    except (ImportError, AttributeError):
        pytest.skip(f"{config['module']} is not installed")

    kwargs = dict(config.get("kwargs") or {})
    # Taken by the loader before the calculator is built.
    for key in ("name", "model_name", "task_name", "overrides", "inference_settings"):
        kwargs.pop(key, None)

    check_kwargs(kwargs, target, config["class_name"])


def test_registry_loads():
    """Every model in the registry can be loaded."""
    models = load_models(filepath=MODELS_FILE, run_mock=False, mock_only=False)

    assert sorted(models) == sorted(REGISTRY)


@pytest.mark.parametrize("name", REGISTRY)
def test_class_name_selects_wrapper(name: str):
    """Every entry is loaded through the wrapper its class_name selects."""
    class_name = REGISTRY[name]["class_name"]
    assert class_name in WRAPPERS, (
        f"'{name}' uses class_name '{class_name}', which has no wrapper in this test. "
        "Add it here when adding a model family."
    )

    model = load_models((name,), filepath=MODELS_FILE, run_mock=False, mock_only=False)

    assert isinstance(model[name], WRAPPERS[class_name])


def test_unknown_precision_rejected():
    """Calculators are only built at a precision ML-PEG defines."""
    model = load_models(
        ("mace-mp-0a",), filepath=MODELS_FILE, run_mock=False, mock_only=False
    )["mace-mp-0a"]

    with pytest.raises(ValueError, match="Invalid precision"):
        model.get_calculator(precision="medium")


#: Wrappers that apply the configured device themselves, rather than passing it to
#: the calculator they build, so it need not be an argument of that calculator.
DEVICE_HANDLED_BY_WRAPPER = ("ORBCalculator", "FAIRChemCalculator", "MLFFCalculator")


@pytest.mark.parametrize("name", REGISTRY)
def test_device_is_set_where_supported(name: str):
    """Models with an unsupported device say so, rather than setting one."""
    config = REGISTRY[name]
    if (
        config.get("device") is None
        or config["class_name"] in DEVICE_HANDLED_BY_WRAPPER
    ):
        return

    try:
        target = import_calculator(config["module"], config["class_name"])
    except (ImportError, AttributeError):
        pytest.skip(f"{config['module']} is not installed")

    assert "device" in accepted_kwargs(target, config["class_name"]), (
        f"'{name}' sets a device, but {config['class_name']} has no device argument. "
        "Set 'device: null' with a comment to record that it is unused."
    )
