"""Get MLIPs to be used for calculations or analysis."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ml_peg.models import models_file

#: Keys that may be set for a model in models.yml. Anything else would be ignored.
MODEL_KEYS = frozenset(
    {
        "module",
        "class_name",
        "device",
        "datasets",
        "additional_supported_elements",
        "level_of_theory",
        "overwrite_dtype",
        "trained_on_dispersion",
        "dispersion_kwargs",
        "kwargs",
    }
)

#: Keys that must be set for a model in models.yml.
REQUIRED_MODEL_KEYS = frozenset({"module", "class_name"})


def validate_config(name: str, cfg: dict[str, Any]) -> None:
    """
    Check that a model's configuration only sets keys that are read.

    Keys are read individually when models are loaded, so a misnamed or outdated key
    would otherwise be ignored, and the model built with the default it was meant to
    override.

    Parameters
    ----------
    name
        Name of the model being checked.
    cfg
        Model configuration, as defined in models.yml.

    Raises
    ------
    ValueError
        If a required key is missing, or an unrecognised key is set.
    """
    from ml_peg.models.models import DISPERSION_KWARGS

    if missing := sorted(REQUIRED_MODEL_KEYS - set(cfg)):
        raise ValueError(f"'{name}' is missing required key(s): {', '.join(missing)}")

    if unknown := sorted(set(cfg) - MODEL_KEYS):
        raise ValueError(
            f"'{name}' sets key(s) that would be ignored: {', '.join(unknown)}. "
            f"Valid keys are: {', '.join(sorted(MODEL_KEYS))}"
        )

    dispersion_kwargs = cfg.get("dispersion_kwargs") or {}
    if unknown := sorted(set(dispersion_kwargs) - DISPERSION_KWARGS):
        raise ValueError(
            f"'{name}' sets dispersion_kwargs that would be ignored: "
            f"{', '.join(unknown)}. "
            f"Valid keys are: {', '.join(sorted(DISPERSION_KWARGS))}"
        )


@lru_cache(maxsize=1)
def _load_models_yaml(
    filepath: Path | str | None = None,
) -> dict[str, Any]:
    """
    Load and cache models.yml to prevent repeated expensive YAML parsing.

    Parameters
    ----------
    filepath
        Path to YAML file with models. Default is `models_file`.

    Returns
    -------
    dict[str, Any]
        Parsed models.yml registry.
    """
    filepath = filepath if filepath else models_file
    with open(filepath, encoding="utf8") as file:
        return yaml.safe_load(file) or {}


def load_model_configs(
    mlips: Iterable[str] | tuple[str, ...],
    filepath: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, str | None]]:
    """
    Load model configurations and level of theory metadata from models.yml.

    Parameters
    ----------
    mlips
        Iterable of model identifiers to load configurations for.
    filepath
        Path to YAML file with models. Default is `models_file`.

    Returns
    -------
    tuple[dict[str, Any], dict[str, str | None]]
        A tuple containing:
        - model_configs: Dictionary mapping model names to their configuration dicts
        - model_levels: Dictionary mapping model names to their level of
          theory (or ``None``)
    """
    filepath = filepath if filepath else models_file
    all_models = _load_models_yaml(filepath)

    model_levels: dict[str, str | None] = {}
    model_configs: dict[str, Any] = {}
    for mlip in mlips:
        cfg = deepcopy(all_models.get(mlip) or {})
        if not isinstance(cfg, dict):
            cfg = {}
        model_configs[mlip] = cfg
        model_levels[mlip] = cfg.get("level_of_theory")

    return model_configs, model_levels


def get_subset(
    all_models: dict[str, Any], models: None | str | Iterable = None
) -> dict[str, Any]:
    """
    Get a subset of models from a dictionary.

    Parameters
    ----------
    all_models
        Dictionary of models to extract a subset of.
    models
        Models to select fromm `all_models`. If `None`, all models will be selected.
        If an iterable, all models with matching keys will be selected. If a string,
        this will be treated as a comma-separated list.

    Returns
    -------
    dict[str, Any]
        Subset of `all_models` matching `models`, as described above.
    """
    if models is None:
        return all_models

    if isinstance(models, str):
        models = models.split(",")

    try:
        return {model: all_models[model] for model in models}
    except KeyError as err:
        for model in models:
            if model not in all_models:
                raise ValueError(
                    f"Model name '{model}' not recognised. Please check models.yml"
                ) from err


def load_models(
    models: None | str | Iterable = None,
    filepath: Path | str | None = None,
    run_mock: bool | None = None,
    mock_only: bool | None = None,
) -> dict[str, Any]:
    """
    Load models for use in calculations.

    Parameters
    ----------
    models
        Models to select from `filepath`. If `None`, all models will be selected.
        If an iterable, all models with matching keys will be selected. If a string,
        this will be treated as a comma-separated list.
    filepath
        Path to YAML file with models. Default is `models_file`.
    run_mock
            Whether to include mock model in the loaded models. Default is False.
    mock_only
            Whether to load only mock model, ignoring `models`. Default is False.

    Returns
    -------
    dict[str, Any]
        Loaded models from models.yml and/or loaded mock model.
    """
    from ml_peg.models.models import (
        DpaCalc,
        FairChemCalc,
        GenericASECalc,
        GraceCalc,
        MatterSimCalc,
        MockCalc,
        OrbCalc,
        SevenNetCalc,
        UPETCalc,
        VivaceCalc,
    )

    if run_mock is None:
        from ml_peg.models import run_mock
    if mock_only is None:
        from ml_peg.models import mock_only

    if mock_only and not run_mock:
        raise ValueError("Cannot set `mock_only` without `run_mock`")

    loaded_models = {"mock": MockCalc()} if run_mock else {}
    if mock_only:
        return loaded_models

    filepath = filepath if filepath else models_file
    all_models = _load_models_yaml(filepath)

    for name, cfg in get_subset(all_models, models).items():
        print(f"Loading model from {filepath}: {name}")

        # Copy, as the parsed registry is cached and shared between loaded models.
        cfg = deepcopy(cfg)
        validate_config(name, cfg)

        match cfg["class_name"]:
            case "DP":
                loaded_models[name] = DpaCalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "FAIRChemCalculator":
                kwargs = cfg.get("kwargs", {})
                model_name = kwargs.pop("model_name")
                task_name = kwargs.pop("task_name", "omat")
                overrides = kwargs.pop("overrides", {})
                loaded_models[name] = FairChemCalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    model_name=model_name,
                    task_name=task_name,
                    device=cfg.get("device", "cpu"),
                    overrides=overrides,
                    kwargs=kwargs,
                    default_dtype=cfg.get("overwrite_dtype", None),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "ORBCalculator":
                kwargs = cfg.get("kwargs", {})
                orb_name = kwargs.pop("name")
                loaded_models[name] = OrbCalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    name=orb_name,
                    device=cfg.get("device", "cpu"),
                    kwargs=kwargs,
                    default_dtype=cfg.get("overwrite_dtype", None),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "mace" | "mace_mp" | "mace_off" | "mace_omol" | "mace_polar":
                loaded_models[name] = GenericASECalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    device=cfg.get("device", "auto"),
                    default_dtype=cfg.get("overwrite_dtype", None),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "MatterSimCalculator":
                loaded_models[name] = MatterSimCalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    device=cfg.get("device", "cpu"),
                    default_dtype=cfg.get("overwrite_dtype", None),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "UPETCalculator":
                loaded_models[name] = UPETCalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    device=cfg.get("device", "cpu"),
                    default_dtype=cfg.get("overwrite_dtype", None),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "SevenNetCalculator":
                loaded_models[name] = SevenNetCalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    device=cfg.get("device", "cpu"),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "MLFFCalculator":
                loaded_models[name] = VivaceCalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    device=cfg.get("device", "auto"),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "grace_fm":
                loaded_models[name] = GraceCalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    device=cfg.get("device", "cpu"),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                    default_dtype=cfg.get("overwrite_dtype", None),
                )
            case _:
                loaded_models[name] = GenericASECalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    device=cfg.get("device", "auto"),
                    default_dtype=cfg.get("overwrite_dtype", None),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )

    return loaded_models


def get_model_names(
    models: None | Iterable = None,
    filepath: Path | str | None = None,
) -> list[str]:
    """
    Load models names for use in analysis.

    Parameters
    ----------
    models
        Models to select from `filepath`. If `None`, all models will be selected.
        If an iterable, all models with matching keys will be selected. If a string,
        this will be treated as a comma-separated list.
    filepath
        Path to YAML file with models. Default is `models_file`.

    Returns
    -------
    list[str]
        Loaded model names from `filepath`.
    """
    filepath = filepath if filepath else models_file
    all_models = _load_models_yaml(filepath)

    model_names = []
    for name in get_subset(all_models, models):
        model_names.append(name)

    return model_names
