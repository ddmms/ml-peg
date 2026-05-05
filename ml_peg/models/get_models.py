"""Get MLIPs to be used for calculations or analysis."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ml_peg.models import models_file


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

    Returns
    -------
    dict[str, Any]
        Loaded models from models.yml.
    """
    from ml_peg.models.models import FairChemCalc, GenericASECalc, OrbCalc, PetMadCalc

    loaded_models = {}

    filepath = filepath if filepath else models_file
    all_models = _load_models_yaml(filepath)

    for name, cfg in get_subset(all_models, models).items():
        print(f"Loading model from {filepath}: {name}")

        match cfg["class_name"]:
            case "FAIRChemCalculator":
                kwargs = cfg.get("kwargs", {})
                loaded_models[name] = FairChemCalc(
                    model_name=kwargs["model_name"],
                    task_name=kwargs.get("task_name", "omat"),
                    device=cfg.get("device", "cpu"),
                    overrides=kwargs.get("overrides", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case "OrbCalc":
                kwargs = cfg.get("kwargs", {})
                loaded_models[name] = OrbCalc(
                    name=kwargs["name"],
                    device=cfg.get("device", "cpu"),
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
            case "PETMADCalculator":
                loaded_models[name] = PetMadCalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    device=cfg.get("device", "cpu"),
                    default_dtype=cfg.get("overwrite_dtype", None),
                    kwargs=cfg.get("kwargs", {}),
                    trained_on_dispersion=cfg.get("trained_on_dispersion", False),
                    dispersion_kwargs=cfg.get("dispersion_kwargs", {}),
                )
            case _:
                loaded_models[name] = GenericASECalc(
                    module=cfg["module"],
                    class_name=cfg["class_name"],
                    device=cfg.get("device", "auto"),
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
