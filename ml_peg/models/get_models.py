"""Get MLIPs to be used for calculations or analysis."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import yaml

from ml_peg.models import MODELS_ROOT


def load_model_configs(
    mlips: Iterable[str] | tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, str | None]]:
    """
    Load model configurations and level of theory metadata from models.yml.

    Parameters
    ----------
    mlips
        Iterable of model identifiers to load configurations for.

    Returns
    -------
    tuple[dict[str, Any], dict[str, str | None]]
        A tuple containing:
        - model_configs: Dictionary mapping model names to their configuration dicts
        - model_levels: Dictionary mapping model names to their level of
          theory (or ``None``)
    """
    with open(MODELS_ROOT / "models.yml", encoding="utf8") as model_file:
        all_models = yaml.safe_load(model_file) or {}

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


def load_models(models: None | str | Iterable = None) -> dict[str, Any]:
    """
    Load models for use in calculations.

    Parameters
    ----------
    models
        Models to select from models.yml. If `None`, all models will be selected.
        If an iterable, all models with matching keys will be selected. If a string,
        this will be treated as a comma-separated list.

    Returns
    -------
    dict[str, Any]
        Loaded models from models.yml.
    """
    from ml_peg.models.models import FairChemCalc, GenericASECalc, OrbCalc, PetMadCalc

    loaded_models = {}

    # Load models from registry YAML: models.yml
    with open(MODELS_ROOT / "models.yml") as file:
        all_models = yaml.safe_load(file)

    for name, cfg in get_subset(all_models, models).items():
        print(f"Loading model from models.yml: {name}")

        if cfg["class_name"] == "FAIRChemCalculator":
            kwargs = cfg.get("kwargs", {})
            loaded_models[name] = FairChemCalc(
                model_name=kwargs["model_name"],
                task_name=kwargs.get("task_name", "omat"),
                device=cfg.get("device", "cpu"),
                overrides=kwargs.get("overrides", {}),
                trained_on_d3=cfg.get("trained_on_d3", False),
                d3_kwargs=cfg.get("d3_kwargs", {}),
            )
        elif cfg["class_name"] == "OrbCalc":
            kwargs = cfg.get("kwargs", {})
            loaded_models[name] = OrbCalc(
                name=kwargs["name"],
                device=cfg.get("device", "cpu"),
                default_dtype=cfg.get("default_dtype", "float32"),
                trained_on_d3=cfg.get("trained_on_d3", False),
                d3_kwargs=cfg.get("d3_kwargs", {}),
            )
        elif cfg["class_name"] == "mace_mp":
            loaded_models[name] = GenericASECalc(
                module=cfg["module"],
                class_name=cfg["class_name"],
                device=cfg.get("device", "auto"),
                default_dtype=cfg.get("default_dtype", "float32"),
                kwargs=cfg.get("kwargs", {}),
                trained_on_d3=cfg.get("trained_on_d3", False),
                d3_kwargs=cfg.get("d3_kwargs", {}),
            )
        elif cfg["class_name"] == "PETMADCalculator":
            loaded_models[name] = PetMadCalc(
                module=cfg["module"],
                class_name=cfg["class_name"],
                device=cfg.get("device", "cpu"),
                default_dtype=cfg.get("default_dtype", "float32"),
                kwargs=cfg.get("kwargs", {}),
                trained_on_d3=cfg.get("trained_on_d3", False),
                d3_kwargs=cfg.get("d3_kwargs", {}),
            )
        else:
            loaded_models[name] = GenericASECalc(
                module=cfg["module"],
                class_name=cfg["class_name"],
                device=cfg.get("device", "auto"),
                kwargs=cfg.get("kwargs", {}),
                trained_on_d3=cfg.get("trained_on_d3", False),
                d3_kwargs=cfg.get("d3_kwargs", {}),
            )

    return loaded_models


def get_model_names(models: None | Iterable = None) -> list[str]:
    """
    Load models names for use in analysis.

    Parameters
    ----------
    models
        Models to select from models.yml. If `None`, all models will be selected.
        If an iterable, all models with matching keys will be selected. If a string,
        this will be treated as a comma-separated list.

    Returns
    -------
    list[str]
        Loaded model names from models.yml.
    """
    # Load models from registry YAML: models.yml
    with open(MODELS_ROOT / "models.yml") as file:
        all_models = yaml.safe_load(file)

    model_names = []
    for name in get_subset(all_models, models):
        model_names.append(name)

    return model_names
