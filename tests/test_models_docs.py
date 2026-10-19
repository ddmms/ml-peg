"""Tests that the documented model list matches the model registry."""

from __future__ import annotations

from pathlib import Path
import re
from textwrap import dedent

from yaml import safe_load

MODELS_FILE = Path(__file__).parents[1] / "ml_peg" / "models" / "models.yml"
MODELS_DOC = Path(__file__).parents[1] / "docs" / "source" / "user_guide" / "models.rst"


def documented_models() -> dict:
    """
    Load every model configuration documented in the user guide.

    Returns
    -------
    dict
        Model configurations parsed from the YAML code blocks in models.rst.
    """
    blocks = re.findall(
        r"\.\. code-block:: yaml\n\n((?:[ \t]+.*\n|\n)+)",
        MODELS_DOC.read_text(encoding="utf8"),
    )

    documented = {}
    for block in blocks:
        config = safe_load(dedent(block))
        if isinstance(config, dict):
            documented |= config

    return documented


def test_documented_models_match_registry():
    """Every model is documented with the configuration it is loaded with."""
    registry = safe_load(MODELS_FILE.read_text(encoding="utf8"))
    documented = documented_models()

    assert sorted(documented) == sorted(registry), (
        "models.rst and models.yml define different models. Update "
        "docs/source/user_guide/models.rst to match the registry."
    )

    for name, config in registry.items():
        assert documented[name] == config, (
            f"Documented configuration for '{name}' does not match models.yml"
        )
