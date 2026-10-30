"""Sphinx directive documenting every model defined in the model registry."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles
import yaml

if TYPE_CHECKING:
    from docutils.nodes import Node
    from sphinx.application import Sphinx

MODELS_FILE = Path(__file__).parents[3] / "ml_peg" / "models" / "models.yml"
NOTES_FILE = Path(__file__).parents[1] / "user_guide" / "model_notes.yml"


def registry_entries(models_file: Path = MODELS_FILE) -> dict[str, str]:
    """
    Read each model's configuration from the registry, as written.

    Read as text rather than parsed, so comments are kept. An entry runs until the
    next top-level line.

    Parameters
    ----------
    models_file
        Path to the YAML file defining models. Default is models.yml.

    Returns
    -------
    dict[str, str]
        Lines defining each model, keyed by model name.
    """
    entries: dict[str, list[str]] = {}
    name = None

    for line in models_file.read_text(encoding="utf8").split("\n"):
        if line and not line[0].isspace():
            name = line.rstrip()[:-1] if line.rstrip().endswith(":") else None
            if name is not None:
                entries[name] = [line]
        elif name is not None:
            entries[name].append(line)

    return {name: "\n".join(lines).rstrip() for name, lines in entries.items()}


def model_families(entries: dict[str, str], notes: dict[str, Any]) -> dict[str, list]:
    """
    Group models into the families they are documented under.

    Parameters
    ----------
    entries
        Lines defining each model, keyed by model name.
    notes
        Family and model descriptions, as defined in model_notes.yml.

    Returns
    -------
    dict[str, list]
        Model names, keyed by family, in the order families are described.

    Raises
    ------
    ValueError
        If a model's `class_name` belongs to no documented family.
    """
    registry = yaml.safe_load(MODELS_FILE.read_text(encoding="utf8"))
    families: dict[str, list] = {family: [] for family in notes["families"]}

    for name in entries:
        class_name = registry[name]["class_name"]
        family = next(
            (
                family
                for family, described in notes["families"].items()
                if class_name in described["class_names"]
            ),
            None,
        )
        if family is None:
            raise ValueError(
                f"'{name}' uses class_name '{class_name}', which belongs to no family "
                f"in {NOTES_FILE.name}. Add it to an existing family, or a new one."
            )
        families[family].append(name)

    return {family: names for family, names in families.items() if names}


def build_rst(entries: dict[str, str], notes: dict[str, Any]) -> str:
    """
    Build the documentation for every model in the registry.

    Parameters
    ----------
    entries
        Lines defining each model, keyed by model name.
    notes
        Family and model descriptions, as defined in model_notes.yml.

    Returns
    -------
    str
        reStructuredText documenting each model, grouped by family.
    """
    model_notes = notes.get("models") or {}
    lines: list[str] = []

    for family, names in model_families(entries, notes).items():
        lines += [family, "=" * len(family), ""]
        if family_note := notes["families"][family].get("notes"):
            lines += [family_note.strip(), ""]

        # A family of one model named after it needs no section of its own
        named_after_family = names == [family]

        for name in names:
            if not named_after_family:
                lines += [name, "-" * len(name), ""]
            if model_note := model_notes.get(name):
                lines += [model_note.strip(), ""]
            lines += [".. code-block:: yaml", ""]
            lines += [
                f"   {line}" if line.strip() else ""
                for line in entries[name].split("\n")
            ]
            lines += [""]

    return "\n".join(lines)


class ModelRegistry(SphinxDirective):
    """Document every model in models.yml, grouped by family."""

    def run(self) -> list[Node]:
        """
        Render the model registry.

        Returns
        -------
        list[Node]
            Documentation for each model in the registry.
        """
        # Rebuild when the registry or its descriptions change
        self.env.note_dependency(str(MODELS_FILE))
        self.env.note_dependency(str(NOTES_FILE))

        notes = yaml.safe_load(NOTES_FILE.read_text(encoding="utf8"))
        text = build_rst(registry_entries(), notes)

        # Parsed with titles, as the text defines a section per model
        parsed = nodes.section()
        parsed.document = self.state.document
        nested_parse_with_titles(
            self.state, StringList(text.split("\n"), source=str(MODELS_FILE)), parsed
        )

        return parsed.children


def setup(app: Sphinx) -> dict[str, Any]:
    """
    Register the directive with Sphinx.

    Parameters
    ----------
    app
        Sphinx application.

    Returns
    -------
    dict[str, Any]
        Extension metadata.
    """
    app.add_directive("model-registry", ModelRegistry)

    return {"parallel_read_safe": True, "parallel_write_safe": True}
