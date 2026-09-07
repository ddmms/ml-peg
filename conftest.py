"""Load the benchmark plugin when testing an uninstalled source checkout."""

from __future__ import annotations

# Match the entry-point name so an installed plugin is not registered twice.
pytest_plugins = ["ml_peg.pytest_plugin"]
