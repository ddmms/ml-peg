"""Run analysis for testing framework."""

from __future__ import annotations

from pathlib import Path

ANALYSIS_ROOT = Path(__file__).parent

# Whether saved tables and plots should be updated in place, preserving results for
# models that are not part of the current analysis run. Set from `--update`.
update_results = False
