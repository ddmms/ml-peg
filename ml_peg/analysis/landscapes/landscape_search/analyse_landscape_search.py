"""Analyse the Landscape17 landscape search results and emit metrics.

Reads the raw per-model outputs written by ``calc_landscape_search.py`` and
derives the quantities declared in ``metrics.yml``. Derived properties are
computed here rather than in the calculation, so the raw outputs stay reusable
if a metric is later redefined.

Distributions are summarised by percentile rather than by the fraction crossing
a threshold; see the calculation module's docstring for why.

Status
------
Skeleton. Not yet implemented.
"""

from __future__ import annotations

raise NotImplementedError("Skeleton only.")
