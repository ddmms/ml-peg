"""Shared sampling and completion criteria for electrolyte NPT runs."""

from __future__ import annotations

from ase import units

NUM_MD_STEPS = 500_000
TIMESTEP = 1 * units.fs
LOG_INTERVAL = 100
LOG_INTERVAL_PS = LOG_INTERVAL * TIMESTEP / units.fs / 1000
EQUILIB_TIME_PS = 300
EXPECTED_FRAMES = 1 + NUM_MD_STEPS // LOG_INTERVAL
