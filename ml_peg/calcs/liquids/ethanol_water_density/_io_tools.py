"""i/o tools for calculations."""

from __future__ import annotations

from collections.abc import Iterable
import csv
from pathlib import Path


def write_density_timeseries_checkpointed(
    ts_path: Path,
    rho_series: Iterable[float],
    *,
    min_match_fraction: float = 0.8,
) -> None:
    """
    Write density_timeseries.csv with checkpoint validation.

    Behavior
    --------
    If file exists:
        - read existing values
        - verify >= min_match_fraction already match rho_series
        - overwrite anyway
        - raise AssertionError if insufficient match

    If file does not exist:
        - just write

    Helps detect broken resume logic while still allowing overwrite.
    """
    rho_series = list(rho_series)

    # -------------------------
    # 1) Validate existing file
    # -------------------------
    if ts_path.exists():
        old_vals: list[float] = []

        try:
            with ts_path.open() as f:
                r = csv.reader(f)
                next(r, None)  # header
                for row in r:
                    if len(row) >= 2:
                        old_vals.append(float(row[1]))
        except Exception:
            old_vals = []

        if old_vals:
            n = min(len(old_vals), len(rho_series))

            matches = sum(abs(old_vals[i] - rho_series[i]) < 1e-6 for i in range(n))

            frac = matches / n if n else 0.0

            if frac < min_match_fraction:
                raise AssertionError(
                    f"{ts_path}: only {frac:.1%} of checkpoint values match "
                    f"(expected ≥ {min_match_fraction:.0%}). "
                    "run_one_case resume likely broken."
                )

    # -------------------------
    # 2) Always rewrite file
    # -------------------------
    with ts_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "rho_g_cm3"])
        for i, rho in enumerate(rho_series):
            w.writerow([i, f"{rho:.8f}"])


class DensityTimeseriesLogger:
    """
    Streaming CSV logger for density time series.

    - deletes existing file on start (optional)
    - writes header once
    - append rows as simulation runs
    - flushes every write (crash-safe)
    - usable as context manager
    """

    def __init__(self, path: Path, *, overwrite: bool = True):
        self.path = Path(path)
        self.overwrite = overwrite
        self._f = None
        self._writer = None
        self._step = 0

    # ---------------------
    # context manager API
    # ---------------------
    def __enter__(self):
        """Open the file."""
        if self.overwrite and self.path.exists():
            self.path.unlink()

        self._f = self.path.open("w", newline="")
        self._writer = csv.writer(self._f)

        self._writer.writerow(["step", "rho_g_cm3"])
        self._f.flush()

        return self

    def __exit__(self, exc_type, exc, tb):
        """Close the file."""
        if self._f:
            self._f.close()

    # ---------------------
    # logging
    # ---------------------
    def write(self, rho: float):
        """Write one density value."""
        self._writer.writerow([self._step, f"{rho:.8f}"])
        self._f.flush()  # critical for crash safety
        self._step += 1
