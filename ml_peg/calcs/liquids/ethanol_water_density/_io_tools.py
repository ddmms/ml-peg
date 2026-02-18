"""I/O tools for calculations."""

from __future__ import annotations

from collections.abc import Iterable
import csv
from pathlib import Path


def write_density_timeseries_checkpointed(
    ts_path: Path,
    rho_series: Iterable[float],
    *,
    min_match_fraction: float = 0.8,
    do_not_raise: bool = False,
) -> None:
    """
    Write ``density_timeseries.csv`` with checkpoint validation.

    Parameters
    ----------
    ts_path : pathlib.Path
        Output CSV path.
    rho_series : collections.abc.Iterable[float]
        Density samples in g/cm^3.
    min_match_fraction : float, optional
        Required fraction of matching values versus existing checkpoint.
    do_not_raise : bool, optional
        If ``True``, skip assertion failures when checkpoint mismatches.

    Returns
    -------
    None
        This function writes the file in-place.
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

            if frac < min_match_fraction and not do_not_raise:
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

    Parameters
    ----------
    path : pathlib.Path
        Output CSV file path.
    overwrite : bool, optional
        Whether to delete pre-existing output when opening.
    """

    def __init__(self, path: Path, *, overwrite: bool = True):
        """
        Initialize the logger.

        Parameters
        ----------
        path : pathlib.Path
            Path to CSV output file.
        overwrite : bool, optional
            If ``True``, delete existing file when opening.
        """
        self.path = Path(path)
        self.overwrite = overwrite
        self._f = None
        self._writer = None
        self._step = 0

    def __enter__(self):
        """
        Open the output file and return logger instance.

        Returns
        -------
        DensityTimeseriesLogger
            Logger ready to write rows.
        """
        if self.overwrite and self.path.exists():
            mode = "w"
            self.path.unlink()
        elif not self.path.exists():
            mode = "w"
        else:
            mode = "a"
            # If appending, recover last step index
            try:
                with self.path.open("r", newline="") as f:
                    last_step = -1
                    for row in csv.reader(f):
                        if not row or row[0] == "step":
                            continue
                        try:
                            last_step = int(row[0])
                        except ValueError:
                            continue
                    self._step = last_step + 1
            except Exception:
                # If file is corrupted or empty, just continue from 0
                self._step = 0

        self._f = self.path.open(mode, newline="")
        self._writer = csv.writer(self._f)

        # Only write header if creating new file
        if mode == "w":
            self._writer.writerow(["step", "rho_g_cm3"])
            self._f.flush()

        return self

    def __exit__(self, exc_type, exc, tb):
        """
        Close the output file.

        Parameters
        ----------
        exc_type : type | None
            Exception type, if raised inside context.
        exc : BaseException | None
            Exception instance, if raised inside context.
        tb : traceback | None
            Traceback, if raised inside context.

        Returns
        -------
        None
            This method performs cleanup only.
        """
        if self._f:
            self._f.close()

    def write(self, rho: float):
        """
        Write one density value to the CSV file.

        Parameters
        ----------
        rho : float
            Density in g/cm^3 for the current sample.

        Returns
        -------
        None
            This method appends one row to disk.
        """
        self._writer.writerow([self._step, f"{rho:.8f}"])
        self._f.flush()
        self._step += 1
