"""Read update times recorded with the application and its benchmark assets."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ml_peg.app import APP_ROOT


def site_updated_at(app_root: Path = APP_ROOT) -> datetime | None:
    """
    Find the latest recorded application build or benchmark-data update.

    Parameters
    ----------
    app_root
        Application directory containing the image build stamp and data assets.

    Returns
    -------
    datetime or None
        Latest UTC timestamp, or None when no update information is available.
        Reading it never substitutes the current time or changes the files.
    """
    timestamps = []
    build_stamp = app_root / "_build_time.txt"
    if build_stamp.is_file():
        timestamps.append(
            datetime.fromisoformat(build_stamp.read_text().strip()).timestamp()
        )
    # Benchmark tables and figures live at data/<category>/<benchmark>/*.json.
    # Exclude UI assets and unrelated top-level metadata. Stat only, never read
    # the potentially large plot files or scan model trajectories.
    data_root = app_root / "data"
    for path in data_root.glob("*/*/*.json"):
        if path.relative_to(data_root).parts[0] != "ui" and path.is_file():
            timestamps.append(path.stat().st_mtime)
    return (
        datetime.fromtimestamp(max(timestamps), tz=timezone.utc) if timestamps else None
    )
