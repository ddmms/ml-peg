"""Shared helpers for the small-cluster force benchmark apps."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

from dash.html import Div
import numpy as np

from ml_peg.analysis.utils.analyse_cluster_forces import (
    CLUSTER_SIZES,
    figure_filename,
    frame_groups_filename,
    metric_name,
    structs_filename,
)
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    clear_from_table_cell,
    plot_from_table_cell,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_density_plot_for_model
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)


class FrameGroups(Sequence):
    """
    Frames shown for each density point, stored compactly in memory.

    Each model has tens of thousands of density points per cluster size, so the
    groups are kept as flat integer arrays rather than nested lists.

    Parameters
    ----------
    path
        JSON file containing a list of frame indices for each density point.
    """

    def __init__(self, path: Path) -> None:
        """
        Load the frame groups.

        Parameters
        ----------
        path
            JSON file containing a list of frame indices for each density point.
        """
        groups = json.loads(path.read_text())
        self._offsets = np.cumsum([0, *map(len, groups)])
        self._frames = np.fromiter(
            (frame for group in groups for frame in group),
            dtype=np.int32,
            count=int(self._offsets[-1]),
        )

    def __len__(self) -> int:
        """
        Get the number of density points.

        Returns
        -------
        int
            Number of density points.
        """
        return len(self._offsets) - 1

    def __getitem__(self, index: int) -> list[int]:
        """
        Get the frames shown for one density point.

        Parameters
        ----------
        index
            Index of the density point.

        Returns
        -------
        list[int]
            Frame indices.
        """
        if not 0 <= index < len(self):
            raise IndexError(index)
        start, end = self._offsets[index], self._offsets[index + 1]
        return self._frames[start:end].tolist()


class ClusterForcesApp(BaseApp):
    """
    Cluster force benchmark app, with one density plot per cluster size.

    Parameters
    ----------
    name
        Benchmark name.
    description
        Benchmark description shown in the UI.
    data_path
        Path to app data written by the analysis.
    docs_url
        Optional documentation link.
    """

    def __init__(
        self,
        name: str,
        description: str,
        data_path: Path,
        docs_url: str | None = None,
    ) -> None:
        """
        Initialise application info.

        Parameters
        ----------
        name
            Benchmark name.
        description
            Benchmark description shown in the UI.
        data_path
            Path to app data written by the analysis.
        docs_url
            Optional documentation link.
        """
        self.data_path = data_path
        super().__init__(
            name=name,
            description=description,
            table_path=data_path / "cluster_forces_metrics_table.json",
            docs_url=docs_url,
            extra_components=[
                Div(id=f"{name}-figure-placeholder"),
                Div(id=f"{name}-struct-placeholder"),
            ],
            info_path=data_path / "info.json",
        )

    def register_callbacks(self) -> None:
        """Register callbacks linking table cells, density plots and structures."""
        assets_prefix = f"/assets/{self.data_path.relative_to(APP_ROOT / 'data')}"
        struct_id = f"{self.name}-struct-placeholder"
        density_plots: dict[str, dict] = {model: {} for model in MODELS}

        for size in CLUSTER_SIZES:
            figure_path = self.data_path / figure_filename(size)
            if not figure_path.exists():
                continue
            for model in MODELS:
                graph = read_density_plot_for_model(
                    filename=figure_path,
                    model=model,
                    id=f"{self.name}-{model}-{size}atoms-density",
                )
                if graph is not None:
                    density_plots[model][metric_name(size)] = graph

            # All models share one file of clusters per size, and each plotted
            # point shows a group of frames from it
            structs = f"{assets_prefix}/{structs_filename(size)}"
            for model in MODELS:
                groups_path = self.data_path / model / frame_groups_filename(size)
                if not groups_path.exists():
                    continue
                struct_from_scatter(
                    scatter_id=f"{self.name}-{model}-{size}atoms-density",
                    struct_id=struct_id,
                    structs=structs,
                    mode="traj",
                    frame_groups=FrameGroups(groups_path),
                )

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{self.name}-figure-placeholder",
            cell_to_plot={
                model: plots for model, plots in density_plots.items() if plots
            },
        )
        clear_from_table_cell(self.table_id, [struct_id])
