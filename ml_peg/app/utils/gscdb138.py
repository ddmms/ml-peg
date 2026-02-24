"""Shared helpers for GSCDB138 Dash applications."""

from __future__ import annotations

from pathlib import Path

from dash.html import Div

from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column, struct_from_scatter
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)


class GSCDB138BenchmarkApp(BaseApp):
    """
    Base Dash app for GSCDB138 datasets across all categories.

    Parameters
    ----------
    name
        Benchmark name.
    description
        Benchmark description shown in the UI.
    data_path
        Path to data for application.
    datasets
        Dataset identifiers for this category.
    docs_url
        Optional documentation link.
    """

    def __init__(
        self,
        name: str,
        description: str,
        data_path: Path,
        datasets: list[str],
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
            Path to data for application.
        datasets
            Dataset identifiers for this category.
        docs_url
            Optional documentation link.
        """
        self.data_path = data_path
        self.datasets = datasets

        super().__init__(
            name=name,
            description=description,
            table_path=self.data_path / "gscdb138_metrics_table.json",
            docs_url=docs_url,
            extra_components=[
                Div(id=f"{name}-figure-placeholder"),
                Div(id=f"{name}-struct-placeholder"),
            ],
        )

    def get_system_paths(self, dataset: str) -> list[Path]:
        """
        Get list of paths to system from the first available model for a dataset.

        Parameters
        ----------
        dataset
            Dataset to get system paths for.

        Returns
        -------
        list[Path]
            List of systems in the dataset.
        """
        for model_name in MODELS:
            model_dir = self.data_path / model_name
            if model_dir.exists():
                system_paths = sorted(model_dir.glob(f"{dataset}_*.xyz"))
                if system_paths:
                    # Get path starting from assets directory
                    return [
                        Path("assets")
                        / system_path.relative_to(self.data_path.parent.parent)
                        for system_path in system_paths
                    ]
        return []

    def register_callbacks(self) -> None:
        """Register callbacks to link metrics, parity plots, and structures."""
        # Load scatter plot for each dataset
        scatter_plots = {
            f"{dataset} MAE": read_plot(
                filename=self.data_path / f"figure_gscdb138_{dataset}.json",
                id=f"{self.name}-{dataset}-figure",
            )
            for dataset in self.datasets
        }

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{self.name}-figure-placeholder",
            column_to_plot=scatter_plots,
        )

        for dataset in self.datasets:
            struct_from_scatter(
                scatter_id=f"{self.name}-{dataset}-figure",
                struct_id=f"{self.name}-struct-placeholder",
                structs=self.get_system_paths(dataset),
                mode="struct",
            )
