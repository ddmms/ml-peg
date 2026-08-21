"""Utilities for the thermodynamic properties app."""

from __future__ import annotations

from pathlib import Path

from dash.dcc import Graph
from dash.html import Div

from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot


def load_property_plots(
    data_path: Path,
    benchmark_name: str,
    properties: tuple[str, ...],
) -> dict[str, Graph]:
    """
    Load plots for thermodynamic properties.

    Parameters
    ----------
    data_path
        Path containing the benchmark plot files.
    benchmark_name
        Name of the benchmark used to construct Dash component IDs.
    properties
        Names of the thermodynamic properties to load.

    Returns
    -------
    dict[str, Graph]
        Mapping from property names to Dash graph components.
    """
    return {
        name: read_plot(
            data_path / f"figure_{name}.json",
            id=f"{benchmark_name}-{name}-figure",
        )
        for name in properties
    }


def map_metrics_to_plots(
    plots: dict[str, Graph],
    metrics: tuple[str, ...] = ("MAE", "MAZE"),
) -> dict[str, Graph]:
    """
    Map metric table columns to property plots.

    Parameters
    ----------
    plots
        Mapping from property names to Dash graph components.
    metrics
        Metric names associated with each property.

    Returns
    -------
    dict[str, Graph]
        Mapping from metric table column names to Dash graph components.
    """
    return {
        f"{property_name}_{metric}": plot
        for property_name, plot in plots.items()
        for metric in metrics
    }


def get_structures(
    data_path: Path,
    model: str,
    asset_path: str,
) -> list[str]:
    """
    Get structure asset paths for a model.

    Parameters
    ----------
    data_path
        Path containing benchmark data.
    model
        Name of the model whose structures should be loaded.
    asset_path
        URL path to the benchmark structure assets.

    Returns
    -------
    list[str]
        Structure asset paths ordered by filename.
    """
    model_dir = data_path / model

    if not model_dir.exists():
        return []

    return [
        f"{asset_path}/{model}/{file.stem}.xyz"
        for file in sorted(model_dir.glob("*.xyz"))
    ]


def thermodynamic_properties_app_factory(
    benchmark_name,
    data_path,
    benchmark_path,
    docs_url,
):
    """
    Create a thermodynamic properties benchmark app.

    Parameters
    ----------
    benchmark_name
        Display name of the benchmark.
    data_path
        Path to the analysed benchmark data.
    benchmark_path
        Benchmark path used to locate structure assets.
    docs_url
        URL of the benchmark documentation.

    Returns
    -------
    BaseApp
        Configured thermodynamic properties benchmark app.
    """

    class ThermodynamicPropertiesApp(BaseApp):
        """
        Thermodynamic properties benchmark app.

        Provides the benchmark layout and callback registration for displaying
        thermodynamic property metrics, parity plots, and molecular structures.
        """

        def register_callbacks(self) -> None:
            """
            Register callbacks for the thermodynamic properties app.

            Returns
            -------
            None
                The callbacks are registered on the app instance.
            """
            plots = {
                property_name: read_plot(
                    data_path / f"figure_{figure_name}.json",
                    id=f"{benchmark_name}-{figure_name}-figure",
                )
                for property_name, figure_name in {
                    "density_MAE": "density",
                    "cp_MAE": "cp",
                    "evaporation_enthalpy_MAE": "evaporation_enthalpy",
                    "compressibility_MAE": "compressibility",
                    "alpha_MAE": "alpha",
                }.items()
            }

            model = plots["density_MAE"].figure.data[0].name
            model_dir = data_path / model

            structs = [
                (f"/assets/molecular_dynamics/{benchmark_path}/{model}/{f.stem}.xyz")
                for f in sorted(model_dir.glob("*-liq.xyz"))
            ]

            plot_from_table_column(
                table_id=self.table_id,
                plot_id=f"{benchmark_name}-figure-placeholder",
                column_to_plot=plots,
            )

            struct_from_scatter(
                scatter_id=f"{benchmark_name}-density-figure",
                struct_id=f"{benchmark_name}-struct-placeholder",
                structs=structs,
                mode="struct",
            )

    return ThermodynamicPropertiesApp(
        name=benchmark_name,
        description=(
            "Performance in predicting thermodynamic properties of organic liquids."
        ),
        docs_url=docs_url,
        table_path=data_path / "thermodynamic_properties_metrics_table.json",
        extra_components=[
            Div(id=f"{benchmark_name}-figure-placeholder"),
            Div(id=f"{benchmark_name}-struct-placeholder"),
        ],
        info_path=data_path / "info.json",
        framework_ids="ml_peg",
    )
