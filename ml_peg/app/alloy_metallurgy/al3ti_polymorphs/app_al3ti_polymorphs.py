"""Run Al3Ti polymorph app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names()
BENCHMARK_NAME = "Al3Ti polymorphs"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/alloy_metallurgy.html#al3ti-polymorphs"
DATA_PATH = APP_ROOT / "data" / "alloy_metallurgy" / "al3ti_polymorphs"
INFO_PATH = DATA_PATH / "info.json"

POLYMORPHS = ("D022", "D023", "L12")


def structure_urls_by_point(figure: dict | None) -> dict[str, str]:
    """
    Label each model's plot points and map them to available structures.

    Parameters
    ----------
    figure
        Formation-energy plot with model names on its traces.

    Returns
    -------
    dict[str, str]
        Structure URLs keyed by model and polymorph; missing files are omitted.
    """
    structures = {}
    for trace in (figure or {}).get("data", []):
        model_name = trace.get("name")
        if model_name not in MODELS:
            continue
        trace["customdata"] = [f"{model_name}:{name}" for name in POLYMORPHS]
        for name, label in zip(POLYMORPHS, trace["customdata"], strict=True):
            path = DATA_PATH / model_name / f"{name}.xyz"
            if path.is_file():
                structures[label] = (
                    f"/assets/alloy_metallurgy/al3ti_polymorphs/{model_name}/{name}.xyz"
                )
    return structures


class Al3TiPolymorphApp(BaseApp):
    """Al3Ti Polymorph benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        formation_scatter = read_plot(
            DATA_PATH / "figure_formation_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        structs = structure_urls_by_point(formation_scatter.figure)

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Formation Energy MAE": formation_scatter,
                "Relative Stability Ordering Accuracy": formation_scatter,
            },
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> Al3TiPolymorphApp:
    """
    Get Al3Ti Polymorph benchmark app layout and callback registration.

    Returns
    -------
    Al3TiPolymorphApp
        Benchmark layout and callback registration.
    """
    return Al3TiPolymorphApp(
        name=BENCHMARK_NAME,
        description=(
            "Formation energies, relaxed cell geometry, and stability ordering "
            "for Al3Ti polymorphs."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "al3ti_metrics.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
