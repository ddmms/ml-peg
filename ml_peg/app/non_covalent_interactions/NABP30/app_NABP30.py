"""Run NABP30 app."""

from __future__ import annotations

from dash.dcc import Graph
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_multi_scatters,
)
from ml_peg.app.utils.load import read_json, read_plot

BENCHMARK_NAME = "NABP30"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "non_covalent_interactions.html#nabp30"
)
DATA_PATH = APP_ROOT / "data" / "non_covalent_interactions" / "NABP30"
INFO_PATH = DATA_PATH / "info.json"
ASSETS_PATH = "/assets/non_covalent_interactions/NABP30"


def get_structs(scatter: Graph, labels: list[str], relaxed: bool) -> list[list[str]]:
    """
    Get the structure to show for each point of a parity plot.

    Parameters
    ----------
    scatter
        Parity plot with one trace per model, and the parity line.
    labels
        Base pair labels, in the order of the points of each trace.
    relaxed
        Whether to show the base pair relaxed by each model, or the input structure,
        which is the same for all models.

    Returns
    -------
    list[list[str]]
        Structure files for each trace, in the order of its points. Empty for traces
        without structures (the parity line), which clears the viewer.
    """
    structs = []
    for trace in (scatter.figure or {}).get("data", []):
        model_name = str(trace.get("name"))
        struct_dir = model_name if relaxed else "mock"
        structs.append(
            [f"{ASSETS_PATH}/{struct_dir}/{label}.xyz" for label in labels]
            if (DATA_PATH / model_name).is_dir()
            else []
        )
    return structs


class NABP30App(BaseApp):
    """NABP30 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_interaction = read_plot(
            DATA_PATH / "figure_interaction_energies.json",
            id=f"{BENCHMARK_NAME}-interaction-figure",
        )
        scatter_binding = read_plot(
            DATA_PATH / "figure_binding_energies.json",
            id=f"{BENCHMARK_NAME}-binding-figure",
        )
        struct_id = f"{BENCHMARK_NAME}-struct-placeholder"
        labels = read_json(INFO_PATH)["filenames"]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Interaction energy MAE": scatter_interaction,
                "Binding energy MAE": scatter_binding,
            },
        )

        # Interaction energies: input structure. Binding energies: base pair relaxed
        # by the model of the selected point. Showing another plot clears the viewer.
        struct_from_multi_scatters(
            scatter_id=f"{BENCHMARK_NAME}-interaction-figure",
            struct_id=struct_id,
            structs=get_structs(scatter_interaction, labels, relaxed=False),
            mode="struct",
        )
        struct_from_multi_scatters(
            scatter_id=f"{BENCHMARK_NAME}-binding-figure",
            struct_id=struct_id,
            structs=get_structs(scatter_binding, labels, relaxed=True),
            mode="struct",
        )


def get_app() -> NABP30App:
    """
    Get NABP30 benchmark app layout and callback registration.

    Returns
    -------
    NABP30App
        Benchmark layout and callback registration.
    """
    return NABP30App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting interaction and binding energies of 30 "
            "hydrogen-bonded nucleic acid base pairs: interaction energies at the "
            "reference geometries, and binding energies with all structures relaxed "
            "by the model. Reference data from MP2/CBS calculations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "nabp30_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
