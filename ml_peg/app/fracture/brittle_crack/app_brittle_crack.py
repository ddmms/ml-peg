"""Run app for the brittle_crack benchmark."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell, struct_from_scatter
from ml_peg.app.utils.load import read_plot
from ml_peg.calcs.fracture.brittle_crack import crack_tools as ct
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Brittle crack"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/fracture.html#brittle-crack"
)
DATA_PATH = APP_ROOT / "data" / "fracture" / "brittle_crack"
INFO_PATH = DATA_PATH / "info.json"
CASES = ct.load_cases()


def trace_structs(model: str, name: str) -> dict[str, str]:
    """
    Map a model's label to its traced structures, if the trace file exists.

    Parameters
    ----------
    model
        Model name.
    name
        Case name.

    Returns
    -------
    dict[str, str]
        ``{model: asset path}``, or empty when the trace failed (no file).
    """
    if not (DATA_PATH / name / model / f"{model}-trace.extxyz").exists():
        return {}
    return {
        model: f"/assets/fracture/brittle_crack/{name}/{model}/{model}-trace.extxyz"
    }


class BrittleCrackApp(BaseApp):
    """Brittle-crack benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks: S-curve plot per table cell, structures per point."""
        plots: dict[str, dict] = {}
        for model in MODELS:
            model_plots = {}
            for name, case in CASES.items():
                path = DATA_PATH / f"figure_{model}_{name}.json"
                if not path.exists():
                    continue
                for term in ("K", "E", "D"):
                    model_plots[f"{case['label']} {term} error"] = read_plot(
                        path, id=f"{BENCHMARK_NAME}-{model}-{name}-figure"
                    )
            if model_plots:
                plots[model] = model_plots

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=plots,
        )
        for model in plots:
            for name in CASES:
                structs = trace_structs(model, name)
                if not structs:
                    continue
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-{name}-figure",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=structs,
                    mode="traj",
                    follow_frames=True,
                )


def get_app() -> BrittleCrackApp:
    """
    Get the brittle_crack app.

    Returns
    -------
    BrittleCrackApp
        App instance.
    """
    return BrittleCrackApp(
        name=BENCHMARK_NAME,
        description=(
            "Lattice-trapping S-curves of clamped Si crack clusters against DFT: "
            "K/K_G, bond-breaking energy and tip displacement field along the path."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "brittle_crack_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
