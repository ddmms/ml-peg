"""Run LIB electrolyte inter intra benchmark app."""

from __future__ import annotations

import json

from dash.html import Div
import numpy as np

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    clear_from_table_cell,
    plot_from_table_cell,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_density_plot_for_model, read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

BENCHMARK_NAME = "LIB electrolyte Inter-Intra Properties"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/electrolytes.html#lib-electrolyte-inter-intra-properties"
DATA_PATH = APP_ROOT / "data" / "electrolytes" / "LIB_electrolyte_inter_intra"
INFO_PATH = (
    APP_ROOT / "data" / "electrolytes" / "LIB_electrolyte_inter_intra" / "info.json"
)


class LIBelectrolyteInterIntraApp(BaseApp):
    """LIB electrolyte inter intra benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        plots = {
            model: {
                "Intra-Forces": read_density_plot_for_model(
                    DATA_PATH / "intra-forces_density_parity.json",
                    model=model,
                    id=f"{BENCHMARK_NAME}-{model}-intra_forces-figure",
                ),
                "Inter-Forces": read_density_plot_for_model(
                    DATA_PATH / "inter-forces_density_parity.json",
                    model=model,
                    id=f"{BENCHMARK_NAME}-{model}-inter_forces-figure",
                ),
                "Inter-Energy": read_plot(
                    DATA_PATH / f"inter-energy_parity_{model}.json",
                    id=f"{BENCHMARK_NAME}-{model}-inter_energy-figure",
                ),
                "Intra-Virial": read_plot(
                    DATA_PATH / f"intra-virial_parity_{model}.json",
                    id=f"{BENCHMARK_NAME}-{model}-intra_virial-figure",
                ),
                "Inter-Virial": read_plot(
                    DATA_PATH / f"inter-virial_parity_{model}.json",
                    id=f"{BENCHMARK_NAME}-{model}-inter_virial-figure",
                ),
            }
            for model in MODELS
        }

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=plots,
        )

        struct_id = f"{BENCHMARK_NAME}-struct-placeholder"
        clear_from_table_cell(self.table_id, [struct_id])
        assets_dir = "/assets/electrolytes/LIB_electrolyte_inter_intra"

        for model in MODELS:
            structs_dir = DATA_PATH / model
            struct_files = list(structs_dir.glob(f"{model}-intra_inter_*.extxyz"))
            frame_structs = [
                f"{assets_dir}/{model}/{model}-intra_inter_{i}.extxyz"
                for i in range(len(struct_files))
            ]
            struct_from_scatter(
                scatter_id=f"{BENCHMARK_NAME}-{model}-inter_energy-figure",
                struct_id=struct_id,
                structs=frame_structs,
                mode="struct",
            )

            virial_structs = np.repeat(frame_structs, 9).tolist()

            for property in ("intra_virial", "inter_virial"):
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-{property}-figure",
                    struct_id=struct_id,
                    structs=virial_structs,
                    mode="struct",
                )

            for property in ("intra_forces", "inter_forces"):
                bins_path = structs_dir / (
                    f"{property.replace('_', '-')}_structure_bins.json"
                )
                if not bins_path.exists():
                    continue
                mapping = json.loads(bins_path.read_text())
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-{property}-figure",
                    struct_id=struct_id,
                    structs=f"{assets_dir}/{model}/{model}-intra_inter.extxyz",
                    mode="traj",
                    follow_frames=False,
                    frame_groups=[mapping["bins"][i] for i in mapping["point_bins"]],
                )


def get_app() -> LIBelectrolyteInterIntraApp:
    """
    Get LIB electrolyte inter intra benchmark app layout and callback registration.

    Returns
    -------
    LIBelectrolyteInterIntraApp
        Benchmark layout and callback registration.
    """
    return LIBelectrolyteInterIntraApp(
        name=BENCHMARK_NAME,
        description=(
            "Evaluate model inter/intra property prediction "
            "for different densities of LIB electrolyte"
            " and neat solvent configs"
        ),
        # docs_url=DOCS_URL,
        table_path=DATA_PATH / "inter_intra_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
