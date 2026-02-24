"""Run diamond phonon dispersion app (bands-only benchmark)."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
import json
from pathlib import Path
from typing import Any

from dash import Dash, dcc, html

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.bulk_crystal.diamond_phonons.diamond_interactive_helpers import (
    render_dispersion_component,
)
from ml_peg.app.utils.build_callbacks import (
    model_asset_from_scatter,
    scatter_and_assets_from_table,
)
from ml_peg.app.utils.plot_helpers import build_serialized_scatter_content
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_github_data

GITHUB_BASE = "https://raw.githubusercontent.com/7radians/ml-peg-data/main"

EXTRACTED_ROOT = Path(
    download_github_data(
        filename="diamond_data/data.zip",
        github_uri=GITHUB_BASE,
    )
)

DATA_PATH = EXTRACTED_ROOT / "data"
DFT_REF_PATH = DATA_PATH / "dft_band.npz"


BENCHMARK_NAME = "diamond_phonons"

DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / BENCHMARK_NAME
TABLE_PATH = DATA_PATH / "diamond_phonons_bands_table.json"
SCATTER_PATH = DATA_PATH / "diamond_phonons_bands_interactive.json"

DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html"
    f"#{BENCHMARK_NAME}"
)

CALC_BASE = CALCS_ROOT / "bulk_crystal" / BENCHMARK_NAME
# DFT_REF_PATH = Path("data") / "dft_band.npz"

PLOT_CONTAINER_ID = f"{BENCHMARK_NAME}-plot-container"
DISPERSION_CONTAINER_ID = f"{BENCHMARK_NAME}-dispersion-container"
LAST_CELL_STORE_ID = f"{BENCHMARK_NAME}-last-cell"
SCATTER_METADATA_STORE_ID = f"{BENCHMARK_NAME}-scatter-meta"
SCATTER_GRAPH_ID = f"{BENCHMARK_NAME}-scatter"


def model_only_lookup(
    click_data: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build a selection context for the dispersion preview.

    For this benchmark we ignore which scatter point was clicked and return a
    single dispersion preview per model. The returned ``band_yaml`` must be
    resolvable under the app's ``calc_root`` passed to the renderer.

    Parameters
    ----------
    click_data
        Dash click payload from the scatter plot. Unused for this benchmark.
    metadata
        Metadata payload produced by the scatter callback helpers. Must contain
        a ``model`` key.

    Returns
    -------
    dict
        Selection context consumed by ``render_dispersion_component``.
    """
    _ = click_data
    model = str(metadata["model"])
    return {
        "model": model,
        "selection": {
            "id": "diamond",
            "label": "Carbon diamond",
            "band_yaml": f"outputs/{model}/band.yaml",
        },
    }


class DiamondPhononApp(BaseApp):
    """Bands-only phonon benchmark app wiring callbacks and layout."""

    def register_callbacks(self) -> None:
        """Register scatter and dispersion callbacks."""
        with SCATTER_PATH.open(encoding="utf8") as handle:
            interactive_data = json.load(handle)

        calc_root = Path(CALC_BASE)
        models_data = interactive_data.get("models", {})
        metric_labels = interactive_data.get("metrics", {})
        label_to_key = {label: key for key, label in metric_labels.items()}

        refresh_msg = (
            "Click on a metric to view DFT vs predicted frequency scatter plots."
        )

        metric_handler = partial(
            build_serialized_scatter_content,
            models_data=models_data,
            label_map=label_to_key,
            scatter_id=SCATTER_GRAPH_ID,
            instructions=refresh_msg,
        )

        # Bands-only benchmark: no BZ violin panel and no stability panel.
        scatter_and_assets_from_table(
            table_id=self.table_id,
            table_data=self.table.data,
            plot_container_id=PLOT_CONTAINER_ID,
            scatter_metadata_store_id=SCATTER_METADATA_STORE_ID,
            last_cell_store_id=LAST_CELL_STORE_ID,
            column_handlers={},  # only metric scatter
            default_handler=metric_handler,
        )

        dispersion_renderer = partial(
            render_dispersion_component,
            calc_root=calc_root,
            frequency_scale=1,
            frequency_unit="THz",
            reference_label="RSCAN",
            reference_band_npz=DFT_REF_PATH,
        )

        model_asset_from_scatter(
            scatter_id=SCATTER_GRAPH_ID,
            metadata_store_id=SCATTER_METADATA_STORE_ID,
            asset_container_id=DISPERSION_CONTAINER_ID,
            data_lookup=model_only_lookup,
            asset_renderer=dispersion_renderer,
            empty_message="Select a model to preview the phonon dispersion.",
            missing_message="No band.yaml found for this model.",
        )


def get_app() -> DiamondPhononApp:
    """
    Construct the diamond phonon PhononApp instance.

    Returns
    -------
    PhononApp
        Configured application with table + scatter/dispersion panels.
    """
    return DiamondPhononApp(
        name=BENCHMARK_NAME,
        description=(
            "Accuracy of MLIPs in predicting phonon dispersions for Carbon diamond "
            "(RSCAN)."
        ),
        docs_url=DOCS_URL,
        table_path=TABLE_PATH,
        extra_components=[
            dcc.Store(id=LAST_CELL_STORE_ID),
            dcc.Store(id=SCATTER_METADATA_STORE_ID),
            html.Div(
                [
                    html.Div(
                        "Click on a metric to view DFT vs predicted frequency scatter "
                        "plots.",
                        id=PLOT_CONTAINER_ID,
                        style={"flex": "1", "minWidth": 0},
                    ),
                    html.Div(
                        "Click on a scatter point to view the dispersion plot.",
                        id=DISPERSION_CONTAINER_ID,
                        style={"flex": "1", "minWidth": 0},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "24px",
                    "alignItems": "stretch",
                    "flexWrap": "wrap",
                },
            ),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    diamond_phonon_app = get_app()
    full_app.layout = diamond_phonon_app.layout
    diamond_phonon_app.register_callbacks()
    full_app.run(port=8060, debug=True)
