"""Base class to construct app layouts and register callbacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
import json
from pathlib import Path
import warnings

from dash.dcc import Store
from dash.development.base_component import Component
from dash.html import Div

from ml_peg.app.utils.build_components import build_test_layout
from ml_peg.app.utils.load import rebuild_table
from ml_peg.app.utils.utils import normalize_framework_id


class BaseApp(ABC):
    """
    Abstract base class to construct app layouts and register callbacks.

    Parameters
    ----------
    name
        Name of application test.
    description
        Description of benchmark.
    table_path
        Path to json file containing Dash table data for application metrics.
    extra_components
        List of other Dash components to add to app.
    docs_url
        URL for online documentation. Default is None.
    framework_ids
        Extra framework identifiers used for benchmark attribution tags, as a single
        string or a sequence of strings. If `include_ml_peg` is True, the  `"ml_peg"`
        tag is also included, so this need only list additional frameworks (e.g.
        `"mace-multihead"`).
    include_ml_peg
        Whether to add the default `"ml_peg"` tag. Set to `False` to opt a
        benchmark out and show only its explicit `framework_ids`. Default is True.
    info_path
        Path to json file containing additional info for filtering. Default is None.
    """

    def __init__(
        self,
        name: str,
        description: str,
        table_path: Path,
        extra_components: list[Component],
        docs_url: str | None = None,
        framework_ids: str | Sequence[str] = (),
        include_ml_peg: bool = True,
        info_path: Path | None = None,
    ):
        """
        Initiaise class.

        Parameters
        ----------
        name
            Name of application test.
        description
            Description of benchmark.
        table_path
            Path to json file containing Dash table data for application metrics.
        extra_components
            List of other Dash components to add to app.
        docs_url
            URL to online documentation. Default is None.
        framework_ids
            Extra framework identifiers used for benchmark attribution tags, as a
            single string or a sequence of strings. If `include_ml_peg` is True, the
            `"ml_peg"` tag is also included, so this need only list additional
            frameworks (e.g. `"mace-multihead"`).
        include_ml_peg
            Whether to add the default `"ml_peg"` tag. Set to `False` to opt a
            benchmark out and show only its explicit `framework_ids`. Default is True.
        info_path
            Path to json file containing additional info for filtering. Default is None.
        """
        self.name = name
        self.description = description
        self.table_path = table_path
        self.extra_components = extra_components
        self.docs_url = docs_url
        # The "ml_peg" tag is shown on every benchmark by default; any extra
        # frameworks (e.g. "mace-multihead") are displayed alongside it.
        self.framework_ids = [
            normalize_framework_id(framework_id)
            for framework_id in (
                [framework_ids] if isinstance(framework_ids, str) else framework_ids
            )
        ]
        if include_ml_peg and "ml_peg" not in self.framework_ids:
            self.framework_ids.insert(0, "ml_peg")
        self.table_id = f"{self.name}-table"
        self.table = rebuild_table(
            self.table_path, id=self.table_id, description=description
        )
        self.metrics = [
            col["id"]
            for col in self.table.columns
            if col["id"] not in ("MLIP", "Score", "id", "link")
        ]
        self.original_table = deepcopy(self.table)
        self.layout = self.build_layout()
        if info_path:
            self.load_info(info_path)
        else:
            self.info = None
            warnings.warn("No info_path provided.", stacklevel=2)
        if self.info is not None and hasattr(self, "set_elements"):
            self.set_elements()
        else:
            self.elements = None

    def load_info(self, info_path: Path) -> None:
        """
        Load additional info for app.

        Parameters
        ----------
        info_path
            Path to json file containing additional info for filtering.
        """
        if not info_path.exists():
            warnings.warn(f"{info_path} does not exist, skipping.", stacklevel=2)
        with open(info_path) as f:
            self.info = json.load(f)

    def build_layout(self) -> Div:
        """
        Build layout for application.

        Returns
        -------
        Div
            Div component with list all components for app.
        """
        # Define all components/placeholders
        # Metric-weight controls defined inside build_test_layout for all benchmarks
        return build_test_layout(
            name=self.name,
            description=self.description,
            docs_url=self.docs_url,
            framework_ids=self.framework_ids,
            table=self.table,
            column_widths=getattr(self.table, "column_widths", None),
            thresholds=self.table.thresholds,
            extra_components=self.extra_components,
        )

    @abstractmethod
    def register_callbacks(self):
        """Register callbacks with app."""
        pass

    def set_elements(self) -> None:
        """Get element sets for filtering."""
        try:
            if isinstance(self.info["elements"][0], list):
                self.elements = {
                    elements
                    for sublist in self.info["elements"]
                    for elements in sublist
                }
            else:
                self.elements = set(self.info["elements"])
        except (AttributeError, KeyError, TypeError, IndexError) as err:
            self.elements = set()
            warnings.warn(
                f"Unable to read elements lists for {self.name}: {err}", stacklevel=2
            )

    def filter_table(self, filter_elements: list[str] | None) -> dict[str, dict]:
        """
        Filter data by elements.

        Parameters
        ----------
        filter_elements
            List of elements to filter out of data.

        Returns
        -------
        dict[str, dict]
            Updated benchmark table.
        """
        if self.elements is None:
            warnings.warn("No elements info available, skipping filter.", stacklevel=2)
            return self.table.data

        filter_elements = set(filter_elements) if filter_elements else set()

        table_data = deepcopy(self.table.data)

        # Get overlap of deselected elements with each system's elements
        if bool(self.elements & filter_elements):
            for row in table_data:
                for metric in self.metrics:
                    row[metric] = None
        else:
            for current_row, original_row in zip(
                table_data, self.original_table.data, strict=True
            ):
                for metric in self.metrics:
                    current_row[metric] = original_row[metric]

        return table_data

    @property
    def stores(self) -> list[Store]:
        """
        List Stores to be registered with full app.

        Returns
        -------
        list[Store]
            List of Stores to be registered with full app.
        """
        return [
            Store(
                id=f"{self.table_id}-computed-store",
                storage_type="session",
                data=self.table.data,
            ),
            Store(
                id=f"{self.table_id}-raw-data-store",
                storage_type="session",
                data=self.table.data,
            ),
            Store(
                id=f"{self.table_id}-weight-store",
                storage_type="session",
                data=self.table.weights,
            ),
            Store(
                id=f"{self.table_id}-thresholds-store",
                storage_type="session",
                data=self.table.thresholds,
            ),
        ]
