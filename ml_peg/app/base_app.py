"""Base class to construct app layouts and register callbacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
import warnings

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
    framework_id
        Framework identifier used for benchmark attribution tags. Default is
        `"ml_peg"`.
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
        framework_id: str = "ml_peg",
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
        framework_id
            Framework identifier used for benchmark attribution tags.
            Default is `"ml_peg"`.
        info_path
            Path to json file containing additional info for filtering. Default is None.
        """
        self.name = name
        self.description = description
        self.table_path = table_path
        self.extra_components = extra_components
        self.docs_url = docs_url
        self.framework_id = normalize_framework_id(framework_id)
        self.table_id = f"{self.name}-table"
        self.table = rebuild_table(
            self.table_path, id=self.table_id, description=description
        )
        self.layout = self.build_layout()
        if info_path:
            self.load_info(info_path)
        else:
            self.info = None
            warnings.warn("No info_path provided.", stacklevel=2)

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
            framework_id=self.framework_id,
            table=self.table,
            column_widths=getattr(self.table, "column_widths", None),
            thresholds=getattr(self.table, "thresholds", None),
            extra_components=self.extra_components,
        )

    @abstractmethod
    def register_callbacks(self):
        """Register callbacks with app."""
        pass

    # @abstractmethod
    def filter_data(self):
        """Filter data by elements."""
        print(f"No filter_data method defined for {self.name}, skipping.")
