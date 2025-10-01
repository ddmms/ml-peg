"""Base class to construct app layouts and register callbacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dash.development.base_component import Component
from dash.html import Div

from ml_peg.app.utils.build_components import build_test_layout
from ml_peg.app.utils.load import rebuild_table


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
    """

    def __init__(
        self,
        name: str,
        description: str,
        table_path: Path,
        extra_components: list[Component],
        column_widths: dict[str, int] | None = None,
        under_table_components: list[Component] | None = None,
        use_outer_scroll: bool | None = None,
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
        """
        self.name = name
        self.description = description
        self.table_path = table_path
        self.extra_components = extra_components
        self.column_widths = column_widths
        self.under_table_components = under_table_components or []
        self.use_outer_scroll = (
            bool(use_outer_scroll)
            if use_outer_scroll is not None
            else bool(self.under_table_components)
        )

        self.table_id = f"{self.name}-table"
        self.table = rebuild_table(
            self.table_path,
            id=self.table_id,
            column_widths=self.column_widths,
            outer_scroll=self.use_outer_scroll,
        )
        self.layout = self.build_layout()

    def build_layout(self) -> Div:
        """
        Build layout for application.

        Returns
        -------
        Div
            Div component with list all components for app.
        """
        # Define all components/placeholders
        return build_test_layout(
            name=self.name,
            description=self.description,
            table=self.table,
            under_table_components=self.under_table_components,
            extra_components=self.extra_components,
        )

    @abstractmethod
    def register_callbacks(self):
        """Register callbacks with app."""
        pass
