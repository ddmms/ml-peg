"""Mock calculator class."""

from __future__ import annotations

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
import numpy as np

# "Global" variables that can be imported and set through pytest options
run_mock = False
mock_only = False


class MockCalculator(Calculator):
    """A mock calculator that returns zero values for all properties."""

    implemented_properties = ["energy", "forces", "stress"]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
        **kwargs,
    ) -> None:
        """
        Define calculation method for mock calculator.

        Parameters
        ----------
        atoms
            Atoms object to calculate properties for.
        properties
            List of properties to calculate.
        system_changes
            List of system changes to consider for calculation.
        **kwargs
            Any additional keyword arguments.
        """
        super().calculate(atoms, properties, system_changes)

        if "energy" in properties:
            self.results["energy"] = 0.0

        if "forces" in properties:
            self.results["forces"] = np.zeros((len(self.atoms), 3))

        if "stress" in properties:
            self.results["stress"] = np.zeros(6)


class MockErrorCalculator(Calculator):
    """A mock calculator that raises an error for all properties."""

    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, *args, **kwargs) -> None:
        """
        Define calculation method for mock calculator.

        Parameters
        ----------
        *args
            Any additional positional arguments.
        **kwargs
            Any additional keyword arguments.
        """
        raise ValueError("This is a mock error calculator. All calculations fail.")
