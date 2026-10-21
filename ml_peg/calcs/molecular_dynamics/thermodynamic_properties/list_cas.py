"""List CAS numbers."""

from __future__ import annotations

from ml_peg.calcs.molecular_dynamics.thermodynamic_properties.utils import (
    get_available_cas,
)


def main() -> None:
    """Print the available CAS numbers."""
    print("\n".join(get_available_cas()))


if __name__ == "__main__":
    main()
