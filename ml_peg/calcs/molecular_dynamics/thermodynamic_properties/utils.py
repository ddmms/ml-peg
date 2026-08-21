"""Utillity functions shared by thermodynamic property calculation methods."""

from __future__ import annotations

from ml_peg.calcs.utils.utils import download_s3_data


def get_available_cas() -> list[str]:
    """
    Return CAS numbers available in the thermodynamic-properties configs.

    Returns
    -------
    list
        List of all CAS numbers available.
    """
    data_path = (
        download_s3_data(
            filename="thermodynamic_properties.zip",
            key=(
                "inputs/molecular_dynamics/"
                "thermodynamic_properties/"
                "thermodynamic_properties.zip"
            ),
        )
        / "thermodynamic_properties"
    )

    return sorted(
        path.name.removesuffix("-liq.xyz")
        for path in (data_path / "equilibrated_structures_xyz").glob("*-liq.xyz")
    )
