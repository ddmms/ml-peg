"""Run the MAD-2 errors by reference app."""

from __future__ import annotations

from ml_peg.app.utils.mad_2_bench import Mad2ErrorsApp


def get_app() -> Mad2ErrorsApp:
    """
    Get the MAD-2 errors by reference app.

    Returns
    -------
    Mad2ErrorsApp
        Benchmark layout and callback registration.
    """
    return Mad2ErrorsApp(
        name="Errors by reference",
        description="Compare model predictions with a selected DFT reference.",
    )
