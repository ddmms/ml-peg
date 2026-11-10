"""Run the MAD-2 reference comparison app."""

from __future__ import annotations

from ml_peg.app.utils.mad_2_bench import Mad2ComparisonApp


def get_app() -> Mad2ComparisonApp:
    """
    Get the MAD-2 reference comparison app.

    Returns
    -------
    Mad2ComparisonApp
        Benchmark layout and callback registration.
    """
    return Mad2ComparisonApp(
        name="Reference comparison",
        description="Compare model errors across six DFT references.",
    )
