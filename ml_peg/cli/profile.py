"""Register the optional developer command without importing browser tooling."""

from __future__ import annotations

from importlib.util import find_spec

from typer import Context, Typer


def register_profile_command(app: Typer) -> None:
    """
    Register browser profiling only when its optional dependencies are installed.

    Parameters
    ----------
    app
        CLI application to extend.
    """
    if any(find_spec(module) is None for module in ("playwright", "matplotlib")):
        return

    @app.command(
        name="profile",
        help="Profile app interactions in Chromium (developer tool).",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        add_help_option=False,
    )
    def profile(ctx: Context) -> None:
        """
        Forward options to the shared browser-profiling argument parser.

        Parameters
        ----------
        ctx
            Command context containing arguments for the shared parser.
        """
        from ml_peg.app.utils.browser_profiling import main

        main(ctx.args, prog="ml_peg profile")
