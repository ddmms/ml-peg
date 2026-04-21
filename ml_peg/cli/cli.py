"""Set up commandline interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, get_args

from typer import Context, Exit, Option, Typer

from ml_peg import __version__
from ml_peg.analysis import ANALYSIS_ROOT
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT


def get_categories(root: Path, script_prefix: str) -> tuple[str, ...]:
    """
    Get current categories.

    Parameters
    ----------
    root
        Root directory to search for categories in.
    script_prefix
        Prefix of script files to match for e.g. "calc" for calc_test.py.

    Returns
    -------
    tuple[str, ...]
        Tuple of sorted category names. Uses glob matches for `script_prefix` within
        the `root` directory.
    """
    return ("*",) + tuple(
        sorted(
            {
                path.parent.parent.name
                for path in root.glob(f"*/*/{script_prefix}_*.py")
                if path.is_file() and path.parent.parent.is_dir()
            }
        )
    )


def get_tests(root: Path, script_prefix: str, category: str = "*") -> tuple[str, ...]:
    """
    Get current tests.

    Parameters
    ----------
    root
        Root directory to search for tests in.
    script_prefix
        Prefix of script files to match for e.g. "calc" for calc_test.py.
    category
        Category in which to search for tests. Default is `*`, corresponding to all
        categories.

    Returns
    -------
    tuple[str, ...]
        Tuple of sorted test names. Uses glob matches for `script_prefix` within
        the `root` directory.
    """
    return tuple(
        sorted(
            {
                path.parent.name
                for path in root.glob(f"{category}/*/{script_prefix}_*.py")
                if path.is_file()
                and path.parent.is_dir()
                and path.name == f"{script_prefix}_{path.parent.name}.py"
            }
        )
    )


AnalysisCategories = Literal[(get_categories(ANALYSIS_ROOT, "analyse"))]
AppCategories = Literal[(get_categories(APP_ROOT, "app"))]
CalcCategories = Literal[(get_categories(CALCS_ROOT, "calc"))]


app = Typer(
    name="ml_peg",
    no_args_is_help=True,
    epilog="Try 'ml_peg COMMAND --help' for subcommand options",
)


@app.command(name="app", help="Run application")
def run_dash_app(
    models: Annotated[
        str | None,
        Option(
            help=(
                "Comma-separated models to build interactivity for. Default is all "
                "models."
            )
        ),
    ] = None,
    category: Annotated[
        AppCategories,
        Option(
            help="Category to build app for. Default is all categories.",
            case_sensitive=False,
        ),
    ] = "*",
    port: Annotated[str, Option(help="Port to run application on.")] = 8050,
    debug: Annotated[bool, Option(help="Whether to run with Dash debugging.")] = True,
) -> None:
    """
    Run Dash application.

    Parameters
    ----------
    models
        Models to run calculations for, in comma-separated list. Default is `None`,
        corresponding to all available models.
    category
        Category to build app for. Default is `*`, corresponding to all categories.
    port
        Port to run application on. Default is 8050.
    debug
        Whether to run with Dash debugging. Default is `True`.
    """
    from ml_peg.models import models as ml_peg_models

    # Overwrite current_models before it is imported elsewhere
    ml_peg_models.current_models = models

    from ml_peg.app.run_app import run_app

    run_app(category=category, port=port, debug=debug)


@app.command(
    name="calc",
    help="Run calculations",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run_calcs(
    ctx: Context,
    models: Annotated[
        str | None,
        Option(
            help="Comma-separated models to run calculations on. Default is all models."
        ),
    ] = None,
    category: Annotated[
        CalcCategories,
        Option(
            help="Category to run calculations for. Default is all categories.",
            case_sensitive=False,
        ),
    ] = "*",
    test: Annotated[
        str, Option(help="Test to run calculations for. Default is all tests.")
    ] = "*",
    run_slow: Annotated[
        bool, Option(help="Whether to run calculations labelled slow.")
    ] = True,
    run_very_slow: Annotated[
        bool, Option(help="Whether to run calculations labelled very slow.")
    ] = False,
    verbose: Annotated[
        bool, Option(help="Whether to run pytest with verbose and stdout printed.")
    ] = True,
) -> None:
    """
    Run calculations through pytest.

    Parameters
    ----------
    ctx
        Typer Context. Automatically set.
    models
        Models to run calculations for, in comma-separated list. Default is `None`,
        corresponding to all available models.
    category
        Category to run calculations for. Default is `*`, corresponding to all
        categories.
    test
        Test to run calculation for. Default is `*`, corresponding to all tests in the
        category.
    run_slow
        Whether to run slow calculations. Default is `True`.
    run_very_slow
        Whether to run very slow calculations. Default is `False`.
    verbose
        Whether to run pytest with verbose and stdout printed. Default is `True`.
    """
    import pytest

    from ml_peg.calcs import CALCS_ROOT

    options = list(CALCS_ROOT.glob(f"{category}/{test}/calc_*.py"))
    if not options:
        raise ValueError(
            f"No tests were found matching {category}/{test}/calc_*.py in {CALCS_ROOT}"
        )

    if verbose:
        options.extend(["-s", "-vvv"])

    if run_slow:
        options.extend(["--run-slow"])

    if run_very_slow:
        options.extend(["--run-very-slow"])

    if models:
        options.extend(["--models", models])

    # Parse any custom options to pytest
    options.extend(ctx.args)

    pytest.main(options)


@app.command(name="analyse", help="Run analysis")
def run_analysis(
    models: Annotated[
        str | None,
        Option(
            help="Comma-separated models to run analysis for. Default is all models."
        ),
    ] = None,
    category: Annotated[
        AnalysisCategories,
        Option(
            help="Category to run analysis for. Default is all categories.",
            case_sensitive=False,
        ),
    ] = "*",
    test: Annotated[
        str, Option(help="Test to run analysis for. Default is all tests.")
    ] = "*",
    verbose: Annotated[
        bool, Option(help="Whether to run pytest with verbose and stdout printed.")
    ] = True,
):
    """
    Run analysis through pytest.

    Parameters
    ----------
    models
        Models to run analysis for, in comma-separated list. Default is `None`,
        corresponding to all available models.
    category
        Category to run analysis for. Default is `*`, corresponding to all categories.
    test
        Test to run analysis for. Default is `*`, corresponding to all tests in the
        category.
    verbose
        Whether to run pytest with verbose and stdout printed. Default is `True`.
    """
    import pytest

    from ml_peg.analysis import ANALYSIS_ROOT

    options = list(ANALYSIS_ROOT.glob(f"{category}/{test}/analyse_*.py"))
    if not options:
        raise ValueError(
            f"No tests were found matching {category}/{test}/analyse_*.py in "
            f"{ANALYSIS_ROOT}"
        )

    if verbose:
        options.extend(["-s", "-vvv"])

    if models:
        options.extend(["--models", models])

    pytest.main(options)


list_app = Typer(
    name="list",
    no_args_is_help=True,
    epilog="Try 'ml_peg list COMMAND --help' for subcommand options",
)
app.add_typer(
    list_app,
    help=(
        "List available categories, tests, and models for calculations, analysis, and "
        "app."
    ),
)


@list_app.command(
    name="calcs", help="List categories and tests available for calculations"
)
def list_calcs(
    category: Annotated[
        CalcCategories,
        Option(
            help="Category to run calculations for. Default is all categories.",
            case_sensitive=False,
        ),
    ] = "*",
):
    """
    List categories and tests available for calculations.

    Parameters
    ----------
    category
        Category to list test for. Default is `*`, corresponding to all categories.
    """
    if category == "*":
        print(
            f"Categories: {
                ', '.join(
                    category for category in get_args(CalcCategories) if category != '*'
                )
            }\n"
        )
    print(f"Tests: {', '.join(get_tests(CALCS_ROOT, 'calc', category))}")


@list_app.command(
    name="analysis", help="List categories and tests available for analysis"
)
def list_analysis(
    category: Annotated[
        AnalysisCategories,
        Option(
            help="Category to list tests for. Default is all categories.",
            case_sensitive=False,
        ),
    ] = "*",
):
    """
    List categories and tests available for analysis.

    Parameters
    ----------
    category
        Category to list tests for. Default is `*`, corresponding to all categories.
    """
    if category == "*":
        print(
            f"Categories: {
                ', '.join(
                    category
                    for category in get_args(AnalysisCategories)
                    if category != '*'
                )
            }\n"
        )
    print(f"Tests: {', '.join(get_tests(ANALYSIS_ROOT, 'analyse', category))}")


@list_app.command(
    name="app", help="List categories and tests available as applications"
)
def list_apps(
    category: Annotated[
        AppCategories,
        Option(
            help="Category to list tests for. Default is all categories.",
            case_sensitive=False,
        ),
    ] = "*",
):
    """
    List categories and tests available as applications.

    Parameters
    ----------
    category
        Category to list tests for. Default is `*`, corresponding to all categories.
    """
    if category == "*":
        print(
            f"Categories: {
                ', '.join(
                    category for category in get_args(AppCategories) if category != '*'
                )
            }\n"
        )
    print(f"Tests: {', '.join(get_tests(ANALYSIS_ROOT, 'analyse', category))}")


@list_app.command(name="models", help="List models currently available")
def list_models() -> None:
    """List currently available models."""
    from ml_peg.models.get_models import get_model_names

    print(f"Available models: {', '.join(get_model_names())}")


@app.command(name="download", help="Download data from S3 bucket")
def download(
    key: Annotated[str, Option(help="File to download")],
    filename: Annotated[str, Option(help="Filename to save download as")],
    bucket: Annotated[str, Option(help="Name of S3 bucket")] = "ml-peg-data",
    endpoint: Annotated[
        str, Option(help="Endpoint URL")
    ] = "https://s3.echo.stfc.ac.uk",
    credentials: Annotated[str | None, Option(help="S3 credentials")] = None,
):
    """
    Download data from S3 bucket.

    Parameters
    ----------
    key
        Name of file in S3 bucket to download.
    filename
        Name of file to save download as locally.
    bucket
        Name of S3 bucket. Default is "ml-peg-data".
    endpoint
        Endpoint URL. Default is "https://s3.echo.stfc.ac.uk".
    credentials
        S3 credentials. Default is `None`, which will only allow downloading public
        data.
    """
    from ml_peg.data.data import download as download_data

    download_data(
        key=key,
        filename=filename,
        bucket=bucket,
        endpoint=endpoint,
        credentials=credentials,
    )
    print(f"Downloaded {filename}")


@app.command(name="upload", help="Upload data to S3 bucket")
def upload(
    key: Annotated[str, Option(help="File to upload")],
    filename: Annotated[str, Option(help="Filename to save download as")],
    credentials: Annotated[str, Option(help="S3 credentials")],
    bucket: Annotated[str, Option(help="Name of S3 bucket")] = "ml-peg-data",
    endpoint: Annotated[
        str, Option(help="Endpoint URL")
    ] = "https://s3.echo.stfc.ac.uk",
    acl: Annotated[str, Option(help="Access control list")] = "public-read",
):
    """
    Upload data from S3 bucket.

    Parameters
    ----------
    key
        Name of file to create in S3 bucket.
    filename
        File to upload.
    credentials
        S3 credentials.
    bucket
        Name of S3 bucket to upload to. Default is "ml-peg-data".
    endpoint
        Endpoint URL. Default is "https://s3.echo.stfc.ac.uk".
    acl
        Access control list. Default is "public-read".
    """
    from ml_peg.data.data import upload as upload_data

    upload_data(
        key=key,
        filename=filename,
        bucket=bucket,
        endpoint=endpoint,
        credentials=credentials,
        acl=acl,
    )
    print(f"Uploaded {filename}")


@app.callback(invoke_without_command=True, help="")
def print_version(
    version: Annotated[
        bool, Option("--version", help="Print ML-PEG version and exit.")
    ] = None,
) -> None:
    """
    Print current ML-PEG version and exit.

    Parameters
    ----------
    version
        Whether to print the current ML-PEG version.
    """
    if version:
        print(f"ML-PEG version: {__version__}")
        raise Exit()
