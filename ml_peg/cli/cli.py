"""Set up commandline interface."""

from __future__ import annotations

from typing import Annotated

from typer import Exit, Option, Typer

from ml_peg import __version__

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
        str, Option(help="Category to build app for. Default is all categories.")
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


@app.command(name="calc", help="Run calculations")
def run_calcs(
    models: Annotated[
        str | None,
        Option(
            help="Comma-separated models to run calculations on. Default is all models."
        ),
    ] = None,
    category: Annotated[
        str, Option(help="Category to run calculations for. Default is all categories.")
    ] = "*",
    test: Annotated[
        str, Option(help="Test to run calculations for. Default is all tests.")
    ] = "*",
    run_slow: Annotated[
        bool, Option(help="Whether to run calculations labelled slow.")
    ] = True,
    verbose: Annotated[
        bool, Option(help="Whether to run pytest with verbose and stdout printed.")
    ] = True,
) -> None:
    """
    Run calculations through pytest.

    Parameters
    ----------
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
    verbose
        Whether to run pytest with verbose and stdout printed. Default is `True`.
    """
    import pytest

    from ml_peg.calcs import CALCS_ROOT

    test_paths = CALCS_ROOT.glob(f"{category}/{test}/calc_*")
    options = list(test_paths)

    if verbose:
        options.extend(["-s", "-vvv"])

    if run_slow:
        options.extend(["--run-slow"])

    if models:
        options.extend(["--models", models])

    pytest.main(options)


@app.command(name="analyse", help="Run calculations")
def run_analysis(
    models: Annotated[
        str | None,
        Option(
            help="Comma-separated models to run analysis for. Default is all models."
        ),
    ] = None,
    category: Annotated[
        str, Option(help="Category to run analysis for. Default is all categories.")
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

    test_paths = ANALYSIS_ROOT.glob(f"{category}/{test}/analyse_*")
    options = list(test_paths)

    if verbose:
        options.extend(["-s", "-vvv"])

    if models:
        options.extend(["--models", models])

    pytest.main(options)


@app.command(name="download", help="Download data from S3 bucket")
def download(
    key: Annotated[str, Option(help="File to download")],
    filename: Annotated[str, Option(help="Filename to save download as")],
    bucket: Annotated[str, Option(help="Name of S3 bucket")] = "ml-peg-test",
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
        Name of S3 bucket. Default is "ml-peg-test".
    endpoint
        Endpoint URL. Default is "https://s3.echo.stfc.ac.uk".
    credentials
        S3 credentials. Default is `None`, which will only allow downloading public
        data.
    """
    import json

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    if credentials is None:
        s3 = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED, s3={"addressing_style": "path"}),
            endpoint_url=endpoint,
        )
    else:
        with open(credentials) as credentials_file:
            user_credentials = json.load(credentials_file)

        s3 = boto3.client(
            "s3",
            config=Config(s3={"addressing_style": "path"}),
            endpoint_url=endpoint,
            aws_access_key_id=user_credentials["access_key"],
            aws_secret_access_key=user_credentials["secret_key"],
        )

    s3.download_file(Bucket=bucket, Filename=filename, Key=key)

    print(f"Downloaded {filename}")


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
