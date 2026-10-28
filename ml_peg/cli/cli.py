"""Set up commandline interface."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, get_args

from typer import Context, Exit, Option, Typer
from yaml import safe_load

from ml_peg import __version__
from ml_peg.analysis import ANALYSIS_ROOT
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.cli.profile import register_profile_command

if TYPE_CHECKING:
    from rich.table import Table


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


_TEST_ROOTS: dict[str, Path] = {
    "calc": CALCS_ROOT,
    "analyse": ANALYSIS_ROOT,
    "app": APP_ROOT,
}


def complete_test(ctx: Context, incomplete: str) -> list[str]:
    """
    Complete test names for the invoked command, scoped to ``--category``.

    The command name (``ctx.info_name``) doubles as the script prefix, e.g.
    ``calc`` -> ``calc_*.py`` in ``CALCS_ROOT``.

    Parameters
    ----------
    ctx
        Typer context, used to identify the command and the selected category.
    incomplete
        Partially typed test name.

    Returns
    -------
    list[str]
        Matching test names.
    """
    root = _TEST_ROOTS.get(ctx.info_name)
    if root is None:
        return []
    category = ctx.params.get("category") or "*"
    return [
        test
        for test in get_tests(root, ctx.info_name, category)
        if test.startswith(incomplete)
    ]


def complete_models(ctx: Context, incomplete: str) -> list[str]:
    """
    Complete comma-separated model names from ``models.yml``.

    Parameters
    ----------
    ctx
        Typer context, used to read ``--models-file`` if provided.
    incomplete
        Partially typed value; only the segment after the last comma is completed.

    Returns
    -------
    list[str]
        Matching model names, preserving any already-typed comma-separated prefix.
    """
    from ml_peg.models.get_models import get_model_names

    names = sorted(get_model_names(filepath=ctx.params.get("models_file")))

    prefix, sep, last = incomplete.rpartition(",")
    head = f"{prefix}{sep}" if sep else ""
    return [
        f"{head}{name}" for name in names if name.startswith(last) and name not in head
    ]


def get_frameworks() -> tuple[str, ...]:
    """
    Get available MLIP framework ids.

    Returns
    -------
    tuple[str, ...]
        Tuple of "*" followed by sorted framework ids from the framework registry.
    """
    frameworks = safe_load((APP_ROOT / "utils" / "frameworks.yml").read_text())
    return ("*",) + tuple(sorted(frameworks))


AnalysisCategories = Literal[(get_categories(ANALYSIS_ROOT, "analyse"))]
AppCategories = Literal[(get_categories(APP_ROOT, "app"))]
CalcCategories = Literal[(get_categories(CALCS_ROOT, "calc"))]
Frameworks = Literal[(get_frameworks())]


app = Typer(
    name="ml_peg",
    no_args_is_help=True,
    epilog="Try 'ml_peg COMMAND --help' for subcommand options",
)

register_profile_command(app)


@app.command(name="app", help="Run application")
def run_dash_app(
    models: Annotated[
        str | None,
        Option(
            help=(
                "Comma-separated models to build interactivity for. Default is all "
                "models."
            ),
            autocompletion=complete_models,
        ),
    ] = None,
    models_file: Annotated[
        Path | None,
        Option(
            help=(
                "Path to model definitions YAML file. Default is models.yml in models "
                "directory."
            ),
        ),
    ] = None,
    category: Annotated[
        AppCategories,
        Option(
            help="Category to build app for. Default is all categories.",
            case_sensitive=False,
        ),
    ] = "*",
    test: Annotated[
        str,
        Option(
            help="Test to build app for. Default is all tests.",
            case_sensitive=False,
            autocompletion=complete_test,
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
    models_file
        Path to model definitions YAML file. Default is models.yml in models directory.
    category
        Category to build app for. Default is `*`, corresponding to all categories.
    test
        Test to build app for. Default is `*`, corresponding to all tests.
    port
        Port to run application on. Default is 8050.
    debug
        Whether to run with Dash debugging. Default is `True`.
    """
    from ml_peg import models as ml_peg_models

    # Overwrite current_models before it is imported elsewhere
    ml_peg_models.current_models = models
    if models_file:
        ml_peg_models.models_file = models_file

    from ml_peg.app.run_app import run_app

    run_app(category=category, test=test, port=port, debug=debug)


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
            help=(
                "Comma-separated models to run calculations on. Default is all models."
            ),
            autocompletion=complete_models,
        ),
    ] = None,
    models_file: Annotated[
        Path | None,
        Option(
            help=(
                "Path to model definitions YAML file. Default is models.yml in models "
                "directory."
            ),
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
        str,
        Option(
            help="Test to run calculations for. Default is all tests.",
            autocompletion=complete_test,
        ),
    ] = "*",
    framework: Annotated[
        Frameworks,
        Option(
            help="MLIP framework to run calculations for. Default is all frameworks.",
            case_sensitive=False,
        ),
    ] = "*",
    run_mock: Annotated[
        bool, Option(help="Whether to run with mock calculator in addition to models.")
    ] = True,
    mock_only: Annotated[
        bool, Option(help="Whether to only run mock calculator with no models.")
    ] = False,
    run_slow: Annotated[
        bool, Option(help="Whether to run calculations labelled slow.")
    ] = False,
    run_very_slow: Annotated[
        bool, Option(help="Whether to run calculations labelled very slow.")
    ] = False,
    run_multi_day: Annotated[
        bool, Option(help="Whether to run calculations labelled multi-day.")
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
    models_file
        Path to model definitions YAML file. Default is models.yml in models directory.
    category
        Category to run calculations for. Default is `*`, corresponding to all
        categories.
    test
        Test to run calculation for. Default is `*`, corresponding to all tests in the
        category.
    framework
        MLIP framework to run calculations for. Default is `*`, corresponding to all
        frameworks.
    run_mock
        Whether to run mock calculations. Default is `True`.
    mock_only
        Whether to only run mock calculations, with no models. Default is `False`.
    run_slow
        Whether to run slow calculations. Default is `False`.
    run_very_slow
        Whether to run very slow calculations. Default is `False`.
    run_multi_day
        Whether to run multi-day calculations. Default is `False`.
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
        options.extend(["-s", "-vvv", "-rs"])

    if run_slow:
        options.extend(["--run-slow"])

    if run_very_slow:
        options.extend(["--run-very-slow"])

    if run_multi_day:
        options.extend(["--run-multi-day"])

    if run_mock:
        options.extend(["--run-mock"])

    if mock_only:
        options.extend(["--mock-only"])

    if models:
        options.extend(["--models", models])

    if models_file:
        options.extend(["--models-file", models_file])

    if framework != "*":
        options.extend(["--framework", framework])

    # Parse any custom options to pytest
    options.extend(ctx.args)

    pytest.main(options)


@app.command(name="analyse", help="Run analysis")
def run_analysis(
    models: Annotated[
        str | None,
        Option(
            help="Comma-separated models to run analysis for. Default is all models.",
            autocompletion=complete_models,
        ),
    ] = None,
    models_file: Annotated[
        Path | None,
        Option(
            help=(
                "Path to model definitions YAML file. Default is models.yml in models "
                "directory."
            ),
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
        str,
        Option(
            help="Test to run analysis for. Default is all tests.",
            autocompletion=complete_test,
        ),
    ] = "*",
    framework: Annotated[
        Frameworks,
        Option(
            help="MLIP framework to run analysis for. Default is all frameworks.",
            case_sensitive=False,
        ),
    ] = "*",
    update: Annotated[
        bool,
        Option(
            help=(
                "Whether to update saved tables and plots, preserving results for "
                "models not being analysed, rather than overwriting them."
            ),
        ),
    ] = False,
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
    models_file
        Path to model definitions YAML file. Default is models.yml in models directory.
    category
        Category to run analysis for. Default is `*`, corresponding to all categories.
    test
        Test to run analysis for. Default is `*`, corresponding to all tests in the
        category.
    framework
        MLIP framework to run analysis for. Default is `*`, corresponding to all
        frameworks.
    update
        Whether to update saved tables and plots, preserving results for models not
        being analysed, rather than overwriting them. Default is `False`.
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
        options.extend(["-s", "-vvv", "-rs"])

    if models:
        options.extend(["--models", models])

    if models_file:
        options.extend(["--models-file", models_file])

    if framework != "*":
        options.extend(["--framework", framework])

    if update:
        options.extend(["--update"])

    pytest.main(options)


def format_duration(seconds: float | None) -> str:
    """
    Format a duration for display.

    Parameters
    ----------
    seconds
        Duration in seconds, or `None` if it was not recorded.

    Returns
    -------
    str
        Duration in hours, minutes and seconds, as appropriate.
    """
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


def format_timestamp(timestamp: str | None) -> str:
    """
    Format a logged timestamp for display.

    Parameters
    ----------
    timestamp
        Timestamp in ISO 8601 format, or `None` if it was not recorded.

    Returns
    -------
    str
        Date and time, to the minute.
    """
    from datetime import datetime

    if not timestamp:
        return "-"
    try:
        return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return timestamp


def describe_run(run: dict | None) -> tuple[str, str]:
    """
    Describe the state of a run, and why it is in that state.

    Parameters
    ----------
    run
        Most recent logged run, or `None` if the calculation has never been run.

    Returns
    -------
    tuple[str, str]
        State to report, and a note explaining it.
    """
    from ml_peg.calcs.utils.run_log import is_stale

    if run is None:
        return "not run", ""

    status = run["status"]
    if status == "running":
        # Nothing finalised the run, so it was interrupted, unless still in progress
        return "incomplete", f"started {format_timestamp(run.get('started'))}"
    if status == "failed":
        error = run.get("error") or {}
        message = " ".join(str(error.get("message", "")).split())
        return "failed", f"{error.get('type', 'error')}: {message[:60]}"
    if status == "completed" and is_stale(run):
        return "stale", "model definition changed since run"
    return status, ""


#: States a calculation is reported in, in the order they are reported.
STATES = ("completed", "stale", "failed", "incomplete", "skipped", "not run")

#: Colour used to report each state of a run.
STATE_COLOURS = {
    "completed": "green",
    "failed": "red",
    "incomplete": "yellow",
    "stale": "yellow",
    "skipped": "dim",
    "not run": "dim",
}


@app.command(
    name="status", help="Report when calculations ran, and whether they completed"
)
def run_status(
    models: Annotated[
        str | None,
        Option(
            help="Comma-separated models to report on. Default is all models.",
            autocompletion=complete_models,
        ),
    ] = None,
    models_file: Annotated[
        Path | None,
        Option(
            help=(
                "Path to model definitions YAML file. Default is models.yml in models "
                "directory."
            ),
        ),
    ] = None,
    category: Annotated[
        CalcCategories,
        Option(
            help="Category to report on. Default is all categories.",
            case_sensitive=False,
        ),
    ] = "*",
    test: Annotated[
        str,
        Option(
            help="Test to report on. Default is all tests.",
            autocompletion=complete_test,
        ),
    ] = "*",
    detail: Annotated[
        bool,
        Option(help="Whether to report every model separately, rather than totals."),
    ] = False,
    state: Annotated[
        str | None,
        Option(
            help=(
                "Comma-separated states to report, such as failed,incomplete. "
                "Default is every state but not-run."
            )
        ),
    ] = None,
) -> None:
    """
    Report the state of logged calculations.

    Parameters
    ----------
    models
        Models to report on, in comma-separated list. Default is `None`,
        corresponding to all available models.
    models_file
        Path to model definitions YAML file. Default is models.yml in models directory.
    category
        Category to report on. Default is `*`, corresponding to all categories.
    test
        Test to report on. Default is `*`, corresponding to all tests in the category.
    detail
        Whether to report each model separately, rather than totals per benchmark.
        Default is `False`.
    state
        States to report, in comma-separated list. Implies `detail`. Default is
        `None`, reporting every state but "not run", which would otherwise be most
        of the table.
    """
    from rich.console import Console
    from rich.table import Table

    from ml_peg.calcs.utils.run_log import collect_runs

    runs = collect_runs(category, test, models, models_file)
    if not runs:
        raise ValueError(
            f"No tests were found matching {category}/{test}/calc_*.py in {CALCS_ROOT}"
        )

    # Hyphens and underscores are easier to type than the space in "not run"
    wanted = (
        {name.strip().replace("-", " ").replace("_", " ") for name in state.split(",")}
        if state
        else set(STATES) - {"not run"}
    )
    console = Console()
    table = Table(title="Calculation status")

    if detail or state:
        hidden = _add_detail_rows(table, runs, wanted)
        console.print(table)
        if hidden:
            console.print(
                f"[dim]{hidden} model(s) in other states. Use --state to report "
                "them, for example --state not-run.[/]"
            )
        return

    _add_summary_rows(table, runs)
    console.print(table)


def _add_detail_rows(
    table: Table, runs: dict[str, dict[str, dict | None]], wanted: set[str]
) -> int:
    """
    Add one row per model to a status table.

    Parameters
    ----------
    table
        Rich table to add rows to.
    runs
        Most recent run of each model, keyed by benchmark then model.
    wanted
        States to report.

    Returns
    -------
    int
        Number of models left out, as they are in a state not being reported.
    """
    table.add_column("Benchmark", overflow="fold")
    for column in ("Model", "State", "Finished", "Duration"):
        table.add_column(column)
    table.add_column("Warnings", justify="right")
    table.add_column("Note")

    hidden = 0
    for benchmark, models in runs.items():
        for model_name, run in models.items():
            run_state, note = describe_run(run)
            if run_state not in wanted:
                hidden += 1
                continue
            warnings = (run or {}).get("warnings") or {}
            table.add_row(
                benchmark,
                model_name,
                f"[{STATE_COLOURS.get(run_state, '')}]{run_state}[/]",
                format_timestamp((run or {}).get("finished")),
                format_duration((run or {}).get("duration_s")),
                str(warnings.get("total", "")) if run else "",
                note,
            )

    return hidden


def _add_summary_rows(table: Table, runs: dict[str, dict[str, dict | None]]) -> None:
    """
    Add one row per benchmark to a status table, totalling the states of its models.

    Parameters
    ----------
    table
        Rich table to add rows to.
    runs
        Most recent run of each model, keyed by benchmark then model.
    """
    from collections import Counter

    table.add_column("Benchmark", overflow="fold")
    for run_state in STATES:
        table.add_column(
            run_state.title(), justify="right", style=STATE_COLOURS.get(run_state, "")
        )

    totals: Counter[str] = Counter()
    for benchmark, models in runs.items():
        counts = Counter(describe_run(run)[0] for run in models.values())
        totals.update(counts)
        table.add_row(
            benchmark, *(str(counts[run_state] or "") for run_state in STATES)
        )

    table.add_section()
    table.add_row(
        "Total", *(f"[bold]{totals[run_state] or ''}[/]" for run_state in STATES)
    )


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
            f"""Categories: {
                ", ".join(
                    category for category in get_args(CalcCategories) if category != "*"
                )
            }\n"""
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
            f""""Categories: {
                ", ".join(
                    category
                    for category in get_args(AnalysisCategories)
                    if category != "*"
                )
            }\n"""
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
            f"""Categories: {
                ", ".join(
                    category for category in get_args(AppCategories) if category != "*"
                )
            }\n"""
        )
    print(f"Tests: {', '.join(get_tests(ANALYSIS_ROOT, 'analyse', category))}")


@list_app.command(name="models", help="List models currently available")
def list_models(
    models_file: Annotated[
        str,
        Option(
            help=(
                "Path to model definitions YAML file. Default is models.yml in models "
                "directory."
            ),
        ),
    ] = None,
) -> None:
    """
    List currently available models.

    Parameters
    ----------
    models_file
        Path to model definitions YAML file. Default is models.yml in models directory.
    """
    from ml_peg.models.get_models import get_model_names

    if models_file is None:
        from ml_peg.models import models_file

    print(f"Available models: {', '.join(get_model_names(filepath=models_file))}")


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
