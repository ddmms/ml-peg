"""Opt-in startup and request profiling for local performance investigations."""

from __future__ import annotations

from contextlib import contextmanager
import cProfile
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from flask import Flask, request
from werkzeug.middleware.profiler import ProfilerMiddleware


def profile_directory() -> Path | None:
    """
    Create a separate directory for each profiling process.

    Returns
    -------
    Path | None
        Output directory, or None when ML_PEG_PROFILE_DIR is unset or empty.
    """
    root = os.environ.get("ML_PEG_PROFILE_DIR")
    if not root:
        return None
    directory = Path(root).expanduser() / f"{os.getpid()}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=True)
    print(f"ML-PEG profiles: {directory.resolve()}")
    return directory


@contextmanager
def profile_startup(directory: Path | None):
    """
    Record app construction, including failed builds.

    Parameters
    ----------
    directory
        Profiling output directory, or None to disable profiling.

    Yields
    ------
    None
        Control to the app builder.
    """
    if directory is None:
        yield
        return
    profiler = cProfile.Profile()
    start = perf_counter()
    try:
        with profiler:
            yield
    finally:
        # Repeated builds must not replace the first, cold-start measurement.
        identifier = f"startup-{uuid4().hex}"
        profiler.dump_stats(str(directory / f"{identifier}.prof"))
        (directory / f"{identifier}.json").write_text(
            json.dumps({"elapsed_seconds": perf_counter() - start}) + "\n"
        )


def enable_request_profiling(server: Flask, directory: Path | None) -> None:
    """
    Profile Dash requests and record callback names and payload sizes.

    Static assets bypass profiling. Each request has separate files so concurrent
    requests cannot overwrite each other. Request values are not saved.

    Parameters
    ----------
    server
        Flask server to instrument before serving any requests.
    directory
        Profiling output directory, or None to leave the server unchanged.
    """
    if directory is None:
        return

    @server.before_request
    def record_callback():
        """Capture the callback output name without recording its inputs."""
        if not request.environ.get("ml_peg.profiling"):
            return
        metadata = {"request_bytes": request.content_length}
        if request.path.endswith("/_dash-update-component"):
            payload = request.get_json(silent=True)
            if isinstance(payload, dict):
                metadata["callback"] = payload.get("output")
        request.environ["ml_peg.profile"] = metadata

    @server.after_request
    def record_response(response):
        """
        Record response status and uncompressed size.

        Parameters
        ----------
        response
            Flask response being returned.

        Returns
        -------
        Response
            Original response.
        """
        if "ml_peg.profile" in request.environ:
            request.environ["ml_peg.profile"].update(
                status=response.status_code,
                response_bytes=response.calculate_content_length(),
            )
        return response

    def filename(environ):
        """
        Save request metadata beside a uniquely named profile.

        Parameters
        ----------
        environ
            WSGI environment with Werkzeug profiling results.

        Returns
        -------
        str
            Unique profile filename.
        """
        identifier = uuid4().hex
        metadata = {
            **environ.get("ml_peg.profile", {}),
            "method": environ["REQUEST_METHOD"],
            "path": environ["PATH_INFO"],
            "elapsed_ms": environ["werkzeug.profiler"]["elapsed"],
            "profile": f"{identifier}.prof",
        }
        (directory / f"{identifier}.json").write_text(json.dumps(metadata) + "\n")
        return metadata["profile"]

    original = server.wsgi_app
    profiled = ProfilerMiddleware(
        original, stream=None, profile_dir=str(directory), filename_format=filename
    )

    def dispatch(environ, start_response):
        """
        Profile only Dash layout, dependency and callback requests.

        Parameters
        ----------
        environ
            WSGI request environment.
        start_response
            WSGI response callback.

        Returns
        -------
        Iterable[bytes]
            Original or profiled response body.
        """
        path = environ.get("PATH_INFO", "")
        if path.endswith(
            ("/_dash-layout", "/_dash-dependencies", "/_dash-update-component")
        ):
            environ["ml_peg.profiling"] = True
            return profiled(environ, start_response)
        return original(environ, start_response)

    server.wsgi_app = dispatch
