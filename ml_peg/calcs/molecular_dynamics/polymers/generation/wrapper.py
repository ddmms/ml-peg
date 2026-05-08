"""
Subprocess wrapper around the EMC polymer-cell builder.

This module defines :class:`Config` (the per-build parameters) and
:func:`prepare`, which renders the EMC `.esh` template, runs ``emc_setup.pl``
followed by the EMC binary in a working directory, and produces the LAMMPS
data file (``system.data``) that downstream code converts to ASE structures.

The EMC binary is shipped via the optional ``emc-pypi`` distribution
(installed via ``pip install "emc-pypi>=2025.8.21"``), which exposes the
``pyemc`` Python module. We import it lazily so that the wider polymers
package can be imported without it.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import re
import signal
import subprocess
import typing as ty

LOG = logging.getLogger(__name__)


def _wrap(name: str, symbol: str = "_") -> str:
    """
    Wrap ``name`` with the placeholder symbol on both sides.

    Parameters
    ----------
    name
        Placeholder key (without the surrounding symbol).
    symbol
        Wrapping symbol. Default: ``"_"``.

    Returns
    -------
    str
        ``f"{symbol}{name}{symbol}"``.
    """
    return symbol + name + symbol


def _render_template(template: str, values: ty.Mapping[str, ty.Any]) -> str:
    """
    Substitute ``_<key>_`` placeholders in ``template`` from ``values``.

    Booleans are rendered as ``"true"`` / ``"false"``; everything else is
    rendered with ``str(...)``. Raises if a key has no matching placeholder
    or if any ``_<word>_`` placeholder remains after substitution.

    Parameters
    ----------
    template
        Raw template string with ``_<key>_`` placeholders.
    values
        Mapping from placeholder key to replacement value.

    Returns
    -------
    str
        ``template`` with every placeholder replaced.
    """
    rendered = template
    for key, value in values.items():
        placeholder = _wrap(key)
        if value is True:
            formatted = "true"
        elif value is False:
            formatted = "false"
        else:
            formatted = str(value)
        new_rendered = rendered.replace(placeholder, formatted)
        if new_rendered == rendered:
            raise RuntimeError(f"Replacement of '{placeholder}' had no effect")
        rendered = new_rendered

    leftover = re.compile(r"\s_(\w+)_\s").search(rendered)
    if leftover is not None:
        raise AssertionError(f"Unresolved placeholder remains: {leftover.group(0)}")

    return rendered


def _emc_root() -> pathlib.Path:
    """
    Return the root EMC directory bundled with ``pyemc``.

    Returns
    -------
    pathlib.Path
        ``<pyemc package dir>/emc``.
    """
    import pyemc  # type: ignore[import-not-found]

    return pathlib.Path(pyemc.__file__).parent / "emc"


def _emc_setup_path() -> pathlib.Path:
    """
    Return the path to the EMC setup Perl script.

    Returns
    -------
    pathlib.Path
        ``<emc root>/scripts/emc_setup.pl``.
    """
    return _emc_root() / "scripts" / "emc_setup.pl"


def _emc_binary() -> pathlib.Path:
    """
    Return the path to the EMC binary.

    Returns
    -------
    pathlib.Path
        ``<emc root>/bin/emc_linux_x86_64``.
    """
    return _emc_root() / "bin" / "emc_linux_x86_64"


def _read_emc_template() -> str:
    """
    Return the bundled EMC ``.esh`` template as a string.

    Returns
    -------
    str
        UTF-8-decoded contents of ``resources/emc_template.esh``.
    """
    template_path = pathlib.Path(__file__).parent / "resources" / "emc_template.esh"
    return template_path.read_text(encoding="utf-8")


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """Per-build EMC configuration."""

    ru_smiles: str
    first_cap: str
    second_cap: str
    n_ru_per_chain: int
    n_total: int
    density: float
    temperature: float
    seed: int = 42
    pdb: bool = False  # PDB output can crash EMC; off by default

    def render(self) -> str:
        """
        Render the EMC ``.esh`` script for this configuration.

        Returns
        -------
        str
            The fully-substituted ``.esh`` script ready to write to disk.
        """
        template = _read_emc_template()
        values = dataclasses.asdict(self)
        return _render_template(template, values)


class EMCError(RuntimeError):
    """
    Generic error raised when EMC (setup or simulation) fails.

    Parameters
    ----------
    return_code
        Process exit code (negative for signal termination).
    stdout
        Captured standard output.
    stderr
        Captured standard error.
    """

    def __init__(self, *, return_code: int, stdout: str, stderr: str) -> None:
        """
        Construct from the EMC process exit metadata.

        Parameters
        ----------
        return_code
            Process exit code (negative for signal termination).
        stdout
            Captured standard output.
        stderr
            Captured standard error.
        """
        super().__init__(f"EMC failed (return_code={return_code})")
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class SegfaultError(EMCError):
    """Raised when the EMC process is killed by SIGSEGV."""


class MissingForceFieldParametersError(EMCError):
    """Raised when EMC reports missing force-field parameters."""


class AmbiguousChargeAssignmentError(EMCError):
    """Raised when EMC reports ambiguous charge assignments for a group."""


class TotalChargeError(EMCError):
    """Raised when the total charge of the EMC system does not equal zero."""


class SimulationError(EMCError):
    """Raised when EMC fails inside the (Monte Carlo) packing simulation."""


class SetupError(EMCError):
    """Raised when ``emc_setup.pl`` itself returns a non-zero exit code."""


def _classify_error(*, return_code: int, stdout: str, stderr: str) -> EMCError:
    """
    Map an EMC failure into the most specific :class:`EMCError` subclass.

    Parameters
    ----------
    return_code
        Process exit code (negative for signal termination).
    stdout
        Captured standard output.
    stderr
        Captured standard error.

    Returns
    -------
    EMCError
        The most-specific EMCError subclass that matches the failure.
    """
    if "Missing force field parameters." in stdout:
        cls: type[EMCError] = MissingForceFieldParametersError
    elif "Ambiguous charge assignments for group" in stdout:
        cls = AmbiguousChargeAssignmentError
    elif "Total charge of system 'main' does not equal zero" in stdout:
        cls = TotalChargeError
    elif "Error: core/types/inverse/angle.c:377 InverseAngleInit:" in stdout:
        cls = SimulationError  # triggered by very high temperatures
    elif return_code == -signal.SIGSEGV:
        cls = SegfaultError
    else:
        cls = EMCError
    return cls(return_code=return_code, stdout=stdout, stderr=stderr)


def _run(
    args: ty.Sequence[str], working_dir: pathlib.Path
) -> subprocess.CompletedProcess[bytes]:
    """
    Run an external command in ``working_dir`` and capture its output.

    Parameters
    ----------
    args
        Argv-style command and arguments.
    working_dir
        Directory the command runs in.

    Returns
    -------
    subprocess.CompletedProcess[bytes]
        Result with stdout/stderr captured as raw bytes.
    """
    return subprocess.run(
        args=list(args), cwd=working_dir, capture_output=True, shell=False
    )


def prepare(
    config: Config,
    directory: pathlib.Path,
    *,
    clean_up: bool = True,
    n_threads: int = 1,
) -> pathlib.Path:
    """
    Render the EMC config and produce a LAMMPS data file for ``config``.

    Parameters
    ----------
    config
        Per-build EMC configuration.
    directory
        Working directory where intermediate and output files are written.
    clean_up
        Whether to remove EMC's intermediate ``.emc.gz`` and ``.in`` artifacts
        after a successful run. Default: True.
    n_threads
        Threads to pass to the EMC binary.

    Returns
    -------
    pathlib.Path
        Path to the produced LAMMPS data file (``<directory>/system.data``).
    """
    directory.mkdir(parents=True, exist_ok=True)

    esh_path = directory / "build.esh"
    esh_path.write_text(config.render(), encoding="utf-8")

    setup = _run([str(_emc_setup_path()), "build.esh"], working_dir=directory)
    if setup.returncode != 0:
        raise SetupError(
            return_code=setup.returncode,
            stdout=setup.stdout.decode(),
            stderr=setup.stderr.decode(),
        )

    emc = _run(
        [str(_emc_binary()), f"-nthreads={n_threads}", "build.emc"],
        working_dir=directory,
    )
    if emc.returncode != 0:
        raise _classify_error(
            return_code=emc.returncode,
            stdout=emc.stdout.decode(),
            stderr=emc.stderr.decode(),
        )

    if clean_up:
        for suffix in (".emc.gz", ".in"):
            stale = directory / f"system{suffix}"
            if stale.exists():
                LOG.debug(f"Removing {stale}")
                stale.unlink()

    return directory / "system.data"
