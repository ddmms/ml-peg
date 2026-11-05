"""Collect calculation outputs into one bundle, and distribute a bundle back."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import groupby
from operator import itemgetter
from pathlib import Path, PurePosixPath
import shutil
import tarfile


@dataclass
class DistributeSummary:
    """
    Files copied and skipped by `distribute_outputs`.

    Attributes
    ----------
    copied
        Number of files copied, keyed by ``<category>/<benchmark>/<outputs>/<model>``
        (or ``<category>/<benchmark>/<outputs>`` for files outside a model folder).
    skipped
        Number of files skipped because they already existed, with the same keys.
    unknown
        Sorted ``<category>/<benchmark>`` entries missing from the calcs tree, whose
        files were not written.
    """

    copied: Counter[str] = field(default_factory=Counter)
    skipped: Counter[str] = field(default_factory=Counter)
    unknown: list[str] = field(default_factory=list)


def _output_files(
    calcs_root: Path, category: str, test: str, models: list[str] | None
) -> Iterator[tuple[str, Path]]:
    """
    Yield each output file to collect, with its benchmark.

    Parameters
    ----------
    calcs_root
        Root of the calcs tree.
    category
        Category glob to collect.
    test
        Benchmark glob to collect.
    models
        Model folders to keep. Default is None, which keeps every model. Files
        directly inside an outputs directory are always kept.

    Yields
    ------
    tuple[str, Path]
        ``<category>/<benchmark>`` and the path of a file to collect.
    """
    for outputs_dir in sorted(calcs_root.glob(f"{category}/{test}/outputs*")):
        if not outputs_dir.is_dir():
            continue
        benchmark = outputs_dir.parent.relative_to(calcs_root).as_posix()
        for path in sorted(outputs_dir.rglob("*")):
            if not path.is_file():
                continue
            parts = path.relative_to(outputs_dir).parts
            if models is not None and len(parts) > 1 and parts[0] not in models:
                continue
            yield benchmark, path


def collect_outputs(
    calcs_root: Path,
    dest: Path,
    category: str = "*",
    test: str = "*",
    models: list[str] | None = None,
    archive: bool = False,
    on_benchmark: Callable[[str, int], None] | None = None,
) -> Path:
    """
    Copy calculation outputs into a folder or ``.tar.gz`` mirroring the calcs tree.

    Every ``outputs*`` directory of each matching benchmark is collected, as
    ``<category>/<benchmark>/<outputs dir>/...`` relative to the bundle root.

    Parameters
    ----------
    calcs_root
        Root of the calcs tree to collect from.
    dest
        Folder to collect into, or the archive path without ``.tar.gz`` when
        `archive` is True.
    category
        Category glob to collect. Default is all categories.
    test
        Benchmark glob to collect. Default is all benchmarks.
    models
        Model folders to collect. Default is None, which collects every model.
    archive
        Whether to write a ``.tar.gz`` directly instead of a folder, which avoids
        a second on-disk copy of the outputs. Default is False.
    on_benchmark
        Called with ``<category>/<benchmark>`` and its file count once each
        benchmark is collected. Default is None.

    Returns
    -------
    Path
        The folder or archive written.

    Raises
    ------
    ValueError
        If no output files match.
    """
    pattern = f"{category}/{test}/outputs*"
    # Loose files outside model folders are always kept, so check the models
    # themselves, or a mistyped model would still collect those files.
    if models is not None and not any(
        path.is_dir()
        for model in models
        for path in calcs_root.glob(f"{pattern}/{model}")
    ):
        raise ValueError(
            f"No outputs found for models {', '.join(models)} matching {pattern} "
            f"in {calcs_root}"
        )

    bundle = dest.with_name(f"{dest.name}.tar.gz") if archive else dest
    counts: Counter[str] = Counter()

    def copy_all(add: Callable[[Path, str], None]) -> None:
        """
        Add every matching output file to the bundle, counting files per benchmark.

        Parameters
        ----------
        add
            Called with each file's path and its name relative to the bundle root.
        """
        files = _output_files(calcs_root, category, test, models)
        for benchmark, group in groupby(files, key=itemgetter(0)):
            for _, path in group:
                add(path, path.relative_to(calcs_root).as_posix())
                counts[benchmark] += 1
            if on_benchmark:
                on_benchmark(benchmark, counts[benchmark])

    if archive:
        bundle.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle, "w:gz") as tar:
            copy_all(lambda path, name: tar.add(path, arcname=name))
    else:

        def copy_file(path: Path, name: str) -> None:
            """
            Copy a file into the bundle folder, creating parent folders as needed.

            Parameters
            ----------
            path
                File to copy.
            name
                Path of the copy relative to the bundle root.
            """
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

        copy_all(copy_file)

    if not counts:
        if archive:
            bundle.unlink()
        raise ValueError(f"No outputs found matching {pattern} in {calcs_root}")
    return bundle


def _bundle_entries(source: Path) -> Iterator[tuple[PurePosixPath, Callable]]:
    """
    Yield each file in a bundle folder or archive with a way to open it.

    Parameters
    ----------
    source
        Bundle folder, or ``.tar.gz`` archive.

    Yields
    ------
    tuple[PurePosixPath, Callable]
        Path of the file relative to the bundle root, and a function returning
        an open binary file for its contents.

    Raises
    ------
    ValueError
        If an archive entry is a link, device or similar, or its path would
        escape the bundle root.
    """
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file():
                rel = PurePosixPath(path.relative_to(source).as_posix())
                yield rel, lambda path=path: path.open("rb")
        return

    with tarfile.open(source, "r:*") as tar:
        for member in tar:
            if member.isdir():
                continue
            rel = PurePosixPath(member.name)
            if not member.isfile() or rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Unsafe archive entry refused: {member.name}")
            yield rel, lambda member=member: tar.extractfile(member)


def distribute_outputs(source: Path, calcs_root: Path) -> DistributeSummary:
    """
    Copy a collected bundle back into the calcs tree, never overwriting files.

    Parameters
    ----------
    source
        Bundle folder, or ``.tar.gz`` archive, written by `collect_outputs`.
    calcs_root
        Root of the calcs tree to distribute into.

    Returns
    -------
    DistributeSummary
        Files copied and skipped, and benchmarks missing from `calcs_root`.

    Raises
    ------
    ValueError
        If an archive entry is unsafe, or a file is not inside an outputs
        directory of a benchmark.
    """
    summary = DistributeSummary()
    unknown = set()

    for rel, open_file in _bundle_entries(source):
        parts = rel.parts
        if len(parts) < 4 or not parts[2].startswith("outputs"):
            raise ValueError(f"Not a benchmark output file: {rel}")
        benchmark = "/".join(parts[:2])
        if not (calcs_root / benchmark).is_dir():
            unknown.add(benchmark)
            continue

        key = "/".join(parts[:4] if len(parts) > 4 else parts[:3])
        target = calcs_root.joinpath(*parts)
        if target.exists():
            summary.skipped[key] += 1
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        with open_file() as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        summary.copied[key] += 1

    summary.unknown = sorted(unknown)
    return summary
