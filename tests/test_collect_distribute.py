"""Tests for collecting calculation outputs and distributing them elsewhere."""

from __future__ import annotations

import io
from pathlib import Path
import tarfile

import pytest
from typer.testing import CliRunner

from ml_peg.cli import cli
from ml_peg.data.outputs import collect_outputs, distribute_outputs

runner = CliRunner()

FILES = (
    "bulk_crystal/phonons/outputs/mace-mp-0a/mp-1.json",
    "bulk_crystal/phonons/outputs/orb-v3/mp-1.json",
    "bulk_crystal/phonons/outputs/DFT/mp-1.json",
    "bulk_crystal/phonons/outputs/summary.json",
    "carbon/GAP_20/outputs_force/mace-mp-0a/results.xyz",
    "molecular/GMTKN55/outputs/mace-mp-0a/nested/dir/results.xyz",
)


def make_calcs_root(root: Path, files: tuple[str, ...] = FILES) -> Path:
    """
    Build a fake calcs tree, with a calc script in every benchmark.

    Parameters
    ----------
    root
        Directory to build the tree in.
    files
        Output files to create, relative to `root`.

    Returns
    -------
    Path
        The root of the fake calcs tree.
    """
    for file in files:
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file)
        category, benchmark = Path(file).parts[:2]
        (root / category / benchmark / f"calc_{benchmark}.py").touch()
    return root


def empty_calcs_root(root: Path) -> Path:
    """
    Build a fake calcs tree with the same benchmarks as `make_calcs_root`, no outputs.

    Parameters
    ----------
    root
        Directory to build the tree in.

    Returns
    -------
    Path
        The root of the fake calcs tree.
    """
    for file in FILES:
        category, benchmark = Path(file).parts[:2]
        (root / category / benchmark).mkdir(parents=True, exist_ok=True)
    return root


def tree(root: Path) -> dict[str, str]:
    """
    Map every output file under `root` to its contents.

    Parameters
    ----------
    root
        Directory to read.

    Returns
    -------
    dict[str, str]
        File contents keyed by path relative to `root`.
    """
    return {
        path.relative_to(root).as_posix(): path.read_text()
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("calc_")
    }


@pytest.mark.parametrize("archive", (False, True))
def test_round_trip(tmp_path, archive):
    """Collecting then distributing reproduces every output file exactly."""
    hpc = make_calcs_root(tmp_path / "hpc")
    local = empty_calcs_root(tmp_path / "local")

    bundle = collect_outputs(hpc, tmp_path / "bundle", archive=archive)
    assert bundle == tmp_path / ("bundle.tar.gz" if archive else "bundle")

    distribute_outputs(bundle, local)
    assert tree(local) == tree(hpc)


def test_collect_counts_files_per_benchmark(tmp_path):
    """Collect reports how many files each benchmark contributed."""
    hpc = make_calcs_root(tmp_path / "hpc")
    counts = {}
    collect_outputs(hpc, tmp_path / "bundle", on_benchmark=counts.__setitem__)
    assert counts == {
        "bulk_crystal/phonons": 4,
        "carbon/GAP_20": 1,
        "molecular/GMTKN55": 1,
    }


def test_collect_includes_output_variants(tmp_path):
    """Sibling outputs_* directories are collected alongside outputs."""
    hpc = make_calcs_root(tmp_path / "hpc")
    bundle = collect_outputs(hpc, tmp_path / "bundle")
    assert (bundle / "carbon/GAP_20/outputs_force/mace-mp-0a/results.xyz").is_file()


def test_collect_filters_category_and_test(tmp_path):
    """Only benchmarks matching the category and test are collected."""
    hpc = make_calcs_root(tmp_path / "hpc")
    bundle = collect_outputs(
        hpc, tmp_path / "bundle", category="bulk_crystal", test="phonons"
    )
    assert {path.split("/")[1] for path in tree(bundle)} == {"phonons"}


@pytest.mark.parametrize("archive", (False, True))
def test_collect_filters_models(tmp_path, archive):
    """Only the requested model folders are kept, plus loose top-level files."""
    hpc = make_calcs_root(tmp_path / "hpc")
    local = empty_calcs_root(tmp_path / "local")
    bundle = collect_outputs(
        hpc, tmp_path / "bundle", models=["mace-mp-0a"], archive=archive
    )
    distribute_outputs(bundle, local)
    assert set(tree(local)) == {
        "bulk_crystal/phonons/outputs/mace-mp-0a/mp-1.json",
        "bulk_crystal/phonons/outputs/summary.json",
        "carbon/GAP_20/outputs_force/mace-mp-0a/results.xyz",
        "molecular/GMTKN55/outputs/mace-mp-0a/nested/dir/results.xyz",
    }


def test_collect_with_no_outputs_raises(tmp_path):
    """Collecting nothing is an error rather than an empty bundle."""
    hpc = make_calcs_root(tmp_path / "hpc")
    with pytest.raises(ValueError, match="No outputs"):
        collect_outputs(hpc, tmp_path / "bundle", models=["not-a-model"])


@pytest.mark.parametrize("archive", (False, True))
def test_distribute_skips_existing_files(tmp_path, archive):
    """Existing local files are never overwritten, and are counted as skipped."""
    hpc = make_calcs_root(tmp_path / "hpc")
    local = empty_calcs_root(tmp_path / "local")
    existing = local / "bulk_crystal/phonons/outputs/mace-mp-0a/mp-1.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("local result")

    bundle = collect_outputs(hpc, tmp_path / "bundle", archive=archive)
    summary = distribute_outputs(bundle, local)

    assert existing.read_text() == "local result"
    assert summary.skipped == {"bulk_crystal/phonons/outputs/mace-mp-0a": 1}
    assert summary.copied["bulk_crystal/phonons/outputs/orb-v3"] == 1
    assert "bulk_crystal/phonons/outputs/mace-mp-0a" not in summary.copied


def test_distribute_reports_unknown_benchmarks(tmp_path):
    """Outputs for benchmarks missing from this checkout are reported, not written."""
    hpc = make_calcs_root(tmp_path / "hpc")
    local = tmp_path / "local"
    (local / "bulk_crystal/phonons").mkdir(parents=True)

    summary = distribute_outputs(collect_outputs(hpc, tmp_path / "bundle"), local)

    assert summary.unknown == ["carbon/GAP_20", "molecular/GMTKN55"]
    assert not (local / "carbon").exists()
    assert not (local / "molecular").exists()


@pytest.mark.parametrize(
    "name", ("../evil.txt", "/tmp/evil.txt", "bulk_crystal/../../evil.txt")
)
def test_distribute_refuses_unsafe_archive_paths(tmp_path, name):
    """Archive entries that would escape the calcs tree are refused."""
    local = empty_calcs_root(tmp_path / "local")
    bundle = tmp_path / "bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = 4
        tar.addfile(info, io.BytesIO(b"evil"))

    with pytest.raises(ValueError, match="Unsafe"):
        distribute_outputs(bundle, local)
    assert not (tmp_path / "evil.txt").exists()


def test_distribute_refuses_archive_links(tmp_path):
    """Symbolic links in an archive are refused."""
    local = empty_calcs_root(tmp_path / "local")
    bundle = tmp_path / "bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        info = tarfile.TarInfo("bulk_crystal/phonons/outputs/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    with pytest.raises(ValueError, match="Unsafe"):
        distribute_outputs(bundle, local)


def test_cli_collect_and_distribute(tmp_path, monkeypatch):
    """The CLI commands collect into an archive and distribute it back."""
    hpc = make_calcs_root(tmp_path / "hpc")
    local = empty_calcs_root(tmp_path / "local")

    monkeypatch.setattr(cli, "CALCS_ROOT", hpc)
    result = runner.invoke(
        cli.app,
        ["collect", "--output", str(tmp_path / "bundle"), "--archive"],
    )
    assert result.exit_code == 0, result.output
    assert "bulk_crystal/phonons: 4 files" in result.output

    monkeypatch.setattr(cli, "CALCS_ROOT", local)
    result = runner.invoke(cli.app, ["distribute", str(tmp_path / "bundle.tar.gz")])
    assert result.exit_code == 0, result.output
    assert tree(local) == tree(hpc)

    result = runner.invoke(cli.app, ["distribute", str(tmp_path / "bundle.tar.gz")])
    assert result.exit_code == 0, result.output
    assert "bulk_crystal/phonons/outputs/mace-mp-0a: skipped 1 existing" in (
        result.output
    )
