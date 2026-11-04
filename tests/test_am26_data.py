"""Test pinned AM26 retrieval and public metadata without network or disk writes."""

from __future__ import annotations

import ast
from hashlib import sha256
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from requests import HTTPError
from test_am26_analysis import memory_files as memory_files
from test_am26_analysis import reference_frames as reference_frames
from test_am26_analysis import xyz_bytes
import yaml

from ml_peg.analysis.amorphous_materials.AM26 import analyse_AM26 as analysis
from ml_peg.app.amorphous_materials.AM26 import app_AM26
from ml_peg.calcs.amorphous_materials.AM26 import calc_AM26 as calculation
from ml_peg.calcs.amorphous_materials.AM26 import data
from ml_peg.calcs.utils import utils as download_utils
from ml_peg.citations.bibliography import citation_fields
from ml_peg.utils.citations import load_benchmark_credits

ROOT = Path(__file__).parents[1]
DOC_PATH = ROOT / "docs/source/user_guide/benchmarks/amorphous_materials.rst"


@pytest.fixture
def cached_reference(memory_files, reference_frames, monkeypatch):
    """Exercise the existing downloader/cache with synthetic response bytes."""
    root, files = memory_files
    payload = xyz_bytes(reference_frames)
    monkeypatch.delenv(data.DATA_PATH_ENV, raising=False)
    monkeypatch.setattr(data, "REFERENCE_SHA256", sha256(payload).hexdigest())
    monkeypatch.setattr(download_utils, "BENCHMARK_DATA_DIR", root / "cache")
    response = Mock(content=payload)
    getter = Mock(return_value=response)
    monkeypatch.setattr(download_utils.requests, "get", getter)
    return root, files, payload, getter


def test_production_identity_is_immutable():
    """Pin the author-controlled canonical file, never an evolving branch."""
    assert data.REFERENCE_REVISION == "9218ee127f8db5ff06edecdf209f741d0eafb7c6"
    assert data.REFERENCE_FILE == "data/am26.extxyz"
    assert f"{data.REFERENCE_URI}/am26.extxyz" == (
        "https://raw.githubusercontent.com/vldgroup/AM26/"
        "9218ee127f8db5ff06edecdf209f741d0eafb7c6/data/am26.extxyz"
    )
    assert data.REFERENCE_SHA256 == (
        "a071f537305e78e3bb1acf5dbffaa9acf1f79e2eb5081cb59218beddaf5a802a"
    )


def test_existing_github_helper_caches_and_accepts_correct_checksum(cached_reference):
    """Only the helper's cache receives synthetic bytes; repeat calls reuse it."""
    root, files, payload, getter = cached_reference
    path = data.get_reference_path()
    assert path == root / "cache/am26.extxyz"
    assert files == {path: payload}
    assert data.get_reference_path() == path
    getter.assert_called_once_with(f"{data.REFERENCE_URI}/am26.extxyz")
    getter.return_value.raise_for_status.assert_called_once()


@pytest.mark.parametrize("cached", [False, True])
def test_wrong_download_or_cache_checksum_is_rejected(cached_reference, cached):
    """Reject bad bytes without deleting, accepting or overwriting the file."""
    root, files, _, getter = cached_reference
    path = root / "cache/am26.extxyz"
    if cached:
        files[path] = b"invalid reference"
    else:
        getter.return_value.content = b"invalid reference"
    with pytest.raises(ValueError, match="AM26 SHA-256 mismatch.*expected.*found"):
        data.get_reference_path()
    assert files[path] == b"invalid reference"
    if cached:
        getter.assert_not_called()


@pytest.mark.parametrize("selection", ["argument", "environment"])
def test_local_reference_is_read_in_place_without_network(
    cached_reference, monkeypatch, selection
):
    """Local development reads a supplied file with no cache or source-tree copy."""
    root, files, payload, getter = cached_reference
    path = root / "author-checkout/data/am26.extxyz"
    files[path] = payload
    before = files.copy()
    if selection == "environment":
        monkeypatch.setenv(data.DATA_PATH_ENV, str(path))
        result = data.get_reference_path()
    else:
        result = data.get_reference_path(path)
    assert result == path
    assert files == before
    getter.assert_not_called()


def test_explicit_local_argument_takes_precedence(cached_reference, monkeypatch):
    """A supplied Python path overrides an unrelated environment selection."""
    root, files, payload, getter = cached_reference
    path = root / "author-checkout/data/am26.extxyz"
    files[path] = payload
    monkeypatch.setenv(data.DATA_PATH_ENV, str(root / "missing.extxyz"))
    assert data.get_reference_path(path) == path
    getter.assert_not_called()


def test_invalid_local_reference_never_falls_back_to_download(cached_reference):
    """A wrong local checksum is a clear error, never a silent data substitution."""
    root, files, _, getter = cached_reference
    path = root / "author-checkout/data/am26.extxyz"
    files[path] = b"wrong local file"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        data.get_reference_path(path)
    getter.assert_not_called()
    assert files[path] == b"wrong local file"


def test_missing_local_reference_never_falls_back_to_download(cached_reference):
    """A missing explicit path remains a local file error."""
    root, _, _, getter = cached_reference
    with pytest.raises(FileNotFoundError):
        data.get_reference_path(root / "missing.extxyz")
    getter.assert_not_called()


def test_http_error_does_not_create_cache_entry(cached_reference):
    """Use the existing helper's HTTP failure rather than manufacturing success."""
    _, files, _, getter = cached_reference
    getter.return_value.raise_for_status.side_effect = HTTPError("synthetic 404")
    with pytest.raises(HTTPError, match="synthetic 404"):
        data.get_reference_path()
    assert not files


def test_shared_zip_extraction_still_returns_archive_parent(monkeypatch):
    """The uncompressed-file fix leaves the existing ZIP branch unchanged."""
    path = ROOT / ".am26-analysis-memory/cache/example.zip"
    archive = MagicMock()
    constructor = Mock(return_value=archive)
    monkeypatch.setattr(download_utils.zipfile, "ZipFile", constructor)
    assert download_utils.extract_zip(path) == path.parent
    constructor.assert_called_once_with(path, "r")
    archive.__enter__.return_value.extractall.assert_called_once_with(path.parent)


def test_calculation_consumes_verified_local_file(
    cached_reference, reference_frames, monkeypatch
):
    """Use real ASE decoding of synthetic cached data, without any calculator."""
    root, files, payload, getter = cached_reference
    path = root / "author-checkout/data/am26.extxyz"
    files[path] = payload
    monkeypatch.setattr(
        calculation, "REFERENCE_COUNTS", dict.fromkeys(analysis.SYSTEM_ORDER, 2)
    )
    frames = calculation.load_reference_structures(path)
    assert [a.info["label"] for a in frames] == [
        a.info["label"] for a in reference_frames
    ]
    getter.assert_not_called()
    assert list(files) == [path]


def test_retrieval_uses_shared_helper_without_s3_or_copy_infrastructure():
    """AM26 adds selection/integrity checks, not a second download framework."""
    tree = ast.parse(Path(data.__file__).read_text())
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "download_github_data" in calls
    imports = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert "ml_peg.calcs.utils.utils" in imports
    source = Path(data.__file__).read_text()
    assert not any(
        word in source
        for word in (
            "download_s3",
            "requests.",
            "urllib",
            "copyfile",
            "write_bytes",
            "unlink",
        )
    )


def test_am26_citations_validate_with_complete_primary_author_order():
    """Use normal citation validation and verified BibTeX metadata."""
    path = Path(calculation.__file__).with_name("citations.yml")
    credits = load_benchmark_credits(path)
    primary = next(c for c in credits.citations if c.role == "benchmark_method")
    assert primary.title == (
        "Amorphous materials as a frontier challenge "
        "for universal interatomic potentials"
    )
    assert primary.authors == ("Natascia L. Fragapane", "Volker L. Deringer")
    assert primary.year == 2026
    assert primary.url == "https://arxiv.org/abs/2607.11384"
    assert primary.doi == "10.48550/arXiv.2607.11384"
    assert len(credits.implementers) == 1
    implementer = credits.implementers[0]
    assert implementer.name == "Natascia L. Fragapane"
    assert implementer.github == "nfragapane"
    assert implementer.email == "natascia.fragapane@chem.ox.ac.uk"
    assert credits.contributors == ()
    roles = {c.bibtex_key: c.role for c in credits.citations}
    assert roles == {
        "fragapane2026am26": "benchmark_method",
        "fragapane2026am26data": "reference_data",
        "rosset2025paracrystallinity": "reference_data",
        "zhou2023phasechange": "reference_data",
        "deringer-2017-amorphous-carbon": "inspired_by",
        "erhard2024siliconoxygen": "inspired_by",
        "fragapane2026lips": "inspired_by",
    }
    details = {c.bibtex_key: c.detail for c in credits.citations}
    assert details["fragapane2026am26data"] == "Canonical AM26 dataset"
    for key, system in (
        ("deringer-2017-amorphous-carbon", "a-C"),
        ("erhard2024siliconoxygen", "a-SiO2"),
        ("fragapane2026lips", "a-LiPS"),
    ):
        assert details[key] == f"{system} structure-generation potential"
    dataset = citation_fields("fragapane2026am26data")
    assert dataset["url"] == "https://github.com/vldgroup/AM26"


def test_documentation_loads_without_warnings_or_external_tools():
    """Parse the existing reST page, including AM26 equations and references."""
    publish_doctree = pytest.importorskip("docutils.core").publish_doctree
    warnings = StringIO()
    document = publish_doctree(
        DOC_PATH.read_text(),
        source_path=str(DOC_PATH),
        settings_overrides={
            "_disable_config": True,
            "warning_stream": warnings,
            "halt_level": 2,
        },
    )
    assert warnings.getvalue() == ""
    assert document.ids["am26"] is not None
    text = document.astext()
    for expected in (
        data.REFERENCE_REVISION,
        data.REFERENCE_SHA256,
        "ddof=0",
        "RAW, UNSHIFTED",
        "under peer review",
        "licensing and redistribution terms remain unsettled",
    ):
        assert expected in text
    assert "in preparation" not in text
    assert app_AM26.DOCS_URL.endswith("amorphous_materials.html#am26")


def test_public_integration_keeps_only_two_scored_metrics():
    """Publication metadata does not introduce additional scientific scores."""
    assert analysis.METRIC_NAMES == ("Mean Energy RelRMSE", "Mean Force RelRMSE")
    for values in analysis.DEFAULT_THRESHOLDS.values():
        assert values["good"] == 0.0
        assert values["bad"] == 100.0
        assert values["unit"] == "%"
        assert values["level_of_theory"] == "PBE"
    category = yaml.safe_load(
        (ROOT / "ml_peg/app/amorphous_materials/amorphous_materials.yml").read_text()
    )
    assert "energy/force transferability" in category["description"]
