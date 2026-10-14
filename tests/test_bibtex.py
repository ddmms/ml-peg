"""Check bibliography selection, deduplication, coverage and download contents."""

from __future__ import annotations

from base64 import b64decode
from pathlib import Path

from dash import dcc, html
from pybtex.database import parse_string
import pytest

from ml_peg.app.utils.build_bibtex import build_bibtex_controls
from ml_peg.app.utils.build_credits import build_benchmark_credit_components
from ml_peg.app.utils.build_frameworks import build_framework_views
from ml_peg.utils import bibtex
from ml_peg.utils.bibtex import collect_bibtex, combine_bibtex
from ml_peg.utils.citations import BenchmarkCredits, Citation, load_benchmark_credits


@pytest.fixture
def library(tmp_path, monkeypatch):
    """Use a small real bibliography containing a DOI duplicate."""
    path = tmp_path / "references.bib"
    path.write_text(
        "@article{paper, title={Original}, author={García, Ana}, year={2025}, "
        "doi={10.1234/example}, url={https://example.org/paper}}\n"
        "@article{duplicate, title={Original}, author={García, Ana}, year={2025}, "
        "doi={10.1234/example}}\n"
        "@misc{other, title={Other}, author={Smith, Jo}, year={2026}}\n"
    )
    registry = tmp_path / "frameworks.yml"
    registry.write_text(
        "mlip_audit:\n  label: Source\n  type: paper\n"
        "  paper_url: https://example.org/paper\n"
    )
    monkeypatch.setattr(bibtex, "BIBTEX_FILE", path)
    monkeypatch.setattr(bibtex, "FRAMEWORKS_FILE", registry)
    bibtex._library.cache_clear()
    yield
    bibtex._library.cache_clear()


def credits(key, title="Original"):
    """Build credits selecting a supplied record."""
    return BenchmarkCredits(
        contributors=(),
        citations=(
            Citation(
                key="display-key", title=title, authors=("Ana García",), bibtex_key=key
            ),
        ),
    )


def walk(component):
    """Walk Dash components without depending on generated clipboard IDs."""
    yield component
    children = getattr(component, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None:
            yield from walk(child)


def test_download_round_trip_and_clipboard(library):
    """The download and clipboard contain the same parseable, deduplicated records."""
    export = combine_bibtex(
        [collect_bibtex(credits("paper")), collect_bibtex(credits("duplicate"))]
    )
    controls = build_bibtex_controls(export, "My benchmark", preview=True)
    nodes = list(walk(controls))
    link = next(n for n in nodes if isinstance(n, html.A))
    copied = next(n.content for n in nodes if isinstance(n, dcc.Clipboard))
    downloaded = b64decode(link.href.split(",", 1)[1]).decode("utf-8")
    assert downloaded == copied == export.text
    assert len(parse_string(downloaded, "bibtex").entries) == 1
    assert "García" in downloaded
    assert link.download == "mlpeg-My-benchmark.bib"


def test_missing_bibtex_is_visible_and_never_fabricated(library):
    """Missing references are named and cannot be downloaded as invented records."""
    export = combine_bibtex(
        [
            collect_bibtex(credits("paper")),
            collect_bibtex(credits("missing", "Unprovided paper")),
        ]
    )
    controls = build_bibtex_controls(export)
    text = str(controls)
    assert "Download is incomplete" in text
    assert "Unprovided paper" in text
    assert "Unprovided paper" not in export.text
    empty = build_bibtex_controls(collect_bibtex(None, label="Empty benchmark"))
    assert not any(isinstance(n, html.A) for n in walk(empty))
    assert "Empty benchmark" in str(empty)


def test_framework_includes_source_once_and_only_its_benchmarks(library):
    """Framework exports exclude unrelated benchmarks and retain the source paper."""
    first = collect_bibtex(credits("paper"), ["mlip_audit", "ml_peg"])
    assert len(first.entries) == 1
    assert not first.missing
    assert collect_bibtex(None, ["mlip_audit"]).entries == first.entries
    views = build_framework_views(
        {
            "Category": {
                "tests": [
                    {
                        "name": "A",
                        "key": "A",
                        "speed": None,
                        "framework_ids": ["mlip_audit"],
                        "layout": html.Div(),
                        "bibtex": first,
                    },
                    {
                        "name": "B",
                        "key": "B",
                        "speed": None,
                        "framework_ids": [],
                        "layout": html.Div(),
                        "bibtex": collect_bibtex(credits("other")),
                    },
                ]
            }
        },
        {"mlip_audit"},
    )
    assert views["mlip_audit"]["bibtex"].entries == first.entries
    assert len(views["mlip_audit"]["benchmarks_by_category"]["Category"]) == 1


def test_benchmark_credit_ui_uses_metadata_key(library, tmp_path):
    """A YAML BibTeX key survives loading and reaches the benchmark controls."""
    p = tmp_path / "citations.yml"
    p.write_text(
        "implementers: []\ncontributors: []\ncitations:\n"
        "  - key: display-key\n    bibtex_key: paper\n    title: Original\n"
        "    authors: [Ana García]\n    role: reference_data\n"
    )
    loaded = load_benchmark_credits(p)
    assert loaded.citations[0].bibtex_key == "paper"
    ui = build_benchmark_credit_components(loaded, name="Example")
    references = next(n for n in walk(ui) if isinstance(n, html.Details))
    preview = next(
        n
        for n in walk(references)
        if isinstance(n, html.Details) and n is not references
    )
    assert preview.className == "mlpeg-bibtex-preview"
    assert any(isinstance(n, dcc.Clipboard) for n in walk(preview))
    assert any(
        isinstance(n, html.A) and getattr(n, "download", None) for n in walk(preview)
    )
    assert not getattr(references, "open", False)
    assert not getattr(preview, "open", False)


def test_repository_bibtex_links_resolve():
    """Every explicit imported key points to a parseable record."""
    records, _ = bibtex._library()
    assert records
    for path in Path("ml_peg/calcs").glob("*/*/citations.yml"):
        for citation in load_benchmark_credits(path).citations:
            if citation.bibtex_key:
                assert citation.bibtex_key in records, path


def test_source_key_works_without_matching_url(library):
    """Explicit source keys include the paper without a false missing warning."""
    bibtex.FRAMEWORKS_FILE.write_text(
        "mlip_audit:\n  label: Source\n  bibtex_key: other\n"
        "  paper_url: https://example.org/a-different-url\n"
        "  citation:\n    title: Other\n"
    )
    bibtex._frameworks.cache_clear()
    source_credits = BenchmarkCredits(
        contributors=(),
        citations=(
            Citation(
                key="mlip_audit",
                title="Other",
                authors=("Jo Smith",),
                role="upstream_framework",
            ),
        ),
    )
    export = collect_bibtex(source_credits, ["mlip_audit"])
    assert set(export.entries) == {"other"}
    assert not export.missing
