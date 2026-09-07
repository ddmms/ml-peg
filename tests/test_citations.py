"""Tests for benchmark citation metadata and generated guidance."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

from dash.dash_table import DataTable
from dash.development.base_component import Component
from dash.html import Details, Summary
import pytest
from yaml import safe_load

from conftest import CitationReporter
from ml_peg.analysis import ANALYSIS_ROOT
from ml_peg.app.utils import build_components
from ml_peg.app.utils.build_components import (
    build_benchmark_credit_components,
    build_framework_attribution,
    build_framework_citation,
    build_test_layout,
)
from ml_peg.calcs import CALCS_ROOT
from ml_peg.citations import (
    CITATION_FILE,
    FRAMEWORKS_FILE,
    SUMMARY_WIDTH,
    BenchmarkCredits,
    Citation,
    CitationMetadataError,
    Contributor,
    build_run_citations,
    collect_benchmark_credits,
    format_citation_summary,
    load_benchmark_credits,
    load_framework_citations,
    load_model_citations,
    ml_peg_citation,
    write_citation_bundle,
)
from ml_peg.models import models_file


def _walk_components(component: Component) -> Iterator[Component]:
    """Yield a component and all nested components."""
    yield component
    children = getattr(component, "children", None)
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if isinstance(child, Component):
            yield from _walk_components(child)


def _write_credits(path: Path, key: str = "source-paper") -> None:
    """Write minimal valid benchmark citation metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""contributors:
  - name: Test Contributor
citations:
  - key: {key}
    role: reference_data
    title: Test reference data
    authors:
      - A. Author
    year: 2026
"""
    )


def _credits(*citations: Citation) -> BenchmarkCredits:
    """Build benchmark credits with one test implementer."""
    return BenchmarkCredits(
        contributors=(Contributor("Test Implementer"),), citations=citations
    )


TEST_CITATION = Citation(
    key="test-source",
    title="Test source",
    authors=("First Author", "Second Author"),
    year=2026,
    role="benchmark_method",
)


def test_load_benchmark_credits(tmp_path: Path) -> None:
    """Citation metadata preserves publication and implementation credit."""
    path = tmp_path / "citations.yml"
    _write_credits(path)

    credits = load_benchmark_credits(path)

    assert credits.contributors == (Contributor(name="Test Contributor"),)
    assert credits.citations[0].title == "Test reference data"
    assert credits.citations[0].role_label == "reference data"


def test_reject_duplicate_citation_keys(tmp_path: Path) -> None:
    """Citation keys must be unique within one benchmark."""
    path = tmp_path / "citations.yml"
    _write_credits(path)
    path.write_text(path.read_text() + path.read_text().split("citations:\n", 1)[1])

    with pytest.raises(CitationMetadataError, match="keys must be unique"):
        load_benchmark_credits(path)


def test_reject_unknown_citation_role(tmp_path: Path) -> None:
    """An unrecognised role is rejected with the offending location."""
    path = tmp_path / "citations.yml"
    _write_credits(path)
    path.write_text(path.read_text().replace("reference_data", "made_up_role"))

    with pytest.raises(CitationMetadataError, match=r"citations\[0\].role must be one"):
        load_benchmark_credits(path)


def test_empty_citations_is_distinct_from_missing_metadata(tmp_path: Path) -> None:
    """An empty citation list loads successfully and is not treated as missing."""
    path = tmp_path / "citations.yml"
    path.write_text("contributors:\n  - name: Test Contributor\ncitations: []\n")

    credits = load_benchmark_credits(path)

    assert credits.citations == ()
    assert credits.contributors == (Contributor(name="Test Contributor"),)


def test_collect_benchmark_credits_reports_missing(tmp_path: Path) -> None:
    """Collection distinguishes populated metadata from missing placeholders."""
    analysis_root = tmp_path / "analysis"
    first_script = tmp_path / "calcs" / "category" / "first" / "calc_first.py"
    second_script = tmp_path / "calcs" / "category" / "second" / "calc_second.py"
    _write_credits(analysis_root / "category" / "first" / "citations.yml")

    credits, missing = collect_benchmark_credits(
        [first_script, second_script], analysis_root
    )

    assert tuple(credits) == ("category/first",)
    assert missing == ("category/second",)


def test_write_citation_bundle_deduplicates_sources(tmp_path: Path) -> None:
    """Repeated sources produce one BibTeX entry but retain per-benchmark credit."""
    first = BenchmarkCredits(
        contributors=(Contributor("First Implementer"),), citations=(TEST_CITATION,)
    )
    second = BenchmarkCredits(
        contributors=(Contributor("Second Implementer"),), citations=(TEST_CITATION,)
    )

    markdown_path, bibtex_path = write_citation_bundle(
        {"category/first": first, "category/second": second},
        tmp_path,
        missing_benchmarks=("category/third",),
    )

    markdown = markdown_path.read_text()
    assert "First Implementer" in markdown
    assert "Second Implementer" in markdown
    assert "category/third" in markdown
    assert bibtex_path.read_text().count("@misc{test-source") == 1


def test_bundle_marks_ml_peg_only_benchmarks(tmp_path: Path) -> None:
    """A benchmark with no external source is not reported as incomplete."""
    markdown_path, _ = write_citation_bundle({"category/own": _credits()}, tmp_path)

    markdown = markdown_path.read_text()
    assert "Devised for ML-PEG" in markdown
    assert "Incomplete metadata" not in markdown


def test_bundle_reports_models_and_missing_model_citations(tmp_path: Path) -> None:
    """Models of the run are listed, and uncited models are flagged as incomplete."""
    model_citation = Citation(
        key="cited-model", title="Model paper", authors=("M. Author",), year=2026
    )

    markdown_path, bibtex_path = write_citation_bundle(
        {},
        tmp_path,
        models={"cited-model": model_citation, "uncited-model": None},
    )

    markdown = markdown_path.read_text()
    assert "M. Author (2026). Model paper." in markdown
    assert "- `uncited-model`: To be added" in markdown
    assert "Model citations have not yet been supplied for:" in markdown
    assert "@misc{cited-model" in bibtex_path.read_text()


def test_framework_citations_cover_only_source_frameworks() -> None:
    """Only frameworks registered as type: framework contribute a citation."""
    registry = safe_load(FRAMEWORKS_FILE.read_text())
    expected = {
        entry["label"]
        for name, entry in registry.items()
        if name != "ml_peg" and entry.get("type") == "framework"
    }

    citations = load_framework_citations(registry)

    # mace-* are registered as type: paper, and ML-PEG is never a source framework
    assert set(citations) == expected
    assert "ML-PEG" not in citations


def test_repository_framework_citations_are_valid() -> None:
    """Every source framework citation in the repository parses."""
    citations = load_framework_citations(safe_load(FRAMEWORKS_FILE.read_text()))

    for label, citation in citations.items():
        if citation is not None:
            assert citation.authors, label
            assert citation.title, label


def test_bundle_reports_source_frameworks(tmp_path: Path) -> None:
    """Framework citations appear once per run, and unfilled ones are flagged."""
    filled = Citation(
        key="mlip_audit",
        title="MLIPAudit",
        authors=("A. Author",),
        year=2025,
        role="upstream_framework",
    )

    markdown_path, bibtex_path = write_citation_bundle(
        {},
        tmp_path,
        frameworks={"MLIP Audit": filled, "MLIP Arena": None},
    )

    markdown = markdown_path.read_text()
    assert "## Source frameworks" in markdown
    assert "- `MLIP Audit`: A. Author (2025). MLIPAudit." in markdown
    assert "Source framework citations have not yet been supplied for:" in markdown
    assert "- `MLIP Arena`" in markdown
    assert "@misc{mlip_audit" in bibtex_path.read_text()


def test_ml_peg_citation_matches_citation_cff(tmp_path: Path) -> None:
    """The repository citation is generated from CITATION.cff, not duplicated."""
    document = safe_load(CITATION_FILE.read_text())
    citation = ml_peg_citation()

    assert citation is not None
    assert citation.doi == "10.5281/zenodo.16904444"
    assert len(citation.authors) == len(document["authors"])
    for author in document["authors"]:
        assert f"{author['given-names']} {author['family-names']}" in citation.authors

    bibtex = write_citation_bundle({}, tmp_path)[1].read_text()
    assert bibtex.startswith("@software{ml_peg,")


def test_benchmark_runs_print_guidance_without_writing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Citation guidance is reported to the terminal only, leaving no files behind."""
    monkeypatch.chdir(tmp_path)

    summary = build_run_citations(
        [ANALYSIS_ROOT / "conformers" / "ACONFL" / "analyse_ACONFL.py"],
        model_names=["mace-mp-0a"],
    )

    assert "CITATION GUIDANCE" in summary
    assert "conformers/ACONFL" in summary
    assert "MODELS (1)" in summary
    # Nothing is written, so a run leaves the working directory untouched
    assert list(tmp_path.rglob("*")) == []


def test_terminal_summary_prints_citations_and_implementers() -> None:
    """Terminal guidance includes paper authors, implementers, and models."""
    summary = format_citation_summary(
        {"category/test": _credits(TEST_CITATION)},
        missing_benchmarks=("category/other",),
        models={"uncited-model": None},
    )

    assert "First Author, Second Author (2026). Test source." in summary
    assert "implemented" in summary
    assert "Test Implementer" in summary
    assert "BENCHMARKS (2)" in summary
    assert "category/other" in summary
    assert "MODELS (1)" in summary


def test_implementer_is_separated_from_the_citation() -> None:
    """The implementer sits under its own heading, not trailing the reference."""
    summary = format_citation_summary({"category/test": _credits(TEST_CITATION)})
    lines = summary.splitlines()

    citation_label = lines.index("      benchmark citation:")
    implementer_label = lines.index("      implemented in ML-PEG by:")

    assert "First Author" in lines[citation_label + 1]
    assert "Test Implementer" in lines[implementer_label + 1]
    # The implementer never shares a line with the work being cited
    assert "Test Implementer" not in lines[citation_label + 1]
    assert citation_label < implementer_label


def test_citation_label_is_pluralised_in_the_summary() -> None:
    """The heading matches the number of sources listed for the benchmark."""
    second = Citation(
        key="second",
        title="Second source",
        authors=("Third Author",),
        role="reference_data",
    )

    one = format_citation_summary({"category/test": _credits(TEST_CITATION)})
    two = format_citation_summary({"category/test": _credits(TEST_CITATION, second)})

    assert "benchmark citation:" in one
    assert "benchmark citations:" in two


def test_doi_is_part_of_the_citation_in_the_summary() -> None:
    """The DOI reads as part of the reference, not as a detached trailing line."""
    cited = Citation(
        key="doi-source",
        title="Short title",
        authors=("A. Author",),
        year=2026,
        role="benchmark_method",
        doi="10.1234/example",
    )

    summary = format_citation_summary({"category/test": _credits(cited)})
    reference = next(line for line in summary.splitlines() if "Short title" in line)

    assert reference.endswith("https://doi.org/10.1234/example")


def test_terminal_summary_is_a_bounded_block() -> None:
    """Guidance is framed and stays within the summary width for terminal output."""
    summary = format_citation_summary(
        {"category/test": _credits(TEST_CITATION)},
        models={"uncited-model": None},
    )
    lines = summary.splitlines()

    assert lines[0] == lines[-1] == "=" * SUMMARY_WIDTH
    assert "CITATION GUIDANCE" in lines[1]
    assert all(len(line) <= SUMMARY_WIDTH for line in lines)
    assert all(line == line.rstrip() for line in lines)


def test_terminal_summary_shows_dois() -> None:
    """Every source with a DOI prints it as a resolvable link."""
    cited = Citation(
        key="doi-source",
        title="Source with a DOI",
        authors=("First Author",),
        year=2026,
        role="benchmark_method",
        doi="10.1234/example",
    )

    summary = format_citation_summary(
        {"category/test": _credits(cited)},
        models={
            "cited-model": Citation(
                key="m", title="Model", authors=("M. Author",), doi="10.5678/model"
            )
        },
    )

    assert "https://doi.org/10.1234/example" in summary
    assert "https://doi.org/10.5678/model" in summary


def test_terminal_summary_never_wraps_a_link() -> None:
    """Links stay on one line so they remain selectable, even past the block width."""
    long_url = "https://proceedings.example.com/" + "path/" * 30
    cited = Citation(
        key="url-source",
        title="Source with a long URL",
        authors=("First Author",),
        role="benchmark_method",
        url=long_url,
    )

    summary = format_citation_summary({"category/test": _credits(cited)})
    prose = [line for line in summary.splitlines() if "http" not in line]

    assert any(line.strip() == long_url for line in summary.splitlines())
    assert all(len(line) <= SUMMARY_WIDTH for line in prose)


def test_terminal_summary_flags_incomplete_metadata() -> None:
    """Missing citations are called out rather than left to be spotted in a list."""
    summary = format_citation_summary(
        {},
        missing_benchmarks=("category/other",),
        models={"uncited-model": None},
        frameworks={"MLIP Arena": None},
    )

    # The warning wraps, so compare against whitespace-normalised text
    flat = " ".join(summary.split())
    assert "incomplete for 1 benchmark(s), 1 framework(s), 1 model(s)" in flat


def test_terminal_summary_does_not_split_names_at_hyphens() -> None:
    """Long references wrap between words, never inside a hyphenated name."""
    citation = Citation(
        key="long",
        title="A title long enough to force the reference onto a second line here",
        authors=("Some Author with a rather long name indeed", "mace-mp-0a Author"),
        year=2026,
        role="benchmark_method",
    )

    summary = format_citation_summary({"category/test": _credits(citation)})

    assert "mace-mp-0a" in summary
    assert "mace-\n" not in summary


def test_load_model_citations_reads_models_file(tmp_path: Path) -> None:
    """Model citations are read from the supplied model definitions file."""
    path = tmp_path / "models.yml"
    path.write_text(
        """cited-model:
  class_name: mace
  citation:
    title: Model paper
    authors:
      - M. Author
    year: 2026
placeholder-model:
  class_name: mace
  citation:
    title: null
    authors: []
    year: null
"""
    )

    citations = load_model_citations(["cited-model", "placeholder-model"], path)

    assert citations["cited-model"].reference == "M. Author (2026). Model paper."
    assert citations["placeholder-model"] is None


def test_repository_model_citations_are_valid() -> None:
    """Every models.yml entry has a citation block that parses."""
    model_names = list(safe_load(models_file.read_text()))

    citations = load_model_citations(model_names)

    assert set(citations) == set(model_names)


def test_benchmark_credit_is_not_hidden_in_a_details_element() -> None:
    """Citation authors and implementers render outside any collapsible section."""
    table = DataTable(
        id="credit-test-table",
        columns=[{"id": "MLIP", "name": "MLIP"}, {"id": "Score", "name": "Score"}],
        data=[],
        tooltip_header={},
    )
    table.weights = {}

    layout = build_test_layout(
        name="Credit test",
        description="Credit layout test",
        framework_ids=[],
        table=table,
        thresholds={},
        credits=_credits(TEST_CITATION),
    )

    collapsed = [
        component
        for details in _walk_components(layout)
        if isinstance(details, (Details, Summary))
        for component in _walk_components(details)
    ]
    collapsed_text = " ".join(str(component) for component in collapsed)
    assert "First Author, Second Author" in str(layout)
    assert "Test Implementer" in str(layout)
    assert "First Author" not in collapsed_text
    assert "Test Implementer" not in collapsed_text


def test_documentation_is_a_direct_link() -> None:
    """The docs link is shown outright, not hidden behind a collapsible summary."""
    table = DataTable(
        id="docs-test-table",
        columns=[{"id": "MLIP", "name": "MLIP"}],
        data=[],
        tooltip_header={},
    )
    table.weights = {}

    def _layout(docs_url: str | None) -> str:
        return str(
            build_test_layout(
                name="Docs test",
                description="Docs layout test",
                framework_ids=[],
                table=table,
                thresholds={},
                docs_url=docs_url,
            )
        )

    linked = _layout("https://example.com/docs")

    assert "Click for more information" not in linked
    assert "https://example.com/docs" in linked
    assert "View documentation" in linked
    assert "View documentation" not in _layout(None)


def test_citation_links_the_title_and_doi_only() -> None:
    """The title and DOI are hyperlinks, so author text is not styled as a link."""
    citation = Citation(
        key="linked",
        title="Linked source",
        authors=("First Author",),
        role="benchmark_method",
        doi="10.1234/example",
    )

    links = [
        component
        for component in _walk_components(
            build_benchmark_credit_components(_credits(citation))
        )
        if type(component).__name__ == "A"
    ]

    assert [link.href for link in links] == ["https://doi.org/10.1234/example"] * 2
    assert all("First Author" not in str(link) for link in links)
    assert "Linked source" in str(links[0])
    assert str(links[1].children) == "10.1234/example"


def test_doi_is_shown_in_full() -> None:
    """The DOI is readable in the credit box, not hidden behind the title link."""
    with_doi = Citation(
        key="linked",
        title="Linked source",
        authors=("First Author",),
        role="benchmark_method",
        doi="10.1234/example",
    )
    without_doi = Citation(
        key="unlinked",
        title="Unlinked source",
        authors=("First Author",),
        role="benchmark_method",
    )

    assert "doi: " in str(build_benchmark_credit_components(_credits(with_doi)))
    assert "doi: " not in str(build_benchmark_credit_components(_credits(without_doi)))


def test_missing_credit_renders_placeholders() -> None:
    """Benchmarks without metadata display explicit credit placeholders."""
    rendered = str(build_benchmark_credit_components(None))

    assert "Original benchmark paper: " in rendered
    assert rendered.count("To be added") == 2


def test_ml_peg_only_benchmark_cites_ml_peg() -> None:
    """A benchmark devised for ML-PEG shows the ML-PEG citation in the paper slot."""
    ml_peg = ml_peg_citation()
    rendered = str(build_benchmark_credit_components(_credits()))

    assert ml_peg is not None
    for author in ml_peg.authors:
        assert author in rendered
    assert ml_peg.link in rendered
    assert "To be added" not in rendered.split("Implemented in ML-PEG by")[0]


def test_citations_are_not_rendered_as_a_bullet_list() -> None:
    """Citations render as plain lines, not as list items."""
    components = list(
        _walk_components(build_benchmark_credit_components(_credits(TEST_CITATION)))
    )
    element_names = {type(component).__name__ for component in components}

    assert not element_names & {"Ul", "Ol", "Li"}
    assert "Original benchmark paper:" in str(components[0])


def test_citation_label_is_pluralised() -> None:
    """The label matches the number of sources listed."""
    second = Citation(
        key="second-source",
        title="Second source",
        authors=("Third Author",),
        role="reference_data",
    )

    one = str(build_benchmark_credit_components(_credits(TEST_CITATION)))
    two = str(build_benchmark_credit_components(_credits(TEST_CITATION, second)))

    assert "Original benchmark paper:" in one
    assert "Original benchmark papers:" not in one
    assert "Original benchmark papers:" in two


def test_role_tag_shown_only_where_it_adds_meaning() -> None:
    """A benchmark paper needs no role tag, other roles say what they contributed."""
    data = Citation(
        key="data-source",
        title="Data source",
        authors=("Third Author",),
        role="reference_data",
    )

    rendered = str(build_benchmark_credit_components(_credits(TEST_CITATION, data)))

    assert "(benchmark paper)" not in rendered
    assert "(reference data)" in rendered


def test_external_framework_attribution_is_prominent() -> None:
    """Framework ports receive a dedicated, always-visible attribution banner."""
    rendered = str(build_framework_attribution(["ml_peg", "mlip_audit"])[0])

    assert "BENCHMARK ADAPTED FROM" in rendered
    assert "MLIP Audit" in rendered


def test_framework_citation_falls_back_to_the_paper_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frameworks without author details still point at their paper."""
    monkeypatch.setattr(
        build_components,
        "get_framework_config",
        lambda framework_id: {
            "label": "Test Framework",
            "type": "framework",
            "paper_url": "https://example.invalid/paper",
            "citation": {"title": "Titled but unattributed", "authors": []},
        },
    )

    citation = build_framework_citation("test_framework")
    links = [
        component
        for component in _walk_components(citation)
        if type(component).__name__ == "A"
    ]

    assert [link.href for link in links] == ["https://example.invalid/paper"]
    assert "Citation to be added" not in str(citation)


def test_framework_citation_renders_authors_when_supplied() -> None:
    """A fully recorded framework citation shows its authors rather than a link."""
    rendered = str(build_framework_citation("mlip_audit"))

    assert "Please cite the" not in rendered
    assert "to be added" not in rendered.lower()


class _Config:
    """Minimal pytest config stub for the citation reporter."""

    def __init__(self, output_dir: Path) -> None:
        self.rootpath = CALCS_ROOT.parent.parent
        self._options = {"--citation-output": str(output_dir)}

    def getoption(self, name: str, default: object = None) -> object:
        """
        Return a recorded option value.

        Parameters
        ----------
        name
            Option name.
        default
            Value returned for options this stub does not define.

        Returns
        -------
        object
            The option value.
        """
        return self._options.get(name, default)


class _Report:
    """Minimal pytest report stub carrying a rootdir-relative path."""

    def __init__(self, when: str, skipped: bool, fspath: str) -> None:
        self.when = when
        self.skipped = skipped
        self.fspath = fspath


def test_citation_reporter_records_only_executed_benchmarks(tmp_path: Path) -> None:
    """Only benchmark tests that actually ran are recorded, from relative paths."""
    reporter = CitationReporter(_Config(tmp_path))

    # fspath is relative to the pytest rootdir, not absolute
    reporter.pytest_runtest_logreport(
        _Report("setup", False, "ml_peg/calcs/conformers/setup_only/calc_setup_only.py")
    )
    reporter.pytest_runtest_logreport(
        _Report("call", True, "ml_peg/calcs/conformers/skipped/calc_skipped.py")
    )
    reporter.pytest_runtest_logreport(
        _Report("call", False, "ml_peg/calcs/conformers/ran/calc_ran.py")
    )
    reporter.pytest_runtest_logreport(
        _Report("call", False, "ml_peg/analysis/conformers/ran/analyse_ran.py")
    )

    assert reporter.script_paths == {
        CALCS_ROOT / "conformers" / "ran" / "calc_ran.py",
        ANALYSIS_ROOT / "conformers" / "ran" / "analyse_ran.py",
    }


class _Item:
    """Minimal pytest item stub carrying framework markers."""

    def __init__(self, fspath: str, *framework_ids: str) -> None:
        self.fspath = fspath
        self._framework_ids = framework_ids

    def iter_markers(self, name: str) -> list[object]:
        """
        Return the framework markers applied to this item.

        Parameters
        ----------
        name
            Marker name requested.

        Returns
        -------
        list[object]
            Markers matching `name`.
        """
        if name != "framework" or not self._framework_ids:
            return []
        return [SimpleNamespace(args=self._framework_ids)]


def test_citation_reporter_records_framework_markers(tmp_path: Path) -> None:
    """Source frameworks are taken from the framework markers of collected tests."""
    reporter = CitationReporter(_Config(tmp_path))

    reporter.pytest_collection_modifyitems(
        [
            _Item("ml_peg/analysis/conformers/ported/analyse_ported.py", "mlip_audit"),
            _Item("ml_peg/analysis/conformers/own/analyse_own.py"),
        ]
    )

    assert reporter.framework_ids == {
        ANALYSIS_ROOT / "conformers" / "ported" / "analyse_ported.py": {"mlip_audit"}
    }


def test_citation_reporter_ignores_non_benchmark_tests(tmp_path: Path) -> None:
    """Running the package's own test suite produces no citation output."""
    reporter = CitationReporter(_Config(tmp_path))

    reporter.pytest_runtest_logreport(_Report("call", False, "tests/test_citations.py"))
    reporter.pytest_runtest_logreport(
        _Report("call", False, "ml_peg/analysis/utils/analyse_gscdb138.py")
    )

    assert reporter.script_paths == set()
    reporter.pytest_terminal_summary(None)
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_repository_citation_metadata_is_valid() -> None:
    """Validate every populated benchmark citation file in the repository."""
    for path in ANALYSIS_ROOT.glob("*/*/citations.yml"):
        load_benchmark_credits(path)
