"""Tests for benchmark citation metadata and generated guidance."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
import subprocess
from types import SimpleNamespace

from dash.dash_table import DataTable
from dash.development.base_component import Component
from dash.html import Details, Summary
import pytest
from yaml import safe_load

from ml_peg.analysis import ANALYSIS_ROOT
from ml_peg.app.utils.build_components import build_test_layout
from ml_peg.app.utils.build_credits import build_benchmark_credit_components
from ml_peg.calcs import CALCS_ROOT
from ml_peg.conftest import CitationReporter
from ml_peg.utils import citations as citations_module
from ml_peg.utils.citations import (
    FRAMEWORKS_FILE,
    SUMMARY_WIDTH,
    BenchmarkCredits,
    Citation,
    CitationMetadataError,
    Contributor,
    add_framework_citations,
    app_citation_metadata_path,
    build_run_citations,
    citation_metadata_path,
    collect_benchmark_credits,
    format_authors,
    format_citation_summary,
    load_benchmark_credits,
    load_framework_citations,
)


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
        f"""implementers:
  - name: Test Implementer
contributors:
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
    """Build benchmark credits with separate implementation and other credit."""
    return BenchmarkCredits(
        implementers=(Contributor("Test Implementer"),),
        contributors=(Contributor("Test Contributor"),),
        citations=citations,
    )


TEST_CITATION = Citation(
    key="test-source",
    title="Test source",
    authors=("First Author", "Second Author"),
    year=2026,
    role="benchmark_method",
)


def test_load_benchmark_credits(tmp_path: Path) -> None:
    """Citation metadata preserves publication and contributor credit."""
    path = tmp_path / "citations.yml"
    _write_credits(path)

    credits = load_benchmark_credits(path)

    assert credits.contributors == (Contributor(name="Test Contributor"),)
    assert credits.implementers == (Contributor(name="Test Implementer"),)
    assert credits.citations[0].title == "Test reference data"
    assert credits.citations[0].role_label == "reference data"


@pytest.mark.parametrize("credit_role", ["implementers", "contributors"])
@pytest.mark.parametrize("github", [None, "asmith"])
def test_email_contacts_load_and_render(
    tmp_path: Path, credit_role: str, github: str | None
) -> None:
    """Both credit groups support email with or without a GitHub account."""
    path = tmp_path / "citations.yml"
    path.write_text(
        f"{credit_role}:\n"
        "  - name: Alice Smith\n"
        f"    github: {github or 'null'}\n"
        "    email: alice+benchmarks@example.org\n"
        "citations: []\n"
    )
    credits = load_benchmark_credits(path)
    person = getattr(credits, credit_role)[0]
    assert person == Contributor(
        "Alice Smith", github=github, email="alice+benchmarks@example.org"
    )
    rendered = build_benchmark_credit_components(credits)
    links = [
        component
        for component in _walk_components(rendered)
        if type(component).__name__ == "A"
    ]
    expected = (["https://github.com/asmith"] if github else []) + [
        "mailto:alice+benchmarks@example.org"
    ]
    assert [link.href for link in links] == expected
    assert links[-1].title == person.email
    assert "Alice Smith <alice+benchmarks@example.org>" in " ".join(
        format_citation_summary({"category/test": credits}).split()
    )


@pytest.mark.parametrize(
    "email",
    [
        "''",
        "42",
        "alice",
        "'Alice <alice@example.org>'",
        "'alice@ example.org'",
        "'mailto:alice@example.org'",
    ],
)
def test_invalid_email_contacts_are_metadata_errors(tmp_path: Path, email: str) -> None:
    """Email metadata requires a plain address, with a useful field location."""
    path = tmp_path / "citations.yml"
    path.write_text(f"contributors:\n  - name: Alice\n    email: {email}\n")
    with pytest.raises(CitationMetadataError, match=r"contributors\[0\].email"):
        load_benchmark_credits(path)


def test_email_contacts_can_be_null(tmp_path: Path) -> None:
    """Null email is treated like an omitted optional field."""
    path = tmp_path / "citations.yml"
    path.write_text("contributors:\n  - name: Alice\n    email: null\n")
    assert load_benchmark_credits(path).contributors == (Contributor("Alice"),)


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


@pytest.mark.parametrize("role", ["implementers", "contributors"])
def test_credit_roles_are_sorted_by_surname(tmp_path: Path, role: str) -> None:
    """Both credit roles are listed by surname, whatever the file order."""
    path = tmp_path / "citations.yml"
    path.write_text(
        f"{role}:\n"
        "  - name: Zoe Adams\n"
        "  - name: Ben Young\n"
        "  - name: Amy Clark\n"
        "citations: []\n"
    )

    names = [item.name for item in getattr(load_benchmark_credits(path), role)]

    assert names == ["Zoe Adams", "Amy Clark", "Ben Young"]


def test_invalid_implementer_list_is_a_metadata_error(tmp_path: Path) -> None:
    """Implementation credit uses the same validated list format as contributors."""
    path = tmp_path / "citations.yml"
    path.write_text("implementers: A. Implementer\ncitations: []\n")
    with pytest.raises(CitationMetadataError, match="implementers must be a list"):
        load_benchmark_credits(path)


def test_other_contributors_are_optional() -> None:
    """Known implementers and citations are complete without additional contributors."""
    credits = BenchmarkCredits(
        implementers=(Contributor("A. Implementer"),),
        contributors=(),
        citations=(TEST_CITATION,),
    )
    summary = format_citation_summary({"category/test": credits})
    rendered = str(build_benchmark_credit_components(credits))
    assert "incomplete" not in summary
    assert "implemented by A. Implementer" in summary
    assert "Implemented by: " in rendered
    assert "Contributors: " not in rendered
    assert "To be added" not in rendered


def test_citation_authors_keep_published_order(tmp_path: Path) -> None:
    """Citation authors are not reordered, as their order is part of the citation."""
    path = tmp_path / "citations.yml"
    path.write_text(
        "citations:\n"
        "  - key: paper\n"
        "    role: benchmark_method\n"
        "    title: Paper\n"
        "    authors: [Zed Last, Abe First]\n"
    )

    credits = load_benchmark_credits(path)

    assert credits.citations[0].authors == ("Zed Last", "Abe First")


def test_et_al_is_not_accepted_as_an_author(tmp_path: Path) -> None:
    """Authors are listed in full, as "et al." is added when displayed."""
    path = tmp_path / "citations.yml"
    path.write_text(
        "citations:\n"
        "  - key: paper\n"
        "    role: benchmark_method\n"
        "    title: Paper\n"
        "    authors: [A. Author, et al.]\n"
    )

    with pytest.raises(CitationMetadataError, match="not 'et al.'"):
        load_benchmark_credits(path)


def test_invalid_framework_citation_is_a_metadata_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed frameworks.yml citation raises a clear metadata error."""
    registry = tmp_path / "frameworks.yml"
    registry.write_text(
        "arena:\n  label: Arena\n  type: framework\n  citation: just a string\n"
    )
    monkeypatch.setattr(citations_module, "FRAMEWORKS_FILE", registry)

    with pytest.raises(CitationMetadataError, match="arena.citation must be a mapping"):
        load_framework_citations(["arena"])


def test_invalid_citations_do_not_break_the_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bad metadata drops the credits with a warning, instead of stopping the app."""
    from ml_peg.app import APP_ROOT, base_app

    table = DataTable(
        id="bad-table",
        columns=[{"id": "MLIP", "name": "MLIP"}],
        data=[],
        tooltip_header={},
    )
    table.weights = {}
    table.thresholds = {}

    # Real loader on a file with a YAML syntax error, the hardest case to catch
    bad_file = tmp_path / "citations.yml"
    bad_file.write_text('citations:\n  - key: "unterminated\n')

    monkeypatch.setattr(base_app, "rebuild_table", lambda *args, **kwargs: table)
    monkeypatch.setattr(
        base_app, "app_citation_metadata_path", lambda *args, **kwargs: bad_file
    )

    class _App(base_app.BaseApp):
        def register_callbacks(self) -> None:
            """Register no callbacks."""

    with pytest.warns(UserWarning, match="Invalid citations for Bad"):
        app = _App(
            name="Bad",
            description="Bad metadata",
            table_path=APP_ROOT / "data" / "category" / "bench" / "table.json",
            extra_components=[],
            info_path=APP_ROOT / "data" / "category" / "bench" / "info.json",
        )

    assert app.credits is None
    assert "Benchmark references: " in str(app.layout)


def test_yaml_syntax_errors_are_metadata_errors(tmp_path: Path) -> None:
    """Malformed YAML raises the same error as invalid metadata, naming the file."""
    path = tmp_path / "citations.yml"
    path.write_text('citations:\n  - key: "unterminated\n    role: [benchmark_method\n')

    with pytest.raises(CitationMetadataError, match="invalid YAML"):
        load_benchmark_credits(path)


def test_incomplete_notice_names_the_file_to_edit() -> None:
    """Gaps point to citations.yml for benchmarks and frameworks.yml for frameworks."""
    empty = {"category/test": BenchmarkCredits(contributors=(), citations=())}

    framework_only = format_citation_summary({}, frameworks={"MLIP Arena": None})
    benchmark_only = format_citation_summary(empty)
    both = format_citation_summary(empty, frameworks={"MLIP Arena": None})

    def notice(summary: str) -> str:
        return " ".join(summary.split("! Citation metadata")[1].split())

    assert "frameworks.yml" in notice(framework_only)
    assert "citations.yml" not in notice(framework_only)
    assert "citations.yml" in notice(benchmark_only)
    assert "frameworks.yml" not in notice(benchmark_only)
    assert "citations.yml and frameworks.yml" in notice(both)


def test_empty_citations_load(tmp_path: Path) -> None:
    """An empty citation list loads successfully, for references not yet added."""
    path = tmp_path / "citations.yml"
    path.write_text("contributors:\n  - name: Test Contributor\ncitations: []\n")

    credits = load_benchmark_credits(path)

    assert credits.citations == ()
    assert credits.contributors == (Contributor(name="Test Contributor"),)


def test_collect_benchmark_credits_reports_missing(tmp_path: Path) -> None:
    """Collection distinguishes populated metadata from missing placeholders."""
    first_script = tmp_path / "calcs" / "category" / "first" / "calc_first.py"
    second_script = tmp_path / "calcs" / "category" / "second" / "calc_second.py"
    _write_credits(first_script.parent / "citations.yml")

    credits, missing = collect_benchmark_credits([first_script, second_script])

    assert tuple(credits) == ("category/first",)
    assert missing == ("category/second",)


def test_citation_metadata_is_owned_by_calculations(tmp_path: Path) -> None:
    """Calc runs and app layouts resolve the same benchmark-owned metadata."""
    aconfl_script = CALCS_ROOT / "conformers" / "ACONFL" / "calc_ACONFL.py"

    assert (
        citation_metadata_path(aconfl_script) == aconfl_script.parent / "citations.yml"
    )
    assert app_citation_metadata_path("conformers", "ACONFL", CALCS_ROOT) == (
        aconfl_script.parent / "citations.yml"
    )
    # Exercise the alias with our own directories, independent of unmerged tests.
    hyphenated = tmp_path / "category" / "hyphenated-name"
    hyphenated.mkdir(parents=True)
    assert app_citation_metadata_path("category", "hyphenated_name", tmp_path) == (
        hyphenated / "citations.yml"
    )
    underscored = tmp_path / "category" / "hyphenated_name"
    underscored.mkdir()
    assert app_citation_metadata_path("category", "hyphenated_name", tmp_path) == (
        underscored / "citations.yml"
    )
    assert app_citation_metadata_path("category", "absent_name", tmp_path) == (
        tmp_path / "category" / "absent_name" / "citations.yml"
    )


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
    """Every registered source framework has a complete, included citation."""
    registry = safe_load(FRAMEWORKS_FILE.read_text())
    source_ids = {
        name
        for name, entry in registry.items()
        if name != "ml_peg" and entry.get("type") != "paper"
    }
    for name in source_ids:
        assert registry[name].get("type") == "framework", name
    citations = load_framework_citations(source_ids)

    assert set(citations) == {registry[name]["label"] for name in source_ids}
    for label, citation in citations.items():
        assert citation is not None, label


def test_repository_benchmarks_have_citation_metadata() -> None:
    """Every tracked benchmark calculation has a citations.yml beside it."""
    tracked = subprocess.check_output(
        ["git", "ls-files", "--", "*/*/calc_*.py"], cwd=CALCS_ROOT, text=True
    ).splitlines()
    scripts = [CALCS_ROOT / path for path in tracked if len(Path(path).parts) == 3]

    assert scripts, "No tracked benchmark calculations found"
    missing = [
        str(script)
        for script in scripts
        if not citation_metadata_path(script).is_file()
    ]
    assert not missing, f"Benchmarks missing citations.yml: {missing}"


def test_framework_references_are_added_to_benchmark_credits() -> None:
    """Framework references supplement benchmark papers and preserve people."""
    original = _credits(TEST_CITATION)
    framework = load_framework_citations(["matbench-discovery"])["Matbench Discovery"]

    credits = add_framework_citations(original, ["matbench-discovery"])

    assert credits is not None
    assert credits.citations == (TEST_CITATION, framework)
    assert credits.implementers == original.implementers
    assert credits.contributors == original.contributors
    assert original.citations == (TEST_CITATION,)


@pytest.mark.parametrize("matching_field", ["key", "title", "url", "doi"])
def test_framework_references_preserve_existing_citations(matching_field: str) -> None:
    """Matching publication identifiers retain the existing record and role."""
    framework = load_framework_citations(["matbench-discovery"])["Matbench Discovery"]
    assert framework is not None
    existing = replace(
        framework,
        key="benchmark-source",
        title="A different display title",
        role="reference_data",
        url=None,
    )
    if matching_field == "key":
        existing = replace(existing, key=framework.key)
    elif matching_field == "title":
        existing = replace(existing, title=f"  {framework.title.upper()}.  ")
    elif matching_field == "url":
        existing = replace(existing, url=f"{framework.url}/")
    else:
        existing = replace(existing, doi="10.1038/S42256-025-01055-1")

    credits = add_framework_citations(
        _credits(existing), ["matbench-discovery", "matbench-discovery"]
    )

    assert credits is not None
    assert credits.citations == (existing,)


def test_paper_tags_are_not_added_as_framework_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even paper tags with complete citation metadata are not added implicitly."""
    registry = tmp_path / "frameworks.yml"
    registry.write_text(
        "paper:\n  label: Paper\n  type: paper\n"
        "  citation:\n    title: Source paper\n    authors: [An Author]\n"
    )
    monkeypatch.setattr(citations_module, "FRAMEWORKS_FILE", registry)
    original = _credits(TEST_CITATION)

    assert add_framework_citations(original, ["paper", "ml_peg"]) == original
    assert add_framework_citations(None, ["paper", "ml_peg"]) is None


def test_missing_benchmark_credits_can_show_framework_references() -> None:
    """A known framework can be cited before benchmark-specific metadata is added."""
    credits = add_framework_citations(None, ["matbench-discovery"])

    assert credits is not None
    assert credits.implementers == ()
    assert credits.contributors == ()
    assert [citation.key for citation in credits.citations] == ["matbench-discovery"]


def test_run_citations_add_frameworks_to_their_own_benchmarks(tmp_path: Path) -> None:
    """Per-script markers associate framework references with the correct benchmark."""
    first = tmp_path / "category" / "first" / "calc_first.py"
    second = tmp_path / "category" / "second" / "calc_second.py"
    _write_credits(first.parent / "citations.yml")
    _write_credits(second.parent / "citations.yml")
    framework = load_framework_citations(["matbench-discovery"])["Matbench Discovery"]
    assert framework is not None

    summary = build_run_citations(
        [first, second],
        framework_ids_by_script={
            first: ["matbench-discovery"],
            second: ["mace-polar-1"],
            tmp_path / "not-run.py": ["mlip_audit"],
        },
    )
    summary = " ".join(summary.split())

    assert summary.count(framework.title) == 1
    assert summary.index("category/first") < summary.index(framework.title)
    assert summary.index(framework.title) < summary.index("category/second")
    assert "(source framework)" in summary
    assert "SOURCE FRAMEWORKS" not in summary
    assert "MLIPAudit" not in summary


@pytest.mark.parametrize("already_cited", [False, True])
def test_app_references_include_frameworks_once(already_cited: bool) -> None:
    """Framework badges add one source reference to the app's benchmark references."""
    framework = load_framework_citations(["matbench-discovery"])["Matbench Discovery"]
    assert framework is not None
    table = DataTable(
        id="framework-credit-table",
        columns=[{"id": "MLIP", "name": "MLIP"}],
        data=[],
        tooltip_header={},
    )
    table.weights = {}
    credits = (
        _credits(TEST_CITATION, framework) if already_cited else _credits(TEST_CITATION)
    )

    layout = build_test_layout(
        name="Framework credits",
        description="Framework reference test",
        framework_ids=["matbench-discovery", "mace-polar-1", "ml_peg"],
        table=table,
        thresholds={},
        credits=credits,
    )
    title_links = [
        component
        for component in _walk_components(layout)
        if type(component).__name__ == "A"
        and isinstance(component.children, Component)
        and component.children.children == framework.title
    ]

    assert len(title_links) == 1
    assert title_links[0].href == framework.link


@pytest.mark.parametrize("path", sorted(CALCS_ROOT.glob("*/*/citations.yml")))
def test_repository_benchmark_citations_are_valid(path: Path) -> None:
    """Benchmark credit files have a calculation script and valid metadata."""
    assert list(path.parent.glob("calc_*.py")), path
    credits = load_benchmark_credits(path)
    implementers = {person.name for person in credits.implementers}
    contributors = {person.name for person in credits.contributors}
    assert len(implementers) == len(credits.implementers), path
    assert len(contributors) == len(credits.contributors), path
    assert implementers.isdisjoint(contributors), path


def test_benchmark_runs_print_guidance_without_writing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Citation guidance is reported to the terminal only, leaving no files behind."""
    monkeypatch.chdir(tmp_path)

    summary = build_run_citations(
        [CALCS_ROOT / "conformers" / "ACONFL" / "calc_ACONFL.py"]
    )

    assert "CITATIONS" in summary
    assert "conformers/ACONFL" in summary
    assert "MODELS (" not in summary
    # Nothing is written, so a run leaves the working directory untouched
    assert list(tmp_path.rglob("*")) == []


def test_terminal_summary_prints_citations_and_contributors() -> None:
    """Terminal guidance includes paper authors and contributors."""
    summary = format_citation_summary(
        {"category/test": _credits(TEST_CITATION)},
        missing_benchmarks=("category/other",),
    )

    assert "First Author, Second Author (2026). Test source." in summary
    assert "Contributors:" in summary
    assert "implemented by Test Implementer" in summary
    assert "Test Contributor" in summary
    assert "BENCHMARKS (2)" in summary
    assert "category/other" in summary


def test_contributor_is_separated_from_the_citation() -> None:
    """The contributor is named beside the benchmark, not in the cited reference."""
    summary = format_citation_summary({"category/test": _credits(TEST_CITATION)})
    lines = summary.splitlines()

    header = next(line for line in lines if "category/test" in line)
    reference = next(line for line in lines if "First Author" in line)

    assert "implemented by Test Implementer" in header
    assert "Contributors: Test Contributor" in summary
    # The contributor never shares a line with the work being cited
    assert "Test Contributor" not in reference
    assert lines.index(header) < lines.index(reference)


def test_references_are_numbered_in_the_summary() -> None:
    """Each benchmark numbers its references, so long lists stay easy to scan."""
    second = Citation(
        key="second",
        title="Second source",
        authors=("Third Author",),
        role="reference_data",
    )

    summary = format_citation_summary(
        {"category/test": _credits(TEST_CITATION, second)}
    )
    lines = summary.splitlines()

    assert any(line.startswith("    [1] First Author") for line in lines)
    assert any(line.startswith("    [2] Third Author") for line in lines)


def test_reference_numbers_are_right_aligned() -> None:
    """References past [9] start in the same column as the earlier ones."""
    citations = [
        Citation(
            key=f"source-{index}",
            title=f"Source {index}",
            authors=("An Author",),
            role="benchmark_method",
        )
        for index in range(1, 11)
    ]

    lines = format_citation_summary(
        {"category/test": _credits(*citations)}
    ).splitlines()
    starts = {line.index("An Author") for line in lines if "An Author" in line}

    assert any(line.startswith("     [1] An Author") for line in lines)
    assert any(line.startswith("    [10] An Author") for line in lines)
    assert len(starts) == 1


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
    summary = format_citation_summary({"category/test": _credits(TEST_CITATION)})
    lines = summary.splitlines()

    assert lines[0] == lines[-1] == "=" * SUMMARY_WIDTH
    assert "CITATIONS" in lines[1]
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

    summary = format_citation_summary({"category/test": _credits(cited)})

    assert "https://doi.org/10.1234/example" in summary


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
        frameworks={"MLIP Arena": None},
    )

    # The warning wraps, so compare against whitespace-normalised text
    flat = " ".join(summary.split())
    assert "incomplete for 1 benchmark(s), 1 framework(s)" in flat


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


def test_contributor_shown_and_references_collapsed() -> None:
    """Contributors are always visible, while the references start collapsed."""
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
    details = [c for c in _walk_components(layout) if isinstance(c, Details)]
    assert "First Author, Second Author" in collapsed_text
    assert "Contributors: " in str(layout)
    assert "Implemented by: " in str(layout)
    assert "Test Implementer" in str(layout)
    assert "Test Implementer" not in collapsed_text
    assert "Test Contributor" in str(layout)
    assert "Test Contributor" not in collapsed_text
    # Closed by default, as a benchmark can cite many sources
    assert details and not any(getattr(d, "open", False) for d in details)


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


INSPIRED_CITATION = Citation(
    key="inspiration",
    title="Prior platform this benchmark builds on",
    authors=("Prior Author",),
    year=2025,
    role="inspired_by",
)


def test_inspired_benchmark_is_not_called_an_original_paper() -> None:
    """A benchmark built on earlier work tags the source it was inspired by."""
    rendered = str(build_benchmark_credit_components(_credits(INSPIRED_CITATION)))
    summary = format_citation_summary({"category/test": _credits(INSPIRED_CITATION)})

    assert "'Benchmark references'" in rendered
    assert "(inspired by)" in rendered
    assert "(inspired by)" in summary


def test_a_benchmark_paper_still_wins_the_heading() -> None:
    """Mixing in an inspiration does not demote a real benchmark paper."""
    both = _credits(TEST_CITATION, INSPIRED_CITATION)

    assert "'Benchmark references'" in str(build_benchmark_credit_components(both))
    summary = format_citation_summary({"category/test": both})
    # Only the inspiration is tagged, as benchmark papers are the default reference
    assert summary.count("(inspired by)") == 1
    assert "(benchmark paper)" not in summary


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


def test_contributor_links_to_github() -> None:
    """Contributors with a recorded handle link to their GitHub account."""
    credits = BenchmarkCredits(
        contributors=(
            Contributor("Alice Smith", github="asmith"),
            Contributor("Bare Name"),
        ),
        citations=(),
    )

    rendered = build_benchmark_credit_components(credits)
    links = [
        component
        for component in _walk_components(rendered)
        if type(component).__name__ == "A"
    ]

    assert [link.href for link in links] == ["https://github.com/asmith"]
    assert "Alice Smith" in str(links[0].children)
    # The icon is inlined, so the credit box makes no external request
    assert "data:image/svg+xml" in str(rendered)
    assert "Alice Smith" in str(rendered)
    assert "Bare Name" in str(rendered)


def test_long_author_lists_are_shortened() -> None:
    """More than five authors collapse to the first name and et al."""

    def cite(n: int) -> Citation:
        return Citation(
            key=f"k{n}",
            title="A source",
            authors=tuple(f"Author {i}" for i in range(1, n + 1)),
            year=2026,
            role="benchmark_method",
        )

    assert format_authors(cite(5).authors) == (
        "Author 1, Author 2, Author 3, Author 4, Author 5"
    )
    assert format_authors(cite(6).authors) == "Author 1 et al."
    # Both surfaces shorten identically
    assert "Author 1 et al." in cite(6).reference
    assert "Author 6" not in str(build_benchmark_credit_components(_credits(cite(6))))
    assert "Author 5" in str(build_benchmark_credit_components(_credits(cite(5))))


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

    assert "Benchmark references: " in rendered
    assert rendered.count("To be added") == 2


def test_empty_citations_are_shown_as_to_be_added() -> None:
    """An empty citation list is a placeholder, like missing metadata."""
    rendered = str(build_benchmark_credit_components(_credits()))
    summary = format_citation_summary({"category/test": _credits()})

    assert "Benchmark references: " in rendered
    # The contributor is known, so the only placeholder is the references
    assert rendered.count("To be added") == 1
    assert "Devised" not in rendered
    assert "! references to be added" in summary
    assert "Test Contributor" in summary
    assert "incomplete for 1 benchmark(s)" in " ".join(summary.split())


def test_citations_are_not_rendered_as_a_bullet_list() -> None:
    """Citations render as plain lines, not as list items."""
    components = list(
        _walk_components(build_benchmark_credit_components(_credits(TEST_CITATION)))
    )
    element_names = {type(component).__name__ for component in components}

    assert not element_names & {"Ul", "Ol", "Li"}
    assert "'Benchmark references'" in str(components[0])


def test_reference_count_is_shown() -> None:
    """The collapsed summary says how many references it hides."""
    second = Citation(
        key="second-source",
        title="Second source",
        authors=("Third Author",),
        role="reference_data",
    )

    one = str(build_benchmark_credit_components(_credits(TEST_CITATION)))
    two = str(build_benchmark_credit_components(_credits(TEST_CITATION, second)))

    assert "' (1)'" in one
    assert "' (2)'" in two


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

    assert reporter.script_paths == {CALCS_ROOT / "conformers" / "ran" / "calc_ran.py"}


class _Item:
    """Minimal pytest item stub carrying an absolute path and framework markers."""

    def __init__(self, path: Path, *framework_ids: str) -> None:
        self.path = path
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
            _Item(CALCS_ROOT / "conformers/ported/calc_ported.py", "mlip_audit"),
            _Item(
                ANALYSIS_ROOT / "conformers/ignored/analyse_ignored.py", "mlip_audit"
            ),
        ]
    )

    assert reporter.framework_ids == {
        CALCS_ROOT / "conformers" / "ported" / "calc_ported.py": {"mlip_audit"}
    }


def test_citation_reporter_includes_framework_in_benchmark_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reporter passes per-benchmark markers into the printed references."""
    monkeypatch.setattr("ml_peg.conftest.CALCS_ROOT", tmp_path)
    script = tmp_path / "category" / "benchmark" / "calc_benchmark.py"
    _write_credits(script.parent / "citations.yml")
    reporter = CitationReporter(_Config(tmp_path))
    reporter.rootpath = tmp_path
    reporter.pytest_collection_modifyitems([_Item(script, "matbench-discovery")])
    reporter.pytest_runtest_logreport(
        _Report("call", False, "category/benchmark/calc_benchmark.py")
    )
    lines = []

    reporter.pytest_terminal_summary(SimpleNamespace(write_line=lines.append))

    summary = " ".join(lines[-1].split())
    assert "category/benchmark" in summary
    assert (
        "A framework to evaluate machine learning crystal stability predictions"
        in summary
    )
    assert "(source framework)" in summary
    assert "SOURCE FRAMEWORKS" not in summary


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
    for path in CALCS_ROOT.glob("*/*/citations.yml"):
        load_benchmark_credits(path)
