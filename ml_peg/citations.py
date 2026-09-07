"""Load benchmark credits and write citation guidance for ML-PEG runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import textwrap
from typing import TYPE_CHECKING, Any

from yaml import safe_load

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

CITATION_FILE = Path(__file__).parent.parent / "CITATION.cff"
FRAMEWORKS_FILE = Path(__file__).parent / "app" / "utils" / "frameworks.yml"

# Width of the citation guidance printed after a benchmark run
SUMMARY_WIDTH = 79

CITATION_ROLES = {
    "benchmark_method",
    "reference_data",
    "reference_method",
    "upstream_framework",
}
CITATION_ROLE_LABELS = {
    "benchmark_method": "benchmark paper",
    "reference_data": "reference data",
    "reference_method": "reference method",
    "upstream_framework": "source framework",
}


class CitationMetadataError(ValueError):
    """Raised when citation metadata is invalid."""


@dataclass(frozen=True)
class Contributor:
    """A person who implemented a benchmark in ML-PEG."""

    name: str
    github: str | None = None
    orcid: str | None = None


@dataclass(frozen=True)
class Citation:
    """A scholarly source or piece of software to be cited."""

    key: str
    title: str
    authors: tuple[str, ...]
    year: int | None = None
    role: str | None = None
    doi: str | None = None
    url: str | None = None
    bibtex: str | None = None
    entry_type: str = "misc"

    @property
    def link(self) -> str | None:
        """
        Return the preferred external link, if supplied.

        Returns
        -------
        str | None
            DOI link if a DOI is set, otherwise the URL, otherwise None.
        """
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return self.url

    @property
    def reference(self) -> str:
        """
        Return a concise human-readable reference.

        Returns
        -------
        str
            Authors, year, and title, formatted for display.
        """
        authors = ", ".join(self.authors)
        year = f" ({self.year})" if self.year is not None else ""
        return f"{authors}{year}. {self.title}."

    @property
    def role_label(self) -> str | None:
        """
        Return a concise human-readable source role, if one is set.

        Returns
        -------
        str | None
            Display label for the role, or None if the citation has no role.
        """
        return CITATION_ROLE_LABELS[self.role] if self.role else None


@dataclass(frozen=True)
class BenchmarkCredits:
    """
    Citations and implementation contributors for one benchmark.

    An empty ``citations`` tuple means the benchmark was devised for ML-PEG and has no
    external source to cite, which is distinct from missing metadata (represented by
    the absence of a ``BenchmarkCredits`` instance).
    """

    contributors: tuple[Contributor, ...]
    citations: tuple[Citation, ...]


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    """
    Validate and return a mapping value.

    Parameters
    ----------
    value
        Raw value loaded from YAML.
    location
        Location of `value` in the metadata, used in error messages.

    Returns
    -------
    Mapping[str, Any]
        The validated mapping.
    """
    if not isinstance(value, Mapping):
        raise CitationMetadataError(f"{location} must be a mapping")
    return value


def _non_empty_string(value: Any, location: str) -> str:
    """
    Validate and return a non-empty string value.

    Parameters
    ----------
    value
        Raw value loaded from YAML.
    location
        Location of `value` in the metadata, used in error messages.

    Returns
    -------
    str
        The validated, stripped string.
    """
    if not isinstance(value, str) or not value.strip():
        raise CitationMetadataError(f"{location} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, location: str) -> str | None:
    """
    Validate and return an optional string value.

    Parameters
    ----------
    value
        Raw value loaded from YAML.
    location
        Location of `value` in the metadata, used in error messages.

    Returns
    -------
    str | None
        The validated, stripped string, or None.
    """
    if value is None:
        return None
    return _non_empty_string(value, location)


def _authors(value: Any, location: str) -> tuple[str, ...]:
    """
    Validate and return a non-empty list of author names.

    Parameters
    ----------
    value
        Raw value loaded from YAML.
    location
        Location of `value` in the metadata, used in error messages.

    Returns
    -------
    tuple[str, ...]
        The validated author names, in order.
    """
    if not isinstance(value, list) or not value:
        raise CitationMetadataError(f"{location} must be a non-empty list")
    return tuple(
        _non_empty_string(author, f"{location}[{index}]")
        for index, author in enumerate(value)
    )


def _year(value: Any, location: str) -> int | None:
    """
    Validate and return an optional publication year.

    Parameters
    ----------
    value
        Raw value loaded from YAML.
    location
        Location of `value` in the metadata, used in error messages.

    Returns
    -------
    int | None
        The validated year, or None.
    """
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise CitationMetadataError(f"{location} must be an integer or null")
    return value


def _parse_contributor(value: Any, location: str) -> Contributor:
    """
    Parse one benchmark contributor.

    Parameters
    ----------
    value
        Raw value loaded from YAML.
    location
        Location of `value` in the metadata, used in error messages.

    Returns
    -------
    Contributor
        The parsed contributor.
    """
    item = _mapping(value, location)
    return Contributor(
        name=_non_empty_string(item.get("name"), f"{location}.name"),
        github=_optional_string(item.get("github"), f"{location}.github"),
        orcid=_optional_string(item.get("orcid"), f"{location}.orcid"),
    )


def _parse_citation(value: Any, location: str) -> Citation:
    """
    Parse one benchmark citation.

    Parameters
    ----------
    value
        Raw value loaded from YAML.
    location
        Location of `value` in the metadata, used in error messages.

    Returns
    -------
    Citation
        The parsed citation.
    """
    item = _mapping(value, location)
    role = item.get("role")
    if role not in CITATION_ROLES:
        raise CitationMetadataError(
            f"{location}.role must be one of {sorted(CITATION_ROLES)}"
        )

    return Citation(
        key=_non_empty_string(item.get("key"), f"{location}.key"),
        title=_non_empty_string(item.get("title"), f"{location}.title"),
        authors=_authors(item.get("authors"), f"{location}.authors"),
        year=_year(item.get("year"), f"{location}.year"),
        role=role,
        doi=_optional_string(item.get("doi"), f"{location}.doi"),
        url=_optional_string(item.get("url"), f"{location}.url"),
        bibtex=_optional_string(item.get("bibtex"), f"{location}.bibtex"),
    )


def load_benchmark_credits(path: str | Path) -> BenchmarkCredits:
    """
    Load and validate one benchmark's ``citations.yml`` file.

    Parameters
    ----------
    path
        Citation metadata file.

    Returns
    -------
    BenchmarkCredits
        Validated benchmark credits.
    """
    path = Path(path)
    document = _mapping(safe_load(path.read_text()), str(path))
    raw_contributors = document.get("contributors", [])
    raw_citations = document.get("citations") or []
    if not isinstance(raw_contributors, list):
        raise CitationMetadataError(f"{path}: contributors must be a list")
    if not isinstance(raw_citations, list):
        raise CitationMetadataError(f"{path}: citations must be a list")

    contributors = tuple(
        _parse_contributor(value, f"{path}: contributors[{index}]")
        for index, value in enumerate(raw_contributors)
    )
    citations = tuple(
        _parse_citation(value, f"{path}: citations[{index}]")
        for index, value in enumerate(raw_citations)
    )
    keys = [citation.key for citation in citations]
    if len(keys) != len(set(keys)):
        raise CitationMetadataError(f"{path}: citation keys must be unique")
    return BenchmarkCredits(contributors=contributors, citations=citations)


def load_optional_benchmark_credits(path: str | Path) -> BenchmarkCredits | None:
    """
    Load benchmark credits when the metadata file exists.

    Parameters
    ----------
    path
        Citation metadata file.

    Returns
    -------
    BenchmarkCredits | None
        Validated benchmark credits, or None if `path` does not exist.
    """
    path = Path(path)
    return load_benchmark_credits(path) if path.is_file() else None


def citation_metadata_path(script_path: str | Path, analysis_root: str | Path) -> Path:
    """
    Return the citation metadata path corresponding to a benchmark script.

    Parameters
    ----------
    script_path
        Path to a ``calc_*.py`` or ``analyse_*.py`` benchmark script.
    analysis_root
        Root of the analysis tree holding the metadata files.

    Returns
    -------
    Path
        Path to the benchmark's ``citations.yml``, which may not exist.
    """
    script_path = Path(script_path)
    return (
        Path(analysis_root)
        / script_path.parent.parent.name
        / script_path.parent.name
        / "citations.yml"
    )


@lru_cache(maxsize=1)
def ml_peg_citation() -> Citation | None:
    """
    Build the ML-PEG software citation from ``CITATION.cff``.

    Returns
    -------
    Citation | None
        Repository citation, or None if ``CITATION.cff`` is not distributed alongside
        the installed package.
    """
    if not CITATION_FILE.is_file():
        return None

    document = safe_load(CITATION_FILE.read_text())
    authors = tuple(
        " ".join(
            part
            for part in (author.get("given-names"), author.get("family-names"))
            if part
        )
        for author in document.get("authors", [])
    )
    doi = next(
        (
            identifier["value"]
            for identifier in document.get("identifiers", [])
            if identifier.get("type") == "doi"
        ),
        None,
    )
    title = document["title"]
    if document.get("abstract"):
        title = f"{title}: {document['abstract']}"

    return Citation(
        key="ml_peg",
        title=title,
        authors=authors,
        doi=doi,
        url=document.get("repository-code"),
        entry_type="software",
    )


def load_model_citations(
    model_names: Iterable[str], filepath: str | Path | None = None
) -> dict[str, Citation | None]:
    """
    Load citations for MLIP models from ``models.yml``.

    Parameters
    ----------
    model_names
        Model identifiers to look up.
    filepath
        Path to model definitions YAML file. Default is models.yml in the models
        directory.

    Returns
    -------
    dict[str, Citation | None]
        Mapping of model name to its citation, or None when not yet supplied.
    """
    from ml_peg.models.get_models import load_model_configs

    model_names = list(model_names)
    configs, _ = load_model_configs(model_names, filepath)

    citations: dict[str, Citation | None] = {}
    for name in model_names:
        location = f"models.yml: {name}.citation"
        raw = configs[name].get("citation")
        if not raw:
            citations[name] = None
            continue
        raw = _mapping(raw, location)
        if not raw.get("title") or not raw.get("authors"):
            citations[name] = None
            continue
        citations[name] = Citation(
            key=name,
            title=_non_empty_string(raw.get("title"), f"{location}.title"),
            authors=_authors(raw.get("authors"), f"{location}.authors"),
            year=_year(raw.get("year"), f"{location}.year"),
            doi=_optional_string(raw.get("doi"), f"{location}.doi"),
            url=_optional_string(raw.get("url"), f"{location}.url"),
        )
    return citations


def load_framework_citations(
    framework_ids: Iterable[str],
) -> dict[str, Citation | None]:
    """
    Load citations for the frameworks a set of benchmarks was adapted from.

    Only entries registered as ``type: framework`` in ``frameworks.yml`` are
    included, matching the benchmarks that receive a prominent source-framework
    banner in the app. ML-PEG itself is never a source framework.

    Parameters
    ----------
    framework_ids
        Framework identifiers attached to the benchmarks of a run.

    Returns
    -------
    dict[str, Citation | None]
        Mapping of framework label to its citation, or None when not yet supplied.
    """
    registry = safe_load(FRAMEWORKS_FILE.read_text()) or {}

    citations: dict[str, Citation | None] = {}
    for framework_id in sorted(set(framework_ids)):
        entry = registry.get(framework_id) or {}
        if framework_id == "ml_peg" or entry.get("type") != "framework":
            continue
        label = entry.get("label", framework_id)
        location = f"frameworks.yml: {framework_id}.citation"
        raw = entry.get("citation") or {}
        if not raw.get("title") or not raw.get("authors"):
            citations[label] = None
            continue
        citations[label] = Citation(
            key=framework_id,
            title=_non_empty_string(raw.get("title"), f"{location}.title"),
            authors=_authors(raw.get("authors"), f"{location}.authors"),
            year=_year(raw.get("year"), f"{location}.year"),
            role="upstream_framework",
            url=entry.get("paper_url"),
        )
    return citations


def _deduplicate_citations(citations: Iterable[Citation]) -> tuple[Citation, ...]:
    """
    Deduplicate sources by DOI, then by citation key.

    Parameters
    ----------
    citations
        Citations to deduplicate, possibly repeated across benchmarks.

    Returns
    -------
    tuple[Citation, ...]
        One citation per distinct source, keeping first occurrences.
    """
    unique: dict[str, Citation] = {}
    for citation in citations:
        identifier = (
            f"doi:{citation.doi.lower()}"
            if citation.doi
            else f"key:{citation.key.lower()}"
        )
        unique.setdefault(identifier, citation)
    return tuple(unique.values())


def _citation_bibtex(citation: Citation) -> str:
    """
    Return supplied BibTeX or a minimal generated entry.

    Parameters
    ----------
    citation
        Citation to render.

    Returns
    -------
    str
        BibTeX entry for `citation`.
    """
    if citation.bibtex:
        return citation.bibtex.strip()

    fields = [
        f"  title = {{{citation.title}}}",
        f"  author = {{{' and '.join(citation.authors)}}}",
    ]
    if citation.year is not None:
        fields.append(f"  year = {{{citation.year}}}")
    if citation.doi:
        fields.append(f"  doi = {{{citation.doi}}}")
    if citation.url:
        fields.append(f"  url = {{{citation.url}}}")
    return f"@{citation.entry_type}{{{citation.key},\n" + ",\n".join(fields) + "\n}"


def _benchmark_citation_lines(credits: BenchmarkCredits) -> list[str]:
    """
    Return Markdown bullets describing one benchmark's sources.

    Parameters
    ----------
    credits
        Credits for one benchmark.

    Returns
    -------
    list[str]
        Markdown bullets, noting where ML-PEG itself is the only citation.
    """
    if not credits.citations:
        return [
            "- Devised for ML-PEG. No source beyond ML-PEG itself needs to be cited."
        ]
    return [
        f"- {citation.reference}"
        + (f" [{citation.role_label}]" if citation.role_label else "")
        for citation in credits.citations
    ]


# Not currently wired into any command: runs print guidance rather than writing files
def write_citation_bundle(
    benchmarks: Mapping[str, BenchmarkCredits],
    output_dir: str | Path,
    missing_benchmarks: Iterable[str] = (),
    models: Mapping[str, Citation | None] | None = None,
    frameworks: Mapping[str, Citation | None] | None = None,
) -> tuple[Path, Path]:
    """
    Write human-readable and BibTeX citation guidance for a benchmark run.

    Parameters
    ----------
    benchmarks
        Mapping of benchmark identifiers to their citation metadata.
    output_dir
        Directory in which to write ``CITATIONS.md`` and ``CITATIONS.bib``.
    missing_benchmarks
        Benchmarks which ran but do not yet provide citation metadata.
    models
        Mapping of MLIP model name to its citation, or None where not yet supplied.
    frameworks
        Mapping of source-framework label to its citation, or None where not yet
        supplied.

    Returns
    -------
    tuple[Path, Path]
        Paths to the Markdown and BibTeX files.
    """
    models = models or {}
    frameworks = frameworks or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "CITATIONS.md"
    bibtex_path = output_dir / "CITATIONS.bib"

    ml_peg = ml_peg_citation()
    lines = [
        "# Citation guidance",
        "",
        "This file covers the benchmarks and models of this run only.",
        "Benchmark implementers are credited separately from publication authors, and",
        "are not authors of the work you cite.",
        "",
        "## ML-PEG",
        "",
        f"- {ml_peg.reference}" if ml_peg else "- See CITATION.cff in the repository.",
    ]

    if benchmarks:
        lines.extend(["", "## Benchmark sources"])
        for benchmark, credits in benchmarks.items():
            lines.extend(
                ["", f"### `{benchmark}`", "", *_benchmark_citation_lines(credits)]
            )

        lines.extend(["", "## Benchmark implementation", ""])
        for benchmark, credits in benchmarks.items():
            names = ", ".join(item.name for item in credits.contributors)
            lines.append(f"- `{benchmark}`: {names or 'Not yet supplied'}")

    if frameworks:
        lines.extend(["", "## Source frameworks", ""])
        for label, citation in frameworks.items():
            lines.append(
                f"- `{label}`: {citation.reference if citation else 'To be added'}"
            )

    if models:
        lines.extend(["", "## Models", ""])
        for name, citation in models.items():
            lines.append(
                f"- `{name}`: {citation.reference if citation else 'To be added'}"
            )

    missing_models = sorted(name for name, citation in models.items() if not citation)
    missing_frameworks = sorted(
        label for label, citation in frameworks.items() if not citation
    )
    missing = sorted(set(missing_benchmarks))
    if missing or missing_models or missing_frameworks:
        lines.extend(["", "## Incomplete metadata", ""])
        if missing:
            lines.extend(
                [
                    "Benchmark citation metadata has not yet been supplied for:",
                    "",
                    *(f"- `{benchmark}`" for benchmark in missing),
                    "",
                ]
            )
        if missing_frameworks:
            lines.extend(
                [
                    "Source framework citations have not yet been supplied for:",
                    "",
                    *(f"- `{label}`" for label in missing_frameworks),
                    "",
                ]
            )
        if missing_models:
            lines.extend(
                [
                    "Model citations have not yet been supplied for:",
                    "",
                    *(f"- `{name}`" for name in missing_models),
                ]
            )

    markdown_path.write_text("\n".join(lines).rstrip() + "\n")

    sources = [citation for citation in (ml_peg,) if citation]
    sources.extend(
        citation for credits in benchmarks.values() for citation in credits.citations
    )
    sources.extend(citation for citation in frameworks.values() if citation)
    sources.extend(citation for citation in models.values() if citation)
    bibtex_path.write_text(
        "\n\n".join(
            _citation_bibtex(citation) for citation in _deduplicate_citations(sources)
        )
        + "\n"
    )
    return markdown_path, bibtex_path


def _wrap(text: str, indent: str, continuation: str | None = None) -> list[str]:
    """
    Wrap one entry to the summary width.

    Parameters
    ----------
    text
        Text to wrap.
    indent
        Indent applied to the first line.
    continuation
        Indent applied to later lines. Default is `indent` plus two spaces, giving a
        hanging indent that keeps multi-line entries visually grouped.

    Returns
    -------
    list[str]
        Wrapped lines.
    """
    return textwrap.wrap(
        text,
        width=SUMMARY_WIDTH,
        initial_indent=indent,
        subsequent_indent=continuation if continuation is not None else f"{indent}  ",
        # Names and identifiers such as "ml-peg" must not be split at a hyphen
        break_on_hyphens=False,
        break_long_words=False,
    )


def _section(title: str) -> list[str]:
    """
    Build a blank-separated section heading with an underline rule.

    Parameters
    ----------
    title
        Section heading text.

    Returns
    -------
    list[str]
        Blank line, heading, and rule.
    """
    return ["", f"  {title}", "  " + "-" * (SUMMARY_WIDTH - 4)]


def _citation_lines(citation: Citation, indent: str) -> list[str]:
    """
    Build the wrapped reference and link lines for one citation.

    Parameters
    ----------
    citation
        Citation to render.
    indent
        Indent applied to the first line of the reference.

    Returns
    -------
    list[str]
        Wrapped reference, ending in the DOI or URL when one is set. Links are never
        broken across lines, so that they stay selectable in the terminal.
    """
    text = citation.reference
    if citation.link:
        text = f"{text} {citation.link}"
    return _wrap(text, indent)


def format_citation_summary(
    benchmarks: Mapping[str, BenchmarkCredits],
    missing_benchmarks: Iterable[str] = (),
    models: Mapping[str, Citation | None] | None = None,
    frameworks: Mapping[str, Citation | None] | None = None,
) -> str:
    """
    Format citation guidance as a self-contained block for the terminal.

    Parameters
    ----------
    benchmarks
        Mapping of benchmark identifiers to their citation metadata.
    missing_benchmarks
        Benchmarks which ran but do not yet provide citation metadata.
    models
        Mapping of MLIP model name to its citation, or None where not yet supplied.
    frameworks
        Mapping of source-framework label to its citation, or None where not yet
        supplied.

    Returns
    -------
    str
        Citation guidance for printing to the terminal.
    """
    models = models or {}
    frameworks = frameworks or {}
    missing = sorted(set(missing_benchmarks))

    rule = "=" * SUMMARY_WIDTH
    lines = [
        rule,
        "CITATION GUIDANCE".center(SUMMARY_WIDTH).rstrip(),
        rule,
        "",
        *_wrap(
            "Please cite the benchmarks below and the models you ran. Benchmark "
            "implementers are credited separately, and are not authors of the work "
            "being cited.",
            "  ",
            "  ",
        ),
    ]

    if benchmarks or missing:
        lines.extend(_section(f"BENCHMARKS ({len(benchmarks) + len(missing)})"))
        for index, benchmark in enumerate(sorted(set(benchmarks) | set(missing))):
            if index:
                lines.append("")
            lines.append(f"    {benchmark}")
            credits = benchmarks.get(benchmark)
            if credits is None:
                lines.extend(
                    [
                        "      benchmark citation:",
                        "        ! to be added",
                        "      implemented in ML-PEG by:",
                        "        ! to be added",
                    ]
                )
                continue
            plural = "s" if len(credits.citations) > 1 else ""
            lines.append(f"      benchmark citation{plural}:")
            for citation in credits.citations:
                lines.extend(_citation_lines(citation, "        "))
            if not credits.citations:
                lines.extend(
                    _wrap("Devised for ML-PEG, no further citation needed.", "        ")
                )
            names = ", ".join(item.name for item in credits.contributors)
            lines.append("      implemented in ML-PEG by:")
            lines.extend(
                _wrap(names, "        ") if names else ["        ! to be added"]
            )

    if frameworks:
        lines.extend(_section(f"SOURCE FRAMEWORKS ({len(frameworks)})"))
        for index, (label, citation) in enumerate(frameworks.items()):
            if index:
                lines.append("")
            lines.append(f"    {label}")
            lines.extend(
                _citation_lines(citation, "      ")
                if citation
                else ["      ! citation to be added"]
            )

    if models:
        lines.extend(_section(f"MODELS ({len(models)})"))
        for index, (name, citation) in enumerate(models.items()):
            if index:
                lines.append("")
            lines.append(f"    {name}")
            lines.extend(
                _citation_lines(citation, "      ")
                if citation
                else ["      ! citation to be added"]
            )

    unfilled_frameworks = sum(1 for c in frameworks.values() if not c)
    unfilled_models = sum(1 for c in models.values() if not c)
    incomplete = []
    if missing:
        incomplete.append(f"{len(missing)} benchmark(s)")
    if unfilled_frameworks:
        incomplete.append(f"{unfilled_frameworks} framework(s)")
    if unfilled_models:
        incomplete.append(f"{unfilled_models} model(s)")
    if incomplete:
        lines.extend(
            [
                "",
                *_wrap(
                    f"! Citation metadata is incomplete for {', '.join(incomplete)}. "
                    "Please help by adding it.",
                    "  ",
                    "    ",
                ),
            ]
        )

    lines.append(rule)
    return "\n".join(lines)


def build_run_citations(
    script_paths: Iterable[str | Path],
    model_names: Iterable[str] = (),
    models_file: str | Path | None = None,
    framework_ids: Iterable[str] = (),
) -> str:
    """
    Build citation guidance for the benchmarks and models of a run.

    Parameters
    ----------
    script_paths
        Benchmark scripts to report citations for.
    model_names
        MLIP models used by the run. Default is no models.
    models_file
        Path to model definitions YAML file. Default is models.yml in the models
        directory.
    framework_ids
        Framework identifiers attached to the benchmarks of the run. Default is none.

    Returns
    -------
    str
        Citation guidance for printing to the terminal.
    """
    from ml_peg.analysis import ANALYSIS_ROOT

    benchmarks, missing = collect_benchmark_credits(sorted(script_paths), ANALYSIS_ROOT)
    return format_citation_summary(
        benchmarks,
        missing,
        load_model_citations(model_names, models_file),
        load_framework_citations(framework_ids),
    )


def collect_benchmark_credits(
    script_paths: Sequence[str | Path], analysis_root: str | Path
) -> tuple[dict[str, BenchmarkCredits], tuple[str, ...]]:
    """
    Collect available credits and missing identifiers for benchmark scripts.

    Parameters
    ----------
    script_paths
        Benchmark scripts to collect credits for.
    analysis_root
        Root of the analysis tree holding the metadata files.

    Returns
    -------
    tuple[dict[str, BenchmarkCredits], tuple[str, ...]]
        Credits keyed by ``category/benchmark``, and the sorted identifiers of
        benchmarks with no metadata file.
    """
    benchmarks: dict[str, BenchmarkCredits] = {}
    missing: set[str] = set()
    for script_path in script_paths:
        script_path = Path(script_path)
        benchmark = f"{script_path.parent.parent.name}/{script_path.parent.name}"
        metadata_path = citation_metadata_path(script_path, analysis_root)
        credits = load_optional_benchmark_credits(metadata_path)
        if credits is None:
            missing.add(benchmark)
        else:
            benchmarks[benchmark] = credits
    return benchmarks, tuple(sorted(missing))
