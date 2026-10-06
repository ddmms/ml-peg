"""Load benchmark credits and write citation guidance for ML-PEG runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import TYPE_CHECKING, Any

from yaml import safe_load

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

FRAMEWORKS_FILE = Path(__file__).parent / "app" / "utils" / "frameworks.yml"

# Width of the citation guidance printed after a benchmark run
SUMMARY_WIDTH = 79

# Author lists longer than this are shortened to "First Author et al."
MAX_AUTHORS = 5

# Joins words that must not be split across lines when wrapping
NBSP = "\u00a0"


def format_authors(authors: Sequence[str]) -> str:
    """
    Join author names, shortening long lists.

    Parameters
    ----------
    authors
        Author names, in order.

    Returns
    -------
    str
        All names, or the first name followed by "et al." when there are more than
        `MAX_AUTHORS`.
    """
    if len(authors) > MAX_AUTHORS:
        return f"{authors[0]} et al."
    return ", ".join(authors)


CITATION_ROLES = {
    "benchmark_method",
    "inspired_by",
    "reference_data",
    "reference_method",
    "upstream_framework",
}
CITATION_ROLE_LABELS = {
    "benchmark_method": "benchmark paper",
    "inspired_by": "inspired by",
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
        year = f" ({self.year})" if self.year is not None else ""
        return f"{format_authors(self.authors)}{year}. {self.title}."

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

    Empty ``citations`` or ``contributors`` mean that information has not been added
    yet, and are shown as placeholders like missing metadata.
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
    authors = tuple(
        _non_empty_string(author, f"{location}[{index}]")
        for index, author in enumerate(value)
    )
    # Long lists are shortened for display, so every entry must be a real author
    if any(author.rstrip(".").casefold() == "et al" for author in authors):
        raise CitationMetadataError(f"{location} must list authors, not 'et al.'")
    return authors


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

    # Implementers are listed alphabetically by surname, unlike citation authors,
    # whose published order is kept
    contributors = tuple(
        sorted(
            (
                _parse_contributor(value, f"{path}: contributors[{index}]")
                for index, value in enumerate(raw_contributors)
            ),
            key=lambda contributor: (
                contributor.name.split()[-1].casefold(),
                contributor.name.casefold(),
            ),
        )
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


def citation_metadata_path(script_path: str | Path) -> Path:
    """
    Return the citation metadata path corresponding to a benchmark script.

    Parameters
    ----------
    script_path
        Path to a ``calc_*.py`` benchmark script.

    Returns
    -------
    Path
        Path to the benchmark's ``citations.yml``, which may not exist.
    """
    return Path(script_path).parent / "citations.yml"


def app_citation_metadata_path(
    category: str, benchmark: str, calcs_root: str | Path
) -> Path:
    """
    Return calc-owned citation metadata for an app benchmark.

    The app and calc trees normally use identical directory names. The fallback
    handles the existing ``CHO_GAP`` app directory whose calc directory is
    ``CHO-GAP``.

    Parameters
    ----------
    category
        Benchmark category, as named in the app tree.
    benchmark
        Benchmark name, as named in the app tree.
    calcs_root
        Root of the calculation tree holding the metadata files.

    Returns
    -------
    Path
        Path to the benchmark's ``citations.yml``, which may not exist.
    """
    category_path = Path(calcs_root) / category
    benchmark_path = category_path / benchmark
    if not benchmark_path.is_dir():
        hyphenated_path = category_path / benchmark.replace("_", "-")
        if hyphenated_path.is_dir():
            benchmark_path = hyphenated_path
    return benchmark_path / "citations.yml"


def load_framework_citations(
    framework_ids: Iterable[str],
) -> dict[str, Citation | None]:
    """
    Load citations for the frameworks a set of benchmarks was taken from.

    Only entries registered as ``type: framework`` in ``frameworks.yml`` are
    included. ML-PEG itself is never a source framework.

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
        raw = _mapping(entry.get("citation") or {}, location)
        if not raw.get("title") or not raw.get("authors"):
            citations[label] = None
            continue
        citations[label] = Citation(
            key=framework_id,
            title=_non_empty_string(raw.get("title"), f"{location}.title"),
            authors=_authors(raw.get("authors"), f"{location}.authors"),
            year=_year(raw.get("year"), f"{location}.year"),
            role="upstream_framework",
            url=_optional_string(
                entry.get("paper_url"), f"frameworks.yml: {framework_id}.paper_url"
            ),
        )
    return citations


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


def _citation_lines(citation: Citation, indent: str, number: str = "") -> list[str]:
    """
    Build the wrapped lines for one citation, optionally numbered.

    Parameters
    ----------
    citation
        Citation to render.
    indent
        Indent applied to the first line.
    number
        Label such as ``"[1]"`` shown before the reference. Default is none.

    Returns
    -------
    list[str]
        Wrapped reference, with later lines aligned under its text and ending in the
        DOI or URL when one is set. Links are never broken across lines, so that they
        stay selectable in the terminal.
    """
    text = citation.reference
    # Benchmark papers are the default kind of reference, and framework citations sit
    # under their own heading, so only other roles are tagged
    if citation.role_label and citation.role not in (
        "benchmark_method",
        "upstream_framework",
    ):
        # Non-breaking spaces keep a tag such as "(inspired by)" on one line
        tag = citation.role_label.replace(" ", NBSP)
        text = f"{text[:-1]} ({tag})."
    if citation.link:
        text = f"{text} {citation.link}"
    if not number:
        lines = _wrap(text, indent)
    else:
        prefix = f"{indent}{number} "
        lines = _wrap(text, prefix, " " * len(prefix))
    return [line.replace(NBSP, " ") for line in lines]


def format_citation_summary(
    benchmarks: Mapping[str, BenchmarkCredits],
    missing_benchmarks: Iterable[str] = (),
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
    frameworks
        Mapping of source-framework label to its citation, or None where not yet
        supplied.

    Returns
    -------
    str
        Citation guidance for printing to the terminal.
    """
    frameworks = frameworks or {}
    names = sorted(set(benchmarks) | set(missing_benchmarks))

    rule = "=" * SUMMARY_WIDTH
    lines = [
        rule,
        "CITATIONS".center(SUMMARY_WIDTH).rstrip(),
        rule,
        "",
        *_wrap(
            "Please cite the references below for the benchmarks you ran. "
            "Implementers are credited for adding each benchmark to ML-PEG.",
            "  ",
            "  ",
        ),
    ]

    unfilled_benchmarks = 0
    if names:
        lines.extend(_section(f"BENCHMARKS ({len(names)})"))
        for index, benchmark in enumerate(names):
            credits = benchmarks.get(benchmark)
            citations = credits.citations if credits else ()
            contributors = ", ".join(
                item.name for item in (credits.contributors if credits else ())
            )
            unfilled_benchmarks += not citations or not contributors

            if index:
                lines.append("")
            implemented = (
                f"implemented by {contributors}"
                if contributors
                else "! implementer to be added"
            )
            lines.extend(_wrap(f"{benchmark}  ({implemented})", "  ", "    "))
            if not citations:
                lines.append("    ! references to be added")
            # Right-align the numbers, so references start in one column past [9]
            width = len(f"[{len(citations)}]")
            for number, citation in enumerate(citations, start=1):
                label = f"[{number}]".rjust(width)
                lines.extend(_citation_lines(citation, "    ", label))

    if frameworks:
        lines.extend(_section(f"SOURCE FRAMEWORKS ({len(frameworks)})"))
        for index, (label, citation) in enumerate(frameworks.items()):
            if index:
                lines.append("")
            lines.append(f"  {label}")
            lines.extend(
                _citation_lines(citation, "    ")
                if citation
                else ["    ! citation to be added"]
            )

    unfilled_frameworks = sum(1 for c in frameworks.values() if not c)
    incomplete = []
    if unfilled_benchmarks:
        incomplete.append(f"{unfilled_benchmarks} benchmark(s)")
    if unfilled_frameworks:
        incomplete.append(f"{unfilled_frameworks} framework(s)")
    if incomplete:
        lines.extend(
            [
                "",
                *_wrap(
                    f"! Citation metadata is incomplete for {', '.join(incomplete)}. "
                    "Please help by adding it to citations.yml.",
                    "  ",
                    "    ",
                ),
            ]
        )

    lines.append(rule)
    return "\n".join(lines)


def build_run_citations(
    script_paths: Iterable[str | Path],
    framework_ids: Iterable[str] = (),
) -> str:
    """
    Build citation guidance for the benchmarks of a run.

    Parameters
    ----------
    script_paths
        Benchmark scripts to report citations for.
    framework_ids
        Framework identifiers attached to the benchmarks of the run. Default is none.

    Returns
    -------
    str
        Citation guidance for printing to the terminal.
    """
    benchmarks, missing = collect_benchmark_credits(sorted(script_paths))
    return format_citation_summary(
        benchmarks,
        missing,
        load_framework_citations(framework_ids),
    )


def collect_benchmark_credits(
    script_paths: Sequence[str | Path],
) -> tuple[dict[str, BenchmarkCredits], tuple[str, ...]]:
    """
    Collect available credits and missing identifiers for benchmark scripts.

    Parameters
    ----------
    script_paths
        Benchmark scripts to collect credits for.

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
        metadata_path = citation_metadata_path(script_path)
        credits = load_optional_benchmark_credits(metadata_path)
        if credits is None:
            missing.add(benchmark)
        else:
            benchmarks[benchmark] = credits
    return benchmarks, tuple(sorted(missing))
