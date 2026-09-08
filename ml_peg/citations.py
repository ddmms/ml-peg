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


def _citation_heading(citations: Sequence[Citation]) -> str:
    """
    Return the heading introducing a benchmark's sources.

    Parameters
    ----------
    citations
        Sources listed for one benchmark.

    Returns
    -------
    str
        Heading matching what the sources are. A benchmark built on earlier work
        rather than taken from it has no benchmark paper to name.
    """
    if not any(citation.role == "benchmark_method" for citation in citations):
        return "built on"
    return f"benchmark citation{'s' if len(citations) > 1 else ''}"


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
    missing = sorted(set(missing_benchmarks))

    rule = "=" * SUMMARY_WIDTH
    lines = [
        rule,
        "CITATION GUIDANCE".center(SUMMARY_WIDTH).rstrip(),
        rule,
        "",
        *_wrap(
            "Please cite the benchmarks below. Benchmark implementers are credited "
            "separately, and are not authors of the work being cited.",
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
            if credits.citations:
                lines.append(f"      {_citation_heading(credits.citations)}:")
                for citation in credits.citations:
                    lines.extend(_citation_lines(citation, "        "))
            else:
                lines.append("      Devised for ML-PEG")
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

    unfilled_frameworks = sum(1 for c in frameworks.values() if not c)
    incomplete = []
    if missing:
        incomplete.append(f"{len(missing)} benchmark(s)")
    if unfilled_frameworks:
        incomplete.append(f"{unfilled_frameworks} framework(s)")
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
