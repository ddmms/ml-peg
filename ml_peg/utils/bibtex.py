"""Collect supplied BibTeX records for benchmark and grouped downloads."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from warnings import warn

from pybtex.database import BibliographyData
from yaml import safe_load

from ml_peg.citations import bibliography
from ml_peg.utils.citations import FRAMEWORKS_FILE, BenchmarkCredits


@dataclass
class BibTeXExport:
    """Unique supplied records and references still awaiting BibTeX."""

    entries: dict[str, str] = field(default_factory=dict)
    missing: set[str] = field(default_factory=set)

    @property
    def text(self) -> str:
        """
        Return records in stable citation-key order.

        Returns
        -------
        str
            Combined BibTeX text, or an empty string when no records are available.
        """
        return "\n\n".join(
            self.entries[key].strip() for key in sorted(self.entries)
        ) + ("\n" if self.entries else "")


@lru_cache(maxsize=1)
def _library() -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """
    Parse supplied records once per app process and identify duplicate DOIs.

    Returns
    -------
    tuple
        Records by key and a DOI/URL lookup for source-framework references.
    """
    try:
        data = bibliography.load_bibliography(bibliography.BIBTEX_FILE)
    except Exception as exc:
        warn(f"BibTeX downloads unavailable: {exc}", stacklevel=2)
        return {}, {}
    records = {}
    links = {}
    doi_keys = {}
    for key, entry in sorted(data.entries.items()):
        doi = entry.fields.get("doi", "").lower().removeprefix("https://doi.org/")
        canonical = doi_keys.setdefault(doi, key) if doi else key
        text = BibliographyData(entries={key: entry}).to_string("bibtex")
        records[key] = records.get(canonical, (canonical, text))
        if doi:
            links[f"https://doi.org/{doi}"] = key
        if url := entry.fields.get("url"):
            links[url.lower().rstrip("/")] = key
        if eprint := entry.fields.get("eprint"):
            links[f"https://arxiv.org/abs/{eprint}"] = key
    return records, links


@lru_cache(maxsize=1)
def _frameworks(path: Path) -> dict:
    """
    Read source-framework metadata once per app process.

    Parameters
    ----------
    path
        Framework registry path.

    Returns
    -------
    dict
        Registered framework and paper metadata.
    """
    return safe_load(path.read_text()) or {}


def collect_bibtex(
    credits: BenchmarkCredits | None,
    framework_ids: Iterable[str] = (),
    *,
    label: str = "Benchmark",
) -> BibTeXExport:
    """
    Collect supplied benchmark and source-framework records without inventing entries.

    Parameters
    ----------
    credits
        Benchmark citation metadata, if available.
    framework_ids
        Associated source frameworks or papers. ML-PEG itself is excluded.
    label
        Benchmark label used when reference metadata is missing.

    Returns
    -------
    BibTeXExport
        Available records and names of references lacking BibTeX.
    """
    records, links = _library()
    result = BibTeXExport()
    registry = _frameworks(FRAMEWORKS_FILE)
    for citation in credits.citations if credits else ():
        config = (
            registry.get(citation.key, {})
            if citation.role == "upstream_framework"
            else {}
        )
        key = (
            citation.bibtex_key
            or config.get("bibtex_key")
            or config.get("citation", {}).get("bibtex_key")
            or citation.key
        )
        if key not in records and citation.link:
            key = links.get(citation.link.lower().rstrip("/"), key)
        if key in records:
            canonical, text = records[key]
            result.entries[canonical] = text
        else:
            result.missing.add(citation.title)
    for framework_id in set(framework_ids) - {"ml_peg"}:
        config = registry.get(framework_id, {})
        url = config.get("paper_url", "")
        key = (
            config.get("bibtex_key")
            or config.get("citation", {}).get("bibtex_key")
            or links.get(url.lower().rstrip("/"))
        )
        if key in records:
            canonical, text = records[key]
            result.entries[canonical] = text
        elif url:
            result.missing.add(
                config.get("citation", {}).get("title")
                or config.get("label", framework_id)
            )
    if (
        not (credits and credits.citations)
        and not result.entries
        and not result.missing
    ):
        result.missing.add(f"{label}: references to be added")
    return result


def combine_bibtex(exports: Iterable[BibTeXExport]) -> BibTeXExport:
    """
    Combine benchmark exports without repeating shared references.

    Parameters
    ----------
    exports
        Exports for every benchmark in the selected group, regardless of UI filters.

    Returns
    -------
    BibTeXExport
        Deduplicated records and missing references.
    """
    result = BibTeXExport()
    for export in exports:
        result.entries.update(export.entries)
        result.missing.update(export.missing)
    return result
