"""Read shared BibTeX records for citation metadata and downloads."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pybtex.database import BibliographyData, parse_file
from pylatexenc.latex2text import LatexNodes2Text

from ml_peg.citations import CITATIONS_ROOT

BIBTEX_FILE = CITATIONS_ROOT / "references.bib"


@lru_cache(maxsize=1)
def load_bibliography(path: Path) -> BibliographyData:
    """
    Parse the shared bibliography once per process.

    Parameters
    ----------
    path
        BibTeX file to read.

    Returns
    -------
    BibliographyData
        Parsed records with their original fields and author order.
    """
    return parse_file(str(path))


def citation_fields(key: str) -> dict[str, Any]:
    """
    Extract display metadata from one BibTeX record.

    Parameters
    ----------
    key
        Key in the shared bibliography.

    Returns
    -------
    dict[str, Any]
        Citation defaults, with LaTeX converted to readable text.

    Raises
    ------
    ValueError
        If the bibliography cannot be read or the record is absent or invalid.
    """
    try:
        data = load_bibliography(BIBTEX_FILE)
    except Exception as err:
        raise ValueError(f"cannot read bibliography {BIBTEX_FILE}: {err}") from err
    if key not in data.entries:
        raise ValueError(f"unknown BibTeX key {key!r} in {BIBTEX_FILE}")
    entry = data.entries[key]
    render = LatexNodes2Text().latex_to_text
    authors = []
    for person in entry.persons.get("author", []):
        name = " ".join(
            person.first_names
            + person.middle_names
            + person.prelast_names
            + person.last_names
        )
        if person.lineage_names:
            name += ", " + " ".join(person.lineage_names)
        authors.append(" ".join(render(name).split()))
    year = entry.fields.get("year")
    try:
        year = int(year) if year else None
    except ValueError as err:
        raise ValueError(f"BibTeX key {key!r} has invalid year {year!r}") from err
    doi = entry.fields.get("doi")
    if doi:
        doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    url = entry.fields.get("url")
    if not url and entry.fields.get("archiveprefix", "").lower() == "arxiv":
        if eprint := entry.fields.get("eprint"):
            url = f"https://arxiv.org/abs/{eprint}"
    return {
        "key": key,
        "title": " ".join(render(entry.fields.get("title", "")).split()),
        "authors": authors,
        "year": year,
        "doi": doi,
        "url": url,
    }
