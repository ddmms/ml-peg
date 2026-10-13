"""Build benchmark citation, implementation, and contributor credit components."""

from __future__ import annotations

from urllib.parse import quote

from dash import html
from dash.development.base_component import Component
from dash.html import Details, Div, Summary

from ml_peg.app.utils.icons import EMAIL_ICON, GITHUB_ICON
from ml_peg.utils.citations import (
    BenchmarkCredits,
    Citation,
    Contributor,
    format_authors,
)

CREDIT_LABEL_STYLE = {"color": "var(--mlpeg-heading)"}
CREDIT_NOTE_STYLE = {"color": "var(--mlpeg-muted)", "fontSize": "0.9em"}


def _citation_reference(citation: Citation) -> Component:
    """
    Build a citation with a linked title and visible author names.

    Parameters
    ----------
    citation
        Citation to render.

    Returns
    -------
    Component
        One citation line, with the title hyperlinked if a DOI or URL is set, followed
        by the DOI itself where there is one.
    """
    authors = format_authors(citation.authors)
    year = f" ({citation.year})" if citation.year is not None else ""
    title = html.Strong(citation.title)
    contents = [
        html.A(title, href=citation.link, target="_blank") if citation.link else title,
        html.Span(f", {authors}{year}"),
    ]
    # Roles say what a source contributed. Benchmark papers are left untagged, as
    # they are the default kind of benchmark reference
    if citation.role_label and citation.role != "benchmark_method":
        contents.append(html.Span(f" ({citation.role_label})", style=CREDIT_NOTE_STYLE))
    if citation.doi:
        # Shown in full so the DOI can be read and copied, not just followed
        contents.append(
            html.Div(
                [
                    html.Span("doi: ", style=CREDIT_NOTE_STYLE),
                    html.A(
                        citation.doi,
                        href=citation.link,
                        target="_blank",
                        style=CREDIT_NOTE_STYLE,
                    ),
                ]
            )
        )
    return html.Div(contents, style={"marginTop": "2px"})


def _credit_line(label: str, value: Component | str, top_margin: str) -> Component:
    """
    Build one labelled credit line for the benchmark credit box.

    Parameters
    ----------
    label
        Bold label introducing the line.
    value
        Content shown after the label.
    top_margin
        CSS top margin for the line.

    Returns
    -------
    Component
        One labelled credit line.
    """
    return html.Div(
        [html.Strong(label, style=CREDIT_LABEL_STYLE), value],
        style={"marginTop": top_margin},
    )


def _contributor(contributor: Contributor) -> list[Component]:
    """
    Build one person's name with optional GitHub and email contact links.

    Parameters
    ----------
    contributor
        Person who contributed to the benchmark's addition to ML-PEG.

    Returns
    -------
    list[Component]
        Name linked to GitHub when known, followed by an optional email link.
    """
    name = html.Span(contributor.name)
    people: list[Component] = [name]
    if contributor.github:
        icon = html.Span(
            style={
                "backgroundColor": "var(--mlpeg-heading)",
                "display": "inline-block",
                "height": "14px",
                "mask": f'url("{GITHUB_ICON}") center / contain no-repeat',
                "verticalAlign": "-2px",
                "WebkitMask": f'url("{GITHUB_ICON}") center / contain no-repeat',
                "width": "14px",
                "marginLeft": "4px",
            },
        )
        people = [
            html.A(
                [name, icon],
                href=f"https://github.com/{contributor.github}",
                target="_blank",
                title=f"@{contributor.github}",
                **{"aria-label": f"{contributor.name} on GitHub"},
            )
        ]
    if contributor.email:
        people.append(
            html.A(
                html.Span(
                    style={
                        "backgroundColor": "var(--mlpeg-heading)",
                        "display": "inline-block",
                        "height": "14px",
                        "mask": f'url("{EMAIL_ICON}") center / contain no-repeat',
                        "verticalAlign": "-2px",
                        "WebkitMask": f'url("{EMAIL_ICON}") center / contain no-repeat',
                        "width": "14px",
                    },
                    **{"aria-hidden": "true"},
                ),
                href=f"mailto:{quote(contributor.email, safe='@.+-_')}",
                title=contributor.email,
                style={"marginLeft": "4px"},
                **{"aria-label": f"Email {contributor.name} at {contributor.email}"},
            )
        )
    return people


def build_benchmark_credit_components(
    credits: BenchmarkCredits | None,
) -> Component:
    """
    Build separate implementation, contributor, and reference credits.

    Missing implementers or citations are shown as "To be added". Other contributors
    are optional. References are collapsed, as a benchmark can cite many sources.

    Parameters
    ----------
    credits
        Validated benchmark credit metadata, or None when no ``citations.yml`` exists.

    Returns
    -------
    Component
        Credit box for the benchmark header.
    """
    if credits is None or not credits.citations:
        citation_line = _credit_line(
            "Benchmark references: ", html.Span("To be added"), "0"
        )
    else:
        # Each reference's role tag says how it relates to the benchmark
        citation_line = Details(
            [
                Summary(
                    [
                        html.Strong("Benchmark references", style=CREDIT_LABEL_STYLE),
                        html.Span(
                            f" ({len(credits.citations)})", style=CREDIT_NOTE_STYLE
                        ),
                    ],
                    style={"cursor": "pointer"},
                ),
                *[_citation_reference(citation) for citation in credits.citations],
            ]
        )

    credit_lines = []
    groups = [("Implemented by: ", credits.implementers if credits else ())]
    if credits and credits.contributors:
        groups.append(("Contributors: ", credits.contributors))
    for label, persons in groups:
        people: list[Component] = []
        for person in persons:
            if people:
                people.append(html.Span(", "))
            people.extend(_contributor(person))
        credit_lines.append(
            _credit_line(label, html.Span(people or "To be added"), "0")
        )
    return Div(
        [
            *credit_lines,
            Div(citation_line, style={"marginTop": "8px"}),
        ],
        style={
            "background": "var(--mlpeg-surface-2)",
            "border": "1px solid var(--mlpeg-border)",
            "borderLeft": "4px solid var(--mlpeg-border-strong)",
            "borderRadius": "6px",
            "margin": "8px 0 12px",
            "maxWidth": "1100px",
            "padding": "10px 12px",
            "width": "fit-content",
        },
    )
