"""Build copy and download controls for supplied benchmark BibTeX."""

from __future__ import annotations

from base64 import b64encode
import re

from dash import dcc, html
from dash.html import Div

from ml_peg.utils.bibtex import BibTeXExport


def build_bibtex_controls(
    export: BibTeXExport,
    label: str = "benchmark",
    *,
    preview: bool = False,
    scope: str = "",
    heading: str | None = None,
) -> Div:
    """
    Build a BibTeX download with optional copyable preview and coverage information.

    Parameters
    ----------
    export
        Available BibTeX records and missing references.
    label
        Name used for the downloaded file.
    preview
        Include a collapsed BibTeX preview with a clipboard control.
    scope
        Visible explanation of which benchmarks are included.
    heading
        Custom heading. Defaults to "BibTeX" for previews and "References" otherwise.

    Returns
    -------
    Div
        Controls requiring no server callbacks or model/element filter state.
    """
    children = []
    if preview and scope:
        children.append(html.Div(scope, className="mlpeg-bibtex-scope"))
    actions = []
    if export.entries:
        text = export.text
        filename = re.sub(r"[^a-zA-Z0-9_-]+", "-", label).strip("-") or "citations"
        encoded = b64encode(text.encode("utf-8")).decode("ascii")
        actions.append(
            html.A(
                [html.Span("↓", **{"aria-hidden": "true"}), "Download BibTeX"],
                href=f"data:application/x-bibtex;charset=utf-8;base64,{encoded}",
                download=f"mlpeg-{filename}.bib",
                className="mlpeg-bibtex-download",
            )
        )
        if preview:
            actions.append(
                dcc.Clipboard(
                    content=text,
                    children="Copy BibTeX",
                    copied_children="Copied!",
                    title="Copy BibTeX",
                    className="mlpeg-bibtex-copy",
                )
            )
        children.append(html.Div(actions, className="mlpeg-bibtex-actions"))
        if preview:
            children.append(html.Pre(text, className="mlpeg-bibtex-code"))
    if export.missing:
        count = len(export.missing)
        children.append(
            html.Details(
                [
                    html.Summary(
                        f"BibTeX unavailable for {count} "
                        + ("reference." if count == 1 else "references.")
                        + (" Download is incomplete." if export.entries else "")
                    ),
                    html.Div([html.Div(title) for title in sorted(export.missing)]),
                ],
                className="mlpeg-bibtex-missing",
            )
        )
    count = len(export.entries)
    count_label = f"{count} " + ("reference" if count == 1 else "references")
    if preview:
        return Div(
            html.Details(
                [
                    html.Summary(
                        [
                            heading or "BibTeX",
                            html.Span(count_label, className="mlpeg-bibtex-count"),
                        ]
                    ),
                    html.Div(children, className="mlpeg-bibtex-body"),
                ],
                className="mlpeg-bibtex-preview",
            ),
            className="mlpeg-bibtex-controls",
        )
    return Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                heading or "References", className="mlpeg-bibtex-label"
                            ),
                            html.Span(count_label, className="mlpeg-bibtex-count"),
                        ]
                    ),
                    html.Div(scope, className="mlpeg-bibtex-scope") if scope else None,
                ],
                className="mlpeg-bibtex-heading",
            ),
            *children,
        ],
        className="mlpeg-bibtex-controls mlpeg-bibtex-group",
    )
