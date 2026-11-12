"""Sidebar search box for jumping to a category, framework or benchmark."""

from __future__ import annotations

from urllib.parse import quote

from dash import ALL, Input, Output, callback, clientside_callback, ctx, no_update
from dash.dcc import Input as SearchInput
from dash.dcc import Link, Store
from dash.html import Div, Label, Span

MAX_RESULTS = 8


def build_search_box() -> Div:
    """
    Build the sidebar search input and its results list.

    Returns
    -------
    Div
        Search input with an empty results container below it.
    """
    return Div(
        [
            Label("Search", htmlFor="sidebar-search"),
            SearchInput(
                id="sidebar-search",
                type="search",
                placeholder="Search benchmarks…",
                debounce=False,
                autoComplete="off",
                className="mlpeg-search-input",
            ),
            Div(id="sidebar-search-results", className="mlpeg-search-results"),
            Store(id="sidebar-search-jump", storage_type="memory"),
        ],
        className="mlpeg-search",
        **{"role": "search"},
    )


def build_search_entries(
    category_paths: dict[str, str],
    framework_paths: dict[str, str],
    framework_labels: dict[str, str],
    category_views: dict[str, dict[str, object]],
) -> list[dict[str, str]]:
    """
    Build the list of searchable pages and benchmarks.

    Parameters
    ----------
    category_paths
        Mapping of category title to its URL path.
    framework_paths
        Mapping of framework ID to its URL path.
    framework_labels
        Mapping of framework ID to display label.
    category_views
        Category metadata, used for the benchmarks within each category.

    Returns
    -------
    list[dict[str, str]]
        Entries with a display label, a kind shown next to it, and a target href.
    """
    entries = [
        {"label": category_title, "kind": "Category", "href": category_path}
        for category_title, category_path in category_paths.items()
    ]

    entries.extend(
        {
            "label": framework_labels[framework_id],
            "kind": "Framework",
            "href": framework_path,
        }
        for framework_id, framework_path in framework_paths.items()
    )

    # benchmarks link to their category page with the card name in the hash
    for category_title, category_view in category_views.items():
        category_path = category_paths[category_title]
        entries.extend(
            {
                "label": test["name"],
                "kind": category_title,
                "href": f"{category_path}#{quote(test['name'])}",
                "search_text": (
                    f"{test['name']} {test.get('description', '')} {category_title}"
                ),
            }
            for test in category_view["tests"]
        )

    return entries


def register_search_callbacks(entries: list[dict[str, str]]) -> None:
    """
    Register the search callbacks.

    Parameters
    ----------
    entries
        Searchable entries, as built by ``build_search_entries``.
    """

    @callback(
        Output("sidebar-search", "value"),
        Output("sidebar-search-results", "children"),
        Input("sidebar-search", "value"),
        Input("app-location", "href"),
        Input("sidebar-search-jump", "data"),
        prevent_initial_call=True,
    )
    def show_matches(
        query: str | None, href: str | None, selection: dict | None
    ) -> tuple[str | object, list[Div] | Div]:
        """
        List the entries matching the query, or clear the box after navigation.

        Parameters
        ----------
        query
            Current text in the search input.
        href
            Current URL, used only to detect navigation.
        selection
            Selection event, including clicks on the current URL.

        Returns
        -------
        tuple[str | object, list[Div] | Div]
            New input value and the results to show below it.
        """
        if ctx.triggered_id in ("app-location", "sidebar-search-jump"):
            return "", []

        query = (query or "").strip().lower()
        if not query:
            return no_update, []

        matches = [
            entry
            for entry in entries
            if query in entry.get("search_text", entry["label"]).lower()
        ]
        matches.sort(
            key=lambda entry: (
                not entry["label"].lower().startswith(query),
                entry["label"].lower(),
            )
        )

        if not matches:
            return no_update, Div("No matches", className="mlpeg-search-empty")

        return no_update, [
            Div(
                Link(
                    [
                        Span(entry["label"], className="mlpeg-search-label"),
                        Span(entry["kind"], className="mlpeg-search-kind"),
                    ],
                    href=entry["href"],
                    className="mlpeg-search-result",
                ),
                id={"type": "search-result", "href": entry["href"]},
                n_clicks=0,
            )
            for entry in matches[:MAX_RESULTS]
        ]

    # the card named in the URL hash is mounted by the page callback a moment
    # after navigation, so poll for its toggle, expand it and scroll to it
    clientside_callback(
        """
        function(href, clicks) {
            var nu = window.dash_clientside.no_update;
            var ctx = window.dash_clientside.callback_context;
            var trigger = (ctx.triggered || [])[0];
            var selected = trigger && trigger.prop_id.startsWith('{');
            var target = href;
            if (selected) {
                // Ignore result-list mounting/removal, which also changes ALL.
                if (!clicks || !clicks.some(n => n > 0)) return nu;
                var id = JSON.parse(trigger.prop_id.slice(0, -9));
                target = id.href;
                document.body.classList.remove('sidebar-open');
            }

            var token = (window.mlpegSearchJumpToken || 0) + 1;
            window.mlpegSearchJumpToken = token;
            var url = new URL(target, window.location.href);
            var selection = selected ? {token: token} : nu;
            if (!url.hash || url.hash.length < 2) return selection;
            var name;
            try {
                name = decodeURIComponent(url.hash.slice(1));
            } catch (e) {
                return selection;
            }
            var toggleId = JSON.stringify({index: name, type: "bench-toggle"});
            var tries = 0;
            function current() {
                return window.mlpegSearchJumpToken === token;
            }
            function openCard() {
                if (!current()) return;
                var arrived = window.location.pathname === url.pathname
                    && window.location.hash === url.hash;
                var toggle = arrived && document.getElementById(toggleId);
                if (!toggle) {
                    if (tries++ < 100) setTimeout(openCard, 100);
                    return;
                }
                if (toggle.getAttribute("aria-expanded") !== "true") toggle.click();
                var card = toggle.closest(".mlpeg-bench-card") || toggle;
                card.scrollIntoView({behavior: "smooth", block: "start"});
                setTimeout(function() {
                    if (current() && card.isConnected) {
                        card.scrollIntoView({behavior: "smooth", block: "start"});
                    }
                }, 800);
            }
            openCard();
            return selection;
        }
        """,
        Output("sidebar-search-jump", "data"),
        Input("app-location", "href"),
        Input({"type": "search-result", "href": ALL}, "n_clicks"),
    )
