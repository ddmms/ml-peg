"""
Clientside callbacks for the app shell (start-up mask + mobile navigation).

Kept in its own file so ``build_app`` stays short (mirrors
``register_storage_callbacks`` and ``register_settings_callbacks``). Both
callbacks are clientside so they add no server load; they only touch the DOM.
The component ids they reference are created in ``build_full_app``'s layout.
"""

from __future__ import annotations

from dash import Input, Output, clientside_callback

# Start-up mask polling, defined together so the safety timeout can't silently
# drift: the layout's Interval (build_app) ticks every STARTUP_MASK_POLL_MS and
# the clientside guard below gives up after STARTUP_MASK_MAX_POLLS ticks
# (250 ms x 80 = 20 s — generous headroom over the heaviest framework page).
STARTUP_MASK_POLL_MS = 250
STARTUP_MASK_MAX_POLLS = 80


def register_shell_callbacks() -> None:
    """Register the start-up mask and mobile-navigation clientside callbacks."""
    # Hide the start-up mask once the page has rendered, or after a timeout as
    # a safety net, then stop polling. Clientside, so it adds no server load.
    # (The progress bar fills via a CSS animation, not this callback.)
    clientside_callback(
        f"""
        function(n) {{
            var nu = window.dash_clientside.no_update;
            var ready = document.querySelector('#page-content table tbody tr');
            // The fill + number are driven by loading_progress.js, which watches
            // for the display:none below and completes the ring; here we only
            // hide the mask once the first table row exists (or after a timeout).
            if (ready || n > {STARTUP_MASK_MAX_POLLS}) {{
                return [{{'display': 'none'}}, true];
            }}
            return [nu, nu];
        }}
        """,
        Output("startup-mask", "style"),
        Output("startup-mask-poll", "disabled"),
        Input("startup-mask-poll", "n_intervals"),
    )

    # Toggle the off-canvas mobile sidebar by adding/removing ``sidebar-open`` on
    # <body> (all styling + breakpoints live in theme.css). Opens on the
    # hamburger, closes on a scrim tap or on navigation. Clientside: no server load.
    clientside_callback(
        """
        function(nToggle, nScrim, pathname) {
            var ctx = window.dash_clientside.callback_context;
            var trig = (ctx.triggered[0] || {}).prop_id || '';
            var open = trig.indexOf('mobile-nav-toggle') === 0
                && !document.body.classList.contains('sidebar-open');
            document.body.classList.toggle('sidebar-open', open);
            return window.dash_clientside.no_update;
        }
        """,
        Output("mobile-nav-state", "data"),
        Input("mobile-nav-toggle", "n_clicks"),
        Input("sidebar-scrim", "n_clicks"),
        Input("app-location", "pathname"),
        prevent_initial_call=True,
    )
