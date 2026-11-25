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
    # Hide the start-up mask once the page has rendered and its model order has
    # settled, or after a timeout as a safety net, then stop polling. Clientside,
    # so it adds no server load. Only random-mode sessions pay the extra polls;
    # alphabetical sessions already match the server-rendered order.
    clientside_callback(
        f"""
        function(n) {{
            var nu = window.dash_clientside.no_update;
            var ready = !!document.querySelector('#page-content table tbody tr');
            var hidden = [{{'display': 'none'}}, true];
            // Rows are server-rendered in alphabetical order, so a session in
            // random mode always has a reorder in flight. Hold the mask until
            // the rendered order stops changing, otherwise the reorder lands
            // after the mask lifts and the table visibly jumps.
            if (ready) {{
                var order = null;
                try {{
                    order = JSON.parse(
                        sessionStorage.getItem('model-order-store')
                    );
                }} catch (err) {{
                    order = null;
                }}
                if (order && order.mode === 'random') {{
                    var signature = Array.prototype.map.call(
                        document.querySelectorAll(
                            '#page-content table tbody tr'
                            + ' td[data-dash-column="MLIP"]'
                        ),
                        function(cell) {{ return cell.textContent; }}
                    ).join('\\u0000');
                    if (window.__mlpegOrderSignature !== signature) {{
                        window.__mlpegOrderSignature = signature;
                        ready = false;
                    }}
                }}
            }}
            // Finish the ring visibly before dismissing the startup overlay.
            if (ready && window.__mlpegLoadingProgress?.finishStartup) {{
                return window.__mlpegLoadingProgress.finishStartup().then(
                    function() {{ return hidden; }}
                );
            }}
            if (ready || n > {STARTUP_MASK_MAX_POLLS}) {{
                return hidden;
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
        function(nToggle, nScrim, href) {
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
        Input("app-location", "href"),
        prevent_initial_call=True,
    )
