"""
Clientside callbacks for the app shell (start-up mask + mobile navigation).

Kept in its own file so ``build_app`` stays short (mirrors
``register_storage_callbacks`` and ``register_settings_callbacks``). Both
callbacks run locally and inspect browser state without adding server load.
The component ids they reference are created in ``build_full_app``'s layout.
"""

from __future__ import annotations

from dash import Input, Output, clientside_callback

# Allow slow devices to finish hydration, with a bounded escape for failed startup.
STARTUP_MASK_POLL_MS = 250
STARTUP_MASK_MAX_POLLS = 480


def register_shell_callbacks() -> None:
    """Register the start-up mask and mobile-navigation clientside callbacks."""
    # Wait for rendering and callback propagation to settle before enabling clicks.
    # Empty pages also finish. The timeout releases the mask after failed startup.
    clientside_callback(
        f"""
        function(n) {{
            var nu = window.dash_clientside.no_update;
            const page = document.getElementById('page-content');
            const callbacks = window.store?.getState()?.callbacks;
            // Dash 4's renderer queues include clientside work. Ignore this poll
            // itself, plus completed callbacks retained for bookkeeping.
            const idle = callbacks && Object.entries(callbacks).every(
                ([key, queue]) => ['stored', 'completed'].includes(key) ||
                    (Array.isArray(queue) && queue.every(cb =>
                        cb.callback?.output?.includes('startup-mask.style'))));
            const loading = page?.querySelector('[data-dash-is-loading="true"]');
            const rendered = page && page.children.length > 0;
            const now = performance.now();
            if (!rendered || !idle || loading) {{
                window.__mlpegStartupQuietSince = null;
            }} else if (window.__mlpegStartupQuietSince == null) {{
                window.__mlpegStartupQuietSince = now;
            }}
            const ready = window.__mlpegStartupQuietSince != null &&
                now - window.__mlpegStartupQuietSince >= 500;
            var hidden = [{{'display': 'none'}}, true];
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
