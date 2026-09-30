"""
Inline ``<head>`` scripts injected into the Dash index template.

Kept in its own file so ``run_app`` stays about running the app (mirrors
``settings``, ``shell`` and ``storage``). These run before first paint, which is
the whole point: they read the persisted preferences straight from
``localStorage`` and apply them to ``<html>`` so a reload doesn't flash the
wrong theme, zoom or font before Dash hydrates.

``run_app`` only needs ``analytics_scripts`` (for ``Dash(external_scripts=...)``)
and ``inject_head_scripts`` (called on the constructed app); everything else here
is an implementation detail of those two.

Dash persists a local ``dcc.Store`` under its component id, so the keys here
("theme-store", "zoom-store", "font-store") must match the ids created in
``ml_peg.app.build_app``; the apply callbacks live in ``ml_peg.app.utils.settings``.
"""

from __future__ import annotations

import os

from dash import Dash

from ml_peg.app.utils.settings import ZOOM_MAX, ZOOM_MIN
from ml_peg.app.utils.storage import build_version_check_script

ANALYTICS_ID = os.environ.get("ML_PEG_ANALYTICS_ID")


def build_no_flash_script() -> str:
    """
    Build the pre-paint script that applies the saved theme, zoom and font.

    Returns
    -------
    str
        A ``<script>`` element, ready to be injected ahead of the parsed head.
    """
    return (
        "    <script>(function(){try{"
        # NOTE: "theme-store" must match the Store id in build_app and the
        # clientside apply callback in settings.py.
        'var t=null;var s=window.localStorage.getItem("theme-store");'
        "if(s){t=JSON.parse(s);}"
        # Light is the default for a first visit. Deliberately NOT
        # prefers-color-scheme: the app should look the same to everyone until
        # the theme toggle is used, and theme.css is light at :root with dark
        # opt-in via [data-theme="dark"].
        'if(t!=="dark"&&t!=="light"){t="light";}'
        'document.documentElement.setAttribute("data-theme",t);'
        '}catch(e){document.documentElement.setAttribute("data-theme","light");}'
        # Apply the saved table zoom before paint too (same store pattern),
        # clamped to the slider range (ZOOM_MIN/ZOOM_MAX in settings.py). Sets a
        # CSS var the tables read (see theme.css) — not page zoom. Its own
        # try/catch: a corrupt zoom value must not reset a valid theme above.
        "try{"
        'var z=null;var zs=window.localStorage.getItem("zoom-store");'
        "if(zs){z=JSON.parse(zs);}"
        'if(typeof z==="number"&&z>0){'
        f"z=Math.max({ZOOM_MIN},Math.min({ZOOM_MAX},z));"
        "document.documentElement.style.setProperty("
        '"--mlpeg-table-zoom",String(z/100));}'
        "}catch(e){}"
        # Apply the saved font choice before paint too (same store pattern); its
        # own try/catch so a corrupt value can't disturb the theme/zoom above.
        # theme.css maps data-font="system" to the native stack; else keeps Inter.
        "try{"
        'var f=null;var fs=window.localStorage.getItem("font-store");'
        "if(fs){f=JSON.parse(fs);}"
        'if(f==="system"||f==="inter"){'
        'document.documentElement.setAttribute("data-font",f);}'
        "}catch(e){}"
        "})();</script>"
    )


def build_analytics_head(analytics_id: str) -> str:
    """
    Build the inline gtag initialisation script.

    The async gtag loader itself is passed to ``Dash(external_scripts=...)``;
    this is the companion inline call that configures it.

    Parameters
    ----------
    analytics_id
        Google Analytics measurement ID.

    Returns
    -------
    str
        A ``<script>`` element, ready to be injected ahead of the parsed head.
    """
    return (
        "    <script>"
        "window.dataLayer=window.dataLayer||[];"
        "function gtag(){dataLayer.push(arguments);}"
        "gtag('js', new Date());"
        f"gtag('config', '{analytics_id}');"
        "</script>"
    )


def analytics_scripts() -> list[dict]:
    """
    Build the ``external_scripts`` entries for Dash.

    Loads the async gtag script in ``<head>``, and only when an analytics ID is
    configured. The companion inline init is added by ``inject_head_scripts``.

    Returns
    -------
    list[dict]
        Script tag attributes, empty if no analytics ID is set.
    """
    if not ANALYTICS_ID:
        return []
    return [
        {
            "src": f"https://www.googletagmanager.com/gtag/js?id={ANALYTICS_ID}",
            "async": True,
        }
    ]


def inject_head_scripts(app: Dash) -> None:
    """
    Prepend the inline head scripts to the app's index template.

    Injected ahead of the parsed ``<head>`` so the browser executes them before
    first paint, and before Dash's own assets.

    Parameters
    ----------
    app
        Dash application to modify in place.
    """
    # Version check first: it may clear the very keys the no-flash script reads.
    scripts = [build_version_check_script(), build_no_flash_script()]
    if ANALYTICS_ID:
        scripts.append(build_analytics_head(ANALYTICS_ID))

    # One replace, so the scripts run in the order listed above (repeating the
    # replace inserts each new script *before* the previous one).
    app.index_string = app.index_string.replace(
        "{%metas%}", "{%metas%}\n" + "\n".join(scripts)
    )
