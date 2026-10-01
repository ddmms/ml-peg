"""Browser tests for the shared plot settings menu.

Driven through IONPI19's parity plot, which the fixture supplies and which the
menu reaches via the usual pattern-matching IDs.
"""

from __future__ import annotations

from typing import Any

from conftest import wait_for_app_ready  # noqa: F401
from playwright.sync_api import Page, expect

TIMEOUT = 60_000

GRAPH_ID = "IONPI19-figure"


def _setting(control: str) -> str:
    """
    Build a CSS selector for one of the menu's pattern-matching controls.

    Parameters
    ----------
    control
        Control name, without the ``plot-settings-`` prefix.

    Returns
    -------
    str
        CSS selector matching that control.
    """
    pattern = f'{{"index":"{GRAPH_ID}","type":"plot-settings-{control}"}}'
    return f"[id='{pattern}']"


def _open_menu(page: Page) -> None:
    """
    Show IONPI19's parity plot and open its plot settings menu.

    Parameters
    ----------
    page
        Loaded, hydrated page.
    """
    page.locator('#sidebar-nav a[href="/category/non-covalent-interactions"]').click()
    expect(page.locator("#IONPI19-table")).to_be_visible(timeout=TIMEOUT)

    # The parity plot is dispatched by clicking a metric cell, not a model name.
    page.locator('#IONPI19-table td.dash-cell[data-dash-column="MAE"]').first.click()
    expect(page.locator(f"#{GRAPH_ID} .js-plotly-plot")).to_be_visible(timeout=TIMEOUT)

    page.locator(_setting("summary")).click()
    expect(page.locator(_setting("apply"))).to_be_visible(timeout=TIMEOUT)
    # Opening the menu also syncs the controls from the plot, and that lands
    # after the panel is visible; typing before it does would be overwritten.
    page.wait_for_timeout(1000)


def _layout(page: Page, path: str) -> Any:
    """
    Read a value out of the live figure's resolved layout.

    Parameters
    ----------
    page
        Page showing the parity plot.
    path
        Dotted path below ``_fullLayout``, e.g. ``"xaxis.type"``.

    Returns
    -------
    Any
        The resolved value, or ``None`` if any step of the path is missing.
    """
    return page.evaluate(
        """(args) => {
          const node = document.querySelector('#' + args.id + ' .js-plotly-plot');
          return args.path.split('.').reduce(
            (value, key) => (value == null ? null : value[key]),
            node && node._fullLayout,
          );
        }""",
        {"id": GRAPH_ID, "path": path},
    )


def _choose(page: Page, control: str, option: str) -> None:
    """
    Pick an option from one of the menu's dropdowns.

    Parameters
    ----------
    page
        Page with the menu open.
    control
        Dropdown name, without the ``plot-settings-`` prefix.
    option
        Visible label of the option to select.
    """
    page.locator(_setting(control)).click()
    # Options are portalled out of the component, so match on the open menu.
    page.locator(".dash-dropdown-content").get_by_text(option, exact=True).click()


def _fill(page: Page, control: str, value: str) -> None:
    """
    Type into one of the menu's number inputs and commit it.

    Parameters
    ----------
    page
        Page with the menu open.
    control
        Input name, without the ``plot-settings-`` prefix.
    value
        Value to enter.
    """
    # The inputs are debounced, so the value only reaches Dash once focus goes.
    page.locator(_setting(control)).fill(value)
    page.locator(_setting(control)).blur()
    page.wait_for_timeout(300)


def _apply(page: Page) -> None:
    """
    Click Apply and let the clientside callback settle.

    Parameters
    ----------
    page
        Page with the menu open.
    """
    page.locator(_setting("apply")).click()
    page.wait_for_timeout(500)


def test_menu_syncs_with_plot(ready_page: Page) -> None:
    """Opening the menu reads the live figure rather than showing defaults."""
    page = ready_page
    _open_menu(page)

    # The parity plot is drawn on linear axes, so that is what must be shown.
    assert _layout(page, "xaxis.type") == "linear"
    expect(page.locator(f"{_setting('x-scale')} .dash-dropdown-value")).to_have_text(
        "Linear"
    )
    expect(page.locator(f"{_setting('y-scale')} .dash-dropdown-value")).to_have_text(
        "Linear"
    )


def test_log_scale_applies_and_resets(ready_page: Page) -> None:
    """Apply switches an axis to log and labels it; Reset all puts it back."""
    page = ready_page
    _open_menu(page)
    original_title = _layout(page, "xaxis.title.text")

    _choose(page, "x-scale", "Log")
    _apply(page)
    assert _layout(page, "xaxis.type") == "log"
    assert _layout(page, "xaxis.title.text") == f"{original_title} (log)"

    page.locator(_setting("reset")).click()
    page.wait_for_timeout(500)
    assert _layout(page, "xaxis.type") == "linear"
    assert _layout(page, "xaxis.title.text") == original_title


def test_explicit_limits_then_autoscale(ready_page: Page) -> None:
    """Both limits pin the range; Autoscale clears the inputs and the range."""
    page = ready_page
    _open_menu(page)

    _fill(page, "x-min", "-5")
    _fill(page, "x-max", "5")
    _apply(page)
    assert _layout(page, "xaxis.range") == [-5, 5]

    page.locator(_setting("x-autoscale")).click()
    page.wait_for_timeout(500)
    expect(page.locator(_setting("x-min"))).to_have_value("")
    expect(page.locator(_setting("x-max"))).to_have_value("")
    assert _layout(page, "xaxis.range") != [-5, 5]


def test_one_limit_is_rejected(ready_page: Page) -> None:
    """A minimum without a maximum reports the problem and changes nothing."""
    page = ready_page
    _open_menu(page)
    before = _layout(page, "xaxis.range")

    _fill(page, "x-min", "-5")
    _apply(page)

    expect(page.locator(_setting("message"))).not_to_be_empty()
    assert _layout(page, "xaxis.range") == before


def test_size_preset_applies(ready_page: Page) -> None:
    """Choosing a preset sizes the figure to that preset's dimensions."""
    page = ready_page
    _open_menu(page)

    _choose(page, "size-preset", "Square (700 × 700)")
    _apply(page)

    assert _layout(page, "width") == 700
    assert _layout(page, "height") == 700
