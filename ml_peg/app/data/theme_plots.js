/* Make every Plotly figure follow the active theme.
 *
 * Figures are built server-side with `template="plotly_white"` (and 71 parity
 * plots are pre-saved with white paper), so on their own they stay light in
 * dark mode. Rather than rebuild them server-side on every theme switch, this
 * repaints only their *chrome* (paper/plot background, font colour, gridlines)
 * from the same `--mlpeg-*` CSS tokens the rest of the app uses — one code path
 * for both themes, no server round-trip. Data-series colours (win/loss blue &
 * amber, the category palette, heatmap cells) are semantic and left untouched.
 *
 * Dash auto-loads any .js under the assets_folder (ml_peg/app/data), after the
 * renderer, so `window.Plotly` and the graphs exist by the time this runs; the
 * observers below also catch figures that mount later from callbacks.
 */
(function () {
  "use strict";

  function token(name, fallback) {
    var v = getComputedStyle(document.documentElement)
      .getPropertyValue(name)
      .trim();
    return v || fallback;
  }

  // Theme-dependent chart chrome, read from the CSS tokens (which already
  // differ per theme), so light and dark share this single path.
  function chartTheme() {
    var dark = document.documentElement.getAttribute("data-theme") === "dark";
    return {
      paper: token("--mlpeg-surface", dark ? "#161a21" : "#ffffff"),
      font: token("--mlpeg-ink", dark ? "#e8eaed" : "#222222"),
      grid: token("--mlpeg-border", dark ? "#2a3140" : "#ebebeb"),
    };
  }

  // Apply the current theme to one Plotly graph div. Chrome only.
  function themeOne(gd, t) {
    if (!window.Plotly || !gd || !gd.layout) return;
    var update = {
      paper_bgcolor: t.paper,
      plot_bgcolor: t.paper,
      "font.color": t.font,
    };
    Object.keys(gd.layout).forEach(function (k) {
      if (/^[xy]axis\d*$/.test(k)) {
        update[k + ".gridcolor"] = t.grid;
        update[k + ".zerolinecolor"] = t.grid;
        update[k + ".linecolor"] = t.grid;
      }
    });
    if (gd.layout.polar) {
      update["polar.bgcolor"] = t.paper;
      update["polar.angularaxis.gridcolor"] = t.grid;
      update["polar.radialaxis.gridcolor"] = t.grid;
    }
    if (gd.layout.ternary) update["ternary.bgcolor"] = t.paper;
    try {
      window.Plotly.relayout(gd, update);
    } catch (e) {
      /* graph not ready yet; an observer will re-fire */
    }
  }

  // `applying` suppresses the childList observer while relayout rewrites the
  // graph's internal SVG, so a repaint can't trigger another repaint.
  var applying = false;
  function applyPlotTheme() {
    applying = true;
    var t = chartTheme();
    var plots = document.querySelectorAll(".js-plotly-plot");
    for (var i = 0; i < plots.length; i++) themeOne(plots[i], t);
    window.setTimeout(function () {
      applying = false;
    }, 0);
  }

  // Debounce bursts (a theme flip plus many graphs mounting together).
  var pending = false;
  function schedule() {
    if (pending) return;
    pending = true;
    window.requestAnimationFrame(function () {
      pending = false;
      applyPlotTheme();
    });
  }

  function start() {
    // 1) Repaint when the theme attribute flips.
    new MutationObserver(function (muts) {
      for (var i = 0; i < muts.length; i++) {
        if (muts[i].attributeName === "data-theme") {
          schedule();
          return;
        }
      }
    }).observe(document.documentElement, { attributes: true });

    // 2) Repaint when a *new* graph is inserted by a callback (e.g. the
    //    lazily-mounted benchmark cards). Relayout's own internal SVG edits never
    //    add a `.js-plotly-plot` node, and `applying` guards the rest.
    new MutationObserver(function (muts) {
      if (applying) return;
      for (var i = 0; i < muts.length; i++) {
        var added = muts[i].addedNodes;
        for (var j = 0; j < added.length; j++) {
          var n = added[j];
          if (
            n.nodeType === 1 &&
            (n.classList.contains("js-plotly-plot") ||
              (n.querySelector && n.querySelector(".js-plotly-plot")))
          ) {
            schedule();
            return;
          }
        }
      }
    }).observe(document.body, { childList: true, subtree: true });

    // 3) Initial pass for server-rendered figures present at load.
    schedule();
  }

  // Dash may run assets before <body> exists; defer wiring until it does.
  if (document.body) {
    start();
  } else {
    document.addEventListener("DOMContentLoaded", start);
  }
})();
