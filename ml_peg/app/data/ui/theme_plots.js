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
 * Two mechanisms, for the two moments a figure can be wrong:
 *
 * 1. Creation. dcc.Graph renders every figure through `Plotly.react(gd, fig)`
 *    (dash/dcc/async-graph.js), including graphs mounted later by callbacks
 *    and figure swaps — both of which reset to the saved template. `react` /
 *    `newPlot` are wrapped so the theme is merged into the layout *before*
 *    plotly draws. Theming at creation is the only race-free option: a
 *    post-hoc relayout during dcc's mount transient re-measures the
 *    still-unlaid-out container and locks a zero height into the figure.
 *    plotly.js is loaded lazily by dcc, so the wrap installs through a
 *    `window.Plotly` setter if plotly is not there yet.
 *
 * 2. A live theme flip. The observer below relayouts the already-rendered
 *    (settled, laid-out) graphs when `data-theme` changes.
 */
(function () {
  "use strict";

  // Theme-dependent chart chrome, read from the CSS tokens (which already
  // differ per theme), so light and dark share this single path.
  function chartTheme() {
    var dark = document.documentElement.getAttribute("data-theme") === "dark";
    var style = getComputedStyle(document.documentElement);
    function token(name, fallback) {
      return style.getPropertyValue(name).trim() || fallback;
    }
    return {
      paper: token("--mlpeg-surface", dark ? "#161a21" : "#ffffff"),
      font: token("--mlpeg-ink", dark ? "#e8eaed" : "#222222"),
      grid: token("--mlpeg-border", dark ? "#2a3140" : "#ebebeb"),
    };
  }

  // Axis keys to recolour: any the figure declares, plus the default pair on
  // cartesian figures — figures built without an explicit xaxis/yaxis dict
  // (violins, heatmaps) still draw gridlines. Polar/ternary figures are left
  // without synthesised cartesian axes: those can allocate an unwanted
  // default subplot.
  function axisKeys(layout) {
    var axes = Object.keys(layout).filter(function (k) {
      return /^[xy]axis\d*$/.test(k);
    });
    if (!axes.length && !layout.polar && !layout.ternary) {
      return ["xaxis", "yaxis"];
    }
    if (axes.length) {
      if (axes.indexOf("xaxis") === -1) axes.push("xaxis");
      if (axes.indexOf("yaxis") === -1) axes.push("yaxis");
    }
    return axes;
  }

  // Merge the current theme into a layout object (creation path).
  function themeLayout(layout) {
    var t = chartTheme();
    layout.paper_bgcolor = t.paper;
    layout.plot_bgcolor = t.paper;
    layout.font = layout.font || {};
    layout.font.color = t.font;
    axisKeys(layout).forEach(function (k) {
      var axis = (layout[k] = layout[k] || {});
      axis.gridcolor = t.grid;
      axis.zerolinecolor = t.grid;
      axis.linecolor = t.grid;
    });
    if (layout.polar) {
      layout.polar.bgcolor = t.paper;
      layout.polar.angularaxis = layout.polar.angularaxis || {};
      layout.polar.angularaxis.gridcolor = t.grid;
      layout.polar.radialaxis = layout.polar.radialaxis || {};
      layout.polar.radialaxis.gridcolor = t.grid;
    }
    if (layout.ternary) layout.ternary.bgcolor = t.paper;
    return layout;
  }

  // Wrap Plotly.react/newPlot so every figure is themed as it is drawn. Both
  // call signatures are handled: (gd, {data, layout, ...}) — the dcc form —
  // and (gd, data[], layout, config).
  function wrapPlotly(P) {
    if (!P || P.__mlpegThemeWrap || !P.react) return false;
    P.__mlpegThemeWrap = true;
    ["react", "newPlot"].forEach(function (name) {
      var orig = P[name];
      if (!orig) return;
      P[name] = function (gd, dataOrFig, layout) {
        var args = Array.prototype.slice.call(arguments);
        if (dataOrFig && !Array.isArray(dataOrFig)) {
          dataOrFig.layout = themeLayout(dataOrFig.layout || {});
        } else {
          args[2] = themeLayout(layout || {});
        }
        return orig.apply(this, args);
      };
    });
    return true;
  }

  if (!wrapPlotly(window.Plotly)) {
    // dcc loads plotly.min.js on demand; intercept the global assignment so
    // the wrap is in place before the first figure is drawn.
    var pendingPlotly = window.Plotly;
    Object.defineProperty(window, "Plotly", {
      configurable: true,
      get: function () {
        return pendingPlotly;
      },
      set: function (value) {
        pendingPlotly = value;
        wrapPlotly(value);
      },
    });
  }

  // Relayout one settled graph after a theme flip. Chrome only.
  function themeOne(gd, t) {
    if (!window.Plotly || !gd || !gd.layout) return;
    // Skip graphs with no laid-out geometry (e.g. inside a collapsed card):
    // a relayout would re-measure the hidden container and lock in a zero
    // height. Such graphs re-render through the wrapped react on their next
    // update — dcc resizes and redraws a graph revealed by expanding a card.
    if (!gd.clientHeight) return;
    // Already current (creation-path figures usually are): skip the relayout.
    if (
      gd.layout.paper_bgcolor === t.paper &&
      gd.layout.font &&
      gd.layout.font.color === t.font
    ) {
      return;
    }
    var update = {
      paper_bgcolor: t.paper,
      plot_bgcolor: t.paper,
      "font.color": t.font,
    };
    axisKeys(gd.layout).forEach(function (k) {
      update[k + ".gridcolor"] = t.grid;
      update[k + ".zerolinecolor"] = t.grid;
      update[k + ".linecolor"] = t.grid;
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
      /* a graph mid-teardown must not break the repaint pass */
    }
  }

  function applyPlotTheme() {
    var t = chartTheme();
    var plots = document.querySelectorAll(".js-plotly-plot");
    for (var i = 0; i < plots.length; i++) themeOne(plots[i], t);
  }

  // Debounce a flip arriving in a burst of other attribute changes.
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
    // Repaint the settled graphs when the theme attribute flips.
    new MutationObserver(function (muts) {
      for (var i = 0; i < muts.length; i++) {
        if (muts[i].attributeName === "data-theme") {
          schedule();
          return;
        }
      }
    }).observe(document.documentElement, { attributes: true });

    // Initial pass for any figure drawn before this asset ran.
    schedule();
  }

  if (document.body) {
    start();
  } else {
    document.addEventListener("DOMContentLoaded", start);
  }
})();
