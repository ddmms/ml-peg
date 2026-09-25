/* Keep a parity plot's y=x guide inside the visible data.

The guide is written into the figure at analysis time spanning every model, so
on its own it holds the axes at the full range and hiding a model with the
legend changes nothing. Shrinking it to the models still shown lets Plotly
autorange to the current selection. */
(function () {
  "use strict";

  // Matches the padding applied when the guide is first built.
  const PAD_FRACTION = 0.05;
  const PARITY_LINE_NAME = "__parity_line__";

  // Prefer the explicit tag used by the shared parity-plot builders. The
  // geometric fallback keeps older, already-generated figure JSON working.
  function parityLineIndex(traces) {
    const taggedIndex = traces.findIndex(
      (trace) => trace.name === PARITY_LINE_NAME
    );
    if (taggedIndex >= 0) return taggedIndex;

    for (let i = 0; i < traces.length; i++) {
      const trace = traces[i];
      if (trace.mode !== "lines" || !trace.x || !trace.y) continue;
      if (trace.x.length !== 2 || trace.y.length !== 2) continue;
      if (trace.x[0] === trace.y[0] && trace.x[1] === trace.y[1]) return i;
    }
    return -1;
  }

  // Extent of every visible trace, ignoring the guide itself and the click
  // highlight, which stays out of the legend and would otherwise hold the
  // range open at a point belonging to a hidden model.
  function visibleExtent(traces, guideIndex, positiveOnly) {
    let low = Infinity;
    let high = -Infinity;
    traces.forEach((trace, index) => {
      if (index === guideIndex || trace.name === "__clicked_point__") return;
      if (trace.visible === false || trace.visible === "legendonly") return;
      [trace.x, trace.y].forEach((values) => {
        for (let i = 0; i < (values || []).length; i++) {
          const value = values[i];
          if (typeof value !== "number" || !isFinite(value)) continue;
          if (positiveOnly && value <= 0) continue;
          if (value < low) low = value;
          if (value > high) high = value;
        }
      });
    });
    return [low, high];
  }

  function usesLogScale(plotNode) {
    const layout = plotNode.layout || {};
    return (
      (layout.xaxis && layout.xaxis.type === "log") ||
      (layout.yaxis && layout.yaxis.type === "log")
    );
  }

  function paddedEnds(low, high, logarithmic) {
    if (logarithmic) {
      const decades = high > low ? Math.log10(high / low) : 1;
      const factor = Math.pow(10, PAD_FRACTION * decades);
      return [low / factor, high * factor];
    }
    const pad = PAD_FRACTION * (high > low ? high - low : 1);
    return [low - pad, high + pad];
  }

  function rescaleParityLine(plotNode) {
    const traces = plotNode.data;
    if (!traces || !traces.length) return;
    const guideIndex = parityLineIndex(traces);
    if (guideIndex < 0) return;
    const logarithmic = usesLogScale(plotNode);
    const [low, high] = visibleExtent(traces, guideIndex, logarithmic);
    if (!isFinite(low) || !isFinite(high)) return;
    const ends = paddedEnds(low, high, logarithmic);
    const guide = traces[guideIndex];
    if (guide.x[0] === ends[0] && guide.x[1] === ends[1]) return;
    window.Plotly.restyle(plotNode, {x: [ends], y: [ends]}, [guideIndex]);
  }

  // Pages are rebuilt as the user navigates, so plots are bound on first
  // pointer contact rather than up front. plotly_restyle fires after the new
  // visibility is applied, so the extent read below is already current.
  function bind(plotNode) {
    if (plotNode.__mlPegParityBound || typeof plotNode.on !== "function") return;
    plotNode.__mlPegParityBound = true;
    plotNode.on("plotly_restyle", (eventData) => {
      const update = Array.isArray(eventData) ? eventData[0] : eventData;
      if (!update || !Object.prototype.hasOwnProperty.call(update, "visible")) {
        return;
      }
      rescaleParityLine(plotNode);
    });
    plotNode.on("plotly_relayout", (update) => {
      if (!update) return;
      if (!("xaxis.type" in update) && !("yaxis.type" in update)) return;
      rescaleParityLine(plotNode);
    });
  }

  document.addEventListener(
    "pointerover",
    (event) => {
      const target = event.target;
      if (!target || !target.closest) return;
      const plotNode = target.closest(".js-plotly-plot");
      if (plotNode) bind(plotNode);
    },
    true
  );
})();
