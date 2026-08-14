/* Apply the shared plot-axis controls without a server round trip. */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  plot_settings: {
    applyAxes: function (
      applyClicks,
      resetClicks,
      xScale,
      yScale,
      xMin,
      xMax,
      yMin,
      yMax,
      graphId,
    ) {
      const dash = window.dash_clientside;
      const noUpdate = dash.no_update;
      const triggered = dash.callback_context.triggered_id;
      const isReset = triggered && triggered.type === "plot-settings-reset";
      const unchangedControls = [
        noUpdate,
        noUpdate,
        noUpdate,
        noUpdate,
        noUpdate,
        noUpdate,
      ];

      if ((!applyClicks && !resetClicks) || !graphId) {
        return [noUpdate, "", ...unchangedControls];
      }

      const graphContainer = document.getElementById(graphId);
      const plotNode =
        graphContainer &&
        (graphContainer.classList.contains("js-plotly-plot")
          ? graphContainer
          : graphContainer.querySelector(".js-plotly-plot"));
      if (!plotNode || !window.Plotly) {
        return [noUpdate, "Plot is not currently available.", ...unchangedControls];
      }

      if (isReset) {
        window.Plotly.relayout(plotNode, {
          "xaxis.type": "linear",
          "xaxis.range": null,
          "xaxis.autorange": true,
          "yaxis.type": "linear",
          "yaxis.range": null,
          "yaxis.autorange": true,
        });
        return [Date.now(), "", "linear", "linear", null, null, null, null];
      }

      const hasValue = (value) => value !== null && value !== undefined && value !== "";
      const axisUpdate = (axis, scale, minimum, maximum) => {
        const hasMinimum = hasValue(minimum);
        const hasMaximum = hasValue(maximum);
        if (hasMinimum !== hasMaximum) {
          throw new Error(`${axis.toUpperCase()} axis requires both minimum and maximum.`);
        }
        if (hasMinimum && Number(minimum) >= Number(maximum)) {
          throw new Error(`${axis.toUpperCase()} minimum must be less than maximum.`);
        }
        if (scale === "log" && hasMinimum && (Number(minimum) <= 0 || Number(maximum) <= 0)) {
          throw new Error(`${axis.toUpperCase()} log limits must both be positive.`);
        }

        const update = {[`${axis}axis.type`]: scale || "linear"};
        if (!hasMinimum) {
          update[`${axis}axis.range`] = null;
          update[`${axis}axis.autorange`] = true;
          return update;
        }
        update[`${axis}axis.autorange`] = false;
        update[`${axis}axis.range`] =
          scale === "log"
            ? [Math.log10(Number(minimum)), Math.log10(Number(maximum))]
            : [Number(minimum), Number(maximum)];
        return update;
      };

      try {
        const update = Object.assign(
          {},
          axisUpdate("x", xScale, xMin, xMax),
          axisUpdate("y", yScale, yMin, yMax),
        );
        window.Plotly.relayout(plotNode, update);
        return [Date.now(), "", ...unchangedControls];
      } catch (error) {
        return [noUpdate, error.message || String(error), ...unchangedControls];
      }
    },
  },
});
