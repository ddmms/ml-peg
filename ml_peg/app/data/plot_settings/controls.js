/* Apply shared plot display controls without a server round trip. */
(function () {
  "use strict";

  const hasValue = (value) => value !== null && value !== undefined && value !== "";
  const isReversed = (value) => Array.isArray(value) && value.includes("reverse");

  // Resolve the Plotly node nested inside a Dash Graph container.
  function plotNodeFor(graphId) {
    const container = document.getElementById(graphId);
    if (!container) return null;
    return container.classList.contains("js-plotly-plot")
      ? container
      : container.querySelector(".js-plotly-plot");
  }

  // Validate one axis and translate form values into Plotly layout updates.
  function axisUpdate(axis, settings) {
    const {scale, minimum, maximum, reversed, tickFormat, precision, spacing} = settings;
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

    const numericPrecision = Number(precision);
    if (!Number.isInteger(numericPrecision) || numericPrecision < 0 || numericPrecision > 10) {
      throw new Error(`${axis.toUpperCase()} tick precision must be an integer from 0 to 10.`);
    }
    if (hasValue(spacing) && Number(spacing) <= 0) {
      throw new Error(`${axis.toUpperCase()} tick spacing must be positive.`);
    }

    const update = {
      [`${axis}axis.type`]: scale || "linear",
      [`${axis}axis.tickformat`]:
        tickFormat === "decimal"
          ? `.${numericPrecision}f`
          : tickFormat === "scientific"
            ? `.${numericPrecision}e`
            : null,
      [`${axis}axis.dtick`]: hasValue(spacing) ? Number(spacing) : null,
    };

    if (!hasMinimum) {
      update[`${axis}axis.range`] = null;
      update[`${axis}axis.autorange`] = reversed ? "reversed" : true;
      return update;
    }

    let range =
      scale === "log"
        ? [Math.log10(Number(minimum)), Math.log10(Number(maximum))]
        : [Number(minimum), Number(maximum)];
    if (reversed) range = range.reverse();
    update[`${axis}axis.autorange`] = false;
    update[`${axis}axis.range`] = range;
    return update;
  }

  // Map responsive, preset, or custom sizing onto Plotly dimensions.
  function sizeUpdate(preset, customWidth, customHeight) {
    const presets = {
      square: [700, 700],
      wide: [1000, 600],
    };
    if (!preset || preset === "responsive") {
      return {autosize: true, width: null, height: null};
    }

    let dimensions = presets[preset];
    if (preset === "custom") {
      if (!hasValue(customWidth) || !hasValue(customHeight)) {
        throw new Error("Custom size requires both width and height.");
      }
      dimensions = [Number(customWidth), Number(customHeight)];
      if (dimensions.some((value) => value < 200 || value > 3000)) {
        throw new Error("Custom width and height must be between 200 and 3000 px.");
      }
    }
    if (!dimensions) throw new Error("Unknown figure-size preset.");
    return {autosize: false, width: dimensions[0], height: dimensions[1]};
  }

  // Add or remove the log suffix without duplicating it on repeated applies.
  function axisTitleUpdate(plotNode, axis, scale) {
    const axisName = `${axis}axis`;
    const layoutAxis = (plotNode.layout && plotNode.layout[axisName]) || {};
    const fullLayoutAxis = (plotNode._fullLayout && plotNode._fullLayout[axisName]) || {};
    const currentTitle =
      (layoutAxis.title && layoutAxis.title.text) ||
      (fullLayoutAxis.title && fullLayoutAxis.title.text) ||
      "";
    const baseTitle = String(currentTitle).replace(/ \(log\)$/, "");
    if (!baseTitle) return {};
    return {[`${axisName}.title.text`]: scale === "log" ? `${baseTitle} (log)` : baseTitle};
  }

  function resetLayout(plotNode) {
    return Object.assign({
      autosize: true,
      width: null,
      height: null,
      "xaxis.type": "linear",
      "xaxis.range": null,
      "xaxis.autorange": true,
      "xaxis.tickformat": null,
      "xaxis.dtick": null,
      "yaxis.type": "linear",
      "yaxis.range": null,
      "yaxis.autorange": true,
      "yaxis.tickformat": null,
      "yaxis.dtick": null,
    }, axisTitleUpdate(plotNode, "x", "linear"), axisTitleUpdate(plotNode, "y", "linear"));
  }

  // One pattern-matching callback serves every graph settings menu.
  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    plot_settings: {
      applyAxes: function (
        applyClicks,
        resetClicks,
        xAutoscaleClicks,
        yAutoscaleClicks,
        xScale,
        yScale,
        xMin,
        xMax,
        yMin,
        yMax,
        sizePreset,
        width,
        height,
        xReverse,
        yReverse,
        xTickFormat,
        xTickPrecision,
        xTickSpacing,
        yTickFormat,
        yTickPrecision,
        yTickSpacing,
        graphId,
      ) {
        const dash = window.dash_clientside;
        const noUpdate = dash.no_update;
        const unchangedControls = Array(17).fill(noUpdate);
        const triggered = dash.callback_context.triggered_id;
        const triggerType = triggered && triggered.type;

        if ((!applyClicks && !resetClicks && !xAutoscaleClicks && !yAutoscaleClicks) || !graphId) {
          return [noUpdate, "", ...unchangedControls];
        }

        const plotNode = plotNodeFor(graphId);
        if (!plotNode || !window.Plotly) {
          return [noUpdate, "Plot is not currently available.", ...unchangedControls];
        }

        if (triggerType === "plot-settings-reset") {
          window.Plotly.relayout(plotNode, resetLayout(plotNode));
          return [
            Date.now(),
            "",
            "linear",
            "linear",
            null,
            null,
            null,
            null,
            "responsive",
            null,
            null,
            [],
            [],
            "auto",
            2,
            null,
            "auto",
            2,
            null,
          ];
        }

        if (triggerType === "plot-settings-x-autoscale" || triggerType === "plot-settings-y-autoscale") {
          const axis = triggerType === "plot-settings-x-autoscale" ? "x" : "y";
          const reversed = axis === "x" ? isReversed(xReverse) : isReversed(yReverse);
          window.Plotly.relayout(plotNode, {
            [`${axis}axis.range`]: null,
            [`${axis}axis.autorange`]: reversed ? "reversed" : true,
          });
          const controls = [...unchangedControls];
          if (axis === "x") {
            controls[2] = null;
            controls[3] = null;
          } else {
            controls[4] = null;
            controls[5] = null;
          }
          return [Date.now(), "", ...controls];
        }

        try {
          const update = Object.assign(
            {},
            sizeUpdate(sizePreset, width, height),
            axisUpdate("x", {
              scale: xScale,
              minimum: xMin,
              maximum: xMax,
              reversed: isReversed(xReverse),
              tickFormat: xTickFormat,
              precision: xTickPrecision,
              spacing: xTickSpacing,
            }),
            axisUpdate("y", {
              scale: yScale,
              minimum: yMin,
              maximum: yMax,
              reversed: isReversed(yReverse),
              tickFormat: yTickFormat,
              precision: yTickPrecision,
              spacing: yTickSpacing,
            }),
            axisTitleUpdate(plotNode, "x", xScale),
            axisTitleUpdate(plotNode, "y", yScale),
          );
          window.Plotly.relayout(plotNode, update);
          return [Date.now(), "", ...unchangedControls];
        } catch (error) {
          return [noUpdate, error.message || String(error), ...unchangedControls];
        }
      },
    },
  });
})();
