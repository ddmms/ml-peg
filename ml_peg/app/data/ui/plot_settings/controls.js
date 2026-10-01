/* Apply shared plot display controls without a server round trip. */
(function () {
  "use strict";

  const hasValue = (value) => value !== null && value !== undefined && value !== "";
  const isReversed = (value) => Array.isArray(value) && value.includes("reverse");
  const copy = (value) => (Array.isArray(value) ? value.slice() : value);

  // Layout keys the menu writes, and therefore has to be able to put back.
  const MANAGED_AXIS_KEYS = ["type", "range", "autorange", "tickformat", "dtick"];
  const SIZE_PRESETS = {square: [700, 700], wide: [1000, 600]};

  // Resolve the Plotly node nested inside a Dash Graph container.
  function plotNodeFor(graphId) {
    const container = document.getElementById(graphId);
    if (!container) return null;
    return container.classList.contains("js-plotly-plot")
      ? container
      : container.querySelector(".js-plotly-plot");
  }

  // Subplot grids number their axes (xaxis, xaxis2, ...) and are driven
  // together; overlaid axes of differing scale would need per-axis controls.
  function axisNames(plotNode, axis) {
    const full = plotNode._fullLayout || {};
    const pattern = new RegExp(`^${axis}axis\\d*$`);
    const names = Object.keys(full).filter((key) => pattern.test(key));
    return names.length ? names.sort() : [`${axis}axis`];
  }

  /* One Graph id hosts a succession of figures, so a snapshot has to be
  dropped when the plot under it is replaced. Series endpoints are sampled as
  well as structure: two models share trace names, types and lengths. Guides
  are two points long and so left unsampled, since parity_line_autorange.js
  rewrites them in place. */
  function figureKey(plotNode) {
    const sample = (values) =>
      values && values.length > 2 ? `${values[0]},${values[values.length - 1]}` : "";
    return (plotNode.data || [])
      .filter((trace) => trace.name !== "__clicked_point__")
      .map((trace) => {
        const length = (trace.x || trace.y || []).length;
        const name = trace.name || "";
        const type = trace.type || "";
        return `${name}:${type}:${length}:${sample(trace.x)}/${sample(trace.y)}`;
      })
      .join("|");
  }

  // Figures author their own log scales, ranges and sizes, so "linear,
  // autoranged, responsive" is not the state to return to. Drives "Reset all".
  function snapshot(plotNode) {
    if (!plotNode._fullLayout) return null;
    const key = figureKey(plotNode);
    const cached = plotNode.__mlPegPlotSettings;
    if (cached && cached.key === key) return cached;
    // A new figure comes with its own size; stop holding the old one's.
    plotNode.__mlPegSize = null;

    const layout = plotNode.layout || {};
    const axes = {};
    ["x", "y"].forEach((axis) => {
      axisNames(plotNode, axis).forEach((name) => {
        const source = layout[name] || {};
        const entry = {title: (source.title && source.title.text) || ""};
        MANAGED_AXIS_KEYS.forEach((key) => {
          entry[key] = key in source ? copy(source[key]) : null;
        });
        // An authored range with no autorange key means a pinned axis.
        if (entry.autorange === null && Array.isArray(entry.range)) {
          entry.autorange = false;
        }
        axes[name] = entry;
      });
    });

    // Plotly only autosizes when no explicit size was given.
    const width = hasValue(layout.width) ? layout.width : null;
    const height = hasValue(layout.height) ? layout.height : null;
    const snap = {
      key,
      axes,
      autosize:
        "autosize" in layout ? layout.autosize : !(width !== null && height !== null),
      width,
      height,
    };
    plotNode.__mlPegPlotSettings = snap;
    return snap;
  }

  // Translate one axis container into the menu's control values.
  function controlsForAxis(axisLayout) {
    const source = axisLayout || {};
    const logarithmic = source.type === "log";
    const toData = (value) => (logarithmic ? Math.pow(10, value) : value);

    let minimum = null;
    let maximum = null;
    let reversed = source.autorange === "reversed";
    if (!source.autorange && Array.isArray(source.range)) {
      const ends = source.range.map(Number);
      reversed = ends[0] > ends[1];
      const low = reversed ? ends[1] : ends[0];
      const high = reversed ? ends[0] : ends[1];
      minimum = toData(low);
      maximum = toData(high);
    }

    let tickFormat = "auto";
    let precision = 2;
    const match = /^\.(\d+)([fe])$/.exec(source.tickformat || "");
    if (match) {
      tickFormat = match[2] === "f" ? "decimal" : "scientific";
      precision = Number(match[1]);
    }

    return {
      scale: logarithmic ? "log" : "linear",
      minimum,
      maximum,
      reversed: reversed ? ["reverse"] : [],
      tickFormat,
      precision,
      spacing: typeof source.dtick === "number" ? source.dtick : null,
    };
  }

  function sizeControls(width, height, autosize) {
    if (autosize !== false || !hasValue(width) || !hasValue(height)) {
      return ["responsive", null, null];
    }
    const preset = Object.keys(SIZE_PRESETS).find(
      (key) => SIZE_PRESETS[key][0] === width && SIZE_PRESETS[key][1] === height
    );
    return preset ? [preset, null, null] : ["custom", width, height];
  }

  // Assemble the 17 control outputs in the order the Python callback declares.
  function controlValues(xAxis, yAxis, width, height, autosize) {
    const x = controlsForAxis(xAxis);
    const y = controlsForAxis(yAxis);
    const size = sizeControls(width, height, autosize);
    return [
      x.scale, y.scale,
      x.minimum, x.maximum,
      y.minimum, y.maximum,
      size[0], size[1], size[2],
      x.reversed, y.reversed,
      x.tickFormat, x.precision, x.spacing,
      y.tickFormat, y.precision, y.spacing,
    ];
  }

  // What the menu should show for the plot as it is rendered right now.
  function controlsFromPlot(plotNode) {
    const full = plotNode._fullLayout || {};
    return controlValues(
      full[axisNames(plotNode, "x")[0]],
      full[axisNames(plotNode, "y")[0]],
      full.width,
      full.height,
      full.autosize
    );
  }

  // What the menu should show once the authored layout has been restored.
  function controlsFromSnapshot(plotNode, snap) {
    return controlValues(
      snap.axes[axisNames(plotNode, "x")[0]],
      snap.axes[axisNames(plotNode, "y")[0]],
      snap.width,
      snap.height,
      snap.autosize
    );
  }

  // Validate one axis and translate form values into Plotly layout updates.
  function axisUpdate(plotNode, axis, settings) {
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

    let range = null;
    if (hasMinimum) {
      range =
        scale === "log"
          ? [Math.log10(Number(minimum)), Math.log10(Number(maximum))]
          : [Number(minimum), Number(maximum)];
      if (reversed) range.reverse();
    }

    const update = {};
    axisNames(plotNode, axis).forEach((name) => {
      update[`${name}.type`] = scale || "linear";
      update[`${name}.tickformat`] =
        tickFormat === "decimal"
          ? `.${numericPrecision}f`
          : tickFormat === "scientific"
            ? `.${numericPrecision}e`
            : null;
      update[`${name}.dtick`] = hasValue(spacing) ? Number(spacing) : null;
      if (range) {
        update[`${name}.autorange`] = false;
        update[`${name}.range`] = range.slice();
      } else {
        update[`${name}.range`] = null;
        update[`${name}.autorange`] = reversed ? "reversed" : true;
      }
      Object.assign(update, titleUpdate(plotNode, name, scale));
    });
    return update;
  }

  // Map responsive, preset, or custom sizing onto Plotly dimensions.
  function sizeUpdate(preset, customWidth, customHeight) {
    if (!preset || preset === "responsive") {
      return {autosize: true, width: null, height: null};
    }

    let dimensions = SIZE_PRESETS[preset];
    if (preset === "custom") {
      if (!hasValue(customWidth) || !hasValue(customHeight)) {
        throw new Error("Custom size requires both width and height.");
      }
      dimensions = [Number(customWidth), Number(customHeight)];
      if (dimensions.some((value) => !(value >= 200 && value <= 3000))) {
        throw new Error("Custom width and height must be between 200 and 3000 px.");
      }
    }
    if (!dimensions) throw new Error("Unknown figure-size preset.");
    return {autosize: false, width: dimensions[0], height: dimensions[1]};
  }

  /* Dash copies relayout changes back into the figure prop, but skips
  autosize, width and height. The next Plotly.react therefore redraws at the
  authored size and silently undoes the chosen one, so re-apply it whenever
  that happens. */
  function enforceSize(plotNode, size) {
    // Kept as its own object, since callers go on to extend what they passed.
    plotNode.__mlPegSize = {
      autosize: size.autosize,
      width: size.width,
      height: size.height,
    };
    if (!plotNode.__mlPegSizeGuard) {
      plotNode.__mlPegSizeGuard = true;
      plotNode.on("plotly_afterplot", () => {
        const wanted = plotNode.__mlPegSize;
        if (!wanted) return;
        // Clearing a dimension drops the key rather than nulling it, so an
        // exact comparison here would never settle and would loop forever.
        const layout = plotNode.layout || {};
        const matches = ["autosize", "width", "height"].every(
          (key) => (hasValue(layout[key]) ? layout[key] : null) === wanted[key]
        );
        if (matches) return;
        window.Plotly.relayout(plotNode, wanted);
      });
    }
    return size;
  }

  // Suffix the authored title, not the rendered one, so repeated applies
  // cannot stack suffixes or eat a title that really ends in "(log)".
  function titleUpdate(plotNode, axisName, scale) {
    const snap = snapshot(plotNode);
    const base = snap && snap.axes[axisName] ? snap.axes[axisName].title : "";
    if (!base) return {};
    return {[`${axisName}.title.text`]: scale === "log" ? `${base} (log)` : base};
  }

  // Put the figure back the way its author wrote it.
  function resetLayout(plotNode, snap) {
    const update = enforceSize(plotNode, {
      autosize: snap.autosize,
      width: snap.width,
      height: snap.height,
    });
    Object.keys(snap.axes).forEach((name) => {
      const entry = snap.axes[name];
      MANAGED_AXIS_KEYS.forEach((key) => {
        update[`${name}.${key}`] = copy(entry[key]);
      });
      if (entry.title) update[`${name}.title.text`] = entry.title;
    });
    return update;
  }

  // Capture on first hover, before a pan or zoom can edit gd.layout. Pages are
  // rebuilt as the user navigates, so this cannot be done once up front.
  document.addEventListener(
    "pointerover",
    (event) => {
      const target = event.target;
      if (!target || !target.closest) return;
      const plotNode = target.closest(".js-plotly-plot");
      if (plotNode) snapshot(plotNode);
    },
    true
  );

  // One pattern-matching callback serves every graph settings menu.
  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    plot_settings: {
      applyAxes: function (
        applyClicks,
        resetClicks,
        xAutoscaleClicks,
        yAutoscaleClicks,
        summaryClicks,
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

        const clicked =
          applyClicks || resetClicks || xAutoscaleClicks || yAutoscaleClicks || summaryClicks;
        if (!clicked || !graphId) {
          return [noUpdate, "", ...unchangedControls];
        }

        const plotNode = plotNodeFor(graphId);
        if (!plotNode || !window.Plotly) {
          return [noUpdate, "Plot is not currently available.", ...unchangedControls];
        }
        const snap = snapshot(plotNode);

        // Show the plot's current state, so applying cannot flatten a log axis.
        if (triggerType === "plot-settings-summary") {
          return [noUpdate, "", ...controlsFromPlot(plotNode)];
        }

        if (triggerType === "plot-settings-reset") {
          if (!snap) {
            return [noUpdate, "Plot is not currently available.", ...unchangedControls];
          }
          window.Plotly.relayout(plotNode, resetLayout(plotNode, snap));
          return [Date.now(), "", ...controlsFromSnapshot(plotNode, snap)];
        }

        if (triggerType === "plot-settings-x-autoscale" || triggerType === "plot-settings-y-autoscale") {
          const axis = triggerType === "plot-settings-x-autoscale" ? "x" : "y";
          const reversed = axis === "x" ? isReversed(xReverse) : isReversed(yReverse);
          const update = {};
          axisNames(plotNode, axis).forEach((name) => {
            update[`${name}.range`] = null;
            update[`${name}.autorange`] = reversed ? "reversed" : true;
          });
          window.Plotly.relayout(plotNode, update);
          const controls = [...unchangedControls];
          controls[axis === "x" ? 2 : 4] = null;
          controls[axis === "x" ? 3 : 5] = null;
          return [Date.now(), "", ...controls];
        }

        try {
          const update = Object.assign(
            {},
            enforceSize(plotNode, sizeUpdate(sizePreset, width, height)),
            axisUpdate(plotNode, "x", {
              scale: xScale,
              minimum: xMin,
              maximum: xMax,
              reversed: isReversed(xReverse),
              tickFormat: xTickFormat,
              precision: xTickPrecision,
              spacing: xTickSpacing,
            }),
            axisUpdate(plotNode, "y", {
              scale: yScale,
              minimum: yMin,
              maximum: yMax,
              reversed: isReversed(yReverse),
              tickFormat: yTickFormat,
              precision: yTickPrecision,
              spacing: yTickSpacing,
            }),
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
