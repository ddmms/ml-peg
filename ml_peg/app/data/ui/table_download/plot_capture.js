/*
 * Dash clientside callback for plot export.
 *
 * Python registers this once as plot_download.downloadPlot. The callback reads
 * the rendered Plotly graph in the browser and exports either the trace data
 * as CSV, the current figure as PNG/SVG, or an interactive HTML file.
 */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  plot_download: {
    downloadPlot: function (nClicks, downloadFormat, graphId) {
      // Kept local: asset scripts share one global scope, so a top-level const
      // would clash with any other asset declaring the same name. Keys match
      // PARITY_LINE_NAME in plot_helpers.py and the scatter click-ring trace.
      const TRACE_LABELS = {
        __parity_line__: "y=x",
        __clicked_point__: "selected point",
      };
      const dash = window.dash_clientside;
      const noUpdate = dash ? dash.no_update : null;
      if (!nClicks || !graphId) {
        return noUpdate;
      }

      const graphContainer = document.getElementById(graphId);
      if (!graphContainer) {
        return noUpdate;
      }

      // dcc.Graph owns an outer container; Plotly draws the actual figure in the
      // inner .js-plotly-plot element.
      const plotNode = graphContainer.classList.contains("js-plotly-plot")
        ? graphContainer
        : graphContainer.querySelector(".js-plotly-plot");
      if (!plotNode) {
        return noUpdate;
      }

      const format = (downloadFormat || "csv").toLowerCase();
      const filenameBase = String(graphId).replace(/[\s_]+/g, "-");

      const jsonScript = (value) =>
        JSON.stringify(value || null).replace(/<\/script/gi, "<\\/script");
      const escapeHtml = (value) =>
        String(value)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;");

      if (format === "csv") {
        const csvValue = (value) => {
          if (value === undefined || value === null) {
            return "";
          }
          const text =
            typeof value === "object" ? JSON.stringify(value) : String(value);
          if (/[",\n\r]/.test(text)) {
            return `"${text.replace(/"/g, '""')}"`;
          }
          return text;
        };

        const isArray = (value) =>
          Array.isArray(value) || ArrayBuffer.isView(value);
        // Plotly keeps encoded NumPy inputs in .data and exposes their decoded
        // arrays on the corresponding full trace after rendering.
        const traces = (plotNode.data || []).map((trace, index) => {
          const decoded = { ...trace };
          for (const field of ["x", "y", "text", "hovertext", "customdata"]) {
            if (trace[field] && trace[field].bdata !== undefined) {
              decoded[field] = plotNode._fullData?.[index]?.[field] ?? trace[field];
            }
          }
          return decoded;
        });
        const asArray = (value) =>
          isArray(value) ? value : value == null ? [] : [value];
        // Scalar text applies to every point, while arrays supply per-point values.
        const pointValue = (value, index) =>
          isArray(value) ? value[index] : value;
        const textFields = ["text", "hovertext"].filter((field) =>
          traces.some((trace) => trace[field] != null),
        );
        const headers = ["trace", "point_index", "x", "y", ...textFields];
        const reserved = new Set(headers);
        // A custom-data position may mean different things in different traces.
        // Match columns by their hover label, not just by the array index.
        const traceColumns = traces.map((trace) => {
          const labels = new Map();
          const templates = asArray(trace.hovertemplate);
          templates.forEach((template) => {
            String(template || "").split(/<br\s*\/?\s*>|\n/i).forEach((line) => {
              // Extract labels like "Energy" from "<b>Energy:</b> %{customdata[0]:.3f}".
              // Ignore format specifiers so CSV values retain their full precision.
              const match = line.match(
                /^([^%]*?)%\{customdata(?:\[(\d+)\])?(?::[^}]*|\|[^}]*)?\}/,
              );
              if (!match) return;
              const label = match[1].replace(/<[^>]*>/g, "")
                .trim().replace(/[:=]\s*$/, "").trim();
              const index = Number(match[2] || 0);
              if (label) {
                if (!labels.has(index)) labels.set(index, new Set());
                labels.get(index).add(label);
              }
            });
          });
          let width = 0;
          asArray(trace.customdata).forEach((value) => {
            width = Math.max(width, isArray(value) ? value.length : 1);
          });
          const used = new Set(reserved);
          return Array.from({ length: width }, (_, index) => {
            const candidates = labels.get(index);
            // Per-point templates can disagree on a label. Fall back if ambiguous.
            const label = candidates?.size === 1
              ? [...candidates][0] : `customdata[${index}]`;
            let header = label;
            let suffix = 2;
            // Disambiguate duplicate labels and names reserved for base columns.
            while (used.has(header)) header = `${label} (${suffix++})`;
            used.add(header);
            if (!headers.includes(header)) headers.push(header);
            return header;
          });
        });
        const rows = [headers];
        traces.forEach((trace, traceIndex) => {
          const xValues = asArray(trace.x);
          const yValues = asArray(trace.y);
          const pointCount = Math.max(xValues.length, yValues.length);
          // Internal tags identify helper traces in the browser; export them
          // under a readable label rather than leaking the sentinel.
          const traceName =
            TRACE_LABELS[trace.name] || trace.name || `trace_${traceIndex}`;
          for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
            const values = new Map([
              ["trace", traceName],
              ["point_index", pointIndex],
              ["x", xValues[pointIndex]],
              ["y", yValues[pointIndex]],
              ...textFields.map((field) => [field, pointValue(trace[field], pointIndex)]),
            ]);
            const custom = pointValue(trace.customdata, pointIndex);
            traceColumns[traceIndex].forEach((header, index) => {
              values.set(header, isArray(custom) ? custom[index] : index === 0 ? custom : "");
            });
            // Use the shared column order, leaving absent fields blank via csvValue.
            rows.push(headers.map((header) => values.get(header)));
          }
        });

        return {
          content: rows.map((row) => row.map(csvValue).join(",")).join("\n"),
          filename: `${filenameBase}.csv`,
          type: "text/csv",
        };
      }

      if (format === "html") {
        const plotlyVersion = window.Plotly && window.Plotly.version;
        const plotlySource = plotlyVersion
          ? `https://cdn.plot.ly/plotly-${plotlyVersion}.min.js`
          : "https://cdn.plot.ly/plotly-latest.min.js";
        const title =
          (plotNode.layout && plotNode.layout.title && plotNode.layout.title.text) ||
          filenameBase;

        return {
          content: `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>${escapeHtml(title)}</title>
  <script src="${plotlySource}"></script>
  <style>
    html, body { height: 100%; margin: 0; }
    #plot { width: 100%; height: 100%; }
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const data = ${jsonScript(plotNode.data || [])};
    const layout = ${jsonScript(plotNode.layout || {})};
    const config = {responsive: true};
    Plotly.newPlot("plot", data, layout, config);
  </script>
</body>
</html>`,
          filename: `${filenameBase}.html`,
          type: "text/html",
        };
      }

      if (
        !["png", "svg"].includes(format) ||
        !window.Plotly ||
        !window.Plotly.toImage
      ) {
        return noUpdate;
      }

      const fullLayout = plotNode._fullLayout || {};
      const width = fullLayout.width || plotNode.clientWidth || 800;
      const height = fullLayout.height || plotNode.clientHeight || 600;

      return window.Plotly.toImage(plotNode, {
        format: format,
        width: width,
        height: height,
      })
        .then((dataUrl) => {
          const parts = String(dataUrl || "").split(",");
          if (parts.length < 2) {
            return noUpdate;
          }

          const payload = parts.slice(1).join(",");
          if (format === "png") {
            return {
              base64: true,
              content: payload,
              filename: `${filenameBase}.png`,
              type: "image/png",
            };
          }

          return {
            content: decodeURIComponent(payload),
            filename: `${filenameBase}.svg`,
            type: "image/svg+xml",
          };
        })
        .catch((error) => {
          console.error("Plot export failed", error);
          return noUpdate;
        });
    },
  },
});
