/*
 * Dash clientside callback for plot export.
 *
 * Python registers this once as plot_download.downloadPlot. The callback reads
 * the rendered Plotly graph in the browser and exports either the trace x/y
 * data as CSV, the current figure as PNG/SVG, or an interactive HTML file.
 */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  plot_download: {
    downloadPlot: function (nClicks, downloadFormat, graphId) {
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

        const rows = [["trace", "point_index", "x", "y"]];
        (plotNode.data || []).forEach((trace, traceIndex) => {
          const xValues = Array.isArray(trace.x)
            ? trace.x
            : trace.x == null
              ? []
              : [trace.x];
          const yValues = Array.isArray(trace.y)
            ? trace.y
            : trace.y == null
              ? []
              : [trace.y];
          const pointCount = Math.max(xValues.length, yValues.length);
          const traceName = trace.name || `trace_${traceIndex}`;
          for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
            rows.push([
              traceName,
              pointIndex,
              pointIndex < xValues.length ? xValues[pointIndex] : "",
              pointIndex < yValues.length ? yValues[pointIndex] : "",
            ]);
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
