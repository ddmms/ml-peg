/*
 * Dash clientside callback for PNG/SVG table export.
 *
 * Python registers this as table_download.captureTable. The callback receives a
 * request from register_download_callbacks, captures the actual Dash table already
 * drawn in the browser, and returns a dcc.Download-compatible payload.
 *
 * This intentionally captures the browser's drawn table instead of rebuilding an
 * image from table data. Rebuiling the table is much much easier, but results in an
 * image that doenst actually look anything like our ml-peg tables.
 * Dash has already applied style_data_conditional,
 * tooltip/header changes, warning colours, column widths, and CSS assets, so
 * html-to-image can export the same visual table the user sees.
 */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  table_download: {
    captureTable: function (request) {
      const dash = window.dash_clientside;
      const noUpdate = dash ? dash.no_update : null;
      if (!request) {
        return noUpdate;
      }

      // register_download_callbacks passes the Dash DataTable component id.
      // Capturing this existing page element preserves the current appearance,
      // including conditional cell colours and any user-adjusted table state.
      const tableNode = document.getElementById(request.element_id);
      if (!tableNode) {
        return noUpdate;
      }

      const source =
        "https://cdn.jsdelivr.net/npm/html-to-image@1.11.11/dist/html-to-image.min.js";

      // Load html-to-image only when the user asks for an image export. Cache the
      // promise so repeated downloads do not append duplicate script tags.
      const ensureLib = () => {
        if (window.htmlToImage) {
          return Promise.resolve(window.htmlToImage);
        }
        if (window._mlpegHtmlToImagePromise) {
          return window._mlpegHtmlToImagePromise;
        }
        window._mlpegHtmlToImagePromise = new Promise((resolve, reject) => {
          const script = document.createElement("script");
          script.src = source;
          script.async = true;
          script.onload = () => resolve(window.htmlToImage);
          script.onerror = () => reject(new Error("Failed to load html-to-image"));
          document.head.appendChild(script);
        });
        return window._mlpegHtmlToImagePromise;
      };

      const fmt = (request.format || "png").toLowerCase();
      const filename = request.filename || `table.${fmt}`;
      const basePixelRatio = window.devicePixelRatio || 1;

      // A higher PNG pixel ratio keeps text and colour blocks crisp in the export.
      // SVG is vector output, so the browser pixel ratio is enough there.
      const options = {
        cacheBust: true,
        pixelRatio: fmt === "png" ? Math.max(3, basePixelRatio * 1.5) : basePixelRatio,
        backgroundColor: "#ffffff",
      };

      return ensureLib()
        .then((htmlToImage) => {
          if (!htmlToImage) {
            throw new Error("html-to-image unavailable");
          }
          // html-to-image reads the table already drawn by the browser, including
          // computed styles, so the export matches the live Dash table instead of
          // just the raw table values.
          if (fmt === "svg") {
            return htmlToImage.toSvg(tableNode, options);
          }
          return htmlToImage.toPng(tableNode, options);
        })
        .then((dataUrl) => {
          // html-to-image returns a data URL. Dash downloads need the payload split
          // into content plus metadata.
          const parts = String(dataUrl || "").split(",");
          if (parts.length < 2) {
            return noUpdate;
          }

          if (fmt === "svg") {
            const content = decodeURIComponent(parts.slice(1).join(","));
            return {
              content: content,
              filename: filename,
              type: "image/svg+xml",
            };
          }

          // PNG content remains base64 encoded so dcc.Download can write it as bytes.
          return {
            base64: true,
            content: parts.slice(1).join(","),
            filename: filename,
            type: "image/png",
          };
        })
        .catch((error) => {
          console.error("Table export failed", error);
          return noUpdate;
        });
    },
  },
});
