window.dash_clientside = Object.assign({}, window.dash_clientside, {
  table_download: {
    captureTable: function (request) {
      const dash = window.dash_clientside;
      const noUpdate = dash ? dash.no_update : null;
      if (!request) {
        return noUpdate;
      }

      const tableNode = document.getElementById(request.element_id);
      if (!tableNode) {
        return noUpdate;
      }

      const source =
        "https://cdn.jsdelivr.net/npm/html-to-image@1.11.11/dist/html-to-image.min.js";
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
          if (fmt === "svg") {
            return htmlToImage.toSvg(tableNode, options);
          }
          return htmlToImage.toPng(tableNode, options);
        })
        .then((dataUrl) => {
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
