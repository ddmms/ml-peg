"""Client-side helpers for download controls."""

from __future__ import annotations

from dash import html

DOWNLOAD_HELPER_SCRIPT = """
(function () {
  const HTML_TO_IMAGE_SRC =
    "https://cdn.jsdelivr.net/npm/html-to-image@1.11.11/dist/html-to-image.min.js";
  let htmlToImagePromise = null;

  function getNoUpdate() {
    if (window.dash_clientside && "no_update" in window.dash_clientside) {
      return window.dash_clientside.no_update;
    }
    return null;
  }

  function ensureHtmlToImage() {
    if (window.htmlToImage) {
      return Promise.resolve(window.htmlToImage);
    }
    if (!htmlToImagePromise) {
      htmlToImagePromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = HTML_TO_IMAGE_SRC;
        script.async = true;
        script.onload = () => resolve(window.htmlToImage);
        script.onerror = () =>
          reject(new Error("Failed to load html-to-image library."));
        document.head.appendChild(script);
      }).catch((error) => {
        console.error(error);
        htmlToImagePromise = null;
        throw error;
      });
    }
    return htmlToImagePromise;
  }

  function encodeTextToBase64(text) {
    if (window.TextEncoder) {
      const utf8 = new TextEncoder().encode(text);
      let binary = "";
      utf8.forEach((byte) => {
        binary += String.fromCharCode(byte);
      });
      return window.btoa(binary);
    }
    return window.btoa(unescape(encodeURIComponent(text)));
  }

  function extractBase64(dataUrl) {
    if (!dataUrl || typeof dataUrl !== "string") {
      return null;
    }
    const [meta, ...rest] = dataUrl.split(",");
    if (!meta || rest.length === 0) {
      return null;
    }
    const payload = rest.join(",");
    if (/;base64/i.test(meta)) {
      return payload;
    }
    try {
      const decoded = decodeURIComponent(payload);
      return encodeTextToBase64(decoded);
    } catch (error) {
      console.error("Failed to convert SVG data URL to base64.", error);
      return null;
    }
  }

  function resolveTableNode(request) {
    if (request.element_id) {
      return document.getElementById(request.element_id);
    }
    if (request.selector) {
      return document.querySelector(request.selector);
    }
    return null;
  }

  function captureTable(request) {
    const noUpdate = getNoUpdate();
    if (!request) {
      return noUpdate;
    }
    const tableNode = resolveTableNode(request);
    if (!tableNode) {
      console.warn("Unable to find table element for download request.", request);
      return noUpdate;
    }

    const format = (request.format || \"png\").toLowerCase();
    const filename = request.filename || `table.${format}`;
    const pixelRatio = request.pixel_ratio || window.devicePixelRatio || 2;
    const options = {
      cacheBust: true,
      pixelRatio,
      backgroundColor: request.background || "#ffffff",
    };

    return ensureHtmlToImage()
      .then((htmlToImage) => {
        if (!htmlToImage) {
          throw new Error("html-to-image library failed to load.");
        }
        if (format === "svg") {
          return htmlToImage.toSvg(tableNode, options);
        }
        return htmlToImage.toPng(tableNode, options);
      })
      .then((dataUrl) => {
        const base64 = extractBase64(dataUrl);
        if (!base64) {
          throw new Error("Invalid data URL returned from html-to-image.");
        }
        const mime = format === "svg" ? "image/svg+xml" : "image/png";
        return {
          base64: true,
          content: base64,
          filename,
          type: mime,
        };
      })
      .catch((error) => {
        console.error("Failed to generate table download:", error);
        return noUpdate;
      });
  }

  window.dash_clientside = window.dash_clientside || {};
  window.dash_clientside.download_helpers =
    window.dash_clientside.download_helpers || {};
  window.dash_clientside.download_helpers.captureTable = captureTable;
})();
"""


def build_download_helper_script():
    """
    Build the inline script element that registers the download helper.

    Returns
    -------
    dash.html.Script
        Script tag containing the helper code.
    """
    return html.Script(DOWNLOAD_HELPER_SCRIPT, type="text/javascript")
