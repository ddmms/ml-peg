"""JavaScript helpers for table downloads."""

from __future__ import annotations

DOWNLOAD_CLIENTSIDE_HANDLER = """
function(request) {
    const dash = window.dash_clientside;
    if (!dash) {
        return null;
    }
    if (!request) {
        return dash.no_update;
    }

    if (!dash._mlpegDownloadHelper) {
        dash._mlpegDownloadHelper = (function(noUpdate) {
            const HTML_TO_IMAGE_SRC =
                "https://cdn.jsdelivr.net/npm/html-to-image@1.11.11/dist/html-to-image.min.js";
            let htmlToImagePromise = null;

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
                const parts = dataUrl.split(",");
                if (parts.length < 2) {
                    return null;
                }
                if (/;base64/i.test(parts[0])) {
                    return parts.slice(1).join(",");
                }
                try {
                    const decoded = decodeURIComponent(parts.slice(1).join(","));
                    return encodeTextToBase64(decoded);
                } catch (error) {
                    console.error("Failed to convert SVG data URL to base64.", error);
                    return null;
                }
            }

            function resolveTableNode(req) {
                if (req.element_id) {
                    return document.getElementById(req.element_id);
                }
                if (req.selector) {
                    return document.querySelector(req.selector);
                }
                return null;
            }

            function captureTable(req) {
                const tableNode = resolveTableNode(req);
                if (!tableNode) {
                    console.warn(
                        "Unable to find table element for download request.",
                        req
                    );
                    return noUpdate;
                }

                const format = (req.format || "png").toLowerCase();
                const filename = req.filename || `table.${format}`;
                const pixelRatio = req.pixel_ratio || window.devicePixelRatio || 2;
                const options = {
                    cacheBust: true,
                    pixelRatio,
                    backgroundColor: req.background || "#ffffff",
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
                            throw new Error(
                                "Invalid data URL returned from html-to-image."
                            );
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

            return { captureTable };
        })(dash.no_update);
    }

    return dash._mlpegDownloadHelper.captureTable(request);
}
"""
