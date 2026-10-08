/* Prepare a GitHub issue locally; screenshots are downloaded for manual attachment. */
(function () {
  "use strict";
  if (window.mlpegBugReport) return;

  const ISSUE_URL = "https://github.com/ddmms/ml-peg/issues/new";
  let dialog, context, screenshot, rectangles = [], startPoint, generation = 0;

  function field(id) { return dialog.querySelector("#bug-report-" + id); }
  function status(message) { field("status").textContent = message; }

  function download(data, filename) {
    const url = URL.createObjectURL(data);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }

  function downloadScreenshot() {
    // Synchronous preparation preserves the user's gesture for the new tab.
    const data = field("canvas").toDataURL("image/png").split(",")[1];
    const bytes = Uint8Array.from(atob(data), char => char.charCodeAt(0));
    download(new Blob([bytes], {type: "image/png"}), "mlpeg-bug-screenshot.png");
  }

  function reportBody(details = context, compact = false) {
    const instructions = attachmentInstructions();
    const sections = [
      "## What happened\n" + field("description").value.trim(),
      "## What I expected\n" + field("expected").value.trim(),
      "## Steps to reproduce\n" + field("steps").value.trim(),
      "## Page details\n" + (compact ? "Only changed weights and thresholds are included.\n" : "")
        + "```json\n" + JSON.stringify(details, null, compact ? undefined : 2) + "\n```",
    ];
    if (instructions) sections.unshift(instructions);
    return sections.join("\n\n");
  }

  function attachmentInstructions() {
    if (!screenshot) return "";
    return "> [!IMPORTANT]\n> **Complete these manual steps before submitting.**\n"
      + "> Attachments are not uploaded automatically.\n\n"
      + "- [ ] **Attach the screenshot:** drag `mlpeg-bug-screenshot.png` from your browser's downloads into this description. If it is missing, return to the ML-PEG reporter and click **Download screenshot**.";
  }

  function draft() {
    const url = new URL(ISSUE_URL);
    url.searchParams.set("title", "Website bug: " + field("title").value.trim());
    url.searchParams.set("body", reportBody());
    if (url.href.length > 7500) {
      // Keep user text and changed settings intact, omitting the full settings grid.
      const concise = {...context};
      delete concise.page_settings;
      url.searchParams.set("body", reportBody(concise, true));
    }
    return url.href.length <= 7500 ? url.href : null;
  }

  function draw(extra) {
    const canvas = field("canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(screenshot, 0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#e11d48";
    ctx.fillStyle = "rgba(225, 29, 72, 0.10)";
    ctx.lineWidth = Math.max(3, canvas.width / 450);
    for (const rect of [...rectangles, ...(extra ? [extra] : [])]) {
      ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
      ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
    }
  }

  function point(event) {
    const canvas = field("canvas");
    const box = canvas.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(canvas.width, (event.clientX - box.left) * canvas.width / box.width)),
      y: Math.max(0, Math.min(canvas.height, (event.clientY - box.top) * canvas.height / box.height)),
    };
  }

  function rectangle(event) {
    const end = point(event);
    return {x: startPoint.x, y: startPoint.y, width: end.x - startPoint.x, height: end.y - startPoint.y};
  }

  async function useScreenshot(source, token) {
    const image = new Image();
    image.src = source;
    await image.decode();
    if (token !== generation || !dialog.open) return;
    screenshot = image;
    rectangles = [];
    const scale = Math.min(1, 1920 / image.width, 1080 / image.height);
    field("canvas").width = Math.round(image.width * scale);
    field("canvas").height = Math.round(image.height * scale);
    field("image").hidden = false;
    draw();
    status("Drag on the screenshot to mark the problem. Opening the GitHub draft downloads the image; you must attach it on GitHub.");
  }

  function imageLibrary() {
    if (window.htmlToImage) return Promise.resolve(window.htmlToImage);
    // Reuse the same library and cached load as the table-export feature.
    if (!window._mlpegHtmlToImagePromise) {
      window._mlpegHtmlToImagePromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/npm/html-to-image@1.11.11/dist/html-to-image.min.js";
        script.onload = () => resolve(window.htmlToImage);
        script.onerror = () => reject(new Error("Screenshot tools could not load."));
        document.head.appendChild(script);
      });
    }
    return window._mlpegHtmlToImagePromise;
  }

  async function capture() {
    const token = generation;
    field("capture").disabled = true;
    status("Capturing the current page…");
    try {
      const library = await imageLibrary();
      const width = window.innerWidth, height = window.innerHeight;
      const data = await library.toPng(document.body, {
        width, height,
        pixelRatio: Math.min(1, 1920 / width, 1080 / height),
        backgroundColor: getComputedStyle(document.body).backgroundColor,
        // Hidden onboarding videos and graphs can produce empty image URLs.
        filter: node => {
          if (node === dialog) return false;
          if (node instanceof HTMLElement && getComputedStyle(node).display === "none") return false;
          if (node instanceof HTMLVideoElement && (!node.clientWidth || !node.clientHeight)) return false;
          return !(node instanceof HTMLCanvasElement && (!node.width || !node.height));
        },
        imagePlaceholder: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a3ioAAAAASUVORK5CYII=",
        style: {
          transform: `translate(${-window.scrollX}px, ${-window.scrollY}px)`,
          transformOrigin: "top left",
        },
      });
      await useScreenshot(data, token);
      if (token === generation && document.querySelector("#page-content iframe")) {
        status("Screenshot ready. Check the preview: 3D viewers may be missing. You can use your own screenshot instead.");
      }
    } catch (error) {
      window._mlpegHtmlToImagePromise = null;
      if (token === generation) status("Could not capture this page. Choose your own screenshot, or report without one.");
    } finally {
      if (token === generation) field("capture").disabled = false;
    }
  }

  function createDialog() {
    dialog = document.createElement("dialog");
    dialog.id = "bug-report-dialog";
    dialog.className = "mlpeg-bug-report";
    dialog.setAttribute("aria-labelledby", "bug-report-heading");
    // Static markup only. Report data is assigned using value/textContent.
    dialog.innerHTML = `
      <form id="bug-report-form">
        <h2 id="bug-report-heading">Report a bug</h2>
        <p>Describe the problem, then review a draft on GitHub. A GitHub account is needed to submit it.
        Optional screenshots download to your device for you to attach. Nothing is submitted automatically.</p>
        <label for="bug-report-title">Short title</label>
        <input id="bug-report-title" type="text" maxlength="120" required>
        <label for="bug-report-description">What happened?</label>
        <textarea id="bug-report-description" maxlength="4000" required></textarea>
        <label for="bug-report-expected">What did you expect?</label>
        <textarea id="bug-report-expected" maxlength="2000"></textarea>
        <label for="bug-report-steps">Steps to reproduce</label>
        <textarea id="bug-report-steps" maxlength="4000"></textarea>
        <details><summary>Included page details</summary><pre id="bug-report-context"></pre></details>
        <div class="bug-report-actions"><button id="bug-report-capture" type="button">Capture this page</button></div>
        <label for="bug-report-upload">Or choose a screenshot (PNG or JPEG, up to 10 MB)</label>
        <input id="bug-report-upload" type="file" accept="image/png,image/jpeg">
        <div id="bug-report-image" hidden>
          <p>Drag to mark an area. Check the preview before attaching it:
          automatic captures can differ in scrolled areas or 3D viewers.</p>
          <canvas id="bug-report-canvas" aria-label="Screenshot: drag to highlight a problem area"></canvas>
          <p id="bug-report-attachment-note">The screenshot is not uploaded automatically.
          Open the GitHub draft, then drag the downloaded <strong>mlpeg-bug-screenshot.png</strong>
          into the issue description before submitting.</p>
          <div class="bug-report-actions">
            <button id="bug-report-download" type="button">Download screenshot</button>
            <button id="bug-report-clear" type="button">Clear marks</button>
            <button id="bug-report-remove" type="button">Remove screenshot</button>
          </div>
        </div>
        <p id="bug-report-status" class="bug-report-status" role="status" aria-live="polite"></p>
        <div class="bug-report-actions">
          <button id="bug-report-cancel" type="button">Cancel</button>
          <button type="submit">Open GitHub draft</button>
        </div>
      </form>`;
    document.body.appendChild(dialog);
    field("capture").addEventListener("click", capture);
    field("download").addEventListener("click", () => {
      if (!screenshot) return;
      downloadScreenshot();
      status("Screenshot downloaded. Drag mlpeg-bug-screenshot.png into the GitHub issue description to attach it.");
    });
    field("cancel").addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", () => {
      generation++;
      screenshot = null;
      startPoint = null;
      document.getElementById("bug-report-button")?.focus({preventScroll: true});
    });
    field("clear").addEventListener("click", () => { rectangles = []; draw(); });
    field("remove").addEventListener("click", () => {
      generation++;
      screenshot = null;
      field("upload").value = "";
      field("image").hidden = true;
      field("capture").disabled = false;
      status("");
    });
    field("upload").addEventListener("change", async event => {
      const file = event.target.files[0];
      if (!file) return;
      if (!["image/png", "image/jpeg"].includes(file.type) || file.size > 10 * 1024 * 1024) {
        status("Choose a PNG or JPEG screenshot smaller than 10 MB.");
        return;
      }
      const token = ++generation;
      const source = URL.createObjectURL(file);
      field("capture").disabled = false;
      try { await useScreenshot(source, token); }
      catch (error) {
        if (token === generation) status("Could not read that screenshot. Try another image.");
      }
      finally { URL.revokeObjectURL(source); }
    });
    const canvas = field("canvas");
    canvas.addEventListener("pointerdown", event => {
      if (!screenshot) return;
      startPoint = point(event);
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", event => { if (startPoint) draw(rectangle(event)); });
    canvas.addEventListener("pointerup", event => {
      if (!startPoint) return;
      const rect = rectangle(event);
      if (Math.abs(rect.width) > 3 && Math.abs(rect.height) > 3) rectangles.push(rect);
      startPoint = null;
      draw();
    });
    canvas.addEventListener("pointercancel", () => { startPoint = null; if (screenshot) draw(); });
    field("form").addEventListener("submit", event => {
      event.preventDefault();
      const url = draft();
      if (!url) {
        status("This report is too long to open as a GitHub draft. Please shorten the description, expected result or steps. Your text is still here.");
        return;
      }
      if (screenshot) downloadScreenshot();
      const link = document.createElement("a");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.click();
      status(screenshot
        ? "GitHub draft opened. Drag mlpeg-bug-screenshot.png from your downloads into the issue description to attach it before submitting."
        : "GitHub draft opened. Review your report, then submit it on GitHub.");
    });
  }

  window.mlpegBugReport = {
    open(config, models, excluded, values) {
      if (!dialog) createDialog();
      if (dialog.open) return;
      generation++;
      context = {
        url: window.location.href,
        version: config.version,
        reported_at: new Date().toISOString(),
        browser: navigator.userAgent,
        viewport: {
          width: window.innerWidth, height: window.innerHeight,
          scroll_x: window.scrollX, scroll_y: window.scrollY,
        },
        selected_models: models || [],
        excluded_elements: excluded || [],
        appearance: {},
        page_selections: {},
        page_settings: {},
        changed_settings: {},
      };
      config.store_ids.forEach((id, index) => {
        const value = values[index];
        if (id.endsWith("-weight-store") || id.endsWith("-thresholds-store")) {
          const table = document.getElementById(id.replace(/-(weight|thresholds)-store$/, ""));
          if (table && document.getElementById("page-content").contains(table)) context.page_settings[id] = value;
          if (JSON.stringify(value) !== JSON.stringify(config.defaults[id])) context.changed_settings[id] = value;
        } else context.appearance[id] = value;
      });
      document.querySelectorAll("#page-content .dash-dropdown[id]").forEach(node => {
        context.page_selections[node.id] = Array.from(node.querySelectorAll(".dash-dropdown-value-item"))
          .map(item => item.textContent.trim());
      });
      const heading = document.querySelector("#page-content h1, #page-content h2");
      context.page = heading ? heading.textContent.trim() : window.location.pathname;
      field("form").reset();
      field("context").textContent = JSON.stringify(context, null, 2);
      field("title").value = context.page.slice(0, 100) + ": ";
      field("image").hidden = true;
      field("capture").disabled = false;
      rectangles = [];
      screenshot = null;
      status("");
      dialog.showModal();
      field("title").focus({preventScroll: true});
      window.scrollTo(context.viewport.scroll_x, context.viewport.scroll_y);
    },
  };
})();
