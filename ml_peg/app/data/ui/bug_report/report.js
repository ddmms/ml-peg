/* Prepare a GitHub issue locally; screenshots are downloaded for manual attachment. */
(function () {
  "use strict";
  if (window.mlpegBugReport) return;

  const ISSUE_URL = "https://github.com/ddmms/ml-peg/issues/new";
  // Issue form fields are prefilled by their ids in .github/ISSUE_TEMPLATE/6_website_bug.yml.
  const ISSUE_TEMPLATE = "6_website_bug.yml";
  // Bound the encoded link, including the title, report text and attachment note.
  const MAX_ISSUE_URL_LENGTH = 7500;
  const MAX_DETAIL_ENTRIES = 20;
  let dialog, context, screenshot, rectangles = [], startPoint, generation = 0, captures = 0;

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

  let screenshotFilename = "";

  function nextScreenshotFilename() {
    screenshotFilename = `mlpeg-bug-screenshot-${crypto.randomUUID()}.png`;
    field("attachment-filename").textContent = screenshotFilename;
    return screenshotFilename;
  }

  function downloadScreenshot(filename = nextScreenshotFilename()) {
    // Synchronous preparation preserves the user's gesture for the new tab.
    const data = field("canvas").toDataURL("image/png").split(",")[1];
    const bytes = Uint8Array.from(atob(data), char => char.charCodeAt(0));
    download(new Blob([bytes], {type: "image/png"}), filename);
  }

  function attachmentInstructions() {
    if (!screenshot) return "";
    return "> [!IMPORTANT]\n> **Complete these manual steps before submitting.**\n"
      + "> Attachments are not uploaded automatically.\n\n"
      + "- [ ] **Attach the screenshot:** drag `" + screenshotFilename + "` from your browser's downloads into this field. If it is missing, return to the ML-PEG reporter and click **Download screenshot**.";
  }

  function draft() {
    const url = new URL(ISSUE_URL);
    url.searchParams.set("template", ISSUE_TEMPLATE);
    url.searchParams.set("title", field("title").value.trim() || context.page);
    for (const id of ["description", "expected", "steps"]) {
      const value = field(id).value.trim();
      if (value) url.searchParams.set(id, value);
    }
    if (screenshot) url.searchParams.set("screenshots", attachmentInstructions());
    const fits = (details, compact = false) => {
      url.searchParams.set("details", JSON.stringify(details, null, compact ? undefined : 2));
      return url.href.length <= MAX_ISSUE_URL_LENGTH;
    };
    if (fits(context)) return url.href;
    const concise = {note: "Only changed weights and thresholds are included.", ...context};
    delete concise.page_settings;
    if (fits(concise, true)) return url.href;

    // Keep report text intact and admit only page details that fit the remaining budget.
    const details = {
      note: "Page details shortened to fit the issue link. Omitted entry counts are listed in omitted_details.",
      omitted_details: {},
    };
    const groups = ["selected_models", "changed_settings", "excluded_elements", "appearance", "page_selections"];
    const scalars = ["url", "page", "version", "reported_at", "browser", "viewport"];
    for (const key of groups) details.omitted_details[key] = Object.keys(context[key]).length;
    details.omitted_details.page_settings = Object.keys(context.page_settings).length;
    // Omission counts reserve room before adding entries, including unusually long values.
    for (const key of scalars) details.omitted_details[key] = 1;
    if (!fits(details, true)) return null;
    for (const key of scalars) {
      details[key] = context[key];
      details.omitted_details[key] = 0;
      if (!fits(details, true)) {
        delete details[key];
        details.omitted_details[key] = 1;
      }
    }
    for (const key of groups) {
      const isArray = Array.isArray(context[key]);
      details[key] = isArray ? [] : {};
      let entries = Object.entries(context[key]);
      if (key === "changed_settings") {
        // Changes on the current page are more useful than those on other pages.
        const onPage = id => Object.prototype.hasOwnProperty.call(context.page_settings, id);
        entries = entries.filter(([id]) => onPage(id)).concat(entries.filter(([id]) => !onPage(id)));
      }
      for (const [id, value] of entries.slice(0, MAX_DETAIL_ENTRIES)) {
        if (isArray) details[key].push(value);
        else details[key][id] = value;
        details.omitted_details[key]--;
        if (!fits(details, true)) {
          if (isArray) details[key].pop();
          else delete details[key][id];
          details.omitted_details[key]++;
        }
      }
      if (Object.keys(details[key]).length === 0) delete details[key];
    }
    for (const key of Object.keys(details.omitted_details)) {
      if (details.omitted_details[key] === 0) delete details.omitted_details[key];
    }
    // The last attempted entry may have been rejected, so rebuild the final URL.
    return fits(details, true) ? url.href : null;
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
    if (token !== generation || !dialog.open) return;
    const image = new Image();
    image.src = source;
    await image.decode();
    if (token !== generation || !dialog.open) return;
    screenshot = image;
    nextScreenshotFilename();
    rectangles = [];
    startPoint = null;
    const scale = Math.min(1, 1920 / image.width, 1080 / image.height);
    field("canvas").width = Math.round(image.width * scale);
    field("canvas").height = Math.round(image.height * scale);
    field("image").hidden = false;
    draw();
    status("Drag on the screenshot to mark the problem. Opening the GitHub issue downloads the image for you to attach on GitHub.");
  }

  function imageLibrary() {
    if (window.htmlToImage) return Promise.resolve(window.htmlToImage);
    // Reuse the same library and cached load as the table-export feature.
    if (!window._mlpegHtmlToImagePromise) {
      const pending = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/npm/html-to-image@1.11.11/dist/html-to-image.min.js";
        script.onload = () => resolve(window.htmlToImage);
        script.onerror = () => reject(new Error("Screenshot tools could not load."));
        document.head.appendChild(script);
      }).catch(error => {
        // Loading is shared with table export and outlives the dialog.
        if (window._mlpegHtmlToImagePromise === pending) window._mlpegHtmlToImagePromise = null;
        throw error;
      });
      window._mlpegHtmlToImagePromise = pending;
    }
    return window._mlpegHtmlToImagePromise;
  }

  async function capture() {
    const token = ++generation;
    let capturing = false;
    field("capture").disabled = true;
    status("Capturing the current page…");
    try {
      const library = await imageLibrary();
      if (token !== generation || !dialog.open) return;
      const width = window.innerWidth, height = window.innerHeight;
      // SVG foreignObject rendering can add scrollbars absent from the live page.
      captures++;
      capturing = true;
      document.body.classList.add("mlpeg-page-capturing");
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
        status("Screenshot ready. Check the preview: 3D viewers may be missing. You can attach your own screenshot directly on GitHub.");
      }
    } catch (error) {
      if (token === generation) {
        status("Could not capture this page. You can attach your own screenshot directly on GitHub, or report without one.");
      }
    } finally {
      if (capturing && --captures === 0) document.body.classList.remove("mlpeg-page-capturing");
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
        <div class="bug-report-header">
          <h2 id="bug-report-heading">Report a bug</h2>
          <button id="bug-report-close" type="button" aria-label="Close bug report">×</button>
        </div>
        <div id="bug-report-body" class="bug-report-body">
        <p>All fields are optional. Review and complete your report on GitHub before submitting with your GitHub account.
        Optional screenshots download to your device for you to attach. Nothing is submitted automatically.</p>
        <label for="bug-report-title">Short title</label>
        <input id="bug-report-title" type="text" maxlength="120">
        <label for="bug-report-description">What happened?</label>
        <textarea id="bug-report-description" maxlength="4000"></textarea>
        <label for="bug-report-expected">What did you expect?</label>
        <textarea id="bug-report-expected" maxlength="2000"></textarea>
        <label for="bug-report-steps">Steps to reproduce</label>
        <textarea id="bug-report-steps" maxlength="4000"></textarea>
        <details><summary>Included page details</summary><pre id="bug-report-context"></pre></details>
        <div class="bug-report-actions"><button id="bug-report-capture" type="button">Capture this page</button></div>
        <div id="bug-report-image" hidden>
          <p>Drag to mark an area. Check the preview before attaching it:
          automatic captures can differ in scrolled areas or 3D viewers.</p>
          <canvas id="bug-report-canvas" aria-label="Screenshot: drag to highlight a problem area"></canvas>
          <p id="bug-report-attachment-note">The screenshot is not uploaded automatically.
          Open the GitHub issue, then drag the downloaded <strong id="bug-report-attachment-filename">screenshot</strong>
          into the Screenshots field before submitting.</p>
          <div class="bug-report-actions">
            <button id="bug-report-download" type="button">Download screenshot</button>
            <button id="bug-report-clear" type="button">Clear marks</button>
            <button id="bug-report-remove" type="button">Remove screenshot</button>
          </div>
        </div>
        <p id="bug-report-status" class="bug-report-status" role="status" aria-live="polite"></p>
        <div class="bug-report-actions">
          <button id="bug-report-cancel" type="button">Cancel</button>
          <button type="submit">Open GitHub issue</button>
        </div>
        </div>
      </form>`;
    document.body.appendChild(dialog);
    field("capture").addEventListener("click", capture);
    field("download").addEventListener("click", () => {
      if (!screenshot) return;
      downloadScreenshot();
      status("Screenshot downloaded. Drag " + screenshotFilename + " into the Screenshots field of the GitHub issue to attach it.");
    });
    field("cancel").addEventListener("click", () => dialog.close());
    field("close").addEventListener("click", () => dialog.close());
    let backdropPointer = false;
    function outsideDialog(event) {
      const box = dialog.getBoundingClientRect();
      return event.clientX < box.left || event.clientX > box.right
        || event.clientY < box.top || event.clientY > box.bottom;
    }
    dialog.addEventListener("pointerdown", event => {
      backdropPointer = event.target === dialog && outsideDialog(event);
    });
    dialog.addEventListener("pointercancel", () => { backdropPointer = false; });
    dialog.addEventListener("click", event => {
      if (backdropPointer && event.target === dialog && outsideDialog(event)) dialog.close();
      backdropPointer = false;
    });
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
      startPoint = null;
      rectangles = [];
      field("image").hidden = true;
      field("capture").disabled = false;
      status("");
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
      const filename = screenshot ? nextScreenshotFilename() : null;
      const url = draft();
      if (!url) {
        status("This report is too long to open as a GitHub issue. Please shorten the description, expected result or steps. Your text is still here.");
        return;
      }
      if (screenshot) downloadScreenshot(filename);
      const link = document.createElement("a");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.click();
      status(screenshot
        ? "GitHub issue opened. Drag " + screenshotFilename + " from your downloads into the Screenshots field to attach it before submitting."
        : "GitHub issue opened. Review your report, then submit it on GitHub.");
    });
  }

  // Python registers this as bug_report.open. Keeping the callback a named asset
  // function rather than an inline source string keeps its Dash identity stable:
  // inline clientside callbacks are named by hashing their source, so embedding
  // the defaults there renamed the function whenever benchmark data changed.
  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    bug_report: {
      open: function (nClicks, config, models, excluded, ...values) {
        // The dialog module and this wrapper ship together, but a failed asset
        // load would otherwise throw on every click with nothing shown.
        if (nClicks && config && window.mlpegBugReport) {
          window.mlpegBugReport.open(config, models, excluded, values);
        }
        return "";
      },
    },
  });

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
