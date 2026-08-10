/* Present DataTable hover tooltips in a viewport-fixed layer at <body> level,
 * and let a click PIN the card so it stays open and interactive.
 *
 * Dash renders the native tooltip (.dash-tooltip) inside the table's
 * .dash-spreadsheet-container, hence inside our .mlpeg-table-scroll wrapper. That
 * wrapper needs overflow-x:auto so a wide table (e.g. the MACE-POLAR-1 framework
 * summary) scrolls inside its card rather than spanning the page — but per the
 * CSS Overflow spec, overflow-x:auto coerces overflow-y:visible to auto, so the
 * wrapper clips any tooltip that rises above / drops below a row. Worse, Dash
 * itself writes an inline display:none on the tooltip whenever the anchor cell
 * nears the container's top/bottom/left edge. Neither is fixable in CSS, and a
 * benchmark card gets a transform on hover (a fixed-positioning containing
 * block), so an in-place position:fixed tooltip would be trapped too — the
 * tooltip must live at <body> level.
 *
 * We never move Dash's React-owned node (reparenting risks reconciliation
 * errors); we mirror its rendered HTML into our own fixed <div>, positioned from
 * the hovered cell's rect with above/below + viewport clamping. Hover shows a
 * transient preview; clicking a body cell PINS it so the card persists and can
 * be moused over to select/copy text or follow links (dismiss with the × button,
 * Escape, or a click outside). Header clicks are left alone (they sort). The
 * native tooltip is hidden only after we mark <html> (see theme.css), so if this
 * asset ever fails to run tooltips fall back to the native (clipped) behaviour.
 *
 * Dash auto-loads any .js under the assets_folder (ml_peg/app/data).
 */
(function () {
  "use strict";

  var GAP = 8; // px between the anchor cell and the card
  var MARGIN = 8; // px keep-clear from the viewport edges
  var HIDE_DELAY = 150; // ms grace so the pointer can travel into the card

  var portal = null; // our body-level container
  var inner = null; // the .dash-table-tooltip clone inside `portal`
  var closeBtn = null; // × shown only when pinned
  var anchorCell = null; // the td/th the card is anchored to
  var lastRect = null; // last anchor rect
  var currentHtml = ""; // mirrored content, so we skip redundant writes
  var looping = false; // transient hover loop active?
  var hideTimer = null;
  var pinned = false; // click-pinned (persistent + interactive)?

  function ensurePortal() {
    if (portal) return;
    portal = document.createElement("div");
    portal.className = "mlpeg-tooltip-portal";
    inner = document.createElement("div");
    inner.className = "dash-table-tooltip";
    portal.appendChild(inner);
    closeBtn = document.createElement("button");
    closeBtn.className = "mlpeg-tooltip-close";
    closeBtn.type = "button";
    closeBtn.setAttribute("aria-label", "Close");
    closeBtn.textContent = "×"; // ×
    closeBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      unpin();
    });
    portal.appendChild(closeBtn);
    // Keep a transient card open while the pointer is inside it (so it can be
    // read/copied); pinned cards ignore this and only close on an explicit action.
    portal.addEventListener("mouseenter", cancelHide);
    portal.addEventListener("mouseleave", function () {
      if (!pinned) scheduleHide();
    });
    document.body.appendChild(portal);
  }

  function hasContent() {
    return currentHtml && currentHtml.trim().length > 0;
  }

  // The native tooltip HTML for a cell's table (display:none near an edge does
  // not clear its innerHTML — surfacing what Dash hides is the whole point).
  function readTooltip(cell) {
    if (!cell) return "";
    var container = cell.closest(".dash-table-container");
    if (!container) return "";
    var src = container.querySelector(".dash-tooltip .dash-table-tooltip");
    return src ? src.innerHTML : "";
  }

  function setContent(html) {
    if (html !== currentHtml) {
      currentHtml = html;
      inner.innerHTML = html;
    }
  }

  function position(rect) {
    var vw = window.innerWidth;
    var vh = window.innerHeight;
    var pw = portal.offsetWidth;
    var ph = portal.offsetHeight;
    // Vertical: prefer below the cell; flip above if it would overflow the foot.
    var top = rect.bottom + GAP;
    if (top + ph > vh - MARGIN) {
      var above = rect.top - ph - GAP;
      top = above >= MARGIN ? above : Math.max(MARGIN, vh - ph - MARGIN);
    }
    // Horizontal: align to the cell's left; shift in if it would overflow.
    var left = rect.left;
    if (left + pw > vw - MARGIN) left = vw - pw - MARGIN;
    if (left < MARGIN) left = MARGIN;
    portal.style.top = Math.round(top) + "px";
    portal.style.left = Math.round(left) + "px";
  }

  function reposition() {
    if (anchorCell) {
      lastRect = anchorCell.getBoundingClientRect();
    }
    if (hasContent() && lastRect) {
      portal.classList.add("is-visible");
      position(lastRect);
    } else {
      portal.classList.remove("is-visible");
    }
  }

  // Transient-hover loop; runs only between a cell mouseover and the hide.
  function frame() {
    if (!looping || pinned) return;
    if (anchorCell) setContent(readTooltip(anchorCell));
    reposition();
    window.requestAnimationFrame(frame);
  }

  function startLoop() {
    if (looping) return;
    looping = true;
    window.requestAnimationFrame(frame);
  }

  function cancelHide() {
    if (hideTimer !== null) {
      window.clearTimeout(hideTimer);
      hideTimer = null;
    }
  }

  function scheduleHide() {
    cancelHide();
    hideTimer = window.setTimeout(hide, HIDE_DELAY);
  }

  function hide() {
    cancelHide();
    looping = false;
    anchorCell = null;
    lastRect = null;
    currentHtml = "";
    if (portal) {
      portal.classList.remove("is-visible");
      inner.innerHTML = "";
    }
  }

  function pin(cell) {
    ensurePortal();
    var html = readTooltip(cell);
    if (!html || !html.trim()) return; // nothing to pin for this cell
    cancelHide();
    looping = false; // switch from the hover loop to event-based positioning
    pinned = true;
    anchorCell = cell;
    lastRect = cell.getBoundingClientRect();
    setContent(html);
    portal.classList.add("is-pinned");
    reposition();
    // Follow the anchor on scroll/resize without a per-frame reflow.
    window.addEventListener("scroll", reposition, true);
    window.addEventListener("resize", reposition, true);
  }

  function unpin() {
    if (!pinned) return;
    pinned = false;
    portal.classList.remove("is-pinned");
    window.removeEventListener("scroll", reposition, true);
    window.removeEventListener("resize", reposition, true);
    hide();
  }

  // The tracked cell for an event target, or null. `bodyOnly` restricts to data
  // cells (td) — used for click-to-pin so header clicks keep sorting the table.
  function cellFrom(target, bodyOnly) {
    if (!target || !target.closest) return null;
    var sel = bodyOnly
      ? ".dash-table-container td"
      : ".dash-table-container td, .dash-table-container th";
    var cell = target.closest(sel);
    return cell && cell.hasAttribute("data-dash-column") ? cell : null;
  }

  function onOver(e) {
    if (pinned) return; // a pinned card owns the display
    var cell = cellFrom(e.target, false);
    if (!cell) return;
    ensurePortal();
    cancelHide();
    if (cell !== anchorCell) {
      anchorCell = cell;
      lastRect = cell.getBoundingClientRect();
      currentHtml = "";
      inner.innerHTML = "";
    }
    startLoop();
  }

  function onOut(e) {
    if (pinned) return;
    if (!cellFrom(e.target, false)) return;
    var to = e.relatedTarget;
    if (to && portal && (to === portal || portal.contains(to))) return; // into the card
    if (to && cellFrom(to, false)) return; // straight onto another cell
    anchorCell = null;
    scheduleHide();
  }

  function onClick(e) {
    // Clicks inside the card (text selection, links, the × button) never dismiss.
    if (portal && (e.target === portal || portal.contains(e.target))) return;
    var cell = cellFrom(e.target, true); // body cells only
    if (cell) {
      // Toggle off if the same cell is re-clicked; otherwise pin (or re-pin).
      if (pinned && cell === anchorCell) unpin();
      else pin(cell);
      return;
    }
    // A click anywhere else dismisses a pinned card.
    if (pinned) unpin();
  }

  function onKey(e) {
    if (e.key === "Escape" && pinned) unpin();
  }

  function start() {
    ensurePortal();
    // Gate the native-tooltip hide on this flag so tooltips degrade gracefully
    // if the asset ever fails to load.
    document.documentElement.classList.add("mlpeg-tooltip-js");
    // Capture phase: the DataTable can stop propagation on its cells. We never
    // stop propagation ourselves, so Dash still receives clicks (detail plots).
    document.addEventListener("mouseover", onOver, true);
    document.addEventListener("mouseout", onOut, true);
    document.addEventListener("click", onClick, true);
    document.addEventListener("keydown", onKey, true);
  }

  // Dash may run assets before <body> exists; defer wiring until it does.
  if (document.body) {
    start();
  } else {
    document.addEventListener("DOMContentLoaded", start);
  }
})();
