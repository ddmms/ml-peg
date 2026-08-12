(function () {
  "use strict";

  const CELL_SELECTOR =
    ".dash-table-container th, .dash-table-container td";
  const TOOLTIP_SELECTOR = ".dash-tooltip";
  const LEAVE_GRACE_PERIOD_MS = 200;

  let sourceCell = null;
  let leaveTimer = null;
  let replayingMouseout = false;

  function closestElement(target, selector) {
    return target instanceof Element ? target.closest(selector) : null;
  }

  function clearLeaveTimer() {
    if (leaveTimer !== null) {
      window.clearTimeout(leaveTimer);
      leaveTimer = null;
    }
  }

  function visibleTooltipFor(cell) {
    const table = cell.closest(".dash-table-container");
    const tooltip = table && table.querySelector(TOOLTIP_SELECTOR);
    if (!tooltip) {
      return null;
    }

    const style = window.getComputedStyle(tooltip);
    return style.visibility !== "hidden" && style.display !== "none"
      ? tooltip
      : null;
  }

  function releaseTooltip(relatedTarget) {
    clearLeaveTimer();

    const cell = sourceCell;
    sourceCell = null;
    if (!cell || !cell.isConnected) {
      return;
    }

    // Dash closes a tooltip in its cell mouseout handler. Replay the event once
    // the pointer has also left the tooltip, or after the grace period expires.
    replayingMouseout = true;
    cell.dispatchEvent(
      new MouseEvent("mouseout", {
        bubbles: true,
        cancelable: true,
        relatedTarget: relatedTarget instanceof Node ? relatedTarget : null,
        view: window,
      }),
    );
    replayingMouseout = false;
  }

  document.addEventListener(
    "mouseout",
    function (event) {
      if (replayingMouseout) {
        return;
      }

      const tooltip = closestElement(event.target, TOOLTIP_SELECTOR);
      if (tooltip) {
        const nextTooltip = closestElement(event.relatedTarget, TOOLTIP_SELECTOR);
        if (!nextTooltip) {
          if (
            sourceCell &&
            event.relatedTarget instanceof Node &&
            sourceCell.contains(event.relatedTarget)
          ) {
            clearLeaveTimer();
            sourceCell = null;
          } else {
            releaseTooltip(event.relatedTarget);
          }
        }
        return;
      }

      const cell = closestElement(event.target, CELL_SELECTOR);
      if (!cell) {
        return;
      }
      if (
        event.relatedTarget instanceof Node &&
        cell.contains(event.relatedTarget)
      ) {
        return;
      }

      // Moving directly to another cell should retain Dash's normal behavior.
      // Otherwise, pause its mouseout handler briefly so the pointer can cross
      // the tooltip arrow/gap without the tooltip disappearing first.
      if (closestElement(event.relatedTarget, CELL_SELECTOR)) {
        return;
      }
      if (!visibleTooltipFor(cell)) {
        return;
      }

      event.stopImmediatePropagation();
      sourceCell = cell;
      clearLeaveTimer();
      leaveTimer = window.setTimeout(
        function () {
          releaseTooltip(event.relatedTarget);
        },
        LEAVE_GRACE_PERIOD_MS,
      );
    },
    true,
  );

  document.addEventListener(
    "mouseover",
    function (event) {
      if (closestElement(event.target, TOOLTIP_SELECTOR)) {
        clearLeaveTimer();
        return;
      }

      const enteredCell = closestElement(event.target, CELL_SELECTOR);
      if (sourceCell && enteredCell && enteredCell !== sourceCell) {
        // Release the old cell before Dash handles entry into the new one. This
        // prevents a pending grace-period timer from closing the new tooltip.
        releaseTooltip(enteredCell);
      }
    },
    true,
  );
})();
