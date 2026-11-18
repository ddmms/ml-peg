/* Close the header settings popover on Escape or an outside click.
 *
 * The popover is a native <details id="settings-details">. A <details> closes
 * only when its own <summary> is toggled — not on Escape and not when the user
 * clicks elsewhere on the page, which is the expected behaviour for a menu.
 * The element mounts after this asset loads, so it is looked up lazily on each
 * event rather than cached.
 */
(function () {
  "use strict";

  // The data tarball ships older flat copies of these assets, so Dash can
  // register this file twice; a second run would double-bind its listeners.
  if (window.__mlpegSettingsPopover) {
    return;
  }
  window.__mlpegSettingsPopover = true;

  function popover() {
    return document.getElementById("settings-details");
  }

  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    var d = popover();
    if (d && d.open) {
      d.open = false;
      var summary = d.querySelector("summary");
      if (summary) summary.focus(); // return focus to the trigger
    }
  });

  // Where the press started. A click event is dispatched on the common ancestor
  // of mousedown and mouseup, so a drag that begins on the zoom slider and ends
  // outside the panel reports a target outside it — closing the popover on every
  // slider drag. Only an interaction that *starts* outside should close it.
  var pressedOutside = false;
  document.addEventListener(
    "pointerdown",
    function (e) {
      var d = popover();
      pressedOutside = !!(d && !d.contains(e.target));
    },
    true
  );

  document.addEventListener("click", function (e) {
    var d = popover();
    // The click that opens it targets the summary (inside d), so opening never
    // immediately re-closes; only a click outside an open popover closes it.
    if (d && d.open && pressedOutside && !d.contains(e.target)) d.open = false;
  });
})();
