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

  document.addEventListener("click", function (e) {
    var d = popover();
    // The click that opens it targets the summary (inside d), so opening never
    // immediately re-closes; only a click outside an open popover closes it.
    if (d && d.open && !d.contains(e.target)) d.open = false;
  });
})();
