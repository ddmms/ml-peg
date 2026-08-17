/* Drives the percentage on every .mlpeg-progress-ring (loading.css).
 *
 * Why this is JS and not a CSS animation: the ring's fill and centre number both
 * read --mlpeg-pct, so animating it needs a style recalc + repaint per frame,
 * i.e. the main thread. During Dash hydration the main thread is blocked for
 * seconds at a time, so a CSS animation completed invisibly in the background
 * and the user saw 0% jump straight to the end value.
 *
 * A requestAnimationFrame loop can only advance on frames that actually run, and
 * MAX_FRAME_MS caps how much progress any single frame may contribute, so a long
 * blocked stretch cannot produce a jump either. The number therefore tracks what
 * the user can actually see being painted.
 *
 * The climb is still an estimate — the server gives no progress events — so it
 * eases towards CEILING and waits there for the real "ready" signal, at which
 * point the ring is completed. Rings that Dash mounts later (the page and table
 * loading overlays) are picked up automatically on the next frame.
 */
(function () {
  "use strict";

  // The tarball that ships the benchmark data also carries older flat copies of
  // these assets, so Dash can register this file twice. Bail on the second run
  // rather than driving two loops over the same rings.
  if (window.__mlpegLoadingProgress) {
    return;
  }
  window.__mlpegLoadingProgress = true;

  var CEILING = 92; // never claim done before the app actually is
  var TIME_CONSTANT_MS = 2600; // how quickly the ease approaches CEILING
  var MAX_FRAME_MS = 100; // ignore time the main thread spent blocked
  // Stop the loop once nothing has needed it for this many consecutive frames
  // (~2s at 60fps); a DOM observer restarts it when a ring next appears.
  var IDLE_FRAMES = 120;

  // Progress per ring, keyed by element, so several rings can coexist.
  var state = new WeakMap();
  var running = false;
  var idle = 0;

  function paint(ring, pct) {
    ring.style.setProperty("--mlpeg-pct", String(Math.round(pct)));
  }

  function tick(now) {
    watchStartupMask();

    var rings = document.querySelectorAll(".mlpeg-progress-ring");
    idle = rings.length ? 0 : idle + 1;

    for (var i = 0; i < rings.length; i++) {
      var ring = rings[i];
      var prev = state.get(ring);

      if (prev === undefined) {
        // First frame this ring is seen: start at 0 rather than guessing how
        // long it was already mounted while the thread was busy.
        state.set(ring, { pct: 0, last: now, done: false });
        paint(ring, 0);
        continue;
      }

      // Completed rings hold at 100 until they are unmounted; without this the
      // next frame would ease them straight back down to the ceiling.
      if (prev.done) {
        continue;
      }

      // Cap the delta: a frame that arrives after a long block represents time
      // the user spent looking at a frozen ring, not progress.
      var delta = Math.min(now - prev.last, MAX_FRAME_MS);
      prev.last = now;

      // Exponential ease towards CEILING — fast at first, then slower, which
      // reads as "working" without ever implying it is nearly finished.
      prev.pct += (CEILING - prev.pct) * (delta / TIME_CONSTANT_MS);
      paint(ring, prev.pct);
    }

    // Idle out rather than burning a querySelectorAll every frame for the life
    // of the page; observeForRings() brings us back when a ring next mounts.
    if (idle > IDLE_FRAMES) {
      running = false;
      return;
    }
    window.requestAnimationFrame(tick);
  }

  function start() {
    idle = 0;
    if (running) {
      return;
    }
    running = true;
    window.requestAnimationFrame(tick);
  }

  /* Dash mounts the page and table loading overlays long after first paint, so
   * the loop has to be able to wake up again after it has idled out. */
  function observeForRings() {
    new MutationObserver(function () {
      if (!running && document.querySelector(".mlpeg-progress-ring")) {
        start();
      }
    }).observe(document.documentElement, { childList: true, subtree: true });
  }

  /* Complete the start-up ring the moment the app is interactive, so the last
   * thing the user sees is 100% rather than the ring vanishing mid-climb. The
   * mask is hidden by a clientside callback in shell.py (it sets display:none);
   * watch for that instead of duplicating its readiness test here.
   *
   * #startup-mask is part of the Dash layout, so it does not exist at
   * DOMContentLoaded — React renders it once the bundle boots. Hence the check
   * runs from the frame loop until the mask turns up. */
  var maskWatched = false;

  function watchStartupMask() {
    if (maskWatched) {
      return;
    }
    var mask = document.getElementById("startup-mask");
    if (!mask) {
      return;
    }
    maskWatched = true;

    var observer = new MutationObserver(function () {
      if (mask.style.display !== "none") {
        return;
      }
      var ring = mask.querySelector(".mlpeg-progress-ring");
      if (ring) {
        // Mark it done as well as painting it: tick() owns --mlpeg-pct and
        // would otherwise ease this straight back to the ceiling next frame.
        var entry = state.get(ring);
        if (entry) {
          entry.pct = 100;
          entry.done = true;
        } else {
          state.set(ring, { pct: 100, last: 0, done: true });
        }
        paint(ring, 100);
      }
      observer.disconnect();
    });
    observer.observe(mask, { attributes: true, attributeFilter: ["style"] });
  }

  function init() {
    observeForRings();
    start();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
