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
  window.__mlpegLoadingProgress = {};

  var CEILING = 92; // never claim done before the app actually is
  var TIME_CONSTANT_MS = 1500; // how quickly the ease approaches CEILING
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

  /* Counting mounted rings would never let the loop idle out: shell.py hides
   * #startup-mask with display:none rather than unmounting it, so its finished
   * ring stays in the document for the life of the page.
   *
   * getClientRects() not offsetParent, which is also null for position:fixed —
   * a fixed ring would read as hidden and stop the loop mid-climb. */
  function needsFrames(ring) {
    var entry = state.get(ring);
    return !(entry && entry.done) && ring.getClientRects().length > 0;
  }

  function tick(now) {
    var rings = document.querySelectorAll(".mlpeg-progress-ring");
    var active = 0;

    for (var i = 0; i < rings.length; i++) {
      var ring = rings[i];
      if (needsFrames(ring)) {
        active++;
      }
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
    // of the page; observeForRings() brings us back when a ring next needs us.
    idle = active ? 0 : idle + 1;
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
   * the loop has to be able to wake up again after it has idled out.
   *
   * childList is enough: a ring only ever appears by being mounted (dcc.Loading
   * renders its spinner conditionally, and #startup-mask only ever hides).
   * Watching attributes would fire on every style write on the page.
   *
   * Mirrors tick()'s stop test, or the two fight: any mutation would restart
   * the loop just because the finished start-up ring is still mounted. */
  function observeForRings() {
    new MutationObserver(function () {
      if (running) {
        return;
      }
      var rings = document.querySelectorAll(".mlpeg-progress-ring");
      for (var i = 0; i < rings.length; i++) {
        if (needsFrames(rings[i])) {
          start();
          return;
        }
      }
    }).observe(document.documentElement, { childList: true, subtree: true });
  }

  // The shell waits for this promise before hiding the mask, so completion is
  // painted while the ring is still visible. Share it across overlapping polls.
  var completion;
  window.__mlpegLoadingProgress.finishStartup = function () {
    if (completion) {
      return completion;
    }
    var ring = document.querySelector("#startup-mask .mlpeg-progress-ring");
    if (!ring) {
      return Promise.resolve();
    }
    var entry = state.get(ring) || { pct: 0, last: 0 };
    entry.done = true; // Stop the estimated-progress loop overwriting completion.
    state.set(ring, entry);
    var initial = entry.pct;
    var duration = window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? 0 : 180;
    completion = new Promise(function (resolve) {
      var started;
      function finish(now) {
        if (started === undefined) {
          started = now;
        }
        var fraction = duration ? Math.min((now - started) / duration, 1) : 1;
        entry.pct = initial + (100 - initial) * fraction;
        paint(ring, entry.pct);
        if (fraction < 1) {
          window.requestAnimationFrame(finish);
        } else {
          // Leave a brief, visible completed frame before revealing the page.
          window.setTimeout(resolve, 100);
        }
      }
      window.requestAnimationFrame(finish);
    });
    return completion;
  };

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
