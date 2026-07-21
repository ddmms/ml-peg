(function () {
    window.addEventListener("message", function (e) {
        var msg = e.data;
        if (!msg || msg.type !== "ml-peg-weas-frame") {
            return;
        }
        var active = window.__mlPegActiveTraj;
        if (!active) {
            return;
        }
        var container = document.getElementById(active.scatterId);
        var gd = container && container.querySelector(".js-plotly-plot");
        if (!gd || !gd.data) {
            return;
        }
        var hi = gd.data.findIndex(function (t) {
            return t.name === "__clicked_point__";
        });
        var f = msg.frame;
        if (hi < 0 || f == null || f < 0 || f >= active.x.length) {
            return;
        }
        // Pin the ring's axes to their current range once, so moving the ring
        // can no longer trigger autorange (and resize the plot).
        var layoutUpdate = {};
        if (!active.pinned) {
            var ring = gd.data[hi];
            var xa = (ring.xaxis || "x").replace("x", "xaxis");
            var ya = (ring.yaxis || "y").replace("y", "yaxis");
            var fl = gd._fullLayout;
            if (fl && fl[xa] && fl[ya]) {
                layoutUpdate[xa + ".range"] = fl[xa].range.slice();
                layoutUpdate[ya + ".range"] = fl[ya].range.slice();
                active.pinned = true;
            }
        }
        // cliponaxis:false keeps the ring fully drawn at the edge without the
        // range needing to grow to fit it.
        window.Plotly.update(
            gd,
            {x: [[active.x[f]]], y: [[active.y[f]]], cliponaxis: [false]},
            layoutUpdate,
            [hi]
        );
    });
})();
