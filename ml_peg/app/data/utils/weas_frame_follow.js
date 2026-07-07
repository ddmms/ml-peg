// Keep a scatter's clicked-point ring in sync with a playing WEAS trajectory.
//
// The WEAS viewer (in a srcDoc iframe, see weas.py) posts its current frame to
// the parent on every change. The clicked curve is recorded on
// window.__mlPegActiveTraj by the highlight callback (build_callbacks.py). Here
// we move that scatter's __clicked_point__ ring to the frame's point, where
// point i corresponds to frame i (e.g. an NEB band).
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
        window.Plotly.restyle(gd, {x: [[active.x[f]]], y: [[active.y[f]]]}, [hi]);
    });
})();
