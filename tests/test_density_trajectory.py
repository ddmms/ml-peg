"""Test selecting frame groups from density plots."""

from __future__ import annotations

from unittest.mock import Mock

from ml_peg.app.utils import build_callbacks


def test_density_frame_group(monkeypatch):
    """Select the whole bin and keep playback independent of plot point indices."""
    callbacks = []

    def register(*args, **kwargs):
        def capture(func):
            callbacks.append(func)
            return func

        return capture

    renderer = Mock(return_value="viewer")
    highlight = Mock()
    monkeypatch.setattr(build_callbacks, "callback", register)
    monkeypatch.setattr(build_callbacks, "generate_weas_html", renderer)
    monkeypatch.setattr(build_callbacks, "_register_point_highlight", highlight)
    build_callbacks.struct_from_scatter(
        "plot", "viewer", "frames.extxyz", mode="traj", frame_groups=[[2], [0, 2, 7]]
    )
    callbacks[0]({"points": [{"pointNumber": 1}]})
    renderer.assert_called_once_with(
        "frames.extxyz", "traj", 0, frame_indices=[0, 2, 7]
    )
    highlight.assert_called_once_with("plot", follow_frames=False)
    assert callbacks[0](None).children is None
