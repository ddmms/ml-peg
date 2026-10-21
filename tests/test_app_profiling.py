"""Check opt-in profiling without loading benchmark data."""

from __future__ import annotations

import json
import pstats

from flask import Flask, jsonify, request
import pytest

from ml_peg.app.utils.profiling import (
    enable_request_profiling,
    profile_directory,
    profile_startup,
)


def test_disabled_profiling(monkeypatch):
    """Normal runs do not change the WSGI application."""
    monkeypatch.delenv("ML_PEG_PROFILE_DIR", raising=False)
    assert profile_directory() is None
    server = Flask(__name__)
    original = server.wsgi_app
    enable_request_profiling(server, None)
    with profile_startup(None):
        assert server.wsgi_app == original


def test_startup_profile_survives_failure(tmp_path):
    """A failed build still produces readable diagnostic output."""
    with pytest.raises(ValueError, match="failed build"), profile_startup(tmp_path):
        raise ValueError("failed build")
    assert pstats.Stats(str(tmp_path / "startup.prof")).total_calls > 0
    assert json.loads((tmp_path / "startup.json").read_text())["elapsed_seconds"] > 0


def test_request_profiles_preserve_responses(tmp_path):
    """Callback diagnostics keep sizes and identity without saving input values."""
    server = Flask(__name__)

    @server.post("/_dash-update-component")
    def callback():
        return jsonify(value=request.json["inputs"][0]["value"])

    @server.get("/assets/example.css")
    def asset():
        return "body {}"

    enable_request_profiling(server, tmp_path)
    with server.test_client() as client:
        for _ in range(2):
            response = client.post(
                "/_dash-update-component",
                json={"output": "table.data", "inputs": [{"value": "private-value"}]},
            )
            assert response.json == {"value": "private-value"}
        assert client.get("/assets/example.css").data == b"body {}"

    records = list(tmp_path.glob("*.json"))
    assert len(records) == 2
    assert len(list(tmp_path.glob("*.prof"))) == 2
    for path in records:
        assert "private-value" not in path.read_text()
        record = json.loads(path.read_text())
        assert record["callback"] == "table.data"
        assert record["response_bytes"] == len(response.data)
        assert record["request_bytes"] > 0
        assert record["status"] == 200
        assert record["elapsed_ms"] >= 0
        assert pstats.Stats(str(tmp_path / record["profile"])).total_calls > 0


def test_profile_runs_are_isolated(monkeypatch, tmp_path):
    """Repeated runs cannot overwrite earlier measurements."""
    monkeypatch.setenv("ML_PEG_PROFILE_DIR", str(tmp_path))
    first = profile_directory()
    second = profile_directory()
    assert first != second
    assert first.is_dir()
    assert second.is_dir()
