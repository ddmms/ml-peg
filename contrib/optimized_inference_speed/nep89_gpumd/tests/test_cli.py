"""Run-ID parsing and strict command-line handling."""

from __future__ import annotations

from pathlib import Path
import re

import gpumd_io
import mlpeg_gpumd
import pytest

POC_DIR = Path(__file__).resolve().parents[1]
GOOD = [
    "--spec",
    "/mlpeg/in/spec.json",
    "--run-id",
    "sio2_quartz-n1125-r1",
    "--precision",
    "mixed",
    "--out",
    "/mlpeg/out",
]


@pytest.mark.parametrize(
    ("run_id", "expected"),
    [
        ("sio2_quartz-n1125-r1", ("sio2_quartz", 1125, 1)),
        ("h2o_liquid-n943296-r3", ("h2o_liquid", 943296, 3)),
        ("a-b-c-n10-r0", ("a-b-c", 10, 0)),
    ],
)
def test_parse_run_id(run_id, expected):
    """Valid run IDs split into material, size and repeat."""
    assert gpumd_io.parse_run_id(run_id) == expected


@pytest.mark.parametrize(
    "run_id", ["sio2_quartz-1125-r1", "sio2_quartz-n1125", "-n10-r1", "x-nA-r1", ""]
)
def test_parse_run_id_rejects(run_id):
    """Malformed run IDs raise."""
    with pytest.raises(ValueError):
        gpumd_io.parse_run_id(run_id)


def test_good_args_parse():
    """The four contract arguments parse."""
    args = mlpeg_gpumd.build_parser().parse_args(GOOD)
    assert args.precision == "mixed"
    assert args.run_id == "sio2_quartz-n1125-r1"


@pytest.mark.parametrize(
    "argv",
    [
        [],
        GOOD[:-2],  # missing --out
        [*GOOD, "--extra", "1"],
        [*GOOD, "positional"],
        [a.replace("--precision", "--prec") for a in GOOD],  # no abbreviations
        [a if a != "mixed" else "fp32" for a in GOOD],  # undeclared mode
        [a if a != "mixed" else "fp64" for a in GOOD],
        [a if a != "mixed" else "bf16" for a in GOOD],
        ["--self-test", *GOOD],
    ],
)
def test_bad_args_exit_nonzero(argv):
    """Anything but the exact contract arguments exits non-zero."""
    with pytest.raises(SystemExit) as exc:
        mlpeg_gpumd.main(argv)
    assert exc.value.code != 0


def test_bad_run_id_exits_nonzero():
    """A malformed run ID is a CLI error, not a result."""
    argv = [a if a != "sio2_quartz-n1125-r1" else "bogus" for a in GOOD]
    assert mlpeg_gpumd.main(argv) != 0


def test_declared_precisions_match_label():
    """The runner's precision list equals the mlpeg.precisions label."""
    text = (POC_DIR / "container.def").read_text()
    label = re.search(r"^\s*mlpeg\.precisions\s+(\S+)\s*$", text, re.MULTILINE)[1]
    assert tuple(label.split(",")) == mlpeg_gpumd.DECLARED_PRECISIONS


def test_required_labels_present():
    """All contract labels are declared in container.def."""
    text = (POC_DIR / "container.def").read_text()
    labels = text.split("%labels")[1].split("%help")[0]
    for key in (
        "mlpeg.contract_version 1.0",
        "mlpeg.model nep89",
        "mlpeg.precisions",
        "mlpeg.weights_sha256",
        "gpumd.version",
        "gpumd.commit",
    ):
        assert key in labels
