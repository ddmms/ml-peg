"""Exercise benchmark hooks through real pytest subprocesses."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture
def benchmark_session(pytester: pytest.Pytester) -> pytest.Pytester:
    """Use temporary benchmark trees without running scientific calculations."""
    pytester.makeconftest(
        """
from pathlib import Path
import ml_peg.analysis as analysis
import ml_peg.pytest_plugin as plugin

plugin.CALCS_ROOT = Path(__file__).parent / "calcs"
plugin.ANALYSIS_ROOT = Path(__file__).parent / "analysis"
analysis.ANALYSIS_ROOT = plugin.ANALYSIS_ROOT
"""
    )
    return pytester


@pytest.mark.parametrize("tree,prefix", [("calcs", "calc"), ("analysis", "analyse")])
def test_reporting_and_filtering(benchmark_session, tree, prefix):
    """Report executed scripts, preserve unrelated tests, and honour selection."""
    session = benchmark_session
    paths = []
    cases = {
        "ran": "",
        "failed": "",
        "skipped": '@pytest.mark.skip(reason="unavailable")',
        "slow": "@pytest.mark.slow",
        "very_slow": "@pytest.mark.very_slow",
        "filtered": '@pytest.mark.framework("other")',
    }
    for name, marker in cases.items():
        path = session.path / tree / "category" / name / f"{prefix}_{name}.py"
        path.parent.mkdir(parents=True)
        framework = "" if name == "filtered" else '@pytest.mark.framework("mlip_audit")'
        path.write_text(
            "import pytest\nfrom ml_peg import models\n"
            'assert models.current_models == "mace-mp-0a"\n'
            "assert models.run_mock and models.mock_only\n"
            f"{framework}\n{marker}\ndef test_benchmark():\n"
            f"    assert {name != 'failed'}\n"
        )
        paths.append(str(path))
    unrelated = session.makepyfile(
        test_unrelated="""
import pytest

@pytest.mark.slow
@pytest.mark.framework("other")
def test_unrelated():
    pass
"""
    )

    result = session.runpytest_subprocess(
        "-p",
        "ml_peg.pytest_plugin",
        *paths,
        str(unrelated),
        "--run-mock",
        "--mock-only",
        "--models",
        "mace-mp-0a",
        "--framework",
        "mlip_audit",
        "-q",
    )

    result.assert_outcomes(passed=2, failed=1, skipped=3, deselected=1)
    output = result.stdout.str()
    if tree == "analysis":
        assert "CITATION GUIDANCE" not in output
        return
    guidance = output.split("CITATION GUIDANCE", 1)[1]
    assert "category/ran" in guidance
    assert "category/failed" in guidance
    assert "MLIP Audit" in guidance
    assert "MODELS (" not in guidance
    for name in ("skipped", "slow", "very_slow", "filtered"):
        assert f"category/{name}" not in guidance


def test_slow_flags_and_source_fallback(benchmark_session):
    """Explicit slow flags work with both plugin discovery and a source fallback."""
    session = benchmark_session
    with (session.path / "conftest.py").open("a") as handle:
        handle.write('\npytest_plugins = ["ml_peg.pytest_plugin"]\n')
    path = session.path / "calcs" / "category" / "slow" / "calc_slow.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        """
import pytest

@pytest.mark.slow
def test_slow():
    pass

@pytest.mark.very_slow
def test_very_slow():
    pass
"""
    )
    result = session.runpytest_subprocess(
        "-p",
        "ml_peg.pytest_plugin",
        str(path),
        "--run-mock",
        "--mock-only",
        "--run-slow",
        "--run-very-slow",
        "-q",
    )
    result.assert_outcomes(passed=2)
    assert result.stdout.str().count("CITATION GUIDANCE") == 1


def test_unrelated_session_has_no_citation_output(pytester):
    """Installing the plugin does not skip unrelated slow tests or print credits."""
    pytester.makepyfile(
        """
import pytest

@pytest.mark.slow
@pytest.mark.very_slow
def test_external():
    pass
"""
    )
    result = pytester.runpytest_subprocess("-p", "ml_peg.pytest_plugin", "-q")
    result.assert_outcomes(passed=1)
    assert "CITATION GUIDANCE" not in result.stdout.str()


@pytest.mark.parametrize("tree,prefix", [("calcs", "calc"), ("analysis", "analyse")])
@pytest.mark.parametrize(
    "benchmark", ["conformers/Folmsbee", "molecular_reactions/tautomers"]
)
def test_audit_benchmark_attribution(benchmark_session, tree, prefix, benchmark):
    """Exercise real benchmark markers without importing scientific dependencies."""
    session = benchmark_session
    name = benchmark.split("/")[-1]
    relative_path = Path(tree) / benchmark / f"{prefix}_{name}.py"
    source = Path(__file__).parents[1] / "ml_peg" / relative_path
    module = ast.parse(source.read_text())
    markers = [
        ast.unparse(decorator)
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and ast.unparse(decorator.func) == "pytest.mark.framework"
    ]
    path = session.path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text(
        "import pytest\n"
        + "".join(f"@{marker}\n" for marker in markers)
        + "def test_benchmark():\n    pass\n"
    )
    result = session.runpytest_subprocess(
        "-p",
        "ml_peg.pytest_plugin",
        str(path),
        "--run-mock",
        "--mock-only",
        "--framework",
        "mlip_audit",
        "-q",
    )
    result.assert_outcomes(passed=1)
    output = result.stdout.str()
    if tree == "analysis":
        assert "CITATION GUIDANCE" not in output
        return
    guidance = output.split("CITATION GUIDANCE", 1)[1]
    assert "SOURCE FRAMEWORKS (1)" in guidance
    assert "MLIP Audit" in guidance
    assert "Leon Wehrhan" in guidance
    assert "https://arxiv.org/abs/2511.20487" in guidance


@pytest.mark.parametrize("include_framework", [False, True])
def test_paper_tags_are_not_framework_citations(benchmark_session, include_framework):
    """Only external framework tags produce framework attribution in calc output."""
    session = benchmark_session
    path = session.path / "calcs" / "category" / "mixed" / "calc_mixed.py"
    path.parent.mkdir(parents=True)
    ids = ["ml_peg", "mace-multihead", "mace-polar-1"]
    if include_framework:
        ids.extend(["mlip_audit", "mlip_audit"])
    path.write_text(
        "import pytest\n"
        f"@pytest.mark.framework(*{ids!r})\n"
        "def test_benchmark():\n    pass\n"
    )
    result = session.runpytest_subprocess(
        "-p",
        "ml_peg.pytest_plugin",
        str(path),
        "--run-mock",
        "--mock-only",
        "-q",
    )
    result.assert_outcomes(passed=1)
    guidance = result.stdout.str().split("CITATION GUIDANCE", 1)[1]
    assert ("SOURCE FRAMEWORKS (1)" in guidance) == include_framework
    assert ("MLIP Audit" in guidance) == include_framework
    assert "Multihead Cross Learning" not in guidance
    assert "MACE-POLAR-1" not in guidance
