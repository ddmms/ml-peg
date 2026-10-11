"""A small synthetic benchmark data tree with hand-calculable Explorer values."""

from __future__ import annotations

import json
from pathlib import Path

# Model -> (training level of theory, registry id, residual scale).
MODELS = {
    "alpha-1": ("PBE", "alpha-1", 1.0),
    "alpha-1-D3": ("PBE", "alpha-1", 1.1),
    "beta-2": ("PBE", "beta-2", 0.8),
    "gamma-3": ("r2SCAN", "gamma-3", 1.5),
}

# (category, benchmark) -> metric reference level and per-structure error scale.
# Every model's error on structure i is its scale times ``errors[i]``, so the median
# model error is 1.05 * errors[i] and each benchmark's last structure is its hardest.
BENCHMARKS = {
    ("bulk_crystal", "lattice"): {"level": "PBE", "errors": (1, 2, 3, 12)},
    ("surfaces", "adsorption"): {"level": "PBE", "errors": (1, 1, 1, 2)},
    ("conformers", "confs"): {
        "level": "DLPNO-CCSD(T)/CBS",
        "errors": (2, 4, 6, 8),
    },
    ("molecular", "atomisation"): {"level": "r2SCAN", "errors": (1, 2, 4, 30)},
}

# Geometry per structure index: lattice/adsorption are metals, the rest molecules.
ELEMENTS = {
    "lattice": (["Fe"], ["Cu"], ["Fe", "Ni"], ["Cr"]),
    "adsorption": (["Pt", "H"], ["Pt", "O"], ["Pt", "C", "O"], ["Pt", "N"]),
    "confs": (["C", "H"], ["C", "H", "O"], ["C", "H", "N"], ["C", "H", "O"]),
    "atomisation": (["H", "O"], ["C", "O"], ["N", "H"], ["C", "H"]),
}

# Physicality scores (no reference level of theory, so lot_class is "none").
PHYSICALITY_SCORES = {"alpha-1": 0.9, "alpha-1-D3": 0.85, "beta-2": 0.4, "gamma-3": 0.7}


def _residual(model: str, index: int, error: float) -> float:
    """
    Return a model's signed residual on a structure.

    Parameters
    ----------
    model
        Model name.
    index
        Structure index within its benchmark.
    error
        The structure's error scale.

    Returns
    -------
    float
        Signed residual; ``gamma-3`` errs in the opposite direction on odd structures
        so its errors correlate differently with the other models'.
    """
    sign = -1.0 if index % 2 else 1.0
    if model == "gamma-3" and index % 2:
        sign = 1.0
    return sign * MODELS[model][2] * error


def _xyz(elements: list[str]) -> str:
    """
    Write a single-frame xyz block with one atom per element.

    Parameters
    ----------
    elements
        Element symbols.

    Returns
    -------
    str
        The xyz text.
    """
    atoms = "".join(f"{el} {i}.0 0.0 0.0\n" for i, el in enumerate(elements))
    return f"{len(elements)}\nsynthetic\n{atoms}"


def write_synthetic_tree(root: Path) -> list[tuple[str, str]]:
    """
    Write a small benchmark data tree in the layout the app ships.

    Parameters
    ----------
    root
        Directory to write the ``<category>/<benchmark>`` tree into.

    Returns
    -------
    list[tuple[str, str]]
        The ``(category, benchmark)`` pairs written.
    """
    written = []
    for (category, benchmark), spec in BENCHMARKS.items():
        bench = root / category / benchmark
        bench.mkdir(parents=True)
        sids = [f"{benchmark}_{i}" for i in range(len(spec["errors"]))]
        traces = []
        rows = []
        for model, (_, model_id, scale) in MODELS.items():
            residuals = [
                _residual(model, i, err) for i, err in enumerate(spec["errors"])
            ]
            traces.append(
                {
                    "mode": "markers",
                    "name": model,
                    "x": [10.0 + r for r in residuals],
                    "y": [10.0] * len(residuals),
                    "customdata": [[sid] for sid in sids],
                }
            )
            score = round(1 - scale / 2, 4)
            rows.append({"MLIP": model, "id": model_id, "Score": score})
            model_dir = bench / model
            model_dir.mkdir()
            for sid, elements in zip(sids, ELEMENTS[benchmark], strict=True):
                (model_dir / f"{sid}.xyz").write_text(_xyz(elements))
        traces.append({"mode": "lines", "name": "y = x", "x": [0, 1], "y": [0, 1]})
        (bench / "figure_energy.json").write_text(json.dumps({"data": traces}))
        table = {
            "data": rows,
            "metric_levels_of_theory": {"energy": spec["level"]},
            "model_levels_of_theory": {m: v[0] for m, v in MODELS.items()},
        }
        (bench / f"{benchmark}_metrics_table.json").write_text(json.dumps(table))
        written.append((category, benchmark))

    # conformers ships aligned per-structure elements; adsorption only a union.
    (root / "conformers" / "confs" / "info.json").write_text(
        json.dumps(
            {
                "filenames": [f"confs_{i}" for i in range(4)],
                "elements": [list(els) for els in ELEMENTS["confs"]],
            }
        )
    )
    (root / "surfaces" / "adsorption" / "info.json").write_text(
        json.dumps({"elements": ["C", "H", "N", "O", "Pt"]})
    )

    phys = root / "physicality" / "smoothness"
    phys.mkdir(parents=True)
    (phys / "smoothness_metrics_table.json").write_text(
        json.dumps(
            {
                "data": [
                    {"MLIP": m, "id": MODELS[m][1], "Score": s}
                    for m, s in PHYSICALITY_SCORES.items()
                ],
                "model_levels_of_theory": {m: v[0] for m, v in MODELS.items()},
            }
        )
    )
    written.append(("physicality", "smoothness"))
    return written
