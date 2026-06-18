"""Run calculations for elasticity benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator
from ase.io import write as ase_write
from matcalc._base import PropCalc
from matcalc._elasticity import ElasticityCalc
from matcalc.benchmark import Benchmark
from matcalc.units import eVA3ToGPa
import numpy as np
from pymatgen.core.elasticity.elastic import ElasticTensor
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pytest

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
OUT_PATH = Path(__file__).parent / "outputs"


def get_crystal_system(struct: Structure) -> str:
    """
    Determine a crystal-system label.

    Refine the crystal-system label returned by pymatgen using
    the crystallographic point group to distinguish symmetry
    subclasses relevant for elastic constants.

    Parameters
    ----------
    struct
        Structure to analyse.

    Returns
    -------
    str
        Crystal-system label.
    """
    sga = SpacegroupAnalyzer(struct)
    crystal_system = sga.get_crystal_system().lower()
    point_group = sga.get_point_group_symbol()
    if crystal_system == "tetragonal":
        return "tetragonal_7" if point_group in {"4", "-4", "4/m"} else "tetragonal_6"

    if crystal_system in {"trigonal", "rhombohedral"}:
        return "trigonal_7" if point_group in {"3", "-3"} else "trigonal_6"

    return crystal_system


class CustomElasticityBenchmark(Benchmark):
    """
    Extend the matcalc Benchmark to output full elastic tensors.

    This allows extraction and comparison of elasticity tensors
    for materials.

    Parameters
    ----------
    index_name
        Name of the index used to identify benchmark records (default is "mp_id").
    benchmark_name
        Name or path of the benchmark dataset,
        default is "mp-binary-pbe-elasticity-2025.1.json.gz".
    **kwargs
        Additional keyword arguments forwarded to the parent class.
    """

    def __init__(
        self,
        index_name: str = "mp_id",
        benchmark_name: str | Path = "mp-binary-pbe-elasticity-2025.1.json.gz",
        **kwargs,
    ) -> None:
        """
        Initialize the elasticity benchmark.

        Parameters
        ----------
        index_name
            Name of the index used to identify benchmark records (default is "mp_id").
        benchmark_name
            Name or path of the benchmark dataset,
            default is "mp-binary-pbe-elasticity-2025.1.json.gz".
        **kwargs
            Additional keyword arguments forwarded to the parent class.
        """
        kwargs.setdefault(
            "properties", ("bulk_modulus_vrh", "shear_modulus_vrh", "elasticity_tensor")
        )
        kwargs.setdefault(
            "property_rename_map", {"bulk_modulus": "K", "shear_modulus": "G"}
        )
        kwargs.setdefault("other_fields", ("formula",))
        super().__init__(
            benchmark_name,
            index_name=index_name,
            **kwargs,
        )
        for struct, row in zip(self.structures, self.ground_truth, strict=False):
            row["crystal_system_DFT"] = get_crystal_system(struct)

    def get_prop_calc(self, calculator: str | Calculator, **kwargs: Any) -> PropCalc:
        """
        Create a property calculation object.

        Parameters
        ----------
        calculator
            Calculator used to evaluate elastic properties.
        **kwargs
            Additional configuration options.

        Returns
        -------
        PropCalc
            Configured property calculation object.
        """
        kwargs.setdefault("fmax", 0.05)
        return ElasticityCalc(calculator, **kwargs)

    def process_result(self, result: dict | None, model_name: str) -> dict:  # noqa: SS05
        """
        Extract data from a benchmark result.

        Parameters
        ----------
        result
            Result dictionary containing elastic properties.
        model_name
            Name of the model used to label output keys.

        Returns
        -------
        dict
            Dictionary of processed elastic properties.
        """
        return {
            f"bulk_modulus_vrh_{model_name}": (
                result["bulk_modulus_vrh"] * eVA3ToGPa
                if result is not None
                else float("nan")
            ),
            f"shear_modulus_vrh_{model_name}": (
                result["shear_modulus_vrh"] * eVA3ToGPa
                if result is not None
                else float("nan")
            ),
            f"elastic_tensor_{model_name}": (
                result["elastic_tensor"] * eVA3ToGPa
                if result is not None
                else float("nan")
            ),
            f"crystal_system_{model_name}": (
                get_crystal_system(result["final_structure"])
                if result is not None
                else np.nan
            ),
        }


def elastic_tensor_to_voigt(tensor: Any) -> np.ndarray | float:
    """
    Convert elastic tensor-like objects to 6x6 Voigt form.

    For some models, matcalc returns the tensor as a plain
    numpy.ndarray with shape (3, 3, 3, 3), not as a pymatgen
    tensor object. NumPy arrays do not have .voigt, so the
    final dataframe conversion fails, so this conversion must handle both forms.

    Parameters
    ----------
    tensor
        Elastic tensor-like object to convert. This may be None, NaN, an object
        with a voigt attribute, a dictionary containing tensor data, a 6x6 array,
        or a 3x3x3x3 array.

    Returns
    -------
    numpy.ndarray or float
        Tensor in 6x6 Voigt form, or NaN if ``tensor`` is missing or
        cannot be converted.
    """
    if tensor is None or (isinstance(tensor, float) and np.isnan(tensor)):
        return np.nan

    # pymatgen Tensor / ElasticTensor case
    if hasattr(tensor, "voigt"):
        return np.asarray(tensor.voigt)

    # Reference data and JSON checkpoints can store tensors as dictionaries.
    if isinstance(tensor, dict):
        if "raw" in tensor:
            tensor = tensor["raw"]
        elif "data" in tensor:
            tensor = tensor["data"]
        else:
            return np.nan

    try:
        arr = np.asarray(tensor, dtype=float)
    except (TypeError, ValueError):
        return np.nan

    if arr.shape == (6, 6):
        return arr
    if arr.shape == (3, 3, 3, 3):
        return np.asarray(ElasticTensor(arr).voigt)
    return np.nan


def run_elasticity_benchmark(
    *,
    calc,
    model_name: str,
    out_dir: Path,
    n_jobs: int = 1,
    norm_strains: tuple[float, float, float, float] = (-0.1, -0.05, 0.05, 0.1),
    shear_strains: tuple[float, float, float, float] = (-0.02, -0.01, 0.01, 0.02),
    relax_structure: bool = True,
    relax_deformed_structures: bool = True,
    use_checkpoint: bool = True,
    n_materials: int | None = None,
    fmax: float = 0.05,
) -> None:
    """
    Run the elasticity benchmark and write results to CSV.

    Parameters
    ----------
    calc
        ASE calculator for evaluating structures.
    model_name
        Name of MLIP model.
    out_dir
        Directory to write per-model outputs.
    n_jobs
        Number of parallel workers for the benchmark.
    norm_strains
        Normal strains to apply.
    shear_strains
        Shear strains to apply.
    relax_structure
        Relax the equilibrium structure before deformations.
    relax_deformed_structures
        Relax each strained structure.
    use_checkpoint
        Whether to write intermediate checkpoints.
    n_materials
        Number of materials to sample. None means all materials.
    fmax
        Force threshold for structural relaxations.
    """
    benchmark = CustomElasticityBenchmark(
        n_samples=n_materials,
        seed=2025,
        fmax=fmax,
        relax_structure=relax_structure,
        relax_deformed_structures=relax_deformed_structures,
        norm_strains=norm_strains,
        shear_strains=shear_strains,
        benchmark_name="mp-pbe-elasticity-2025.3.json.gz",
        properties=("bulk_modulus_vrh", "shear_modulus_vrh", "elastic_tensor"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = None
    if use_checkpoint:
        checkpoint_dir = out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{model_name}_checkpoint.json"

    results = benchmark.run(
        calc,
        model_name,
        n_jobs=n_jobs,
        checkpoint_file=checkpoint_file if checkpoint_file else None,
        checkpoint_freq=100,
        delete_checkpoint_on_finish=False,
        include_full_results=True,
    )

    # Save relaxed structures to extxyz for visualisation.
    # Mock calc produces zero stress so final_structure is never populated;
    # fall back to reference structures so info.json can record elements later.
    mock_ref_map = (
        {
            gt_row[benchmark.index_name]: struct
            for struct, gt_row in zip(
                benchmark.structures, benchmark.ground_truth, strict=True
            )
        }
        if model_name == "mock"
        else {}
    )
    atoms_list = []
    # ⚡ Bolt: Use to_dict("records") instead of iterrows() for faster iteration
    # while preserving support for dynamic column names
    for row in results.to_dict("records"):
        struct = row.get("final_structure")
        if not isinstance(struct, Structure):
            struct = mock_ref_map.get(row[benchmark.index_name])
        if not isinstance(struct, Structure):
            continue
        atoms = AseAtomsAdaptor.get_atoms(struct).copy()
        atoms.calc = None
        atoms.info = {"mp_id": row[benchmark.index_name]}
        atoms_list.append(atoms)
    if atoms_list:
        ase_write(
            out_dir / "relaxed_structures.extxyz",
            atoms_list,
            columns=["symbols", "positions"],
            write_results=False,
        )

    # Drop structure column before CSV processing
    results = results.drop(columns=["final_structure"], errors="ignore")

    for col in results.columns:
        if col.startswith("elastic_tensor_"):
            results[col] = results[col].apply(elastic_tensor_to_voigt)

    results.to_csv(out_dir / "moduli_results.csv", index=False)


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_elasticity(mlip: tuple[str, Any]) -> None:
    """
    Run the elasticity benchmark for a single model.

    Parameters
    ----------
    mlip
        Model entry containing name and object capable of providing a calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="low")
    run_elasticity_benchmark(
        calc=calc,
        model_name=model_name,
        out_dir=OUT_PATH / model_name,
    )
