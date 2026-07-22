"""Crystal-symmetry analysis and structure comparison."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import pandas as pd

from ml_peg.analysis.bulk_crystal.geo_opt.schema import (
    ANGLE_TOLERANCE,
    HALL_NUM,
    HALL_SYMBOL,
    INTERNATIONAL_SPG_NAME,
    MATERIAL_ID,
    MAX_PAIR_DIST,
    N_ROT_SYMS,
    N_SYM_OPS,
    N_SYM_OPS_DIFF,
    N_TRANS_SYMS,
    SITE_SYMMETRY_SYMBOLS,
    SPG_NUM,
    SPG_NUM_DIFF,
    STRUCTURE_RMSD_VS_DFT,
    SYMMETRY_FIELDS,
    SYMPREC,
    WYCKOFF_SYMBOLS,
)

ProgressConfig = bool | dict[str, Any]


def _validate_symmetry_parameters(
    symprec: float, angle_tolerance: float | None
) -> None:
    """Validate numerical symmetry tolerances passed to moyopy."""
    if not math.isfinite(symprec) or symprec <= 0:
        raise ValueError(f"symprec must be positive and finite, got {symprec!r}")
    if angle_tolerance is not None and (
        not math.isfinite(angle_tolerance) or angle_tolerance < 0
    ):
        raise ValueError(
            "angle_tolerance must be non-negative and finite when provided, "
            f"got {angle_tolerance!r}"
        )


def _progress_iterator(
    iterable: Any,
    *,
    total: int,
    pbar: ProgressConfig,
    default_description: str,
    leave: bool | None = None,
) -> Any:
    """Optionally wrap an iterable in a configured tqdm progress bar."""
    if not pbar:
        return iterable

    from tqdm.auto import tqdm

    progress_kwargs = dict(pbar) if isinstance(pbar, dict) else {}
    progress_kwargs.setdefault("desc", default_description)
    if leave is not None:
        progress_kwargs.setdefault("leave", leave)
    return tqdm(iterable, total=total, **progress_kwargs)


def get_sym_info_from_structs(
    structures: Mapping[str, object],
    *,
    pbar: ProgressConfig = True,
    symprec: float = 1e-2,
    angle_tolerance: float | None = None,
) -> pd.DataFrame:
    """Compile symmetry information for pymatgen or ASE crystal structures.

    Moyopy's ``angle_tolerance`` is in radians, unlike spglib's degree convention.
    """
    _validate_symmetry_parameters(symprec, angle_tolerance)
    if not isinstance(structures, Mapping):
        raise TypeError(
            f"structures must be a mapping, got {type(structures).__name__}"
        )

    try:
        from ase import Atoms
        import moyopy
        from moyopy.interface import MoyoAdapter
        from pymatgen.core import Structure
    except ImportError as exc:
        raise ImportError(
            "Crystal symmetry analysis requires ASE, pymatgen, and moyopy"
        ) from exc

    results: dict[str, dict[str, object]] = {}
    structure_items = _progress_iterator(
        structures.items(),
        total=len(structures),
        pbar=pbar,
        default_description="Analyzing symmetry",
    )

    for material_id, structure in structure_items:
        if not isinstance(material_id, str) or not material_id.strip():
            raise ValueError(
                f"Material identifiers must be non-empty strings, got {material_id!r}"
            )
        if not isinstance(structure, (Structure, Atoms)):
            raise TypeError(
                "Expected pymatgen Structure or ASE Atoms for "
                f"{material_id!r}, got {type(structure).__name__}"
            )

        moyo_cell = MoyoAdapter.from_py_obj(structure)
        symmetry_dataset = moyopy.MoyoDataset(
            moyo_cell,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        symmetry_operations = symmetry_dataset.operations
        hall_symbol_entry = moyopy.HallSymbolEntry(
            hall_number=symmetry_dataset.hall_number
        )

        results[material_id] = {
            SPG_NUM: symmetry_dataset.number,
            HALL_NUM: hall_symbol_entry.hall_number,
            INTERNATIONAL_SPG_NAME: hall_symbol_entry.hm_short,
            SITE_SYMMETRY_SYMBOLS: symmetry_dataset.site_symmetry_symbols,
            WYCKOFF_SYMBOLS: symmetry_dataset.wyckoffs,
            N_SYM_OPS: symmetry_operations.num_operations,
            N_ROT_SYMS: len(symmetry_operations.rotations),
            N_TRANS_SYMS: len(symmetry_operations.translations),
            HALL_SYMBOL: hall_symbol_entry.hall_symbol,
            SYMPREC: symprec,
            ANGLE_TOLERANCE: angle_tolerance,
        }

    dataframe = pd.DataFrame.from_dict(results, orient="index")
    if dataframe.empty:
        dataframe = pd.DataFrame(columns=SYMMETRY_FIELDS)
    dataframe.index.name = MATERIAL_ID
    return dataframe


def _validate_symmetry_dataframe(
    dataframe: pd.DataFrame, *, dataframe_name: str
) -> None:
    """Validate a symmetry table needed for structure comparison."""
    if dataframe.index.name != MATERIAL_ID:
        raise ValueError(
            f"{dataframe_name}.index.name={dataframe.index.name!r} "
            f"must be {MATERIAL_ID!r}"
        )
    if dataframe.index.has_duplicates:
        duplicate_ids = dataframe.index[dataframe.index.duplicated()].tolist()
        raise ValueError(
            f"{dataframe_name} has duplicate material IDs: {duplicate_ids!r}"
        )
    missing_columns = [
        column for column in (SPG_NUM, N_SYM_OPS) if column not in dataframe
    ]
    if missing_columns:
        raise ValueError(
            f"{dataframe_name} is missing symmetry columns: {missing_columns!r}"
        )


def _matching_structure_ids(
    dataframe: pd.DataFrame,
    structures: Mapping[str, object],
    *,
    source_name: str,
) -> set[str]:
    """Require a structure mapping and symmetry table to contain identical IDs."""
    structure_ids = set(structures)
    symmetry_ids = set(dataframe.index)
    if symmetry_ids != structure_ids:
        raise ValueError(
            f"{source_name} structures and symmetry rows must contain identical IDs; "
            f"only in structures={sorted(structure_ids - symmetry_ids)!r}, "
            f"only in symmetry={sorted(symmetry_ids - structure_ids)!r}"
        )
    return structure_ids


def _as_pymatgen_structure(
    structure: object, *, material_id: str, source_name: str
) -> object:
    """Convert a pymatgen Structure or ASE Atoms for ``StructureMatcher``."""
    try:
        from ase import Atoms
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        raise ImportError("Structure comparison requires ASE and pymatgen") from exc

    if isinstance(structure, Structure):
        return structure
    if isinstance(structure, Atoms):
        return AseAtomsAdaptor.get_structure(structure)
    raise TypeError(
        f"{source_name}[{material_id!r}] must be pymatgen Structure or ASE Atoms, "
        f"got {type(structure).__name__}"
    )


def pred_vs_ref_struct_symmetry(
    df_sym_pred: pd.DataFrame,
    df_sym_ref: pd.DataFrame,
    pred_structs: Mapping[str, object],
    ref_structs: Mapping[str, object],
    *,
    pbar: ProgressConfig = True,
) -> pd.DataFrame:
    """Compare predicted structures and symmetries with references.

    With ``StructureMatcher(stol=1.0, scale=False)``, RMSD and maximum pair
    distance are normalized by the free length per atom.
    Predicted-only IDs remain in the result with missing comparison values, which
    gives them the 1.0 RMSD penalty during aggregation.
    """
    _validate_symmetry_dataframe(df_sym_pred, dataframe_name="df_sym_pred")
    _validate_symmetry_dataframe(df_sym_ref, dataframe_name="df_sym_ref")
    if not isinstance(pred_structs, Mapping) or not isinstance(ref_structs, Mapping):
        raise TypeError("pred_structs and ref_structs must both be mappings")

    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher
    except ImportError as exc:
        raise ImportError("Structure comparison requires pymatgen") from exc

    predicted_ids = _matching_structure_ids(
        df_sym_pred, pred_structs, source_name="Predicted"
    )
    reference_ids = _matching_structure_ids(
        df_sym_ref, ref_structs, source_name="Reference"
    )
    shared_ids = predicted_ids & reference_ids
    if not shared_ids:
        raise ValueError(
            "No shared material IDs between predicted and reference structures: "
            f"predicted={sorted(predicted_ids)!r}, "
            f"reference={sorted(reference_ids)!r}"
        )

    shared_index = df_sym_pred.index[df_sym_pred.index.isin(shared_ids)]
    dataframe_result = df_sym_pred.copy()
    dataframe_result[SPG_NUM_DIFF] = df_sym_pred[SPG_NUM] - df_sym_ref[SPG_NUM]
    dataframe_result[N_SYM_OPS_DIFF] = df_sym_pred[N_SYM_OPS] - df_sym_ref[N_SYM_OPS]
    dataframe_result[[STRUCTURE_RMSD_VS_DFT, MAX_PAIR_DIST]] = float("nan")

    structure_matcher = StructureMatcher(stol=1.0, scale=False)
    material_ids = list(shared_index)
    material_ids = _progress_iterator(
        material_ids,
        total=len(material_ids),
        pbar=pbar,
        default_description="Calculating RMSD",
        leave=False,
    )
    distances_by_id: dict[str, tuple[float, float]] = {}
    for material_id in material_ids:
        predicted_structure = _as_pymatgen_structure(
            pred_structs[material_id],
            material_id=material_id,
            source_name="pred_structs",
        )
        reference_structure = _as_pymatgen_structure(
            ref_structs[material_id],
            material_id=material_id,
            source_name="ref_structs",
        )
        distances_by_id[material_id] = structure_matcher.get_rms_dist(
            predicted_structure, reference_structure
        ) or (
            float("nan"),
            float("nan"),
        )
    dataframe_result.loc[
        list(distances_by_id), [STRUCTURE_RMSD_VS_DFT, MAX_PAIR_DIST]
    ] = list(distances_by_id.values())

    return dataframe_result
