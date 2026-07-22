"""Aggregate geometry-optimization structure metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_peg.analysis.bulk_crystal.geo_opt.schema import (
    N_SYM_OPS_DIFF,
    SPG_NUM_DIFF,
    STRUCTURE_RMSD_VS_DFT,
)

N_SYM_OPS_MAE = "n_sym_ops_mae"
SYMMETRY_DECREASE = "symmetry_decrease"
SYMMETRY_MATCH = "symmetry_match"
SYMMETRY_INCREASE = "symmetry_increase"
N_STRUCTURES = "n_structures"

REQUIRED_METRIC_COLUMNS = (
    STRUCTURE_RMSD_VS_DFT,
    SPG_NUM_DIFF,
    N_SYM_OPS_DIFF,
)


def calc_geo_opt_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, float | int]:
    """Calculate aggregate geometry-optimization metrics.

    Invalid RMSDs receive the ``stol=1.0`` penalty. Symmetry fractions use only
    valid space-group rows, and a match requires an unchanged space-group number.
    """
    missing_columns = [
        column for column in REQUIRED_METRIC_COLUMNS if column not in dataframe
    ]
    if missing_columns:
        raise ValueError(f"Missing geo-opt metric columns: {missing_columns!r}")

    spg_num_diff = dataframe[SPG_NUM_DIFF]
    n_sym_ops_diff = dataframe[N_SYM_OPS_DIFF]

    # Exclude rows where symmetry detection failed.
    valid_symmetry_mask = spg_num_diff.notna()
    n_valid_symmetry = int(valid_symmetry_mask.sum())

    # Penalize invalid RMSDs with StructureMatcher's stol.
    numeric_rmsd = pd.to_numeric(dataframe[STRUCTURE_RMSD_VS_DFT], errors="coerce")
    valid_rmsd = numeric_rmsd.ge(0) & np.isfinite(numeric_rmsd)
    mean_rmsd = numeric_rmsd.where(valid_rmsd, 1.0).mean()

    valid_n_sym_ops_diff = n_sym_ops_diff[valid_symmetry_mask]
    n_sym_ops_mae = valid_n_sym_ops_diff.abs().mean()

    changed_space_group_mask = (spg_num_diff != 0) & valid_symmetry_mask
    symmetry_decreased_mask = (n_sym_ops_diff < 0) & changed_space_group_mask
    symmetry_increased_mask = (n_sym_ops_diff > 0) & changed_space_group_mask
    # A changed space group with the same operation count belongs to no category.
    symmetry_matched_mask = ~changed_space_group_mask & valid_symmetry_mask

    if n_valid_symmetry:
        symmetry_decrease = float(symmetry_decreased_mask.sum() / n_valid_symmetry)
        symmetry_match = float(symmetry_matched_mask.sum() / n_valid_symmetry)
        symmetry_increase = float(symmetry_increased_mask.sum() / n_valid_symmetry)
    else:
        symmetry_decrease = float("nan")
        symmetry_match = float("nan")
        symmetry_increase = float("nan")

    return {
        STRUCTURE_RMSD_VS_DFT: float(mean_rmsd),
        N_SYM_OPS_MAE: float(n_sym_ops_mae),
        SYMMETRY_DECREASE: symmetry_decrease,
        SYMMETRY_MATCH: symmetry_match,
        SYMMETRY_INCREASE: symmetry_increase,
        N_STRUCTURES: n_valid_symmetry,
    }
