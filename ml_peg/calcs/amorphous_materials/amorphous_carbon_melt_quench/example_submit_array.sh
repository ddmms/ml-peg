#!/bin/bash
# Example SLURM GPU array job: one model, one array task per density.
#
# DENSITY_GRID has 5 entries, so the array runs indices 0-4 (one density each).
# Match --array to len(DENSITY_GRID) in calc_amorphous_carbon_melt_quench.py.
#
# Submit with:  sbatch submit_array.sh

#SBATCH --job-name=ml-peg-amorphous-c-melt-quench
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-4

MODEL="mace-mp-0a"

# SLURM_ARRAY_TASK_ID (0..4) selects the density via --density-index.
# Extra args after the ml_peg options are forwarded straight to pytest.
srun ml_peg calc \
    --models "$MODEL" \
    --category amorphous_materials \
    --test amorphous_carbon_melt_quench \
    --run-very-slow \
    --density-index "$SLURM_ARRAY_TASK_ID"
