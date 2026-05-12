#!/bin/bash

#SBATCH --job-name=mlpeg-polymers
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=1-10

POLY_SET="${POLY_SET:-small}"
MODEL="${MODEL:-mace-mp-0a}"

ml_peg calc \
  --category molecular_dynamics \
  --test polymers \
  --models "${MODEL}" \
  --poly-set "${POLY_SET}" \
  --poly-index "${SLURM_ARRAY_TASK_ID}"
