#!/bin/bash

#SBATCH --job-name=mlpeg-polymers
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=1-10

POLY_SET="${POLY_SET:-small}"
MODEL="${MODEL:-mace-mp-0a}"
SET_FILE="ml_peg/calcs/molecular_dynamics/polymers/resources/polymer_sets/${POLY_SET}.txt"
POLY_ID="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SET_FILE}")"

ml_peg calc \
  --category molecular_dynamics \
  --test polymers \
  --models "${MODEL}" \
  --poly-id "${POLY_ID}"
