#!/bin/bash

#SBATCH --job-name=mlpeg-polymers
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# Set --array to match the chosen POLY_SET (override at submit time with
# sbatch --array=1-N ...):
#   small:   --array=1-7    (7 polymers;  common to all precomputed outputs)
#   medium:  --array=1-30   (30 polymers; common to MACE-OFF23-S, PCFF, vivace)
#   large:   --array=1-85   (85 polymers; common to MACE-OFF23-S, PCFF, vivace)
#   x-large: --array=1-130  (full 130-polymer set)
#SBATCH --array=1-7

POLY_SET="${POLY_SET:-small}"
MODEL="${MODEL:-mace-mp-0a}"
SET_FILE="ml_peg/calcs/molecular_dynamics/polymers/resources/polymer_sets/${POLY_SET}.txt"
POLY_ID="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SET_FILE}")"

ml_peg calc \
  --category molecular_dynamics \
  --test polymers \
  --models "${MODEL}" \
  --poly-id "${POLY_ID}"
