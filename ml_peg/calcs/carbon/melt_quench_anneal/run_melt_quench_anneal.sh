#!/usr/bin/env bash
# Run the carbon melt-quench-anneal benchmark as one Slurm array task per
# trajectory. RUNS in calc_melt_quench_anneal.py defines the array bounds:
# 10 trajectories (C and CHO, rho = 1 g cm^-3, 5 runs each), indices 0-9.
#
# Submit one model at a time:
#   MODEL=orb-v3-consv-inf-omat sbatch run_melt_quench_anneal.sh
#
# Long stages are split into chunks of at most MAX_CHUNK_TIME_PS, each writing
# its own final structure. Resubmitting the array is safe and cheap: completed
# trajectories and completed chunks are skipped, so a task killed at the
# walltime resumes at the last finished chunk.

#SBATCH --job-name=mqa
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=slurm-logs/mqa-%A_%a.out
#SBATCH --error=slurm-logs/mqa-%A_%a.err

set -euo pipefail

model="${MODEL:-orb-v3-consv-inf-omat}"
run_id="${SLURM_ARRAY_TASK_ID:-0}"

# Match to the cluster: module loads, venv activation, scratch paths.
source /home/jh2536/venvs/venv_mace/bin/activate

echo "model=${model} run_id=${run_id} host=$(hostname)"

ml_peg calc \
    --category carbon \
    --test melt_quench_anneal \
    --models "${model}" \
    --no-run-mock \
    --run-very-slow \
    --run-id "${run_id}"
