#!/usr/bin/env bash
# Run the carbon melt-quench-anneal benchmark as one Slurm array task per
# trajectory, so each task is a single MD on a single GPU.
#
# Submit one composition and one model at a time. With N_RUNS = 5 the array is
# 0-4, indexing the runs within that composition:
#
#   COMPOSITION=C   MODEL=orb-v3-consv-inf-omat sbatch run_melt_quench_anneal.sh
#   COMPOSITION=CHO MODEL=orb-v3-consv-inf-omat sbatch run_melt_quench_anneal.sh
#
# Leaving COMPOSITION empty runs both compositions from one array, in which case
# the array must be 0-9 (0-4 are C, 5-9 are CHO).
#
# A full trajectory takes longer than the 24 h walltime, so it needs more than
# one job. Long stages are split into chunks of at most MAX_CHUNK_TIME_PS, each
# writing its own final structure, and resubmitting is idempotent: completed
# trajectories and completed chunks are skipped, so a task killed at the
# walltime picks up at the last finished chunk. Chain the follow-on jobs with
#
#   jid=$(COMPOSITION=C MODEL=orb-v3-consv-inf-omat sbatch --parsable \
#       run_melt_quench_anneal.sh)
#   COMPOSITION=C MODEL=orb-v3-consv-inf-omat sbatch \
#       --dependency=afterany:"${jid}" run_melt_quench_anneal.sh
#
# afterany (not afterok) is what you want, since the first job is expected to be
# killed at the walltime. Chaining a third job costs nothing if it has no work.

#SBATCH --job-name=mqa
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/mqa-%A_%a.out
#SBATCH --error=slurm-logs/mqa-%A_%a.err

set -euo pipefail

model="${MODEL:-orb-v3-consv-inf-omat}"
composition="${COMPOSITION:-C}"
run_id="${SLURM_ARRAY_TASK_ID:-0}"

# Match to the cluster: module loads, venv activation, scratch paths.
source /home/jh2536/venvs/venv_mace/bin/activate

echo "model=${model} composition=${composition} run_id=${run_id} host=$(hostname)"

ml_peg calc \
    --category carbon \
    --test melt_quench_anneal \
    --models "${model}" \
    --no-run-mock \
    --run-very-slow \
    --composition "${composition}" \
    --run-id "${run_id}"
