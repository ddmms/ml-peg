#!/bin/bash
#SBATCH --job-name=mlpeg_A
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --qos=boost_qos_lprod
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --account=IscrB_MAMBO
#SBATCH --array=0-116
#SBATCH --output=/leonardo_work/IscrC_MLP-NPSS/ml-peg-outputs/aluminosilicates_densities/logs/slurm_A_%A_%a.out
#SBATCH --error=/leonardo_work/IscrC_MLP-NPSS/ml-peg-outputs/aluminosilicates_densities/logs/slurm_A_%A_%a.err

source /leonardo/home/userexternal/mbenass1/ml-peg/.venv/bin/activate
cd /leonardo/home/userexternal/mbenass1/ml-peg
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export MLPEG_OUT_PATH=/leonardo/home/userexternal/mbenass1/ml-peg/ml_peg/calcs/molecular_dynamics/aluminosilicates_densities/outputs/

TEST_ID=$(python3 -c "
import os
models = [
    'mace-mp-0a', 'mace-mp-0b3', 'mace-mpa-0', 'mace-omat-0',
    'mace-matpes-r2scan', 'mace-mh-1-omat', 'mace-omol', 'mace-mh-1-omol',
    'mace-polar-1-s', 'mace-polar-1-m', 'mace-polar-1-l',
    'orb-v3-consv-inf-omat', 'pet-mad',
]
compositions = ['albite', 'anorthite', 'sanidine']
replicas = range(3)
combos = [f'{r}-{c}-{m}' for m in models for c in compositions for r in replicas]
print(combos[int(os.environ['SLURM_ARRAY_TASK_ID'])])
")

echo "Task: $SLURM_ARRAY_TASK_ID | Test: $TEST_ID | Nodo: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

pytest -v -s \
    ml_peg/calcs/molecular_dynamics/aluminosilicates_densities/calc_aluminosilicates_densities.py \
    -k "$TEST_ID"

echo "Task $SLURM_ARRAY_TASK_ID completato."
date
