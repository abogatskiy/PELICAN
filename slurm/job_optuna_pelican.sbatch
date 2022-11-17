#!/bin/bash

#SBATCH --job-name=optuna
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=32G

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

nvidia-smi

CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate py39
A=(optuna-{a..z})
# python3 ../optuna_pelican_classifier.py --datadir=../data/v0 --cuda --nobj=126 --num-epoch=80 --batch-size=40 --num-train=6000 --num-valid=40000 --no-summarize --lr-decay-type=warm --no-textlog --no-predict --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --sampler=random --pruner=hyperband --storage remote --host worker1031 --port 35719 --study-name=optuna2 --optuna-test
python3 ../optuna_pelican_cov.py --datadir=../data/btW_1_d --target=truth_Pmu_2 --cuda --nobj=48 --nobj-avg=21 --num-epoch=35 --batch-size=128 --num-train=100000 --num-valid=60000 --num-test=100000 --no-summarize --lr-decay-type=cos --no-textlog --no-predict --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --sampler=tpe --pruner=median --storage remote --host worker1026 --port 35719 --study-name=btW1_0 --optuna-test