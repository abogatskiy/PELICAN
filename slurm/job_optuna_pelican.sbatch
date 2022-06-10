#!/bin/bash

#SBATCH --job-name=invariants
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

# python3 ~/ceph/NBodyJetNets/NetworkDesign/scripts/train_lgn.py --datadir=./data/sample_data/v0 --batch-size=50 --ir-safe=True

nvidia-smi

CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate py39
A=(pelican-{a..z})
python3 ../optuna_pelican_classifier.py --datadir=../data/v0 --num-epoch=140 --num-train=6000 --num-valid=60000 --batch-size=128 --dropout --batchnorm=b --no-summarize --prefix="${A[$SLURM_ARRAY_TASK_ID]}"