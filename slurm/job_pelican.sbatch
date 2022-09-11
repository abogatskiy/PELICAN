#!/bin/bash

#SBATCH --job-name=learn361_100p
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err
#SBATCH --array=0-7
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
A=(learn361_100p-{a..z})
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/v0 --cuda --nobj=85 --num-epoch=35 --num-train=-1 --num-valid=100000 --batch-size=100 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --lr-decay-type=warm --config1=M --config2=M --activation=leakyrelu --factorize --masked --lr-init=0.0025 --lr-final=1e-7 --scale=1. --beam-mass=0. --drop-rate=0.01 --drop-rate-out=0.01 --weight-decay=0.01 --reproducible --no-fix-data
