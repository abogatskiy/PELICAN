#!/bin/bash

#SBATCH --job-name=436_05p
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
A=(436_05p-{a..z})

CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/v0 --cuda --nobj=80 --nobj-avg=49 --num-epoch=90 --num-train=6000 --num-valid=60000 --batch-size=32 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu --factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.15 --reproducible --no-fix-data
