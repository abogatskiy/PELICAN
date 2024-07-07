#!/bin/bash

#SBATCH --job-name=jc_3
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --partition=gpuxl
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=1
#SBATCH --ntasks-per-node=4

#--cpus-per-task=1


nvidia-smi

CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate py312
# A=(jc_0-{a..z}) --prefix="${A[$SLURM_ARRAY_TASK_ID]}"

# Top-tagging dataset
CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun --nnodes=1 --nproc-per-node=4 ../train_pelican_classifier.py --yaml=../config/jc.yaml --yaml=../config/48k.yaml --cuda --log-every=1000 --save-every=1000 --alpha=5 --stabilizer=so2 --method=spurions --prefix=jc_3-a --RAMdataset 