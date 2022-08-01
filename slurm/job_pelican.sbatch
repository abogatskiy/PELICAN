#!/bin/bash

#SBATCH --job-name=learn173
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err
#SBATCH --array=0-9
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
A=(learn173-{a..z})
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/v0 --cuda --nobj=85 --num-epoch=100 --num-train=6000 --num-valid=30000 --batch-size=32 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --lr-decay-type=warm --config1=M --config2=M --activation=leakyrelu --no-factorize --masked --lr-init=0.0025 --lr-final=5e-6 --scale=0.33 --drop-rate=0.05 --weight-decay=0.01 --reproducible
