#!/bin/bash

#SBATCH --job-name=btW1_0
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
A=(btW1_0-{a..z})
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --target=truth_Pmu_2 --num-epoch=40 --batch-size=32 --num-train=120000 --num-valid=20000 --nobj=80 --optim=demon --lr-decay-type=exp --config1=M --config2=M --activation=leakyrelu --factorize --masked --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --nobj-avg=40
