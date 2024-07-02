#!/bin/bash

#SBATCH --job-name=btwd_0_11i
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=96:00:00
#SBATCH --partition=gpu
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# python3 ~/ceph/NBodyJetNets/NetworkDesign/scripts/train_lgn.py --datadir=./data/sample_data/v0 --batch-size=50 --ir-safe=True

nvidia-smi

CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate py310
A=(btwd_0_11i-{a..z})

# cov
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/nofilter --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=contained_daughter_sum_Pmu --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.005 --reproducible --no-fix-data --config=M --config-out=M --no-test --method=input --stabilizer=11 --yaml=../config/48k.yaml
