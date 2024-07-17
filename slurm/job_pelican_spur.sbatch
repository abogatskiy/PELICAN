#!/bin/bash

#SBATCH --job-name=v0_3_1s
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

nvidia-smi

CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate py310
A=(v0_3_1s-{a..z})

#spurions
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/nofilter   \
#     --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=contained_daughter_sum_Pmu --num-epoch=35 --batch-size=128 \
#     --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm \
#     --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.005 \
#     --reproducible --no-fix-data --config=M --config-out=M --no-test\
#     --no-add-beams --stabilizer=so2 --method=spurions --seed=$SLURM_ARRAY_TASK_ID

# QG
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/QG \
#     --nobj=80 --nobj-avg=49 --num-epoch=35 --num-train=-1 --num-valid=60000 --batch-size=512 \
#     --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu \
#     --factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.01 --drop-rate-out=0.01 --weight-decay=0. \
#     --reproducible --no-fix-data --no-summarize --config=M --config-out=M \
#     --stabilizer=1 --method=input --seed=$SLURM_ARRAY_TASK_ID \
#     #--task=eval --testfile=../data/QG/QG_jets_valid_new.h5 

# toptag
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/v0 \
#     --nobj=80 --nobj-avg=49 --num-epoch=35 --num-train=-1 --num-valid=60000 --batch-size=100 \
#     --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu \
#     --factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.01 --drop-rate-out=0.01 --weight-decay=0.005 \
#     --reproducible --no-fix-data --no-summarize --config=M --config-out=M \
#     --stabilizer=1 --method=spurions --seed=$SLURM_ARRAY_TASK_ID \
#     #--task=eval --testfile=../data/QG/QG_jets_valid_new.h5 
