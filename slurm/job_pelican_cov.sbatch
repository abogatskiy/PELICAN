#!/bin/bash

#SBATCH --job-name=btW1_6
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err
#SBATCH --array=2-3
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
A=(btW1_6-{a..z})

# train
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --config1=M --config2=M --activation=leakyrelu --factorize --masked --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.005 --reproducible --no-fix-data --double
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1_d --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=200 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --config1=M --config2=M --activation=leakyrelu --factorize --masked --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.005 --reproducible --no-fix-data --double

# evaluate on btW test datasets
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1 --target=truth_Pmu_2 --num-epoch=35 --batch-size=32 --num-train=-1 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --config1=M --config2=M --activation=leakyrelu --factorize --masked --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --double --task=eval --prefix=btW1_1-b
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1_d --target=truth_Pmu_2 --num-epoch=35 --batch-size=32 --num-train=-1 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --config1=M --config2=M --activation=leakyrelu --factorize --masked --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --double --task=eval --prefix=btW1d_1-c

# evaluate on variable W mass datasets
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1_mW/m6 --prefix=btW1_1-b_cluster_m6 --target=truth_Pmu_2 --num-epoch=35 --batch-size=32 --num-train=-1 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --config1=M --config2=M --activation=leakyrelu --factorize --masked --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --double --task=eval --loadfile=./model/btW1_1-b_best.pt
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1_mW_d/m1 --prefix=btW1d_1-c_cluster_m1 --target=truth_Pmu_2 --num-epoch=35 --batch-size=32 --num-train=-1 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --config1=M --config2=M --activation=leakyrelu --factorize --masked --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --double --task=eval --loadfile=./model/btW1d_1-c_best.pt
