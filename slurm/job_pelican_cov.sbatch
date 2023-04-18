#!/bin/bash

#SBATCH --job-name=btW4d_2
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err
#SBATCH --array=5-8
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
conda activate py310
A=(btW4d_2-{a..z})

# train
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config1=M --config2=M --test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1_d --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config1=M --config2=M --test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW4 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=80 --nobj-avg=39 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config1=M --config2=M --test
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --no-add-beam --datadir=../data/btW_4_d --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config1=M --config2=M --test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1_10x --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=50 --nobj-avg=23 --optim=adamw --lr-decay-type=warm --config1=M --config2=M --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_5_d --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --no-add-beams --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config1=M --config2=M --test

# evaluate on btW test datasets
# B=(./model/btW1_1-{a..z}_best.pt)
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1 --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --task=eval --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --loadfile="${B[$SLURM_ARRAY_TASK_ID]}" --testfile=../data/btW_2_mW/m0/events_nodelphes.h5
# B=(./model/btW5d_1-{a..z}_best.pt)
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --no-add-beams --datadir=../data/btW_1_d --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --task=eval --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --loadfile="${B[$SLURM_ARRAY_TASK_ID]}" --testfile=../data/btW_3/events_mini.h5 #2_mW/m0/events_delphes.h5
# B=(./model/btW1_10x_2-{a..z}_best.pt)
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1_10x --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --nobj=50 --nobj-avg=23 --optim=adamw --lr-decay-type=warm --config1=M --config2=M --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --task=eval --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --loadfile="${B[$SLURM_ARRAY_TASK_ID]}" --testfile=../data/btW_1_10x/test.h5 --test

# evaluate on variable W mass datasets
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_2_mW/m6 --prefix=btW1_22_m6-y --target=truth_Pmu_2 --num-epoch=35 --batch-size=32 --num-train=-1 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --task=eval --loadfile=./model/btW1_22-y_best.pt --testfile=../data/btW_2_mW/m6/events_nodelphes.h5
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_2_mW_d/m0 --prefix=btW5d_2_m0-a --add-beams --target=truth_Pmu_2 --num-epoch=35 --batch-size=32 --num-train=-1 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --config1=M --config2=M --task=eval --loadfile=./model/btW5d_2-a_best.pt --testfile=../data/btW_2_mW/m0/events_delphes.h5

# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW4 --prefix=btW1_22_4-y --target=truth_Pmu_2 --num-epoch=35 --batch-size=32 --num-train=-1 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --double --task=eval --loadfile=./model/btW1_22-y_best.pt --testfile=../data/btW4/btW4.h5
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_4_d --prefix=btW4d_1-e --target=truth_Pmu_2 --num-epoch=35 --batch-size=32 --num-train=-1 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --double --task=eval --loadfile=./model/btW1d_23-f_best.pt --testfile=../data/btW_2_mW/m0/events_delphes.h5
