#!/bin/bash

<<<<<<< Updated upstream
#SBATCH --job-name=test_cov
=======
#SBATCH --job-name=btW7d_1_top
>>>>>>> Stashed changes
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err
#SBATCH --array=0
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

nvidia-smi

CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate py310
<<<<<<< Updated upstream
A=(test_cov-{a..z})
=======
A=(btW7d_1_top-{a..z})
>>>>>>> Stashed changes

# train
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=60000 --num-valid=60000 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/v0_signal --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu --num-epoch=35 --batch-size=128 --num-train=60000 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --no-test -v
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW4 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=80 --nobj-avg=39 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_4_d --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1_10x --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=50 --nobj-avg=23 --optim=adamw --lr-decay-type=warm --config=M --config-out=M --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_5_d --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --no-add-beams --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test

# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6/nofilter   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6/nofilter   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=eval --target=contained_daughter_sum_Pmu --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M

# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6/mW_nofilter   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=contained_daughter_sum_Pmu --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test --ir-safe --c-safe
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6/mW_nofilter   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=eval --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=s --config-out=s --test --ir-safe

# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/nofilter   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=contained_daughter_sum_Pmu --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.005 --reproducible --no-fix-data --config=M --config-out=M --no-test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/nofilter   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=eval --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/nofilter   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M

# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/mW_nofilter   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/mW_nofilter   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=eval --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test

# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/mW_contained   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=contained_daughter_sum_Pmu --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=50 --nobj-avg=23 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/mW_contained   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=eval --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=50 --nobj-avg=23 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/mW_nofilter    --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=eval --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=50 --nobj-avg=23 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M


# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/contained   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=contained_daughter_sum_Pmu --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=50 --nobj-avg=23 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/contained   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=50 --nobj-avg=23 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/contained   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=eval --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=50 --nobj-avg=23 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/nofilter    --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=eval --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=50 --nobj-avg=23 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M

# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/nomag   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=contained_daughter_sum_Pmu --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --no-test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/nomag   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=eval --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/nomag   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_3_noB   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=contained_daughter_sum_Pmu --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M


# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_6_d/nofilter --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test



<<<<<<< Updated upstream
# CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun --nnodes=1 --nproc-per-node=4 ../train_pelican_cov.py --datadir=../data/btW_7_d   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_0 --num-epoch=1 --batch-size=128 --num-train=1000 --num-valid=30000 --nobj=60 --nobj-avg=35 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.005 --reproducible --no-fix-data --config=M --config-out=M --no-test
=======
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_7_d   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_0 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=30000 --nobj=60 --nobj-avg=35 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.005 --reproducible --no-fix-data --config=M --config-out=M --no-test
>>>>>>> Stashed changes
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_7   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=contained_daughter_sum_Pmu --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=30000 --nobj=100 --nobj-avg=35 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.005 --reproducible --no-fix-data --config=M --config-out=M --no-test



<<<<<<< Updated upstream
=======


>>>>>>> Stashed changes
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_3_studies/btW_3_clean --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M #--test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_3_studies/btW_3_d_clean --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_3_studies/000_clean --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=eval --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M #--test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_3_studies/001_clean --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_3_studies/010_clean --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=eval --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M #--test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_3_studies/101_clean --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=eval --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M #--test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_3_studies/110_clean --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_3_studies/111_clean --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=eval --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M #--test
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_3_agg   --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --task=train --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --num-valid=60000 --nobj=48 --nobj-avg=28 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --scale=1 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M



# evaluate on btW test datasets
# B=(./model/btW1_1-{a..z}_best.pt)
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1 --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --task=eval --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --loadfile="${B[$SLURM_ARRAY_TASK_ID]}" --testfile=../data/btW_2_mW/m0/events_nodelphes.h5
# B=(./model/btW5d_1-{a..z}_best.pt)
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --no-add-beams --datadir=../data/btW_1_d --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --task=eval --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --loadfile="${B[$SLURM_ARRAY_TASK_ID]}" --testfile=../data/btW_3/events_mini.h5 #2_mW/m0/events_delphes.h5
# B=(./model/btW1_10x_2-{a..z}_best.pt)
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_1_10x --target=truth_Pmu_2 --num-epoch=35 --batch-size=128 --num-train=-1 --nobj=50 --nobj-avg=23 --optim=adamw --lr-decay-type=warm --config=M --config-out=M --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --task=eval --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --loadfile="${B[$SLURM_ARRAY_TASK_ID]}" --testfile=../data/btW_1_10x/test.h5 --test

# evaluate on variable W mass datasets
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_2_mW/m6 --prefix=btW1_22_m6-y --target=truth_Pmu_2 --num-epoch=35 --batch-size=32 --num-train=-1 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --task=eval --loadfile=./model/btW1_22-y_best.pt --testfile=../data/btW_2_mW/m6/events_nodelphes.h5
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_2_mW_d/m0 --prefix=btW5d_2_m0-a --add-beams --target=truth_Pmu_2 --num-epoch=35 --batch-size=32 --num-train=-1 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --config=M --config-out=M --task=eval --loadfile=./model/btW5d_2-a_best.pt --testfile=../data/btW_2_mW/m0/events_delphes.h5

# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW4 --prefix=btW1_22_4-y --target=truth_Pmu_2 --num-epoch=35 --batch-size=32 --num-train=-1 --nobj=74 --nobj-avg=40 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --double --task=eval --loadfile=./model/btW1_22-y_best.pt --testfile=../data/btW4/btW4.h5
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_cov.py --datadir=../data/btW_4_d --prefix=btW4d_1-f --target=truth_Pmu_2 --num-epoch=35 --batch-size=32 --num-train=-1 --nobj=48 --nobj-avg=21 --optim=adamw --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-5 --scale=1 --drop-rate=0.025 --drop-rate-out=0.05 --weight-decay=0.01 --reproducible --no-fix-data --double --task=eval --loadfile=./model/btW4d_1-f_best.pt --prefix=btW4d_1-f_m0 --testfile=../data/btW_2_mW/m0/events_delphes.h5
