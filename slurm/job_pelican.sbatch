#!/bin/bash

#SBATCH --job-name=555_100p
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

# python3 ~/ceph/NBodyJetNets/NetworkDesign/scripts/train_lgn.py --datadir=./data/sample_data/v0 --batch-size=50 --ir-safe=True

nvidia-smi

CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate py310
A=(555-{a..z})
S=(167114966692745 167114966696777 167114966696765 167114966700910 167114966701334 167114966700691 167114966700678 167114966698619 167114966698985 167114966698629)
S=(166619173361423 166619173361420 166619173361348 166619173357645 166619173357650)

# Top-tagging dataset
CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun --nnodes=1 --nproc-per-node=4 ../train_pelican_classifier.py --datadir=../data/v0 --num-workers=0 --stabilizer=1 --cuda --nobj=80 --nobj-avg=49 --num-epoch=35 --num-valid=60000 --batch-size=64 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu --factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.005 --reproducible --no-fix-data --no-summarize #--seed="${S[$SLURM_ARRAY_TASK_ID]}" 
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/v0 --cuda --nobj=80 --nobj-avg=49 --num-epoch=35 --num-train=-1 --num-valid=60000 --batch-size=64 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu --factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.005 --reproducible --no-fix-data --no-summarize #--seed="${S[$SLURM_ARRAY_TASK_ID]}" 
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/v0 --cuda --nobj=80 --nobj-avg=49 --num-epoch=70 --num-train=60000 --num-valid=40000 --batch-size=32 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu --factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.025 --reproducible --no-fix-data --no-summarize #--seed="${S[$SLURM_ARRAY_TASK_ID]}"
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/v0 --cuda --nobj=80 --nobj-avg=49 --num-epoch=70 --num-train=12000 --num-valid=40000 --batch-size=32 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu --factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.025 --reproducible --no-fix-data --no-summarize #--seed="${S[$SLURM_ARRAY_TASK_ID]}"
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/v0 --cuda --nobj=80 --nobj-avg=49 --num-epoch=70 --num-train=6000 --num-valid=40000 --batch-size=32 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu --factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.1 --reproducible --no-fix-data --no-summarize #--seed="${S[$SLURM_ARRAY_TASK_ID]}"
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/v0 --cuda --nobj=80 --nobj-avg=49 --num-epoch=140 --num-train=600 --num-valid=1000 --num-test=5000 --batch-size=16 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu --factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.005 --reproducible --no-fix-data --no-summarize #--seed="${S[$SLURM_ARRAY_TASK_ID]}"

# Quark-gluon dataset
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/QG --cuda --nobj=80 --nobj-avg=49 --num-epoch=35 --num-train=-1 --num-valid=60000 --batch-size=512 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu --factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.01 --drop-rate-out=0.01 --weight-decay=0.005 --reproducible --no-fix-data --no-summarize --config=s --config-out=s --irc-safe #--task=eval --testfile=../data/QG/QG_jets_valid_new.h5 #--seed="${S[$SLURM_ARRAY_TASK_ID]}" 
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 -m torchrun --nproc_per_node=2 ../train_pelican_classifier.py --datadir=../data/QG --cuda --nobj=80 --nobj-avg=49 --num-epoch=35 --num-train=-1 --num-valid=60000 --batch-size=512 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu --factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.01 --drop-rate-out=0.01 --weight-decay=0.005 --reproducible --no-fix-data --no-summarize --config=s --config-out=s --irc-safe #--task=eval --testfile=../data/QG/QG_jets_valid_new.h5 #--seed="${S[$SLURM_ARRAY_TASK_ID]}" 

# btW6+qcd dataset
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/btW_6_d/nofilter --cuda --nobj=48 --nobj-avg=21 --num-epoch=35 --num-train=-1 --num-valid=20000 --batch-size=128 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu --factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.005 --reproducible --no-fix-data --no-summarize --config=s --config-out=s --irc-safe #--seed="${S[$SLURM_ARRAY_TASK_ID]}" 

# btW7+qcd
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/btW_7_d --cuda --nobj=60 --nobj-avg=35 --num-epoch=35 --num-train=-1 --num-valid=30000 --batch-size=128 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu --factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.005 --reproducible --no-fix-data --no-summarize --config=M --config-out=M #--irc-safe #--seed="${S[$SLURM_ARRAY_TASK_ID]}" 
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/btW_7 --cuda --nobj=100 --nobj-avg=35 --num-epoch=35 --num-train=-1 --num-valid=30000 --batch-size=128 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu --factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.025 --drop-rate-out=0.025 --weight-decay=0.005 --reproducible --no-fix-data --no-summarize --config=M --config-out=M #--irc-safe #--seed="${S[$SLURM_ARRAY_TASK_ID]}" 


# nano
# CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_nano.py --datadir=../data/v0 --cuda --nobj=80 --nobj-avg=49 --num-epoch=35 --num-train=-1 --num-valid=60000 --batch-size=100 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --lr-decay-type=warm --activation=leakyrelu --no-factorize --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.01 --drop-rate-out=0.01 --weight-decay=0.005 --reproducible --no-fix-data --no-summarize --config=s --config-out=s --no-mlp-out --add-beams --no-activate-agg-out --no-activate-lin --activation=relu --batchnorm=False --task=eval #--seed="${S[$SLURM_ARRAY_TASK_ID]}" 
