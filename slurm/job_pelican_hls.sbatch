#!/bin/bash

#SBATCH --job-name=hls_5
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH -C h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=32G

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

nvidia-smi

CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate py310
A=(hls_5-{a..z})
S=(167114966692745 167114966696777 167114966696765 167114966700910 167114966701334 167114966700691 167114966700678 167114966698619 167114966698985 167114966698629)
S=(166619173361423 166619173361420 166619173361348 166619173357645 166619173357650)

# HLS dataset
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 ../train_pelican_classifier.py --datadir=../data/HLS4ML_100 --num-classes=5 --target=jet_tag --cuda --nobj=100 --nobj-avg=49 --num-epoch=35 --lr-decay-type=warm --num-train=-1 --num-valid=-1 --batch-size=100 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --activation=leakyrelu --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.01 --drop-rate-out=0.01 --weight-decay=0.005 --reproducible --no-fix-data --no-summarize --config=M --config-out=M --factorize --task=eval #--seed="${S[$SLURM_ARRAY_TASK_ID]}" 
