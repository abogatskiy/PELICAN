#!/bin/bash

#SBATCH --job-name=jc_1
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err
#SBATCH --array=1
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Where was I?
wwi=$(pwd)
# Create a temp directory dedicated to the processing of this sample. 
tmpDirRoot=${JC_TMPDIRROOT:-/tmp}
wd=$(mktemp -d ${tmpDirRoot}/${USER}_JC_XXXXXXXXXX)
[[ -d ${wd} ]] || { echo "Failed to create temp directory. Exiting." ; exit 1 ; }
function cleanup {
    # Try to go back to where we started from.
    cd ${wwi}
    rm -rf ${wd} || echo "Check for \"$wd\" on $(hostname)."
}
# This runs the cleanup function when the script exits (normally or 
# due to an error). 
trap cleanup EXIT

data_path=$(realpath "${wd}")
rsync -av ../data/JetClass ${data_path}
ls ${data_path}

nvidia-smi

CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate py310
A=(jc_1-{a..z})

# Top-tagging dataset
CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun --nnodes=1 --nproc-per-node=8 ../train_pelican_classifier.py --datadir=${data_path}/JetClass --num-classes=10 --target=signal_type --cuda --nobj=80 --nobj-avg=49 --num-epoch=5 --warmup=0 --cooldown=0 --lr-decay-type=cos --num-train=-1 --num-valid=100000 --batch-size=64 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --optim=adamw --activation=leakyrelu --masked --scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.01 --drop-rate-out=0.01 --weight-decay=0.005 --reproducible --no-fix-data --no-summarize --config=M --config-out=M --factorize --no-RAM-dataset --log-every=100 --save-every=20000 --alpha=10 --load
