#!/bin/bash

#SBATCH --job-name=jc_6
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err
#SBATCH --array=0
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1

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
conda activate py312

# source /mnt/home/abogatskii/ceph/venv/py/bin/activate

# A=(jc_0-{a..z}) --prefix="${A[$SLURM_ARRAY_TASK_ID]}"

# CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun --nnodes=1 --nproc-per-node=4 -m pelican eval --config=../config/48k.yaml --config=../config/jc.yaml --num-channels-scalar=20 --cuda --log-every=1000 --save-every=1000 --stabilizer=so2 --method=spurions --prefix=jc_2-a --no-ram-dataset --datadir=${data_path}/JetClass --num-test=1000000 --num-workers=2 --runs-dir=./runs

CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun --nnodes=1 --nproc-per-node=8 -m pelican train --config=../config/JCxl.yaml --config=../config/jc.yaml --cuda --batch-size=100 --log-every=1000 --save-every=1000 --stabilizer=so2 --method=spurions --prefix=jc_6-a --no-ram-dataset --datadir=${data_path}/JetClass --load --runs-dir=./runs

# CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun --nnodes=1 --nproc-per-node=8 -m pelican eval --config=../config/208k.yaml --config=../config/jc2p.yaml --nobj=90 --cuda --stabilizer=so2 --method=spurions --prefix=jc_5-a --no-ram-dataset --datadir=${data_path}/JetClass --num-workers=4 --runs-dir=./runs

# CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun --nnodes=1 --nproc-per-node=8 -m pelican train --config=../config/208k.yaml --config=../config/jc10p.yaml --nobj=80 --cuda --stabilizer=so2 --method=spurions --prefix=jc_5_10p-a --no-ram-dataset --datadir=${data_path}/JetClass --num-workers=4 --runs-dir=./runs


# CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun --nnodes=1 --nproc-per-node=8 -m pelican train --datadir=../data/JetClass --config=../config/jc.yaml --config=../config/48k.yaml --num-channels-scalar=20 --cuda --log-every=1 --save-every=-1 --stabilizer=so2 --method=spurions --no-ram-dataset --num-train=1024 --num-valid=1024 --num-epoch=5 --prefix=jc_test_1 --num-test=10000 --runs-dir=./runs
