#!/bin/bash

#SBATCH --job-name=jc_2
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err
#SBATCH --array=0
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH -C ib-h100p
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
conda activate py310
# A=(jc_0-{a..z}) --prefix="${A[$SLURM_ARRAY_TASK_ID]}"

CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun --nnodes=1 --nproc-per-node=8 ../train_pelican_classifier.py --yaml=../config/jc.yaml --yaml=../config/48k.yaml --num-channels-scalar=10 --cuda --log-every=1000 --save-every=5000 --alpha=5 --stabilizer=so2 --method=spurions --prefix=jc_2-a --no-RAMdataset --datadir=${data_path}/JetClass --load

# CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun --nnodes=1 --nproc-per-node=8 ../train_pelican_classifier.py --yaml=../config/jc.yaml --yaml=../config/208k.yaml --num-channels-scalar=20 --cuda --log-every=1000 --save-every=5000 --alpha=5 --stabilizer=so2 --method=spurions --prefix=jc_3-b --no-RAMdataset --datadir=${data_path}/JetClass --load

# CUBLAS_WORKSPACE_CONFIG=:16:8 torchrun --nnodes=1 --nproc-per-node=8 ../train_pelican_classifier.py --datadir=${data_path}/JetClass --yaml=../config/jc.yaml --yaml=../config/48k.yaml --num-channels-scalar=20 --cuda --log-every=1 --save-every=-1 --alpha=5 --stabilizer=so2 --method=spurions --no-RAMdataset --num-train=1024 --prefix=jc_test
