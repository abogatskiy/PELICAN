#!/bin/bash

#SBATCH --job-name=arcsinh1
#SBATCH --output=./out/array_%A_%a.out
#SBATCH --error=./err/array_%A_%a.err
#SBATCH --array=0-4
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
A=(arcsinh1-{a..z})
python3 ../train_pelican_classifier.py --datadir=../data/v0 --cuda --nobj=126 --num-epoch=80 --num-train=6000 --num-valid=20000 --batch-size=16 --prefix="${A[$SLURM_ARRAY_TASK_ID]}" --lr-decay-type=warm --config1=SM --config2=M --lr-init=0.0025 --lr-final=1e-5 --scale=0.33 --drop-rate=0.25
