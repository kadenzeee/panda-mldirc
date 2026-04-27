#!/bin/bash

#SBATCH --job-name=panda_gnn
#SBATCH --output=logs/panda_gnn_%A_%a.out
#SBATCH --error=logs/panda_gnn_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G

# -----------------------------
# Paths
# -----------------------------

INPUT_DIR=/lustre/nyx/panda/markhoff/dirc/mldirc/graph/data/small_randhits-p-mag-norm
SIF=/lustre/nyx/panda/markhoff/containers/pytorch-rocm/pytorch_rocm6.3.3_ubuntu24.04_py3.12_pytorch_release_2.4.0.sif
WORKDIR=/lustre/nyx/panda/markhoff/dirc/mldirc/graph
LOGDIR=$WORKDIR/logs

singularity exec --bind $WORKDIR:$WORKDIR $SIF bash -c "
    pip install --quiet torch-geometric &&
    python panda_gnn.py -i $INPUT_DIR -o p-mag-norm -nepochs 500 --batch
"