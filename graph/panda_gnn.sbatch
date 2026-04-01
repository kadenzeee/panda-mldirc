#!/bin/bash

#SBATCH --job-name=gnn
#SBATCH --output=logs/gnn_%A_%a.out
#SBATCH --error=logs/gnn_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# -----------------------------
# Paths
# -----------------------------

INPUT_DIR=/lustre/nyx/panda/markhoff/dirc/mldirc/graph/data/randhits
SIF=/lustre/nyx/panda/markhoff/containers/pytorch-rocm/pytorch_rocm6.3.3_ubuntu24.04_py3.12_pytorch_release_2.4.0.sif
WORKDIR=/lustre/nyx/panda/markhoff/dirc/mldirc/graph
LOGDIR=$WORKDIR/logs

singularity exec --bind $WORKDIR:$WORKDIR $SIF bash -c "
    pip install --quiet torch-geometric &&
    python panda_gnn.py -i $INPUT_DIR -o batch_test --batch
"