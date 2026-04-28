#!/bin/bash

# Takes a directory containing .root files from the prtdirc simulation and converts each file individually into a .pkl file suitable for gnn training.

#SBATCH --job-name=fwlmgnnclconv
#SBATCH --output=logs/fwlmgnnclconv_%A_%a.out
#SBATCH --error=logs/fwlmgnnclconv_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --array=0-99

# -----------------------------
# Paths
# -----------------------------

INPUT_DIR=/lustre/nyx/panda/markhoff/dirc/prtdirc/macro/randhitsout5
OUTPUT_DIR=/lustre/nyx/panda/markhoff/dirc/mldirc/graph/fwlmgnn/data/randhits
SIF=/lustre/nyx/panda/markhoff/containers/root/root_6.30.04-ubuntu22.04.sif
WORKDIR=/lustre/nyx/panda/markhoff/dirc/mldirc/graph/fwlmgnn
LOGDIR=$WORKDIR/logs

mkdir -p $LOGDIR
mkdir -p $OUTPUT_DIR
cd $WORKDIR

# -----------------------------
# Get file list
# -----------------------------

FILES=($INPUT_DIR/*.root)

NFILES=${#FILES[@]}

if [ $SLURM_ARRAY_TASK_ID -ge $NFILES ]; then
    echo "Array index exceeds number of ROOT files ($NFILES)"
    exit 0
fi

INPUT_FILE=${FILES[$SLURM_ARRAY_TASK_ID]}
OUTPUT_FILE=$OUTPUT_DIR/batch_${SLURM_ARRAY_TASK_ID}.pkl

echo "[INFO] Processing file: $INPUT_FILE"

# -----------------------------
# Run conversion (1 file/job)
# -----------------------------

singularity exec \
    $SIF \
    python convert_fwlmgnn.py \
        -i $INPUT_FILE \
        -o $OUTPUT_FILE
