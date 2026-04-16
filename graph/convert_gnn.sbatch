#!/bin/bash

#SBATCH --job-name=gnnclconv
#SBATCH --output=logs/gnnclconv_%A_%a.out
#SBATCH --error=logs/gnnclconv_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=long
#SBATCH --mem=32G
#SBATCH --array=0-99

# -----------------------------
# Paths
# -----------------------------

INPUT_DIR=/lustre/nyx/panda/markhoff/dirc/prtdirc/macro/randhitsout5
OUTPUT_DIR=/lustre/nyx/panda/markhoff/dirc/mldirc/graph/data/randhits
SIF=/lustre/nyx/panda/markhoff/containers/root/root_6.30.04-ubuntu22.04.sif
WORKDIR=/lustre/nyx/panda/markhoff/dirc/mldirc/graph
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
    python convert_gnn.py \
        -i $INPUT_FILE \
        -o $OUTPUT_FILE
