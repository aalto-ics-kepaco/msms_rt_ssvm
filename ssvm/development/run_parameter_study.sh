#!/bin/bash

# We should reserve at least 12GB of RAM for the candidate database

#SBATCH --cpus-per-task=32 --mem-per-cpu=5000
#SBATCH --time=24:00:00
# -- SBATCH --time=01:00:00 --partition=debug

N_THREADS=4
N_JOBS=8

PROJECTDIR="/scratch/cs/kepaco/bache1/projects/rt_msms_ssvm/"
DB_DIR="$PROJECTDIR/_CASMI_DB/"
DB_FN="DB_LATEST.db"
LOGDIR="$PROJECTDIR/src/ssvm/experiments/development/logs_triton/parameter_study"
SCRIPTPATH="$PROJECTDIR/src/ssvm/experiments/development/ssvm_fixedms2__parameter_study.py"

# Load the conda environment
module load miniconda
eval "$(conda shell.bash hook)"
conda activate ssvm_environment

# Create temporary output directory for results on local disk of node
LOCAL_DB_DIR="/dev/shm/$SLURM_JOB_ID"
mkdir "$LOCAL_DB_DIR" || exit 1

# Set up trap to remove my results on exit from the local disk
trap "rm -rf $LOCAL_DB_DIR; exit" TERM EXIT

# Copy the DB file to the node's local disk
cp "$DB_DIR/$DB_FN" "$LOCAL_DB_DIR"

NUMBA_NUM_THREADS=$N_THREADS;OMP_NUM_THREADS=$N_THREADS;OPENBLAS_NUM_THREADS=$N_THREADS; \
    srun python $SCRIPTPATH \
    "label_loss" "binary_tanimoto" "count_minmax" "hamming" \
  --n_jobs="$N_JOBS" \
  --db_fn="$LOCAL_DB_DIR/$DB_FN" \
  --output_dir="$LOGDIR" \
  --mol_kernel="minmax_numba"
