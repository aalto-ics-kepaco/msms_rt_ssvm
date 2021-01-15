#!/bin/bash

# We should reserve at least 12GB of RAM for the candidate database

#SBATCH --cpus-per-task=32 --mem-per-cpu=2000
#SBATCH --time=04:00:00
# -- SBATCH --time=01:00:00 --partition=debug
#SBATCH --array=0-2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35,37,38,40,41,43,44
# -- SBATCH --array=4

N_THREADS=4
N_JOBS=8

PROJECTDIR="/scratch/cs/kepaco/bache1/projects/rt_msms_ssvm/"
DB_DIR="$PROJECTDIR/_CASMI_DB/"
DB_FN="DB_LATEST.db"
LOGDIR="$PROJECTDIR/src/ssvm/development/logs_triton/fixedms2_rtw_fix/"
SCRIPTPATH="$PROJECTDIR/src/ssvm/development/ssvm_fixedms2__parameter_grid.py"

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
  --n_jobs="$N_JOBS" \
  --param_tuple_index="$SLURM_ARRAY_TASK_ID" \
  --db_fn="$LOCAL_DB_DIR/$DB_FN" \
  --output_dir="$LOGDIR" \
  --n_samples_train=250 \
  --n_samples_test=250 \
  --n_epoch=6 \
  --mol_kernel="minmax_numba"

# --ms2scorer="IOKR__696a17f3" \
