#!/bin/bash

# We should reserve at least 12GB of RAM for the candidate database

#SBATCH --cpus-per-task=32 --mem-per-cpu=2000
#SBATCH --time=72:00:00
# -- SBATCH --time=01:00:00 --partition=debug
# -- SBATCH --array=0-249
#SBATCH --array=0,101

N_THREADS=4
N_JOBS=8

PROJECTDIR="/scratch/cs/kepaco/bache1/projects/rt_msms_ssvm/"
DB_DIR="$PROJECTDIR/_CASMI_DB/"
DB_FN="DB_LATEST.db"
LOGDIR="$PROJECTDIR/src/ssvm/experiments/bioinf_redo/logs_triton/version_01"
SCRIPTPATH="$PROJECTDIR/src/ssvm/experiments/bioinf_redo/fixedms2__gridsearch.py"

# Load the conda environment
module load miniconda
eval "$(conda shell.bash hook)"
conda activate ssvm_environment

# Create temporary output directory for results on local disk of node
LOCAL_DB_DIR="/dev/shm/$SLURM_JOB_ID"
mkdir "$LOCAL_DB_DIR" || exit 1

# Directory to store joblib-cache files
CACHE_DIR="$LOCAL_DB_DIR/cache"
mkdir "$CACHE_DIR" || exit 1

# Set up trap to remove my results on exit from the local disk
trap "rm -rf $LOCAL_DB_DIR; exit" TERM EXIT

# Copy the DB file to the node's local disk
cp "$DB_DIR/$DB_FN" "$LOCAL_DB_DIR"

NUMBA_NUM_THREADS=$N_THREADS;OMP_NUM_THREADS=$N_THREADS;OPENBLAS_NUM_THREADS=$N_THREADS; \
JOBLIB_MEMORY_CACHE_LOCATION=$CACHE_DIR; \
    srun python $SCRIPTPATH \
    "$SLURM_ARRAY_TASK_ID" \
  --n_jobs="$N_JOBS" \
  --db_fn="$LOCAL_DB_DIR/$DB_FN" \
  --output_dir="$LOGDIR" \
  --n_samples_train=1000 \
  --n_epoch=6 \
  --mol_kernel="minmax_numba" \
  --ms2scorer="MetFrag_2.4.5__8afe4a14" \
  --lloss_fps_mode="count" \
  --n_trees_for_scoring=128 \
  --n_init_per_example=4 \
  --batch_size=16
