#!/bin/bash

# We should reserve at least 12GB of RAM for the candidate database

#SBATCH --cpus-per-task=32 --mem-per-cpu=5000 --time=24:00:00

# -- SBATCH --cpus-per-task=4 --mem-per-cpu=5000
# -- SBATCH --time=01:00:00 --partition=batch

N_THREADS=2
N_JOBS=16

if [ $# -lt 2 ] ; then
    echo "USAGE: sbatch run_parameter_study.sh MS2SCORER_ID PARAMETER_TO_STUDY"
    exit 1
else
    MS2SCORER_ID=$1
    PARAMETER_TO_STUDY=$2
fi

# -----------------
# Choose MS2-scorer
#------------------
if [ $MS2SCORER_ID = "metfrag" ] ; then
    MS2SCORER="MetFrag_2.4.5__8afe4a14"
elif [ $MS2SCORER_ID = "iokr" ] ; then
    MS2SCORER="IOKR__696a17f3"
else
    echo "Invalid MS2-scorer ID: ${MS2SCORER_ID}. Choices are 'metfrag' and 'iokr'."
    exit 1
fi

# --------------------
# Set up grid to study
# --------------------
if [ $PARAMETER_TO_STUDY = "ssvm_update_direction" ] ; then
    PARAMETER_GRID=("map" "random")
elif [ $PARAMETER_TO_STUDY = "C" ] ; then
    PARAMETER_GRID=("0.125" "0.25" "0.5" "1" "2" "4" "8" "16")
elif [ $PARAMETER_TO_STUDY = "potential_options" ] ; then 
    PARAMETER_GRID=("no_avg__no_log" "no_avg__log" "avg__no_log" "avg__log")
elif [ $PARAMETER_TO_STUDY = "max_n_candidates_train" ] ; then
    PARAMETER_GRID=("3" "5" "10" "25" "50" "100")    
elif [ $PARAMETER_TO_STUDY = "n_epochs" ] ; then
    PARAMETER_GRID=("1" "2" "3" "5" "8")
elif [ $PARAMETER_TO_STUDY = "batch_size" ] ; then 
    PARAMETER_GRID=("4" "8" "16" "32" "64")
elif [ $PARAMETER_TO_STUDY = "label_loss" ] ; then
    PARAMETER_GRID=("hamming" "binary_tanimoto" "count_minmax")
elif [ $PARAMETER_TO_STUDY = "L_train" ] ; then
    PARAMETER_GRID=("4" "8" "16" "24" "32")
else
    echo "Invalid parameter to study: ${PARAMETER_TO_STUDY}."
fi


PROJECTDIR="/scratch/cs/kepaco/bache1/projects/rt_msms_ssvm/"
DB_DIR="$PROJECTDIR/_CASMI_DB/"
DB_FN="DB_LATEST.db"
LOGDIR="$PROJECTDIR/src/ssvm/development/logs_triton/parameter_study/avg__log/$MS2SCORER_ID"
SCRIPTPATH="$PROJECTDIR/src/ssvm/development/ssvm_fixedms2__parameter_study.py"

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
    "$PARAMETER_TO_STUDY" ${PARAMETER_GRID[*]} \
  --n_jobs="$N_JOBS" \
  --db_fn="$LOCAL_DB_DIR/$DB_FN" \
  --output_dir="$LOGDIR" \
  --mol_kernel="minmax" \
  --ms2scorer="$MS2SCORER" \
  --average_node_and_edge_potentials=1 \
  --log_transform_node_potentials=1
