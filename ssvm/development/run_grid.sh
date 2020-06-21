#!/bin/bash

#SBATCH --cpus-per-task=18 --mem-per-cpu=3000
#SBATCH --time=56:00:00
# -- SBATCH --time=01:00:00 --partition=debug
# -- SBATCH --array=0-59
#SBATCH --array=32,34,35

module load anaconda

PROJECTDIR="/scratch/cs/kepaco/bache1/projects/rt_msms_ssvm/"
DATADIR="$PROJECTDIR/_ISMB2016_DATA/"
LOGDIR="$PROJECTDIR/src/ssvm/development/logs_triton/"
SCRIPTPATH="$PROJECTDIR/src/ssvm/development/ssvm_metident__parameter_grid.py"

srun python $SCRIPTPATH --param_tuple_index="$SLURM_ARRAY_TASK_ID" \
  --input_data_dir="$DATADIR" \
  --output_dir="$LOGDIR" \
  --n_samples=1000 \
  --n_epoch=1000
