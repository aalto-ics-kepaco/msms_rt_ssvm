#!/bin/bash

#SBATCH --cpus-per-task=8 --mem-per-cpu=5000
#SBATCH --time=12:00:00
#SBATCH --array=0-71

module load annaconda

PROJECTDIR="/scratch/cs/kepaco/bache1/projects/rt_msms_ssvm/"
DATADIR="$PROJECTDIR/_ISMB2016_DATA/"
LOGDIR="$PROJECTDIR/src/ssvm/development/logs_triton/"
SCRIPTPATH="$PROJECTDIR/src/ssvm/development/ssvm_metident__parameter_grid.py"

srun python $SCRIPTPATH --param_tuple_index="$SLURM_ARRAY_TASK_ID" \
  --input_data_dir="$DATADIR" \
  --output_dir="$LOGDIR" \
  --n_samples=800