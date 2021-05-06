#!/bin/bash

#SBATCH --partition=batch --time=00:30:00 --mem-per-cpu=1000 --cpus-per-task=4

module load git
module load miniconda

# For conda environment
LOCAL_DIR="/dev/shm/$SLURM_JOB_ID"
LOCAL_CONDA_DIR="$LOCAL_DIR/miniconda/"
mkdir -p "$LOCAL_CONDA_DIR" || exit 1

# Set up trap to remove my results on exit from the local disk
trap "rm -rf $LOCAL_DIR; exit" TERM EXIT

# Clone the SSVM library from the master branch
GIT_REPO_URL=git@github.com:aalto-ics-kepaco/msms_rt_ssvm.git
SSVM_LIB_DIR="$LOCAL_DIR/msms_rt_ssvm"
git clone "$GIT_REPO_URL" "$SSVM_LIB_DIR"

# Create the conda environment based on the SSVM library's requirements
eval "$(conda shell.bash hook)"
cd "$LOCAL_CONDA_DIR" || exit 1
conda create --clone msms_rt_ssvm__base --prefix msms_rt_ssvm__local
conda activate "$LOCAL_CONDA_DIR/msms_rt_ssvm__local"
cd - || exit 1

# Install SSVM library
pip install --no-deps "$SSVM_LIB_DIR"

# Run the kernel performance evaluation
srun python "$SSVM_LIB_DIR/ssvm/kernel_utils.py" $1 --na=1000 --nb=1000 --d=1000 --nrep=10
