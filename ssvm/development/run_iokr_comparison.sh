#!/bin/bash

# Number of samples is first argument
N_SAMPLES=$1

CMDSTR="iokr_param = struct('center', 1, 'mkl', 'unimkl');\
ky_param = struct('type', 'tanimoto');\
select_param = struct('cv_type', 'loocv', 'lambda', [0.01, 0.1, 1, 10, 100]);\
datadir = '/home/bach/Documents/doctoral/data/metindent_ismb2016/';\
resdir  = '/home/bach/Documents/doctoral/projects/rt_msms_ssvm/src/ssvm/development/reproducible/n_samples=$N_SAMPLES/';\
addpath('/home/bach/Documents/doctoral/projects/rt_msms_ssvm/iokr');\
addpath('/home/bach/Documents/doctoral/projects/rt_msms_ssvm/iokr/general_functions');\
run_IOKR_GNPS__comparison_to_ssvm(datadir, resdir, iokr_param, select_param, ky_param);\
exit;"

echo "$CMDSTR"

matlab -nojvm -r "$CMDSTR"