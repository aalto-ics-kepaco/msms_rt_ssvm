####
#
# The MIT License (MIT)
#
# Copyright 2020 Eric Bach <eric.bach@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####
import os
import numpy as np
import pandas as pd
import itertools as it

from sklearn.model_selection import GroupKFold
from timeit import default_timer as timer

from ssvm.data_structures import CandidateSetMetIdent
from ssvm.ssvm import StructuredSVMMetIdent
from ssvm.examples.utils import read_data
from ssvm.development.utils import get_git_revision_short_hash

N_SAMPLES = 200
N_REPS = 3
BATCH_SIZE = 8
STEPSIZE = "linesearch"

ODIR = "./profiling/"

if __name__ == "__main__":

    # Read in training spectra, fingerprints and candidate set information
    idir = "/home/bach/Documents/doctoral/data/metindent_ismb2016"
    X, fps, mols, mols2cand = read_data(idir)

    # Extract a subset of the full training data
    subset = np.sort(
        np.random.RandomState(1989).choice(len(X), size=np.minimum(len(X), N_SAMPLES).astype(int), replace=False)
    )
    X = X[np.ix_(subset, subset)]
    fps = fps[subset]
    mols = mols[subset]
    print("Number of examples:", len(X))

    # Separate a training and test set: 75% / 25%. If a molecular structure appears multiple times, we add them to the
    # same fold.
    train, test = next(GroupKFold(n_splits=4).split(X, groups=mols))
    X_train = X[np.ix_(train, train)]
    X_test = X[np.ix_(test, train)]
    mols_train = mols[train]
    mols_test = mols[test]
    assert not np.any(np.isin(mols_test, mols_train))

    # Wrap the candidate sets for easier access
    cand = CandidateSetMetIdent(mols, fps, mols2cand, idir=os.path.join(idir, "candidates"), preload_data=True)

    ts = []
    for i, (label_losses, L_Ci_S_matrices, L_Ci_matrices, L_matrices) in enumerate(it.product([True, False], repeat=4)):
        print("Config: %d/%d" % (i + 1, 2**4))
        print(label_losses, L_Ci_S_matrices, L_Ci_matrices, L_matrices)

        for r in range(N_REPS):
            start = timer()
            ssvm = StructuredSVMMetIdent(C=128, rs=928, max_n_epochs=5, batch_size=BATCH_SIZE, stepsize=STEPSIZE) \
                .fit(X_train, mols_train, candidates=cand, num_init_active_vars_per_seq=1,
                     pre_calc_args={"pre_calc_label_losses": label_losses,
                                    "pre_calc_L_Ci_S_matrices": L_Ci_S_matrices,
                                    "pre_calc_L_Ci_matrices": L_Ci_matrices,
                                    "pre_calc_L_matrices": L_matrices},
                     debug_args={"track_objectives": False})
            topkacc = ssvm.score(X_test, mols_test, candidates=cand)
            ts.append([label_losses, L_Ci_S_matrices, L_Ci_matrices, L_matrices, timer() - start])

            print(ts[-1])

    ts = pd.DataFrame(ts, columns=["pre_calc_label_losses", "pre_calc_L_Ci_S_matrices", "pre_calc_L_Ci_matrices",
                                   "pre_calc_L_matrices", "Time (s)"]) \
        .groupby(["pre_calc_label_losses", "pre_calc_L_Ci_S_matrices", "pre_calc_L_Ci_matrices", "pre_calc_L_matrices"]) \
        .aggregate(func=[np.median, np.min]) \
        .round(2) \
        .reset_index()  # type: pd.DataFrame
    print(ts)

    ts.to_csv(os.path.join(ODIR, "run_time__%s__n_samples=%d__n_rep=%d__batch_size=%d__stepsize=%s.csv" % (
        get_git_revision_short_hash(), N_SAMPLES, N_REPS, BATCH_SIZE, STEPSIZE)), index=False, sep="|")
