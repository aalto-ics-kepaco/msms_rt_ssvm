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


from sklearn.model_selection import GroupKFold

from ssvm.data_structures import CandidateSetMetIdent
from ssvm.ssvm import StructuredSVMMetIdent
from ssvm.examples.utils import read_data


if __name__ == "__main__":
    # Read in training spectra, fingerprints and candidate set information
    idir = "/home/bach/Documents/doctoral/data/metindent_ismb2016"
    X, fps, mols, mols2cand = read_data(idir)

    # Extract a subset of the full training data
    subset = np.sort(
        np.random.RandomState(1989).choice(len(X), size=np.minimum(len(X), 500).astype(int), replace=True)
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

    svm = StructuredSVMMetIdent(C=64, rs=928, max_n_epochs=10, batch_size=8, stepsize="linesearch",
                                conv_criteria="rel_duality_gap_decay") \
        .fit(X_train, mols_train, candidates=cand, num_init_active_vars_per_seq=2,
             pre_calc_args={"pre_calc_label_losses": True,
                            "pre_calc_L_Ci_S_matrices": True,
                            "pre_calc_L_Ci_matrices": False,
                            "pre_calc_L_matrices": False},
             debug_args={"track_topk_acc": False})

    for score_type in ["predicted", "random", "first_candidate"]:
        print(score_type)
        print("Top-1=%.2f, Top-5=%.2f, Top-10=%.2f, Top-20=%.2f" %
              tuple(svm.score(X_test, mols_test, candidates=cand, score_type=score_type)[[0, 4, 9, 19]]))


