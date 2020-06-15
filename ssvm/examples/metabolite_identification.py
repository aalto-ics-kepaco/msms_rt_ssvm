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
import datetime
import tensorflow.summary as tf_summary

from scipy.io import loadmat
from sklearn.model_selection import GroupKFold, ShuffleSplit
from ssvm.data_structures import CandidateSetMetIdent
from ssvm.ssvm import StructuredSVMMetIdent


def read_data(idir):
    # Read fingerprints
    data = loadmat(os.path.join(idir, "data_GNPS.mat"))
    fps = data["fp"].toarray().T

    # Get inchis as molecule identifiers
    inchis = data["inchi"].flatten()
    n_samples = len(inchis)
    assert (n_samples == fps.shape[0])
    inchis = [inchis[i][0] for i in range(n_samples)]

    # Get molecular formulas to identify the candidate sets
    mfs = data["mf"].flatten()
    mfs = [mfs[i][0] for i in range(n_samples)]
    assert len(mfs) == len(inchis)

    # Read PPKr kernel
    K = np.load(os.path.join(idir, "input_kernels", "PPKr.npy"))
    spec_df = pd.read_csv(os.path.join(idir, "spectra.txt"), sep="\t")
    assert np.all(np.equal(spec_df.INCHI.values, inchis))

    # Read indices of the examples that where used for the evaluation (only those have a candidate set)
    ind_eval = sorted(np.genfromtxt(os.path.join(idir, "ind_eval.txt")).astype("int") - 1)
    fps = fps[ind_eval, :]
    # TODO: Check, is correct order selected?
    inchis = [inchi for i, inchi in enumerate(inchis) if i in ind_eval]
    mfs = [mf for i, mf in enumerate(mfs) if i in ind_eval]
    K = K[np.ix_(ind_eval, ind_eval)]

    # Map of the candidate inchis to the candidate set
    inchi2mf = {inchi: mf for inchi, mf in zip(inchis, mfs)}

    return K, fps, np.array(inchis), inchi2mf


if __name__ == "__main__":
    from timeit import default_timer as timer

    # Read in training spectra, fingerprints and candidate set information
    idir = "/home/bach/Documents/doctoral/data/metindent_ismb2016"  # "/run/media/bach/EVO500GB/data/metident_ismb2016"
    X, fps, mols, mols2cand = read_data(idir)

    # Get a smaller subset
    _, subset = next(ShuffleSplit(n_splits=1, test_size=0.075, random_state=1989).split(X))
    X = X[np.ix_(subset, subset)]
    fps = fps[subset]
    mols = mols[subset]
    print("Total number of examples:", len(mols))

    # Wrap the candidate sets for easier access
    cand = CandidateSetMetIdent(mols, fps, mols2cand, idir=os.path.join(idir, "candidates"), preload_data=True)

    # Get train test split
    train, test = next(GroupKFold(n_splits=4).split(X, groups=mols))
    # train, test = next(ShuffleSplit(n_splits=1, test_size=0.25, random_state=20201).split(X))
    # np.savetxt(os.path.join(idir, "train_set.txt"), train)
    # np.savetxt(os.path.join(idir, "test_set.txt"), test)

    X_train = X[np.ix_(train, train)]
    X_test = X[np.ix_(test, train)]
    mols_train = mols[train]
    mols_test = mols[test]
    assert not np.any(np.isin(mols_test, mols_train))

    start = timer()

    # Tensorflow training summary log-file
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/' + current_time + '/train'
    train_summary_writer = tf_summary.create_file_writer(train_log_dir)

    svm = StructuredSVMMetIdent(C=300, rs=102, n_epochs=100, batch_size=1) \
        .fit(X_train, mols_train, candidates=cand, num_init_active_vars_per_seq=3,
             train_summary_writer=train_summary_writer)

    for score_type in ["predicted", "random", "first_candidate"]:
        print(score_type)
        print("Top-1=%.2f, Top-5=%.2f, Top-10=%.2f, Top-20=%.2f" %
              tuple(svm.score(X_test, mols_test, candidates=cand, score_type=score_type)[[0, 4, 9, 19]]))

    end = timer()
    print("version 03: %fs" % (end - start))
