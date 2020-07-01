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
import datetime
import tensorflow.summary as tf_summary

from sklearn.model_selection import GroupKFold, ShuffleSplit

from ssvm.data_structures import CandidateSetMetIdent
from ssvm.ssvm import StructuredSVMMetIdent
from ssvm.examples.utils import read_data


if __name__ == "__main__":
    from timeit import default_timer as timer

    # Read in training spectra, fingerprints and candidate set information
    idir = "/home/bach/Documents/doctoral/data/metindent_ismb2016"  # "/run/media/bach/EVO500GB/data/metident_ismb2016"
    X, fps, mols, mols2cand = read_data(idir)

    # Get a smaller subset
    _, subset = next(ShuffleSplit(n_splits=1, test_size=0.1, random_state=1989).split(X))
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

    # Tensorflow training summary log-file
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/' + current_time + '/train'
    train_summary_writer = None  # tf_summary.create_file_writer(train_log_dir)

    start = timer()

    svm = StructuredSVMMetIdent(C=128, rs=928, n_epochs=3, batch_size=8, stepsize="linesearch") \
        .fit(X_train, mols_train, candidates=cand, num_init_active_vars_per_seq=1,
             train_summary_writer=train_summary_writer)

    end = timer()
    print("version 03: %fs" % (end - start))

    for score_type in ["predicted", "random", "first_candidate"]:
        print(score_type)
        print("Top-1=%.2f, Top-5=%.2f, Top-10=%.2f, Top-20=%.2f" %
              tuple(svm.score(X_test, mols_test, candidates=cand, score_type=score_type)[[0, 4, 9, 19]]))


