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

from timeit import default_timer as timer

from ssvm.data_structures import CandidateSetMetIdent
from ssvm.ssvm import StructuredSVMMetIdent
from ssvm.examples.utils import read_data
from ssvm.development.utils import get_git_revision_short_hash

N_SAMPLES = 100
N_REPS = 3
BATCH_SIZE = 4

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

    # Wrap the candidate sets for easier access
    cand = CandidateSetMetIdent(mols, fps, mols2cand, idir=os.path.join(idir, "candidates"), preload_data=True)

    ts = []
    for r in range(N_REPS):
        start = timer()
        StructuredSVMMetIdent(C=128, rs=928, n_epochs=40, batch_size=BATCH_SIZE, stepsize="linesearch") \
            .fit(X, mols, candidates=cand, num_init_active_vars_per_seq=1)
        ts.append([timer() - start])

    ts = pd.DataFrame(ts, columns=["Time (s)"]).aggregate(func=[np.mean, np.median, np.max])  # type: pd.DataFrame
    print(ts)

    ts.to_csv(os.path.join(ODIR, "run_time__%s__n_samples=%d__n_rep=%d__batch_size=%d.csv" % (
        get_git_revision_short_hash(), N_SAMPLES, N_REPS, BATCH_SIZE)), index=False)
