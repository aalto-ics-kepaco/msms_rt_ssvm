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

from scipy.io import loadmat


def read_data(idir):
    # Read fingerprints
    data = loadmat(os.path.join(idir, "data_GNPS.mat"))
    fps = data["fp"].toarray().T

    # Get inchis as molecule identifiers
    inchis = data["inchi"].flatten()
    n_samples = len(inchis)
    assert (n_samples == fps.shape[0])
    inchis = np.array([inchis[i][0] for i in range(n_samples)])

    # Get molecular formulas to identify the candidate sets
    mfs = data["mf"].flatten()
    mfs = np.array([mfs[i][0] for i in range(n_samples)])
    assert len(mfs) == len(inchis)

    # Read PPKr kernel
    K = np.load(os.path.join(idir, "input_kernels", "PPKr.npy"))
    spec_df = pd.read_csv(os.path.join(idir, "spectra.txt"), sep="\t")
    assert np.all(np.equal(spec_df.INCHI.values, inchis))

    # Read indices of the examples that where used for the evaluation (only those have a candidate set)
    ind_eval = sorted(np.genfromtxt(os.path.join(idir, "ind_eval.txt")).astype("int") - 1)
    fps = fps[ind_eval, :]
    inchis = inchis[ind_eval]
    mfs = mfs[ind_eval]
    K = K[np.ix_(ind_eval, ind_eval)]

    # Map of the candidate inchis to the candidate set
    inchi2mf = {inchi: mf for inchi, mf in zip(inchis, mfs)}

    return K, fps, inchis, inchi2mf