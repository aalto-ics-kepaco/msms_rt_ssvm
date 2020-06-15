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
import numpy as np

from scipy.sparse import issparse


def hamming_loss(y: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Function calculating the hamming loss between a single binary vector and a set of binary vectors.

    :param y: array-like, shape = (d,), binary vector (e.g. ground truth fingerprint)

    :param Y: array-like, shape = (n, d), matrix of binary vectors stored row-wise (e.g. candidate fingerprints)

    :return: array-like, shape = (n,), hamming loss values between the y and all vectors in Y
    """
    assert len(y.shape) == 1
    d = y.shape[0]
    assert len(Y.shape) == 2
    assert Y.shape[1] == d
    assert not issparse(y)
    assert not issparse(Y)

    loss = np.sum(y != Y, axis=1) / d

    return loss
