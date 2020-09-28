####
#
# The MIT License (MIT)
#
# Copyright 2019, 2020 Eric Bach <eric.bach@aalto.fi>
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
import scipy.sparse as sp
import itertools as it

from sklearn.metrics.pairwise import manhattan_distances, pairwise_distances
from joblib import delayed, Parallel

"""
Kernel functions here are optimized to work on matrix inputs. 
"""


def check_input(X, Y, datatype=None, shallow=False):
    """
    Function to check whether the two input sets A and B are compatible.

    :param X: array-like, shape = (n_samples_A, n_features), examples A
    :param Y: array-like, shape = (n_samples_B, n_features), examples B
    :param datatype: string, used to specify constraints on the input data (type)
    :param shallow: boolean, indicating whether checks regarding features values, e.g. >= 0, should be skipped.

    :return: X, Y, is_sparse. X is simply passed through. If Y is None, than it
        will be equal X otherwise it is also just passed through. is_sparse is
        a boolean indicating whether X and Y are sparse matrices
    """
    if Y is None:
        Y = X

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Number of features for set A and B must match: %d vs %d." % (
            X.shape[1], Y.shape[1]))

    if isinstance(X, np.ndarray):
        if not isinstance(Y, np.ndarray):
            raise ValueError("Input matrices must be of same type.")
        is_sparse = False
    elif isinstance(X, sp.csr_matrix):
        if not isinstance(Y, sp.csr_matrix):
            raise ValueError("Input matrices must be of same type.")
        is_sparse = True
    else:
        raise ValueError("Input matrices only allowed to be of type 'np.ndarray' or 'scipy.sparse.csr_matrix'.")

    if not shallow:
        if datatype == "binary":
            if is_sparse:
                val_X = np.unique(X.data)
                val_Y = np.unique(Y.data)
            else:
                val_X = np.unique(X)
                val_Y = np.unique(Y)

            if not np.all(np.in1d(val_X, [0, 1])) or not np.all(np.in1d(val_Y, [0, 1])):
                raise ValueError("Input data must be binary.")
        elif datatype == "positive":
            if is_sparse:
                any_neg_X = (X.data < 0).any()
                any_neg_Y = (Y.data < 0).any()
            else:
                any_neg_X = (X < 0).any()
                any_neg_Y = (Y < 0).any()

            if any_neg_X or any_neg_Y:
                raise ValueError("Input data must be positive.")
        elif datatype == "real":
            pass

    return X, Y, is_sparse


def minmax_kernel(X, Y=None, shallow_input_check=False, n_jobs=4):
    """
    Calculates the minmax kernel value for two sets of examples
    represented by their feature vectors.

    :param X: array-like, shape = (n_samples_A, n_features), examples A
    :param Y: array-like, shape = (n_samples_B, n_features), examples B
    :param shallow_input_check: boolean, indicating whether checks regarding features values, e.g. >= 0, should be
        skipped.
    :param n_jobs: scalar, number of jobs used for the kernel calculation from sparse input

    :return: array-like, shape = (n_samples_A, n_samples_B), kernel matrix
             with minmax kernel values:

                K[i,j] = k_mm(A_i, B_j)

    :source: https://github.com/gmum/pykernels/blob/master/pykernels/regular.py
    """
    X, Y, is_sparse = check_input(X, Y, datatype="positive", shallow=shallow_input_check)  # Handle for example Y = None

    if is_sparse:
        K_mm = _min_max_sparse_csr(X, Y, n_jobs=n_jobs)
    else:
        K_mm = _min_max_dense(X, Y)

    return K_mm


def _min_max_dense(X, Y):
    """
    MinMax-Kernel implementation for dense feature vectors.
    """
    n_A, n_B = X.shape[0], Y.shape[0]

    min_K = np.zeros((n_A, n_B))
    max_K = np.zeros((n_A, n_B))

    for s in range(X.shape[1]):  # loop if the feature dimensions
        c_s_A = X[:, s].reshape(-1, 1)
        c_s_B = Y[:, s].reshape(-1, 1)

        # Check for empty features dimension
        if np.all(c_s_A == 0) and np.all(c_s_B == 0):
            continue

        min_K += np.minimum(c_s_A, c_s_B.T)
        max_K += np.maximum(c_s_A, c_s_B.T)

    return min_K / max_K


@delayed
def _min_max_sparse_csr_single_element(x_i, y_j, nonz_idc_x_i, nonz_idc_y_j):
    min_k = 0
    max_k = 0

    # In the indices intersection we need to check min and max
    for s in nonz_idc_x_i & nonz_idc_y_j:
        max_k += np.maximum(x_i[0, s], y_j[0, s])
        min_k += np.minimum(x_i[0, s], y_j[0, s])

    # Indices that appear only in X[i]: minimum is zero, maximum comes from X[i]
    for s in nonz_idc_x_i - nonz_idc_y_j:
        max_k += x_i[0, s]

    # Indices that appear only in Y[j]: minimum is zero, maximum comes from Y[j]
    for s in nonz_idc_y_j - nonz_idc_x_i:
        max_k += y_j[0, s]

    return np.sum(min_k), np.sum(max_k)


def _min_max_sparse_csr(X, Y, n_jobs=1):
    """
    MinMax-Kernel implementation for sparse feature vectors.
    """
    # Find the non-zero indices for each row and put them into set-objects
    n_x, n_y = X.shape[0], Y.shape[0]
    nonz_idc_x = [set() for _ in range(n_x)]
    nonz_idc_y = [set() for _ in range(n_y)]

    for i in range(n_x):
        nonz_idc_x[i].update(X.indices[X.indptr[i]:X.indptr[i + 1]])  # non-zero indices of matrix X in row

    for i in range(n_y):
        nonz_idc_y[i].update(Y.indices[Y.indptr[i]:Y.indptr[i + 1]])  # non-zero indices of matrix X in row

    # Calculate kernel values
    res = Parallel(n_jobs=n_jobs)(_min_max_sparse_csr_single_element(X[i], Y[j], nonz_idc_x[i], nonz_idc_y[j])
                                  for i, j in it.product(range(n_x), range(n_y)))

    min_k, max_k = zip(*res)
    min_k = np.array(min_k).reshape((n_x, n_y))
    max_k = np.array(max_k).reshape((n_x, n_y))

    return min_k / max_k


def tanimoto_kernel(X, Y=None, shallow_input_check=False):
    """
    Tanimoto kernel function

    :param X: array-like, shape=(n_samples_A, n_features), binary feature matrix of set A
    :param Y: array-like, shape=(n_samples_B, n_features), binary feature matrix of set B
        or None, than Y = X
    :param shallow_input_check: boolean, indicating whether checks regarding features values, e.g. >= 0, should be
        skipped.

    :return array-like, shape=(n_samples_A, n_samples_B), tanimoto kernel matrix
    """
    X, Y, is_sparse = check_input(X, Y, datatype="binary", shallow=shallow_input_check)

    XY = X @ Y.T
    XX = X.sum(axis=1).reshape(-1, 1)
    YY = Y.sum(axis=1).reshape(-1, 1)

    K_tan = XY / (XX + YY.T - XY)

    assert (not sp.issparse(K_tan)), "Kernel matrix should not be sparse."

    return K_tan


def generalized_tanimoto_kernel(X, Y=None, shallow_input_check=False, n_jobs=1):
    """
    Generalized tanimoto kernel function

    :param X:
    :param Y:
    :return:
    """
    X, Y, is_sparse = check_input(X, Y, datatype="real", shallow=shallow_input_check)

    if is_sparse:
        raise NotImplementedError("Sparse matrices not supported.")

    XL1 = np.sum(np.abs(X), axis=1)[:, np.newaxis]
    YL1 = np.sum(np.abs(Y), axis=1)[:, np.newaxis]

    XmYL1 = pairwise_distances(X, Y, metric="manhattan", n_jobs=n_jobs)

    K_gtan = (XL1 + YL1.T - XmYL1) / (XL1 + YL1.T + XmYL1)

    return K_gtan


def get_kernel(s1, s2, kernel, raw_data_db, cache_db=None):
    """
    Calculate the kernel values for all elements in S1 x S2

    :param s1:
    :param kernel:
    :param raw_data_db:
    :param cache_db:
    :return:
    """
    K = np.full((len(s1), len(s2)), fill_value=np.nan)
    f_desc = raw_data_db.get_f_desc()

    for idx, (y_1, y_2) in enumerate(it.product(s1, s2)):
        i, j = np.unravel_index(idx, shape=K.shape)
        try:
            K[i, j] = cache_db(y_1, y_2, kernel, f_desc)
        except NotInCacheDBError:
            f_1 = raw_data_db(y_1)
            f_2 = raw_data_db(y_2)
            K[i, j] = kernel(f_1, f_2)

    return K
