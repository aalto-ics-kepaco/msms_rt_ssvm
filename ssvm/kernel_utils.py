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

from numba import jit, prange, guvectorize, float64, int64
from sklearn.metrics.pairwise import manhattan_distances, pairwise_distances
from joblib import delayed, Parallel
from scipy.spatial._distance_wrap import cdist_cityblock_double_wrap

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


def minmax_kernel(X, Y=None, shallow_input_check=False, n_jobs=4, use_numba=True):
    """
    Calculates the minmax kernel value for two sets of examples
    represented by their feature vectors.

    :param X: array-like, shape = (n_samples_A, n_features), examples A
    :param Y: array-like, shape = (n_samples_B, n_features), examples B
    :param shallow_input_check: boolean, indicating whether checks regarding features values, e.g. >= 0, should be
        skipped.
    :param n_jobs: scalar, number of jobs used for the kernel calculation from sparse input
    :param use_numba:

    :return: array-like, shape = (n_samples_A, n_samples_B), kernel matrix
             with minmax kernel values:

                K[i,j] = k_mm(A_i, B_j)

    :source: https://github.com/gmum/pykernels/blob/master/pykernels/regular.py
    """
    X, Y, is_sparse = check_input(X, Y, datatype="positive", shallow=shallow_input_check)  # Handle for example Y = None

    if is_sparse:
        K_mm = _min_max_sparse_csr(X, Y, n_jobs=n_jobs)
    else:
        if use_numba:
            K_mm = minmax_kernel_1__numba(X, Y)
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


def generalized_tanimoto_kernel(X, Y=None, shallow_input_check=False, n_jobs=4):
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


def generalized_tanimoto_kernel_OLD(X, Y=None, shallow_input_check=False):
    """
    Generalized tanimoto kernel function

    :param X:
    :param Y:
    :return:
    """
    X, Y, is_sparse = check_input(X, Y, datatype="real", shallow=shallow_input_check)

    if is_sparse:
        raise NotImplementedError("Sparse matrices not supported.")

    XL1 = np.linalg.norm(X, ord=1, axis=1)[:, np.newaxis]
    YL1 = np.linalg.norm(Y, ord=1, axis=1)[:, np.newaxis]
    XL1pYL1t = XL1 + YL1.T

    XmYL1 = manhattan_distances(X, Y)

    K_gtan = (XL1pYL1t - XmYL1) / (XL1pYL1t + XmYL1)

    return K_gtan


def generalized_tanimoto_kernel_FAST(X, Y):
    """
    Generalized tanimoto kernel function

    :param X:
    :param Y:
    :return:
    """
    XL1 = np.core.add.reduce(np.abs(X), axis=1)[:, np.newaxis]
    YL1 = np.core.add.reduce(np.abs(Y), axis=1)[:, np.newaxis]
    XmYL1 = np.empty((X.shape[0], Y.shape[0]), dtype=np.double)
    cdist_cityblock_double_wrap(np.array(X, order="c", dtype=np.double),
                                np.array(Y, order="c", dtype=np.double), XmYL1)

    XL1pYL1t = XL1 + YL1.T
    K_gtan = (XL1pYL1t - XmYL1) / (XL1pYL1t + XmYL1)

    return K_gtan


# --------------------------
# Implementation using Numba
# --------------------------
@guvectorize([(int64[:], int64[:], float64[:])], '(d),(d)->()', nopython=True, target="parallel")
def _minmax__ufunc(x, y, res):
    s = np.sum(x) + np.sum(y)
    t = np.sum(np.abs(x - y))
    res[0] = (s - t) / (s + t)


def minmax_kernel__ufunc(X, Y):
    return _minmax__ufunc(X.astype(np.int)[:, np.newaxis, :], Y.astype(np.int)[np.newaxis, :, :])


@jit(nopython=True, parallel=True)
def _minmax_kernel_1__numba(X, Y, n_Y):
    """
    MinMax-Kernel implementation for dense feature vectors.
    """
    XL1 = np.sum(X, axis=1).reshape(-1, 1)
    YL1 = np.sum(Y, axis=1).reshape(-1, 1)

    XL1pYL1t = XL1 + YL1.T

    XmYL1 = np.zeros_like(XL1pYL1t)
    for i in prange(n_Y):
        XmYL1[:, i] = np.sum(np.abs(X - Y[i]), axis=1)

    K_mm = (XL1pYL1t - XmYL1) / (XL1pYL1t + XmYL1)

    return K_mm


def minmax_kernel_1__numba(X, Y):
    return _minmax_kernel_1__numba(X, Y, len(Y))


def run_time():
    n_rep = 25

    for n_A, n_B, d in [(100, 5000, 307), (1000, 1000, 307), (100, 5000, 7500), (1000, 1000, 5000)]:
        print(_run_time(n_A, n_B, d, n_rep))


def _run_time(n_A, n_B, d, n_rep):
    # Create random data
    X_A = np.random.RandomState(5943).randint(0, 200, size=(n_A, d))
    X_B = np.random.RandomState(842).randint(0, 200, size=(n_B, d))

    # Define functions to test
    fun_list = [("alg_1__numba", minmax_kernel_1__numba), ("ufunc", minmax_kernel__ufunc)]

    # Compile using numba
    K_ref = generalized_tanimoto_kernel_FAST(X_A, X_B)
    print("Compile ... ", end="")
    for fun_name, fun in fun_list:
        print(fun_name, end="; ", flush=True)
        np.testing.assert_allclose(K_ref, fun(X_A, X_B))
    print()

    # Run timing
    df = []
    for fun_name, fun in fun_list + [("cdist__FAST", generalized_tanimoto_kernel_FAST)]:
        print("RUN - %s" % fun_name)
        for _ in range(n_rep):
            start = time.time()
            fun(X_A, X_B)
            df.append([fun_name, n_A, n_B, d, (time.time() - start)])

    df = pd.DataFrame(df, columns=["function", "n_A", "n_B", "d", "time"]) \
        .groupby(["function", "n_A", "n_B", "d"]) \
        .aggregate(np.mean) \
        .reset_index()

    return df


def profiling(fun, n_A=100, n_B=5000, d=307):
    X_A = np.random.RandomState(5943).randint(0, 200, size=(n_A, d))
    X_B = np.random.RandomState(842).randint(0, 200, size=(n_B, d))

    fun(np.random.randint(0, 200, size=(10, 2)), np.random.randint(0, 200, size=(10, 2)))

    for _ in range(50):
        fun(X_A, X_B)


if __name__ == "__main__":
    import time
    import pandas as pd

    run_time()
