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


def minmax_kernel(X, Y=None, shallow_input_check=True, n_jobs=1, use_numba=False):
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
            K_mm = _min_max_dense_jit(X, Y)
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


def tanimoto_kernel(X, Y=None, shallow_input_check=True):
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

    K_tan = tanimoto_kernel_FAST(X, Y)

    assert (not sp.issparse(K_tan)), "Kernel matrix should not be sparse."

    return K_tan


def tanimoto_kernel_FAST(X, Y):
    """
    Tanimoto kernel function
    """
    XY = X @ Y.T
    XX = np.sum(X, axis=1).reshape(-1, 1)
    YY = np.sum(Y, axis=1).reshape(-1, 1)

    K_tan = XY / (XX + YY.T - XY)

    return K_tan


def generalized_tanimoto_kernel(X, Y=None, shallow_input_check=True, n_jobs=1):
    """
    Generalized tanimoto kernel function

    :param X: array-like, shape=(n_samples_A, n_features), binary feature matrix of set A
    :param Y: array-like, shape=(n_samples_B, n_features), binary feature matrix of set B
        or None, than Y = X
    :param shallow_input_check: boolean, indicating whether checks regarding features values, e.g. >= 0, should be
        skipped.
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


# ===========================
# Implementations using Numba
# ===========================

def _min_max_dense_ufunc(X, Y):
    """
    Min-Max Kernel using reduction
    """
    return _minmax__ufunc(X.astype(np.int)[:, np.newaxis, :], Y.astype(np.int)[np.newaxis, :, :])


def _min_max_dense_ufunc_int(X, Y):
    """
    Min-Max Kernel using reduction
    """
    return _minmax__ufunc(X[:, np.newaxis, :], Y[np.newaxis, :, :])


@guvectorize([(int64[:], int64[:], float64[:])], '(d),(d)->()', nopython=True, target="parallel")
def _minmax__ufunc(x, y, res):
    s = np.sum(x) + np.sum(y)
    t = np.sum(np.abs(x - y))
    res[0] = (s - t) / (s + t)

# ---------------------------


@jit(nopython=True, nogil=True)
def _minmax_nogil(X, Y):
    XL1 = np.sum(X, axis=1).reshape(-1, 1)
    YL1 = np.sum(Y, axis=1).reshape(1, -1)

    XL1pYL1t = XL1 + YL1

    XmYL1 = np.abs(np.reshape(X, (X.shape[0], 1, X.shape[1])) - np.reshape(Y, (1, Y.shape[0], Y.shape[1])))
    XmYL1 = np.sum(XmYL1, axis=2)

    K_mm = (XL1pYL1t - XmYL1) / (XL1pYL1t + XmYL1)

    return K_mm


def _min_max_dense_jit(X, Y):
    """
    MinMax-Kernel implementation for dense feature vectors using Jit
    """
    return _minmax_jit(X, Y, len(Y))


# @jit('float64[:,:](int64[:,:],int64[:,:],int64)', nopython=True, parallel=True)
@jit(nopython=True, parallel=True)
def _minmax_jit(X, Y, n_Y):
    XL1 = np.sum(X, axis=1).reshape(-1, 1)
    YL1 = np.sum(Y, axis=1).reshape(1, -1)

    XL1pYL1t = XL1 + YL1

    XmYL1 = np.zeros_like(XL1pYL1t)
    for i in prange(n_Y):
        XmYL1[:, i] = np.sum(np.abs(X - Y[i]), axis=1)

    K_mm = (XL1pYL1t - XmYL1)
    K_mm = K_mm / (XL1pYL1t + XmYL1)  # type: np.ndarray

    return K_mm

# ---------------------------


def run_time__practical_dimensions():
    n_rep = 25

    for n_A, n_B, d in [(30, 800, 7600), (300, 800, 7600)]:
        print(_run_time(n_A, n_B, d, n_rep, float))

#              function  n_A  n_B     d      time
# 0        minmax_dense   30  800  7600  1.384670
# 1       minmax_gentan   30  800  7600  0.169328
# 2  minmax_gentan_fast   30  800  7600  0.166315
# 3          minmax_jit   30  800  7600  0.122361
# 4        minmax_ufunc   30  800  7600  0.113622

#              function  n_A  n_B     d       time
# 0        minmax_dense  300  800  7600  13.629886
# 1       minmax_gentan  300  800  7600   1.441526
# 2  minmax_gentan_fast  300  800  7600   1.414300
# 3          minmax_jit  300  800  7600   2.277988
# 4        minmax_ufunc  300  800  7600   1.081151


def run_time():
    n_rep = 25

    for n_A, n_B, d in [(160, 400, 7600)]:
        print(_run_time(n_A, n_B, d, n_rep, float))

#              function  n_A   n_B    d      time
# 0        minmax_dense  100  5000  307  1.399382
# 1       minmax_gentan  100  5000  307  0.118079
# 2  minmax_gentan_fast  100  5000  307  0.116313
# 3          minmax_jit  100  5000  307  0.042270
# 4        minmax_ufunc  100  5000  307  0.134376

#              function   n_A   n_B    d      time
# 0        minmax_dense  1000  1000  307  2.262268
# 1       minmax_gentan  1000  1000  307  0.187768
# 2  minmax_gentan_fast  1000  1000  307  0.185888
# 3          minmax_jit  1000  1000  307  0.302484
# 4        minmax_ufunc  1000  1000  307  0.315622

#              function   n_A  n_B    d      time
# 0        minmax_dense  5000  100  307  1.032933
# 1       minmax_gentan  5000  100  307  0.097594
# 2  minmax_gentan_fast  5000  100  307  0.096002
# 3          minmax_jit  5000  100  307  0.284397
# 4        minmax_ufunc  5000  100  307  0.679844

#              function  n_A   n_B     d       time
# 0        minmax_dense  100  5000  5000  22.194602
# 1       minmax_gentan  100  5000  5000   1.974525
# 2  minmax_gentan_fast  100  5000  5000   1.973238
# 3          minmax_jit  100  5000  5000   2.538056
# 4        minmax_ufunc  100  5000  5000   1.632805

#              function  n_A  n_B     d      time
# 0  minmax_gentan_fast   16  400  7600  0.047978
# 1          minmax_jit   16  400  7600  0.022564
# 2    minmax_ufunc_int   16  400  7600  0.029898

#              function  n_A  n_B     d      time
# 0  minmax_gentan_fast  160  400  7600  0.392368
# 1          minmax_jit  160  400  7600  0.600239
# 2    minmax_ufunc_int  160  400  7600  0.296664

#              function  n_A    n_B     d      time
# 0  minmax_gentan_fast   16  30000  7600  3.782805
# 1          minmax_jit   16  30000  7600  2.161236
# 2    minmax_ufunc_int   16  30000  7600  2.370235

#              function  n_A    n_B     d       time
# 0  minmax_gentan_fast  160  30000  7600  29.436921
# 1          minmax_jit  160  30000  7600  44.021622
# 2    minmax_ufunc_int  160  30000  7600  23.798634

#             function  n_A  n_B     d      time
# 0  minmax_gentan_fast   16  400  7600  0.047792
# 1          minmax_jit   16  400  7600  0.022469
# 2    minmax_ufunc_int   16  400  7600  0.032123

#              function  n_A  n_B     d      time
# 0  minmax_gentan_fast  160  400  7600  0.386905
# 1          minmax_jit  160  400  7600  0.568091
# 2    minmax_ufunc_int  160  400  7600  0.286317

#              function   n_A  n_B     d      time
# 0  minmax_gentan_fast  1600  400  7600  3.856638
# 1          minmax_jit  1600  400  7600  9.321267
# 2    minmax_ufunc_int  1600  400  7600  3.158656


def run_time_n_vs_m():
    n_rep = 25
    n_A = 100

    for fac in [1, 2, 5, 10, 50, 100, 1000]:
        print(fac)
        print(_run_time(n_A, fac * n_A, 307, n_rep, float))


def _run_time(n_A, n_B, d, n_rep, dtype):
    # Create random data
    X_A = np.random.RandomState(5943).randint(0, 200, size=(n_A, d))
    X_B = np.random.RandomState(842).randint(0, 200, size=(n_B, d))

    # Define functions to test
    fun_list = [("minmax_jit", _min_max_dense_jit, dtype),
                ("minmax_ufunc_int", _min_max_dense_ufunc_int, int),
                # ("minmax__nogil", _minmax_nogil, dtype),
                # ("minmax_dense", _min_max_dense, dtype),
                # ("minmax_gentan", generalized_tanimoto_kernel, dtype),
                ("minmax_gentan_fast", generalized_tanimoto_kernel_FAST, dtype)]

    # Compile using numba
    K_ref = generalized_tanimoto_kernel_FAST(X_A, X_B)
    print("Compile ... ", end="")
    for fun_name, fun, fun_dtype in fun_list:
        print(fun_name, end="; ", flush=True)
        np.testing.assert_allclose(K_ref, fun(X_A.astype(fun_dtype), X_B.astype(fun_dtype)))
    print()

    # Run timing
    df = []
    for fun_name, fun, fun_dtype in fun_list:
        print("RUN - %s" % fun_name)

        _X_A = X_A.astype(fun_dtype)
        _X_B = X_B.astype(fun_dtype)

        for _ in range(n_rep):
            start = time.time()
            fun(_X_A, _X_B)
            df.append([fun_name, n_A, n_B, d, (time.time() - start), str(fun_dtype)])

    df = pd.DataFrame(df, columns=["function", "n_A", "n_B", "d", "time", "dtype"]) \
        .groupby(["function", "n_A", "n_B", "d", "dtype"]) \
        .aggregate(np.mean) \
        .reset_index()

    return df


def profiling(fun, n_A=100, n_B=5000, d=307):
    X_A = np.random.RandomState(5943).randint(0, 200, size=(n_A, d))
    X_B = np.random.RandomState(842).randint(0, 200, size=(n_B, d))

    fun(np.random.randint(0, 200, size=(10, 2)), np.random.randint(0, 200, size=(10, 2)))

    for _ in range(50):
        fun(X_A, X_B)


def _use_binary_encoding(n_A, n_B, d, n_rep, dtype):
    X_A = (np.random.RandomState(93).rand(n_A, d) > 0.5).astype(dtype)
    X_B = (np.random.RandomState(92).rand(n_B, d) > 0.5).astype(dtype)

    # Define functions to test
    fun_list = [
        ("minmax_gentan_fast", generalized_tanimoto_kernel_FAST),
        ("tanimoto", tanimoto_kernel_FAST),
    ]

    # Compile using numba
    K_ref = generalized_tanimoto_kernel_FAST(X_A, X_B)
    print("Compile ... ", end="")
    for fun_name, fun in fun_list:
        print(fun_name, end="; ", flush=True)
        np.testing.assert_allclose(K_ref, fun(X_A, X_B))
    print()

    # Run timing
    df = []
    for fun_name, fun in fun_list:
        print("RUN - %s" % fun_name)
        for _ in range(n_rep):
            start = time.time()
            fun(X_A, X_B)
            df.append([fun_name, n_A, n_B, d, (time.time() - start), dtype])

    df = pd.DataFrame(df, columns=["function", "n_A", "n_B", "d", "time", "dtype"]) \
        .groupby(["function", "n_A", "n_B", "d", "dtype"]) \
        .aggregate(np.mean) \
        .reset_index()

    return df


#              function  n_A  n_B      d      time
# 0  minmax_gentan_fast  160  400  38000  1.952541
# 1        minmax_ufunc  160  400  38000  1.564143
# 2            tanimoto  160  400  38000  0.071877
#
#              function  n_A  n_B     d      time
# 0  minmax_gentan_fast  160  400  1208  0.061411
# 1        minmax_ufunc  160  400  1208  0.075999
# 2            tanimoto  160  400  1208  0.002795


if __name__ == "__main__":
    import time
    import pandas as pd
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "scenario",
        help="Which performance evaluation to run.",
        choices=["rt_binarized_tanimoto", "rt_counting_minmax"]
    )
    arg_parser.add_argument("--na", default=1000, type=int)
    arg_parser.add_argument("--nb", default=1000, type=int)
    arg_parser.add_argument("--d", default=15000, type=int)
    arg_parser.add_argument("--nrep", default=25, type=int)
    args = arg_parser.parse_args()

    # Parameters
    n_rep = args.nrep

    n_A = args.na
    n_B = args.nb

    d = args.d  # 15000 is approx. the size of the binarized iokr fingerprints

    if args.scenario == "rt_binarized_tanimoto":
        for dtype in [np.int64, np.int32, np.int16, np.float64, np.float32]:
            print(_use_binary_encoding(n_A, n_B, d, n_rep, dtype))
    elif args.scenario == "rt_counting_minmax":
        for dtype in [np.int64, np.int32, np.int16, np.float64, np.float32]:
            print(_run_time(n_A, n_B, d, n_rep, dtype))
    else:
        raise ValueError()

    # run_time()

# Comparison of binary vs. counting encoding

# COUNTING
#              function  n_A  n_B    d      time
# 0  minmax_gentan_fast  160  400  307  0.012799
# 1          minmax_jit  160  400  307  0.007002
# 2    minmax_ufunc_int  160  400  307  0.030934

#              function  n_A  n_B     d      time
# 0  minmax_gentan_fast  160  400  7600  0.382427
# 1          minmax_jit  160  400  7600  0.579165
# 2    minmax_ufunc_int  160  400  7600  0.296164


# BINARY
#              function  n_A  n_B     d      time
# 0  minmax_gentan_fast  160  400  1208  0.054993
# 1          minmax_jit  160  400  1208  0.048108
# 2        minmax_ufunc  160  400  1208  0.061900
# 3            tanimoto  160  400  1208  0.002066

#              function  n_A  n_B      d      time
# 0  minmax_gentan_fast  160  400  20000  1.020886
# 1          minmax_jit  160  400  20000  1.565648
# 2        minmax_ufunc  160  400  20000  0.830727
# 3            tanimoto  160  400  20000  0.039696

# Binary encoding (despite being longer) is 3x as fast as the counting encoding.
# (2nd) Binary encoding (despite being longer) is 8x as fast as the counting encoding.
