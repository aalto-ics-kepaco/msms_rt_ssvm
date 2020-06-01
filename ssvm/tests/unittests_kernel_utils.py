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

import unittest
import numpy as np
import itertools as it

from scipy.sparse import csr_matrix

from ssvm.kernel_utils import minmax_kernel, tanimoto_kernel, check_input, generalized_tanimoto_kernel


class TestCheckInput(unittest.TestCase):
    def test_datatype_checkup_binary(self):
        __X_A = np.array([[0, 1, 2], [1, 0, 0], [3, 4, 0]])

        with self.assertRaises(ValueError):
            check_input(__X_A, None, datatype="binary")
        with self.assertRaises(ValueError):
            check_input(csr_matrix(__X_A), None, datatype="binary")

    def test_datatype_checkup_postive(self):
        __X_A = np.array([[0, 1, 2], [1, 0, 0], [3, 4, -1]])

        with self.assertRaises(ValueError):
            check_input(__X_A, None, datatype="positive")
        with self.assertRaises(ValueError):
            check_input(csr_matrix(__X_A), None, datatype="positive")


class TestMinMaxKernel(unittest.TestCase):
    @staticmethod
    def _minmax_kernel_slow(X, Y=None):
            if Y is None:
                Y = X

            n_A, n_B = X.shape[0], Y.shape[0]
            d = X.shape[1]
            assert d == Y.shape[1]

            K = np.zeros((n_A, n_B))

            for i in range(n_A):
                for j in range(n_B):
                    min_s = 0
                    max_s = 0
                    for s in range(d):
                        min_s += np.minimum(X[i, s], Y[j, s])
                        max_s += np.maximum(X[i, s], Y[j, s])
                    K[i, j] = min_s / max_s

            # for (i, j) in it.product(range(n_A), range(n_B)):
            #     K[i, j] = np.sum(np.minimum(X[i], Y[j])) / np.sum(np.maximum(X[i], Y[j]))

            return K

    def test_corner_cases(self):
        # Empty features
        # ---------------------------------------------
        __X_A = np.array([[0, 1, 2, 0], [1, 0, 0, 0], [3, 4, 0, 0]])
        __X_B = np.array([[0, 0, 1, 0], [3, 1, 0, 0]])

        __K = minmax_kernel(__X_A, __X_B)
        np.testing.assert_equal(__K.shape, (3, 2))
        assert (np.max(__K) <= 1.), "Kernel values must be <= 1"
        assert (np.min(__K) >= 0.), "Kernel values must be >= 0"
        np.testing.assert_equal(__K[0, 1], 1. / 6.)
        np.testing.assert_equal(__K[1, 0], 0.)
        np.testing.assert_equal(__K[1, 1], 1. / 4.)
        np.testing.assert_equal(__K[2, 1], 4. / 7.)

        # Empty features on larger random set
        # ---------------------------------------------
        n_A = 92
        n_B = 29
        d = 103
        X_A = np.random.RandomState(59).randint(0, 90, size=(n_A, d))
        X_B = np.random.RandomState(8443).randint(0, 43, size=(n_B, d))

        X_A[:, 12] = 0
        X_A[:, 10] = 0
        X_A[:, 54] = 0

        X_B[:, 12] = 0
        X_B[:, 10] = 0
        X_B[:, 21] = 0

        __K = minmax_kernel(X_A)
        np.testing.assert_array_equal(__K, self._minmax_kernel_slow(X_A))
        assert (np.max(__K) <= 1.), "Kernel values must be <= 1"
        assert (np.min(__K) >= 0.), "Kernel values must be >= 0"
        assert all(np.diag(__K) == 1.0)

        __K = minmax_kernel(X_A, X_B)
        np.testing.assert_array_equal(__K, self._minmax_kernel_slow(X_A, X_B))
        assert (np.max(__K) <= 1.), "Kernel values must be <= 1"
        assert (np.min(__K) >= 0.), "Kernel values must be >= 0"

    def test_corner_cases_with_sparse_matrix(self):
        # Empty features
        # ---------------------------------------------
        __X_A = csr_matrix(np.array([[0, 1, 2, 0], [1, 0, 0, 0], [3, 4, 0, 0]]))
        __X_B = csr_matrix(np.array([[0, 0, 1, 0], [3, 1, 0, 0]]))

        __K = minmax_kernel(__X_A, __X_B)
        np.testing.assert_equal(__K.shape, (3, 2))
        assert (np.max(__K) <= 1.), "Kernel values must be <= 1"
        assert (np.min(__K) >= 0.), "Kernel values must be >= 0"
        np.testing.assert_equal(__K[0, 1], 1. / 6.)
        np.testing.assert_equal(__K[1, 0], 0.)
        np.testing.assert_equal(__K[1, 1], 1. / 4.)
        np.testing.assert_equal(__K[2, 1], 4. / 7.)

    def test_on_small_data(self):
        # ---------------------------------------------
        __X_A = np.array([[0, 1, 2], [1, 0, 0], [3, 4, 0]])
        __X_B = np.array([[0, 0, 1], [3, 1, 0]])

        __K = minmax_kernel(__X_A)
        np.testing.assert_array_equal(np.diag(__K), np.ones((3,)))
        np.testing.assert_equal(__K.shape, (3, 3))
        assert (np.max(__K) <= 1.), "Kernel values must be <= 1"
        assert (np.min(__K) >= 0.), "Kernel values must be >= 0"
        np.testing.assert_equal(__K[0, 1], 0.)
        np.testing.assert_equal(__K[1, 0], 0.)
        np.testing.assert_equal(__K[0, 2], 1. / 9.)
        np.testing.assert_equal(__K[2, 0], 1. / 9.)
        np.testing.assert_equal(__K[1, 2], 1. / 7.)
        np.testing.assert_equal(__K[2, 1], 1. / 7.)

        __K = minmax_kernel(__X_A, __X_B)
        np.testing.assert_equal(__K.shape, (3, 2))
        assert (np.max(__K) <= 1.), "Kernel values must be <= 1"
        assert (np.min(__K) >= 0.), "Kernel values must be >= 0"
        np.testing.assert_equal(__K[0, 1], 1. / 6.)
        np.testing.assert_equal(__K[1, 0], 0.)
        np.testing.assert_equal(__K[1, 1], 1. / 4.)
        np.testing.assert_equal(__K[2, 1], 4. / 7.)

    def test_on_larger_random_data(self):
        n_A = 45
        n_B = 21
        d = 121
        X_A = np.random.RandomState(5943).randint(0, 43, size=(n_A, d))
        X_B = np.random.RandomState(842).randint(0, 43, size=(n_B, d))

        # Symmetric kernel
        K = minmax_kernel(X_A)
        np.testing.assert_array_equal(K, self._minmax_kernel_slow(X_A))
        np.testing.assert_array_equal(np.diag(K), np.ones((n_A,)))
        assert (np.max(K) <= 1.)
        assert (np.min(K) >= 0.)

        # Non-symmetric kernel
        K = minmax_kernel(X_A, X_B)
        np.testing.assert_array_equal(K, self._minmax_kernel_slow(X_A, X_B))
        assert (np.max(K) <= 1.)
        assert (np.min(K) >= 0.)

    def test_compatibility_with_sparse_matrix(self):
        # ---------------------------------------------
        __X_A = csr_matrix(np.array([[0, 1, 2], [1, 0, 0], [3, 4, 0]]))
        __X_B = csr_matrix(np.array([[0, 0, 1], [3, 1, 0]]))

        __K = minmax_kernel(__X_A)
        np.testing.assert_array_equal(np.diag(__K), np.ones((3,)))
        np.testing.assert_equal(__K.shape, (3, 3))
        assert (np.max(__K) <= 1.), "Kernel values must be <= 1"
        assert (np.min(__K) >= 0.), "Kernel values must be >= 0"
        np.testing.assert_equal(__K[0, 1], 0.)
        np.testing.assert_equal(__K[1, 0], 0.)
        np.testing.assert_equal(__K[0, 2], 1. / 9.)
        np.testing.assert_equal(__K[2, 0], 1. / 9.)
        np.testing.assert_equal(__K[1, 2], 1. / 7.)
        np.testing.assert_equal(__K[2, 1], 1. / 7.)

        __K = minmax_kernel(__X_A, __X_B)
        np.testing.assert_equal(__K.shape, (3, 2))
        assert (np.max(__K) <= 1.), "Kernel values must be <= 1"
        assert (np.min(__K) >= 0.), "Kernel values must be >= 0"
        np.testing.assert_equal(__K[0, 1], 1. / 6.)
        np.testing.assert_equal(__K[1, 0], 0.)
        np.testing.assert_equal(__K[1, 1], 1. / 4.)
        np.testing.assert_equal(__K[2, 1], 4. / 7.)

    def test_on_larger_random_data_with_sparse_matrix(self):
        n_A = 45
        n_B = 21
        d = 121
        X_A = np.random.RandomState(5943).randint(0, 43, size=(n_A, d))
        X_B = np.random.RandomState(842).randint(0, 43, size=(n_B, d))

        # Symmetric kernel
        K = minmax_kernel(csr_matrix(X_A))
        np.testing.assert_array_equal(K, self._minmax_kernel_slow(X_A))
        np.testing.assert_array_equal(np.diag(K), np.ones((n_A,)))
        assert (np.max(K) <= 1.)
        assert (np.min(K) >= 0.)

        # Non-symmetric kernel
        K = minmax_kernel(csr_matrix(X_A), csr_matrix(X_B))
        np.testing.assert_array_equal(K, self._minmax_kernel_slow(X_A, X_B))
        assert (np.max(K) <= 1.)
        assert (np.min(K) >= 0.)


class TestTanimotoKernel(unittest.TestCase):
    def test_on_small_data(self):
        X_A = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]])
        X_B = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0], [1, 1, 0]])

        # symmetric kernel
        K = tanimoto_kernel(X_A)
        np.testing.assert_equal(K.shape, (3, 3))
        np.testing.assert_equal(np.diag(K), np.ones((3,)))
        np.testing.assert_equal(K[0, 1], 1. / 3.)
        np.testing.assert_equal(K[1, 0], 1. / 3.)
        np.testing.assert_equal(K[0, 2], 1. / 2.)
        np.testing.assert_equal(K[2, 0], 1. / 2.)
        assert (np.max(K) <= 1.), "Kernel values must be <= 1"
        assert (np.min(K) >= 0.), "Kernel values must be >= 0"

        # non-symmetric kernel
        K = tanimoto_kernel(X_A, X_B)
        np.testing.assert_equal(K.shape, (3, 4))
        np.testing.assert_equal(K[0, 1], 2. / 3.)
        np.testing.assert_equal(K[1, 0], 1. / 3.)
        np.testing.assert_equal(K[0, 2], 0.)
        np.testing.assert_equal(K[2, 0], 1. / 2.)
        assert (np.max(K) <= 1.), "Kernel values must be <= 1"
        assert (np.min(K) >= 0.), "Kernel values must be >= 0"

    def test_on_larger_random_data(self):
        def jacscr(x, y):
            """
            Calculate Jaccard score
            """
            x, y = set(np.where(x)[0]), set(np.where(y)[0])

            n_inter = len(x & y)
            n_union = len(x | y)

            return n_inter / n_union

        X_A = np.random.RandomState(493).randint(0, 2, size=(51, 32))
        X_B = np.random.RandomState(493).randint(0, 2, size=(12, 32))

        # symmetric kernel
        K = tanimoto_kernel(X_A)
        np.testing.assert_equal(K.shape, (51, 51))
        np.testing.assert_equal(K[3, 6], jacscr(X_A[3], X_A[6]))
        np.testing.assert_equal(K[1, 1], jacscr(X_A[1], X_A[1]))
        np.testing.assert_equal(K[0, 9], jacscr(X_A[0], X_A[9]))
        np.testing.assert_equal(K[5, 10], jacscr(X_A[5], X_A[10]))
        np.testing.assert_equal(K[6, 3], K[3, 6])
        np.testing.assert_equal(K[0, 9], K[9, 0])
        np.testing.assert_equal(K[5, 10], K[10, 5])
        np.testing.assert_equal(np.diag(K), np.ones((51,)))

        # non-symmetric kernel
        K = tanimoto_kernel(X_A, X_B)
        np.testing.assert_equal(K.shape, (51, 12))
        np.testing.assert_equal(K[3, 6], jacscr(X_A[3], X_B[6]))
        np.testing.assert_equal(K[1, 1], jacscr(X_A[1], X_B[1]))

    def test_compatibility_with_sparse_matrix(self):
        self.skipTest("Not implemented")

        X_A = csr_matrix(np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]]))
        X_B = csr_matrix(np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0], [1, 1, 0]]))

        # symmetric kernel
        K = tanimoto_kernel(X_A)
        np.testing.assert_equal(K.shape, (3, 3))
        np.testing.assert_equal(np.diag(K), np.ones((3,)))
        np.testing.assert_equal(K[0, 1], 1. / 3.)
        np.testing.assert_equal(K[1, 0], 1. / 3.)
        np.testing.assert_equal(K[0, 2], 1. / 2.)
        np.testing.assert_equal(K[2, 0], 1. / 2.)
        assert (np.max(K) <= 1.), "Kernel values must be <= 1"
        assert (np.min(K) >= 0.), "Kernel values must be >= 0"

        # non-symmetric kernel
        K = tanimoto_kernel(X_A, X_B)
        np.testing.assert_equal(K.shape, (3, 4))
        np.testing.assert_equal(K[0, 1], 2. / 3.)
        np.testing.assert_equal(K[1, 0], 1. / 3.)
        np.testing.assert_equal(K[0, 2], 0.)
        np.testing.assert_equal(K[2, 0], 1. / 2.)
        assert (np.max(K) <= 1.), "Kernel values must be <= 1"
        assert (np.min(K) >= 0.), "Kernel values must be >= 0"

    def test_on_larger_random_data_with_sparse_matrix(self):
        self.skipTest("Not implemented")

        def jacscr(x, y):
            """
            Calculate Jaccard score
            """
            x, y = set(np.where(x)[0]), set(np.where(y)[0])

            n_inter = len(x & y)
            n_union = len(x | y)

            return n_inter / n_union

        X_A = np.random.RandomState(493).randint(0, 2, size=(51, 32))
        X_B = np.random.RandomState(493).randint(0, 2, size=(12, 32))

        # symmetric kernel
        K = tanimoto_kernel(csr_matrix(X_A))
        np.testing.assert_equal(K.shape, (51, 51))
        np.testing.assert_equal(K[3, 6], jacscr(X_A[3], X_A[6]))
        np.testing.assert_equal(K[1, 1], jacscr(X_A[1], X_A[1]))
        np.testing.assert_equal(K[0, 9], jacscr(X_A[0], X_A[9]))
        np.testing.assert_equal(K[5, 10], jacscr(X_A[5], X_A[10]))
        np.testing.assert_equal(K[6, 3], K[3, 6])
        np.testing.assert_equal(K[0, 9], K[9, 0])
        np.testing.assert_equal(K[5, 10], K[10, 5])
        np.testing.assert_equal(np.diag(K), np.ones((51,)))

        # non-symmetric kernel
        K = tanimoto_kernel(csr_matrix(X_A), csr_matrix(X_B))
        np.testing.assert_equal(K.shape, (51, 12))
        np.testing.assert_equal(K[3, 6], jacscr(X_A[3], X_B[6]))
        np.testing.assert_equal(K[1, 1], jacscr(X_A[1], X_B[1]))


class TestGeneralizedTanimotoKernel(unittest.TestCase):
    def test_on_small_data(self):
        X = np.array([[0, 1.75, -3.25, 0, 1], [2.35, 1, 0, 0, 2], [0, 7.45, 0, 3, 0]])
        # |X|_1 = (6, 5.35, 10.45)
        Y = np.array([[0, 19, 5.64, 2, 0], [2.35, 1, 9, 0, -2], [0, 0, 0, 0, 1], [-1, 2, -3, 4, -5]])
        # |Y|_1 = (26.64, 14.35, 1, 15)

        # Square kernel matrix
        KX = generalized_tanimoto_kernel(X)

        np.testing.assert_equal(KX.shape, (3, 3))
        np.testing.assert_equal(np.diag(KX), np.ones((3,)))
        self.assertTrue((np.all(KX) >= 0) & (np.all(KX) <= 1))
        np.testing.assert_equal(KX, KX.T)

        np.testing.assert_allclose(
            KX[0, 1], (6 + 5.35 - np.sum(np.abs(X[0] - X[1]))) / (6 + 5.35 + np.sum(np.abs(X[0] - X[1]))))
        np.testing.assert_allclose(
            KX[0, 2], (6 + 10.45 - np.sum(np.abs(X[0] - X[2]))) / (6 + 10.45 + np.sum(np.abs(X[0] - X[2]))))
        np.testing.assert_allclose(
            KX[1, 2], (5.35 + 10.45 - np.sum(np.abs(X[1] - X[2]))) / (5.35 + 10.45 + np.sum(np.abs(X[1] - X[2]))))

        # Non-square kernel matrix
        KXY = generalized_tanimoto_kernel(X, Y)

        np.testing.assert_equal(KXY.shape, (3, 4))
        self.assertTrue((np.all(KX) >= 0) & (np.all(KX) <= 1))

        np.testing.assert_allclose(
            KXY[0, 0], (6 + 26.64 - np.sum(np.abs(X[0] - Y[0]))) / (6 + 26.64 + np.sum(np.abs(X[0] - Y[0]))))
        np.testing.assert_allclose(
            KXY[0, 1], (6 + 14.35 - np.sum(np.abs(X[0] - Y[1]))) / (6 + 14.35 + np.sum(np.abs(X[0] - Y[1]))))
        np.testing.assert_allclose(
            KXY[1, 0], (5.35 + 26.64 - np.sum(np.abs(X[1] - Y[0]))) / (5.35 + 26.64 + np.sum(np.abs(X[1] - Y[0]))))
        np.testing.assert_allclose(
            KXY[1, 2], (5.35 + 1 - np.sum(np.abs(X[1] - Y[2]))) / (5.35 + 1 + np.sum(np.abs(X[1] - Y[2]))))

    def test_on_large_random_data(self):
        def _gentan(x, y):
            xl1 = np.sum(np.abs(x))
            yl1 = np.sum(np.abs(y))
            xmyl1 = np.sum(np.abs(x - y))
            return (xl1 + yl1 - xmyl1) / (xl1 + yl1 + xmyl1)

        for _ in range(50):
            X = np.random.RandomState(33).randn(10, 302)
            Y = np.random.RandomState(331).randn(8, 302)

            # Non-square kernel matrix
            K = generalized_tanimoto_kernel(X)
            np.testing.assert_equal(K.shape, (10, 10))
            self.assertTrue((np.all(K) >= 0) & (np.all(K) <= 1))
            np.testing.assert_equal(K, K.T)
            np.testing.assert_equal(np.diag(K), np.ones((10,)))

            for (i, j) in it.combinations(range(10), 2):
                np.testing.assert_allclose(K[i, j], _gentan(X[i], X[j]))

            # Non-square kernel matrix
            K = generalized_tanimoto_kernel(X, Y)
            np.testing.assert_equal(K.shape, (10, 8))
            self.assertTrue((np.all(K) >= 0) & (np.all(K) <= 1))

            for (i, j) in it.product(range(10), range(8)):
                np.testing.assert_allclose(K[i, j], _gentan(X[i], Y[j]))


if __name__ == '__main__':
    unittest.main()
