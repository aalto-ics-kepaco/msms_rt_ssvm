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
import time
import unittest
import numpy as np

from ssvm.kernel_utils import minmax_kernel, tanimoto_kernel
from ssvm.feature_utils import CountingFpsBinarizer, RemoveCorrelatedFeatures


class TestCorrelatedFeatureRemoval(unittest.TestCase):
    def test_corner_cases(self):
        # No feature exceeds the correlation threshold
        X = np.random.RandomState(13).rand(1010, 29)
        mask = RemoveCorrelatedFeatures().fit(X).get_support()
        self.assertTrue(np.all(mask))

        # All features are correlated
        X = np.array([
            np.random.RandomState(101).rand(5),
            np.random.RandomState(101).rand(5),
            np.random.RandomState(101).rand(5),
        ]).T

        mask = RemoveCorrelatedFeatures().fit(X).get_support()
        self.assertEqual(1, np.sum(mask))

    def test_correct_feature_removal(self):
        X = np.random.RandomState(43).random((3, 4))
        # array([[0.11505457, 0.60906654, 0.13339096, 0.24058962],
        #        [0.32713906, 0.85913749, 0.66609021, 0.54116221],
        #        [0.02901382, 0.7337483 , 0.39495002, 0.80204712]])

        R = np.corrcoef(X.T)
        # array([[1., 0.69228233, 0.69857039, -0.24099928],
        #        [0.69228233, 1., 0.99996171, 0.53351747],
        #        [0.69857039, 0.99996171, 1., 0.52609601],
        #        [-0.24099928, 0.53351747, 0.52609601, 1.]])

        mask_ref = [True, True, False, True]

        mask = RemoveCorrelatedFeatures().fit(X).get_support()

        np.testing.assert_array_equal(mask_ref, mask)


class TestCountingFpsBinarizer(unittest.TestCase):
    def setUp(self) -> None:
        self.X1 = np.array(
            [
                [1, 0, 0, 3, 4, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [12, 12, 12, 12, 12, 12, 12],
                [0, 1, 2, 4, 0, 12, 5]
            ]
        )

        self.X2 = np.array(
            [
                [1, 0, 0, 3, 4, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 2, 2, 0, 4, 12, 12],
                [0, 1, 2, 4, 0, 12, 5]
            ]
        )

        self.X3 = np.random.RandomState(111).randint(0, 12, size=(22, 45))

        self.X4 = np.random.RandomState(111).randint(0, 16, size=(300000, 307))

        self.X5 = np.array(
            [
                [1, 1, 0, 3, 4, 1, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [1, 1, 2, 0, 4, 5, 0],
                [0, 1, 2, 4, 0, 5, 0]
            ]
        )

    def test_length(self):
        trans = CountingFpsBinarizer(bin_centers=np.array([1, 2, 3, 4, 8]), compress=True)
        trans.fit(self.X2)

        self.assertEqual(1 + 2 + 2 + 4 + 4 + 5 + 5, len(trans))

    def test_conversion(self):
        trans = CountingFpsBinarizer(bin_centers=np.array([1, 2, 3, 4, 8]))
        Z = trans.fit_transform(self.X1)

        self.assertEqual((len(self.X1), self.X1.shape[1] * 5), Z.shape)
        np.testing.assert_array_equal(
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                ]
            ),
            Z
        )

    def test_edge_cases(self):
        trans = CountingFpsBinarizer(bin_centers=np.array([1]))
        Z = trans.fit_transform(self.X1)

        self.assertEqual(self.X1.shape, Z.shape)
        np.testing.assert_array_equal(self.X1 > 0, Z)

    def test_compression(self):
        trans = CountingFpsBinarizer(bin_centers=np.array([1, 2, 3, 4, 8]), compress=True)
        Z = trans.fit_transform(self.X2)

        self.assertEqual((len(self.X1), 1 + 2 + 2 + 4 + 4 + 5 + 5), Z.shape)
        np.testing.assert_array_equal(
            np.array(
                [
                    [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                ]
            ),
            Z
        )

    def test_removing_constant_features(self):
        trans = CountingFpsBinarizer(bin_centers=np.arange(np.max(self.X5)) + 1, remove_constant_features=True)
        Z = trans.fit_transform(self.X5)

        self.assertEqual((len(self.X5), (self.X5.shape[1] - 2) * np.max(self.X5)), Z.shape)
        np.testing.assert_array_equal(
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
                ]
            ),
            Z
        )

        # Are bin-centers correctly removed?
        bin_centers = [
            np.array([1, 2]),
            np.array([1]),  # should be removed
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5, 10]),
            np.array([2, 4, 8]),
            np.array([3, 5])  # should be removed
        ]
        trans = CountingFpsBinarizer(bin_centers=bin_centers, remove_constant_features=True)
        Z = trans.fit_transform(self.X5)

        self.assertEqual((len(self.X5), 2 + 5 + 5 + 6 + 3), Z.shape)
        np.testing.assert_array_equal(
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                ]
            ),
            Z
        )

        # Now also with compression
        trans = CountingFpsBinarizer(bin_centers=bin_centers, remove_constant_features=True, compress=True)
        Z = trans.fit_transform(self.X5)

        self.assertEqual((len(self.X5), 1 + 2 + 4 + 4 + 2), Z.shape)
        np.testing.assert_array_equal(
            np.array(
                [
                    [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                ]
            ),
            Z
        )

    def test_binary_representation_equal_counting_in_edge_case(self):
        # Bin centers are defined as: [1, 2, ..., max_value]
        trans = CountingFpsBinarizer(bin_centers=(np.arange(np.max(self.X3)) + 1))
        Z = trans.fit_transform(self.X3)
        # print("[1, 2, ..., max_value]: \t", Z.shape)

        self.assertEqual((len(self.X3), self.X3.shape[1] * np.max(self.X3)), Z.shape)
        np.testing.assert_array_equal(minmax_kernel(self.X3), tanimoto_kernel(Z))

        # Use compression
        trans = CountingFpsBinarizer(bin_centers=(np.arange(np.max(self.X3)) + 1), compress=True)
        Z_cpr = trans.fit_transform(self.X3)
        # print("With compression: \t\t\t", Z_cpr.shape)

        self.assertEqual((len(self.X3), np.sum(np.max(self.X3, axis=0))), Z_cpr.shape)
        np.testing.assert_array_equal(tanimoto_kernel(Z), tanimoto_kernel(Z_cpr))

        # Pass in max-values per column
        _bin_centers = [np.arange(np.max(self.X3[:, d])) + 1 for d in range(self.X3.shape[1])]
        trans = CountingFpsBinarizer(bin_centers=_bin_centers)
        Z_max = trans.fit_transform(self.X3)
        # print("With max-values: \t\t\t", Z_max.shape)

        self.assertEqual((len(self.X3), np.sum(np.max(self.X3, axis=0))), Z_max.shape)
        np.testing.assert_array_equal(tanimoto_kernel(Z), tanimoto_kernel(Z_max))

    def test_run_time(self):
        self.skipTest("Only needed for run-time performance evaluation.")

        trans = CountingFpsBinarizer(bin_centers=(np.arange(np.max(self.X4)) + 1), compress=True)

        s = time.time()
        Z = trans.fit_transform(self.X4)
        print("%.3fs" % (time.time() - s))  # 2s


if __name__ == '__main__':
    unittest.main()
