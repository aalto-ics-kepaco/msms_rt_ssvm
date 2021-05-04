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
from ssvm.feature_utils import CountingFpsBinarizer


class TestCountingFpsBinarizer(unittest.TestCase):
    def setUp(self) -> None:
        self.X1 = np.array(
            [
                [ 1,  0,  0,  3,  4,  1,  0],
                [ 0,  0,  0,  0,  0,  0,  0],
                [12, 12, 12, 12, 12, 12, 12],
                [ 0,  1,  2,  4,  0, 12,  5]
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

        self.X4 = np.random.RandomState(111).randint(0, 16, size=(30000, 307))

    def test_conversion(self):
        trans = CountingFpsBinarizer(bin_centers=[1, 2, 3, 4, 8])
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
        trans = CountingFpsBinarizer(bin_centers=[1])
        Z = trans.fit_transform(self.X1)

        self.assertEqual(self.X1.shape, Z.shape)
        np.testing.assert_array_equal(self.X1 > 0, Z)

    def test_compression(self):
        trans = CountingFpsBinarizer(bin_centers=[1, 2, 3, 4, 8], compress=True)
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

    def test_binary_representation_equal_counting_in_edge_case(self):
        # Bin centers are defined as: [1, 2, ..., max_value]
        trans = CountingFpsBinarizer(bin_centers=(np.arange(np.max(self.X3)) + 1).tolist())
        Z = trans.fit_transform(self.X3)
        print("[1, 2, ..., max_value]: \t", Z.shape)

        self.assertEqual((len(self.X3), self.X3.shape[1] * np.max(self.X3)), Z.shape)
        np.testing.assert_array_equal(minmax_kernel(self.X3), tanimoto_kernel(Z))

        # Use compression
        trans = CountingFpsBinarizer(bin_centers=(np.arange(np.max(self.X3)) + 1).tolist(), compress=True)
        Z_cpr = trans.fit_transform(self.X3)
        print("With compression: \t\t\t", Z_cpr.shape)

        self.assertEqual((len(self.X3), np.sum(np.max(self.X3, axis=0))), Z_cpr.shape)
        np.testing.assert_array_equal(tanimoto_kernel(Z), tanimoto_kernel(Z_cpr))

        # Pass in max-values per column
        _bin_centers = [(np.arange(np.max(self.X3[:, d])) + 1).tolist() for d in range(self.X3.shape[1])]
        trans = CountingFpsBinarizer(bin_centers=_bin_centers)
        Z_max = trans.fit_transform(self.X3)
        print("With max-values: \t\t\t", Z_max.shape)

        self.assertEqual((len(self.X3), np.sum(np.max(self.X3, axis=0))), Z_max.shape)
        np.testing.assert_array_equal(tanimoto_kernel(Z), tanimoto_kernel(Z_max))

    def test_run_time(self):
        trans = CountingFpsBinarizer(bin_centers=(np.arange(np.max(self.X3)) + 1).tolist(), compress=True)

        s = time.time()
        Z = trans.fit_transform(self.X4)
        print("%.3fs" % (time.time() - s))  # 32s


if __name__ == '__main__':
    unittest.main()
