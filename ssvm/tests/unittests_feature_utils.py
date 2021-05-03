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


if __name__ == '__main__':
    unittest.main()
