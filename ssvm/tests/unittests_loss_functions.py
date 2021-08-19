####
#
# The MIT License (MIT)
#
# Copyright 2020, 2021 Eric Bach <eric.bach@aalto.fi>
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

from sklearn.metrics import hamming_loss as hamming_loss_sk

from ssvm.loss_functions import hamming_loss, tanimoto_loss, zeroone_loss


class TestZeroOneLoss(unittest.TestCase):
    def test_correctness(self):
        y = "ABC"

        # Correct example is in the label list
        Y = ["VPB", "ASL", "ABC", "ASF"]
        np.testing.assert_equal(zeroone_loss(y, Y), [1, 1, 0, 1])

        # Correct example is not in the label list
        Y = ["VPB", "ASL", "CBA", "ASF"]
        np.testing.assert_equal(zeroone_loss(y, Y), [1, 1, 1, 1])

        # Correct example is multiple times in the label list
        Y = ["VPB", "ABC", "ABC", "ASF"]
        np.testing.assert_equal(zeroone_loss(y, Y), [1, 0, 0, 1])


class TestHammingLoss(unittest.TestCase):
    def test_against_sklearn(self):
        rs = np.random.RandomState(1920)

        for _ in range(100):
            d = rs.randint(1, 30)
            y = rs.randint(0, 2, size=d)
            n = rs.randint(1, 31)
            Y = rs.randint(0, 2, size=(n, d))

            loss = hamming_loss(y, Y)
            loss_sk = np.array([hamming_loss_sk(y, y_i) for y_i in Y])

            np.testing.assert_equal(loss, loss_sk)

            for y_i in Y:
                np.testing.assert_equal(hamming_loss(y, y_i), hamming_loss_sk(y, y_i))


class TestTanimotoLoss(unittest.TestCase):
    @staticmethod
    def _tanimoto_loss_element(x, y):
        x, y = set(np.where(x)[0]), set(np.where(y)[0])

        n_inter = len(x & y)
        n_union = len(x | y)

        return 1 - (n_inter / n_union)

    def test_correctness(self):
        rs = np.random.RandomState(1921)

        for _ in range(100):
            d = rs.randint(1, 200)
            y = (rs.rand(d) > 0.5).astype(float)

            n = rs.randint(1, 500)
            Y = (rs.rand(n, d) > 0.5).astype(float)

            # There should be fingerprint where all entries are zero
            y[rs.randint(d)] = 1
            for i in range(n):
                Y[i, rs.randint(d)] = 1

            # add the ground-truth fingerprint
            Y[-1] = y

            # Calculate the tanimoto loss
            L = tanimoto_loss(y, Y)

            self.assertEqual((n, ), L.shape)
            self.assertEqual(0, L[-1])
            self.assertTrue(np.all(L <= 1))
            self.assertTrue(np.all(L >= 0))

            for i in range(n):
                self.assertEqual(self._tanimoto_loss_element(y, Y[i]), L[i])


if __name__ == '__main__':
    unittest.main()
