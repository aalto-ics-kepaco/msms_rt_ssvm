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
import unittest
import numpy as np

from sklearn.metrics import hamming_loss as hamming_loss_sk

from ssvm.loss_functions import hamming_loss


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


if __name__ == '__main__':
    unittest.main()
