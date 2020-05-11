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
from typing import List
from sklearn.base import BaseEstimator, ClassifierMixin

from ssvm.sequence import Sequence


class StructuredSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        """
        Structured Support Vector Machine (SSVM) class.

        :param C: scalar, SVM regularization parameter. Must be > 0.
        """
        self.C = C

    def fit(self, X: List[Sequence], y: List[List[str]]):
        """
        Train the SSVM given a dataset.

        :param X: list of tuples, of length N. Each element X_i, i.e. X[i], is a tuple (x_i, t_i, C_i) and represents
            a (MS, RT) training sequence:

            x_i: list of array-likes, of length L. Each element x_{is}, i.e. X[i][0][s], corresponds the either to the
                feature or kernel similarity vector of a sequence element s with the other training data
            t_i: array-like, of length L. Each element t_{is}, i.e. X[i][1][s], is the retention time of sequence
                element s. Therefore, it is a scalar value.
            C_i: list of array-likes. of length L. Each element C_{is}, i.e. X[i][2][s], corresponds to either to the
                feature or kernel similarity vector of the molecular candidates of the sequence element s
                TODO: Here we need to versions. (1) The embedding for the MS, and (2) for the Retention Order

        :param y: list of array-likes,
        :return:
        """