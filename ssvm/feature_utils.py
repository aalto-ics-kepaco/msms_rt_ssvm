####
#
# The MIT License (MIT)
#
# Copyright 2021 Eric Bach <eric.bach@aalto.fi>
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

from typing import Union, List, Optional
from sklearn.base import TransformerMixin


class CountingFpsBinarizer(TransformerMixin):
    def __init__(self, bins_centers: Union[np.ndarray, List[int]]):
        self.bins_centers = bins_centers

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params):
        self.n_bins_ = len(self.bins_centers)
        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        Z = np.zeros((len(X), X.shape[1] * len(self.bins_centers)))
        for i in range(len(X)):
            for d in range(len(X[i])):
                v = X[i, d]
                if v > 0:
                    for e, t in enumerate(self.bins_centers):
                        if v >= t:
                            Z[i, d * self.n_bins_ + e] = 1
        return Z

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> np.ndarray:
        return self.fit(X, y, **fit_params).transform(X, y)
