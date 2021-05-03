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
from sklearn.utils.validation import check_is_fitted


class CountingFpsBinarizer(TransformerMixin):
    def __init__(self, bin_centers: Union[np.ndarray, List[int]], compress: bool = False):
        self.bin_centers = bin_centers
        self.compress = compress

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params):
        self.n_bins_ = len(self.bin_centers)

        if self.compress:
            max_val_per_feature = np.max(X, axis=0)
            self.bin_centers_list_ = [
                [
                    v for v in self.bin_centers if v <= max_val_per_feature[d]
                ]
                for d in range(X.shape[1])
            ]
            self.d_out_ = sum(map(len, self.bin_centers_list_))
        else:
            self.d_out_ = X.shape[1] * len(self.bin_centers)

        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        Z = np.zeros((len(X), self.d_out_))

        if self.compress:
            check_is_fitted(self, "bin_centers_list_")
            for i in range(len(X)):
                e = 0
                for d in range(len(X[i])):
                    v = X[i, d]
                    if v > 0:
                        for t in self.bin_centers_list_[d]:
                            if v >= t:
                                Z[i, e] = 1

                            e += 1
                    else:
                        e += len(self.bin_centers_list_[d])
        else:
            for i in range(len(X)):
                e = 0
                for d in range(len(X[i])):
                    v = X[i, d]
                    if v > 0:
                        for t in self.bin_centers:
                            if v >= t:
                                Z[i, e] = 1

                            e += 1
                    else:
                        e += self.n_bins_

        return Z

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> np.ndarray:
        return self.fit(X, y, **fit_params).transform(X, y)
