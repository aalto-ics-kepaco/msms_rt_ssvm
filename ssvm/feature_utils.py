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
from numba import jit, prange
from numba.typed import List as NumbaList


class CountingFpsBinarizer(TransformerMixin):
    def __init__(self, bin_centers: Union[np.ndarray, List[np.ndarray]], compress: bool = False):
        self.bin_centers = bin_centers
        self.compress = compress

        if not isinstance(self.bin_centers, np.ndarray) and \
                not (isinstance(self.bin_centers, list) and isinstance(self.bin_centers[0], np.ndarray)):
            raise TypeError(
                "Bin-centers must be provided as numpy array or list of numpy arrays. Object of type {} passed in."
                .format(type(self.bin_centers))
            )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params):
        # Convert the list into a Numba list to suppress Numba warnings about deprecated data types.
        if isinstance(self.bin_centers, list):
            assert isinstance(self.bin_centers[0], np.ndarray)
            _bin_centers = NumbaList()
            for d in range(X.shape[1]):
                _bin_centers.append(self.bin_centers[d].astype(X.dtype))
            self.bin_centers = _bin_centers
        else:
            assert isinstance(self.bin_centers, np.ndarray)
            _bin_centers = NumbaList()
            for _ in range(X.shape[1]):
                _bin_centers.append(np.array(self.bin_centers, dtype=X.dtype))
            self.bin_centers = _bin_centers

        if self.compress:
            max_val_per_feature = np.max(X, axis=0)
            _bin_centers = NumbaList()
            for d in range(X.shape[1]):
                _bin_centers.append(
                    np.array([v for v in self.bin_centers[d] if v <= max_val_per_feature[d]], dtype=X.dtype)
                )
            self.bin_centers = _bin_centers

        self.d_out_ = sum(map(len, self.bin_centers))

        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        check_is_fitted(self, ["d_out_"])

        return self._transform(X, self.bin_centers, self.d_out_, len(X))

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _transform(X: np.ndarray, bin_centers: List[np.ndarray], d_out: int, n_samples: int) -> np.ndarray:
        Z = np.zeros((len(X), d_out))

        for i in prange(n_samples):
            e = 0
            for d in range(len(X[i])):
                Z[i, e:(e + len(bin_centers[d]))] = (X[i, d] >= bin_centers[d])
                e += len(bin_centers[d])

        return Z

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> np.ndarray:
        return self.fit(X, y, **fit_params).transform(X, y)
