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
import networkx as nx

from typing import Union, List, Optional

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances

from numba import jit, prange
from numba.typed import List as NumbaList


def get_rbf_gamma_based_in_median_heuristic(X: np.array, standardize: bool = False, n_jobs: int = 1) -> float:
    """
    Function implementing a heuristic to estimate the width of an RBF kernel (as defined in the Scikit-learn package)
    from data.

    :param X: array-like, shape = (n_samples, n_features), feature matrix

    :param standardize: boolean, indicating whether the data should be normalized (z-transformation) before the gamma is
        estimated.

    :param n_jobs: scalar, number of parallel jobs used to compute the pairwise distances

    :return: scalar, gamma (of the sklearn RBF kernel) estimated from the data
    """
    # Z-transform the data if requested
    if standardize:
        X = StandardScaler(copy=True).fit_transform(X)

    # Compute all pairwise euclidean distances
    D = pairwise_distances(X, n_jobs=n_jobs).flatten()

    # Get the median of the distances
    sigma = np.median(D)

    # Convert to sigma to gamma as defined in the sklearn package
    gamma = 1 / (2 * sigma**2)

    return gamma


class RemoveCorrelatedFeatures(SelectorMixin, BaseEstimator):
    def __init__(self, corr_threshold: float = 0.98):
        """
        The Bouwmeester feature selection based on the feature correlation. For that, we group highly correlated
        features and keep only one of each group.

        :param corr_threshold: scalar, correlation threshold over which the two features are considered to belong to the
            same group.
        """
        self.corr_threshold = corr_threshold

    def fit(self, X, y=None):
        """
        :param X: array-like, shape = (n_samples, n_features), feature matrix

        :param y: ignored

        :return: self
        """
        # Compute the absolute correlation between features
        R = np.abs(np.corrcoef(X.T))

        # Check for invalid correlation coefficients
        if np.any(np.isnan(R)):
            raise ValueError(
                "Nan-values in the correlation coefficients. Perhaps there are constant feature dimensions. Apply "
                "variance feature selector first."
            )

        # Generate a graph encoding the connection between highly correlated features
        G = nx.from_numpy_array(R > self.corr_threshold)

        # Fill the support mask: An index is one if the feature will be kept
        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)

        for cc in nx.connected_components(G):
            # Keep one node / feature per group of correlated features
            self.support_mask_[next(iter(cc))] = True

        return self

    def _get_support_mask(self) -> np.ndarray:
        """
        :return: array-like, shape = (n_features, ), binary mask indicating which features to keep
        """
        check_is_fitted(self)

        return self.support_mask_


class CountingFpsBinarizer(TransformerMixin, BaseEstimator):
    """
    Feature transformer to convert counting fingerprints to binary vectors, like:

        bin-centers [1, 4, 5]: cnt_fp = [0, 1, 15, 4] -> bin_fp = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0]

    The len of the bin-centers defines the number of bits per input feature dimension. If the count is zero, all bits
    will be set to zero, i.e. 0 -> 0, 0, 0. For values greater zero the conversion works like:

        - 1 -> 1, 0, 0
        - 2 -> 1, 0, 0
        - 3 -> 1, 0, 0
        - 4 -> 1, 1, 0
        - 5 -> 1, 1, 1
        - 6 -> 1, 1, 1
        ...

    This kind of feature conversion allows to use the the Tanimoto kernel to (approximately) calculate the
    minmax-similarity between molecules represented with counting vectors.
    """
    def __init__(self, bin_centers: Union[np.ndarray, List[np.ndarray]], compress: bool = False,
                 remove_constant_features: bool = False):
        """
        Constructor

        :param bin_centers: List of array-like | array-like, thresholds at which a one will be added to the encoding.
            Those thresholds can be specific for each feature dimension (List of array-like). This parameter is better
            understood by presenting an example:

            bin-centers [1, 4, 5]: cnt_fp = [0, 1, 15, 4] -> bin_fp = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0]

            If the count is zero, all bits will be set to zero, i.e. 0 -> 0, 0, 0. For values greater zero the
            conversion works like:

                - 1 -> 1, 0, 0
                - 2 -> 1, 0, 0
                - 3 -> 1, 0, 0
                - 4 -> 1, 1, 0
                - 5 -> 1, 1, 1
                - 6 -> 1, 1, 1
                ...

        :param compress: boolean, indicating whether thresholds are limited to the maximum value in each feature
            dimension. This can shorten the length of the output feature vectors.

        :param remove_constant_features: boolean, indicating whether constant feature dimensions (along axis=0) should
            be removed.
        """
        self.bin_centers = bin_centers
        self.compress = compress
        self.remove_constant_features = remove_constant_features

        if not isinstance(self.bin_centers, np.ndarray) and \
                not (isinstance(self.bin_centers, list) and isinstance(self.bin_centers[0], np.ndarray)):
            raise TypeError(
                "Bin-centers must be provided as numpy array or list of numpy arrays. Object of type {} passed in."
                .format(type(self.bin_centers))
            )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params):
        """
        Fit the binary encoding.
        """
        if isinstance(self.bin_centers, list):
            if len(self.bin_centers) != X.shape[1]:
                raise ValueError(
                    "If a list of bin-centers is provided, its length must be equal to the number of features."
                )
        else:
            # Repeat the bin-centers for all feature dimensions
            self.bin_centers = [np.array(self.bin_centers, dtype=X.dtype) for _ in range(X.shape[1])]

        # Fit a variance threshold feature selector to remove constant feature dimensions
        if self.remove_constant_features:
            self.d_mask_ = np.ptp(X, axis=0) > 0

            # Remove constant feature dimensions from the data
            X = X[:, self.d_mask_]

            # Remove corresponding bin-centers
            self.bin_centers = [self.bin_centers[d] for d in range(len(self.bin_centers)) if self.d_mask_[d]]

            assert len(self.bin_centers) == X.shape[1]

        # Restrict the binary encoding for each dimension to the maximum value in this dimension, that can shorten
        # the binary representations significantly
        if self.compress:
            max_val_per_feature = np.max(X, axis=0)
            self.bin_centers = [
                np.array([v for v in self.bin_centers[d] if v <= max_val_per_feature[d]], dtype=X.dtype)
                for d in range(X.shape[1])
            ]

        # Convert the list into a Numba list to suppress Numba warnings about deprecated data types.
        _bin_centers = NumbaList()
        for bc in self.bin_centers:
            _bin_centers.append(bc.astype(X.dtype))  # elements should have same data-type as X
        self.bin_centers = _bin_centers

        # Determine the final output dimension of the binary encoding
        self.d_out_ = sum(map(len, self.bin_centers))

        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        check_is_fitted(self, ["d_out_"])

        if self.remove_constant_features:
            check_is_fitted(self, ["d_mask_"])
            X = X[:, self.d_mask_]

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

    def __len__(self) -> int:
        check_is_fitted(self, ["d_out_"])
        return self.d_out_


if __name__ == "__main__":
    # trans = RemoveCorrelatedFeatures()
    #
    # X = np.random.RandomState(1092).rand(40, 10)
    #
    # X[:, 9] = X[:, 0]
    #
    # print(trans.fit(X).get_support())

    print(get_rbf_gamma_based_in_median_heuristic(np.random.rand(100, 10), standardize=True))
