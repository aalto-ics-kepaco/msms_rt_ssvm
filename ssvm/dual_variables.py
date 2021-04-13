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
import numpy as np

from copy import deepcopy
from scipy.sparse import csc_matrix, lil_matrix, csr_matrix
from sklearn.utils.validation import check_random_state

from typing import TypeVar, Optional, Union, List, Tuple

DUALVARIABLES_T = TypeVar('DUALVARIABLES_T', bound='DualVariables')


class DualVariables(object):
    def __init__(self, C: Union[int, float], label_space: List[List[List[str]]], num_init_active_vars: int = 1,
                 random_state: Optional[Union[int, np.random.RandomState]] = None, initialize: bool = True):
        """

        :param C: scalar, regularization parameter of the Structured SVM

        :param label_space: list of list of lists, of candidate identifiers. Each sequence element has an associated
            candidate set. The candidates are identified by a string.

            Example:

            [                               # Training sequence i
                [                           # Sequence element
                    [M1, M2, M3],           # Candidate set for sigma = 1
                    [M6, M3, M5, M7],       # Candidate set for sigma = 2
                    ...
                ]
                ...
            ]

        :param num_init_active_vars: scalar, number of initially active dual variables per

        :param random_state: None | int | instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.

        :param initialize: boolean, indicating whether the dual variables should be initialized upon object
            construction.
        """
        self.C = C
        assert self.C > 0, "The regularization parameter must be positive."
        self.N = len(label_space)
        self.random_state = random_state

        self.num_init_active_vars = num_init_active_vars
        assert self.num_init_active_vars > 0

        # Store a shuffled version of the candidate sets for each example and sequence element
        self.label_space = label_space

        if initialize:
            # Initialize the dual variables
            self._alphas, self._y2col, self._iy = self.initialize_alphas()
        else:
            self._alphas, self._y2col, self._iy = None, None, None

    def _assert_input_iy(self, i: int, y_seq: Tuple):
        """
        :raises: ValueError, if the label sequence 'y_seq' is not valid for example 'i'.

        :param i, scalar, sequence example index
        :param y_seq: tuple of strings, sequence of candidate indices identifying the dual variable
        """
        # Test: Are we trying to update a valid label sequence for the current example
        for sigma, y_sigma in enumerate(y_seq):
            if y_sigma not in self.label_space[i][sigma]:
                raise ValueError("For example %d at sequence element %d the label %s does not exist." %
                                 (i, sigma, y_sigma))

    def _is_initialized(self):
        if (self._alphas is None) or (self._y2col is None) or (self._iy is None):
            return False

        return True

    def assert_is_initialized(self):
        if not self._is_initialized():
            raise RuntimeError("Dual variables are not initialized. Call 'initialize_alphas' first.")

    def initialize_alphas(self):
        """
        Return initialized dual variables.

        :return: dictionary: key = label sequence, value = dual variable value
        """
        _alphas = lil_matrix((self.N, self.N * self.num_init_active_vars))
        _y2col = [{} for _ in range(self.N)]
        _iy = []

        col = 0
        for i in range(self.N):
            rs_i = check_random_state(self.random_state)

            n_added = 0
            n_rejections = 0
            # Lazy generation of sample sequences, no problem with exponential space here
            while (n_added < self.num_init_active_vars) and (n_rejections < 1000):
                y_seq = tuple(rs_i.choice(label_space_is, 1).item() for label_space_is in self.label_space[i])

                if y_seq not in _y2col[i]:
                    _alphas[i, col + n_added] = self.C / self.N
                    _y2col[i][y_seq] = col + n_added
                    _iy.append((i, y_seq))
                    n_added += 1
                else:
                    n_rejections += 1

            # Make initial values feasible and sum up to C / N
            _alphas[i, col:(col + n_added)] /= n_added

            col += n_added

        # We might have added less active variables than the maximum, because there where not enough distinct label
        # sequences. We can shrink the alpha-matrix.
        assert col <= _alphas.shape[1]
        _alphas.resize(self.N, col)
        assert not np.any(_alphas.tocsc().sum(axis=0) == 0)

        return _alphas, _y2col, _iy

    def set_alphas(self, iy: List[Tuple[int, Tuple[str, ...]]], alphas: Union[float, List[float], np.ndarray]) \
            -> DUALVARIABLES_T:
        """
        Sets the values for the active dual variables. Class must not be initialized before. This function does not
        do any checks on the feasibility of the resulting DualVariable object.

        :param iy: List of tuples, collecting the active dual variables.

        :param alphas: List or scalar, dual value(s) corresponding to the active variables. If a scalar is provided, its
            repeated to the length of 'iy'.

        :return: The DualVariable class itself.
        """
        if isinstance(alphas, List) or isinstance(alphas, np.ndarray):
            if len(iy) != len(alphas):
                raise ValueError("Number of dual variables values must be equal the number of active dual variables.")
        elif np.isscalar(alphas):
            alphas = np.full((len(iy),), fill_value=alphas)
        else:
            raise ValueError("Dual variables must be passed as list or scalar.")

        if self._is_initialized():
            raise RuntimeError("Cannot set alphas of an already initialized dual variable. Use 'update' function "
                               "instead.")

        _alphas = lil_matrix((self.N, len(iy)))
        _y2col = [{} for _ in range(self.N)]

        for col, ((i, y_seq), a) in enumerate(zip(iy, alphas)):
            # Check that the dual variables that are supposed to be set, are actually valid, i.e. the corresponding
            # label sets contain the provided sequences.
            self._assert_input_iy(i, y_seq)

            if y_seq in _y2col[i]:
                raise ValueError("Each active dual variable should only appear ones in the provided set. Does not hold"
                                 "for example 'i=%d' with sequence 'y_seq=%s'" % (i, y_seq))

            _alphas[i, col] = a
            _y2col[i][y_seq] = col

        self._alphas = _alphas
        self._y2col = _y2col
        self._iy = iy

        return self

    def update(self, i: int, y_seq: Tuple, gamma: float) -> bool:
        """
        Update the value of a dual variable.

        :param i, scalar, sequence example index

        :param y_seq: tuple of strings, sequence of candidate indices identifying the dual variable

        :param gamma: scalar, step-width used to update the dual variable value

        :return: boolean, indicating whether a new variable was added to the set of active variables.
        """
        self.assert_is_initialized()
        self._assert_input_iy(i, y_seq)

        try:
            # Update an active dual variable
            is_new = False
            col = self._y2col[i][y_seq]  # This might throw a KeyError, if the dual variable is not active right now.
            self._alphas[i, col] += gamma * ((self.C / self.N) - self._alphas[i, col])
        except KeyError:
            # Add a new active dual variable by adding a new column
            is_new = True
            self._alphas.resize(self._alphas.shape[0], self._alphas.shape[1] + 1)
            col = self._alphas.shape[1] - 1
            self._alphas[i, col] = gamma * (self.C / self.N)
            self._y2col[i][y_seq] = col
            self._iy.append((i, y_seq))

        # Update all remaining dual variables for the current example
        _r = gamma * self._alphas[i]
        _r[0, col] = 0  # we already updated the selected candidate y_seq
        self._alphas[i] -= _r

        return is_new

    def get_dual_variable(self, i: int, y_seq: tuple) -> float:
        """
        :param i, scalar, sequence example index
        :param y_seq: tuple of strings, sequence of candidate indices identifying the dual variable

        :return: scalar, dual variable value
        """
        self.assert_is_initialized()
        self._assert_input_iy(i, y_seq)

        try:
            col = self._y2col[i][y_seq]  # This might throw a KeyError, if the dual variable is not active right now.
            val = self._alphas[i, col]
        except KeyError:
            val = 0.0

        return val

    def get_dual_variable_matrix(self, type="csr") -> Union[csr_matrix, csc_matrix]:
        """
        Returns the dual variable matrix as a csr-sparse matrix.

        :return: csr_matrix, shape = (N, n_active_dual)
        """
        self.assert_is_initialized()

        if type == "csr":
            return csr_matrix(self._alphas)
        elif type == "csc":
            return csc_matrix(self._alphas)
        elif type == "dense":
            return self._alphas.toarray()
        else:
            raise ValueError("Invalid sparse matrix type: '%s'. Choices are 'csr' and 'csc'.")

    def n_active(self, i: Optional[int] = None) -> int:
        """
        Numer of active dual variables.

        :return: scalar, number of active dual variables (columns in the dual variable matrix)
        """
        self.assert_is_initialized()

        if i is None:
            return self._alphas.getnnz()
        else:
            return self._alphas[i].getnnz()

    def get_iy_for_col(self, c: int) -> Tuple[int, Tuple]:
        """
        Return the example index and candidate sequence corresponding to the specified column in the dual variable
        matrix.

        :param c: scalar, column in the dual variable matrix, i.e. the index of the active dual variable for which the
            corresponding example index and candidate sequence should be returned.

        :return: tuple (
            scalar, example index in {1, ..., N}
            tuple, candidate sequence associated with the active dual variable
        )
        """
        self.assert_is_initialized()

        return self._iy[c]

    def get_blocks(self, i: Optional[Union[List[int], np.ndarray, int]] = None) \
            -> Tuple[List[Tuple[int, Tuple[str, ...]]], List[float]]:
        """
        Returns the active sequences and dual variable values for the requested example(s).

        :param i: scalar or array-like, example sequence index or indices. If None, the active sequences and dual values
            for all examples are returned.

        :return: tuple (
            list of tuples, (example index, label sequence),
            list of scalars, associated dual variable values
        )
        """
        if i is None:
            i = np.arange(self.N)
        else:
            i = np.atleast_1d(i)

        iy = []
        a = []
        for _i in i:
            for y_seq in self._y2col[_i]:
                _a = self._alphas[_i, self._y2col[_i][y_seq]]
                if _a != 0:
                    iy.append((_i, y_seq))
                    a.append(_a)

        return iy, a

    def iter(self, i: int):
        assert 0 <= i < self.N

        for y_seq in self._y2col[i]:
            yield y_seq, self._alphas[i, self._y2col[i][y_seq]]

    def __mul__(self, fac: Union[float, int]) -> DUALVARIABLES_T:
        if not np.isscalar(fac):
            raise ValueError("Can only multiply with scalar.")

        if fac == 0:
            return DualVariables(self.C, self.label_space, initialize=False).set_alphas([], [])
        else:
            out = deepcopy(self)
            out._alphas *= fac
            return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other: DUALVARIABLES_T) -> DUALVARIABLES_T:
        return self._add(-1 * other)

    def __add__(self, other: DUALVARIABLES_T) -> DUALVARIABLES_T:
        return self._add(other)

    def _add(self, other: DUALVARIABLES_T) -> DUALVARIABLES_T:
        """
        Creates an DualVariable object where:

            a_sub(i, y) = a_self(i, y) + a_other(i, y)      for all i and y in Sigma_i

        :param other: DualVariable, add to 'self'

        :return: DualVariable, defined over the same domain. The active dual variables are defined by the union of the
            active dual variables of 'self' and 'other'. Only non-zero dual variables are active. The dual values
            represent the sum of the dual values.

        Note: The resulting DualVariable object can store the dual values of the active variables in an arbitrary order.
              The columns of the dual variable matrix containing the added values, can have a different order than
              the original matrix. Therefore, the active fingerprint set needs to be rebuild after the subtraction
              operation to ensure that the dual values are in the correct order.
        """

        self.assert_is_initialized()
        other.assert_is_initialized()

        # Check the dual variables in 'other' are defined over the same examples and label sets
        if not self._eq_dual_domain(self, other):
            raise ValueError("Dual variables must be defined over the same domain and with the same regularization"
                             "parameter C.")

        # Determine the union of the active dual variables for both sets
        _iy_union = list(set(self._iy + other._iy))
        _alphas_union = []
        for col, (i, y_seq) in enumerate(_iy_union):
            # Get dual variable value from 'self' (or left)
            try:
                col = self._y2col[i][y_seq]
                a_left = self._alphas[i, col]
            except KeyError:
                a_left = 0.0

            # Get dual variable value from 'other' (or right)
            try:
                col = other._y2col[i][y_seq]
                a_right = other._alphas[i, col]
            except KeyError:
                a_right = 0.0

            _alphas_union.append(a_left + a_right)

        # Remove all dual variables those value got zero by the subtraction
        _iy = [(i, y_seq) for (i, y_seq), a in zip(_iy_union, _alphas_union) if a != 0]
        _alphas = [a for a in _alphas_union if a != 0]

        return DualVariables(C=self.C, label_space=self.label_space, initialize=False).set_alphas(_iy, _alphas)

    @staticmethod
    def _eq_dual_domain(left: DUALVARIABLES_T, right: DUALVARIABLES_T) -> bool:
        """
        Checks whether two DualVariables classes are defined over the same domain.

        :param left: DualVariable, first object

        :param right: DualVariable, second object

        :return: boolean, indicating whether the two DualVariable classes are defined over the same domain.
        """
        if left.C != right.C:
            return False

        if left.N != right.N:
            return False

        for i in range(left.N):
            if len(left.label_space[i]) != len(right.label_space[i]):
                return False

            for sigma in range(len(left.label_space[i])):
                if set(left.label_space[i][sigma]) != set(right.label_space[i][sigma]):
                    return False

        return True

    @staticmethod
    def get_s(alphas: DUALVARIABLES_T, y_I_hat: List[Tuple[str, ...]], I_batch: Optional[List[int]] = None) \
            -> DUALVARIABLES_T:
        """
        """
        C = alphas.C
        N = alphas.N

        if I_batch is None:
            assert len(y_I_hat) == N
            I_batch = list(range(N))

        i2idx_batch = {i: idx for idx, i in enumerate(I_batch)}

        iy, a = [], []
        for i in range(N):
            try:
                # Example i was updated in the batch (I_batch)
                iy.append((i, y_I_hat[i2idx_batch[i]]))
                a.append(C / N)
            except KeyError:
                # Example i was not updated
                _iy, _a = alphas.get_blocks(i)
                iy += _iy
                a += _a

        return DualVariables(C=C, label_space=alphas.label_space, initialize=False).set_alphas(iy, a)