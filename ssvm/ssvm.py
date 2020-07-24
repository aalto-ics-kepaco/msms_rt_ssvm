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
import numpy as np
import itertools as it
import more_itertools as mit
import tensorflow as tf

from typing import List, ItemsView, Tuple, ValuesView, KeysView, Iterator, Dict, Union, Optional, TypeVar
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import GroupKFold
from scipy.sparse import csc_matrix, lil_matrix, csr_matrix
from tensorflow.python.ops.summary_ops_v2 import ResourceSummaryWriter
from tqdm import tqdm

from ssvm.data_structures import SequenceSample, CandidateSetMetIdent
from ssvm.loss_functions import hamming_loss
from ssvm.evaluation_tools import get_topk_performance_csifingerid

DUALVARIABLES_T = TypeVar('DUALVARIABLES_T', bound='DualVariables')


class DualVariables(object):
    def __init__(self, C: float, cand_ids: List[List[List[str]]], num_init_active_vars=1, rs=None, initialize=True):
        """

        :param C: scalar, regularization parameter of the Structured SVM

        :param cand_ids: list of list of lists, of candidate identifiers. Each sequence element has an associated
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

        :param rs: None | int | instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.

        :param initialize: boolean, indicating whether the dual variables should be initialized upon object
            construction.
        """
        self.C = C
        assert self.C > 0, "The regularization parameter must be positive."
        self.N = len(cand_ids)
        # Store a shuffled version of the candidate sets for each example and sequence element
        self.rs = check_random_state(rs)

        self.num_init_active_vars = num_init_active_vars
        assert self.num_init_active_vars > 0

        if initialize:
            self.l_cand_ids = []
            for i in range(self.N):
                self.l_cand_ids.append([self.rs.permutation(_cand_ids).tolist() for _cand_ids in cand_ids[i]])
            # Initialize the dual variables
            self.initialize_alphas()
        else:
            self.l_cand_ids = cand_ids
            self._alphas = None  # type: lil_matrix
            self._y2col = None
            self._iy = None

    def _assert_input_iy(self, i: int, y_seq: Tuple):
        """
        :raises: ValueError, if the label sequence 'y_seq' is not valid for example 'i'.

        :param i, scalar, sequence example index
        :param y_seq: tuple of strings, sequence of candidate indices identifying the dual variable
        """
        # Test: Are we trying to update a valid label sequence for the current example
        for sigma, y_sigma in enumerate(y_seq):
            if y_sigma not in self.l_cand_ids[i][sigma]:
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
            n_added = 0
            # Lazy generation of sample sequences, no problem with exponential space here
            # FIXME: Produces heavily biased sequences, as last indices is increased first, then the second last ...
            for y_seq in it.product(*self.l_cand_ids[i]):
                assert y_seq not in _y2col[i], "What, that should not happen."

                _alphas[i, col + n_added] = self.C / self.N
                _y2col[i][y_seq] = col + n_added
                _iy.append((i, y_seq))
                n_added += 1

                # Stop after we have added the desired amount of active variables. This probably happens before we run
                # out of new label sequences.
                if n_added == self.num_init_active_vars:
                    break

            # Make initial values feasible and sum up to C / N
            _alphas[i, col:(col + n_added)] /= n_added

            col += n_added

        # We might have added less active variables than the maximum, because there where not enough distinct label
        # sequences. We can shrink the alpha-matrix.
        assert col <= _alphas.shape[1]
        _alphas.resize(self.N, col)
        assert not np.any(_alphas.tocsc().sum(axis=0) == 0)

        # TODO For faster verification of the label sequence correctness, we transform all candidate set lists in to sets
        # self.l_cand_ids = [for cand_ids_seq in (for cand_ids_exp in self.l_cand_ids)]

        self._alphas = _alphas
        self._y2col = _y2col
        self._iy = _iy

    def set_alphas(self, iy: List[Tuple[int, Tuple]], alphas: Union[float, List[float], np.ndarray]) -> DUALVARIABLES_T:
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
            alphas = [alphas] * len(iy)
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

    def n_active(self) -> int:
        """
        Numer of active dual variables.

        :return: scalar, number of active dual variables (columns in the dual variable matrix)
        """
        self.assert_is_initialized()

        return self._alphas.shape[1]

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

    def get_blocks(self, i: Union[List[int], np.ndarray, int]) -> Tuple[List[Tuple[int, Tuple]], List[float]]:
        i = np.atleast_1d(i)
        assert np.all(np.isin(i, np.arange(self.N)))

        iy = []
        a = []
        for _i in i:
            for y_seq in self._y2col[_i]:
                iy.append((_i, y_seq))
                a.append(self.get_dual_variable(_i, y_seq))

        return iy, a

    def __sub__(self, other: DUALVARIABLES_T) -> DUALVARIABLES_T:
        """
        Creates an DualVariable object where:

            a_sub(i, y) = a_self(i, y) - a_other(i, y)      for all i and y in Sigma_i

        :param other: DualVariable, to subtract from 'self'

        :return: DualVariable, defined over the same domain. The active dual variables are defined by the union of the
            active dual variables of 'self' and 'other'. Only non-zero dual variables are active. The dual values
            represent the difference between the dual values.
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

            _alphas_union.append(a_left - a_right)

        # Remove all dual variables those value got zero by the subtraction
        _iy = [(i, y_seq) for (i, y_seq), a in zip(_iy_union, _alphas_union) if a != 0]
        _alphas = [a for a in _alphas_union if a != 0]

        return DualVariables(C=self.C, cand_ids=self.l_cand_ids, initialize=False).set_alphas(_iy, _alphas)

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
            if len(left.l_cand_ids[i]) != len(right.l_cand_ids[i]):
                return False

            for sigma in range(len(left.l_cand_ids[i])):
                if set(left.l_cand_ids[i][sigma]) != set(right.l_cand_ids[i][sigma]):
                    return False

        return True


class DualVariablesForExample(object):
    def __init__(self, C: float, N: int, cand_ids: List[List[str]], num_init_active_vars=1, rs=None):
        """

        :param C: scalar, regularization parameter of the Structured SVM

        :param N: scalar, total number of sequences used for training

        :param cand_ids: list of lists, of candidate identifiers. Each sequence element has an associated candidate set.
            The candidates are identified by a string.

            Example:

                [
                    [M1, M2, M3],
                    [M6, M3, M5, M7],
                    ...
                ]

        :param num_init_active_vars: scalar, number of initially active dual variables per

        :param rs: None | int | instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.
        """
        self.C = C
        assert self.C > 0, "The regularization parameter must be positive."
        self.N = N
        assert self.N > 0
        self.l_cand_ids = cand_ids
        self.num_init_active_vars = num_init_active_vars
        assert self.num_init_active_vars > 0
        self.rs = check_random_state(rs)

        # Initialize the dual variables
        self._alphas = self._get_initial_alphas()

    def _get_initial_alphas(self) -> dict:
        """
        Return initialized dual variables.

        :return: dictionary: key = label sequence, value = dual variable value
        """
        # Active dual variables are stored in a dictionary: key = label sequence, value = variable value
        _alphas = {}

        # Initial dual variable value ensuring feasibility
        init_var_val = self.C / (self.N * self.num_init_active_vars)  # (C / N) / num_act = C / (N * num_act)

        for i in range(self.num_init_active_vars):
            y_seq = tuple(self.rs.choice(cand_ids) for cand_ids in self.l_cand_ids)
            assert y_seq not in _alphas, "Oups, we sampled the same active dual variable again."

            _alphas[y_seq] = init_var_val

        return _alphas

    def __len__(self) -> int:
        """
        :return: scalar, number of active dual variables
        """
        return len(self._alphas)

    def __iter__(self) -> Iterator:
        return self._alphas.__iter__()

    def __getitem__(self, y_seq: tuple) -> float:
        try:
            return self._alphas[y_seq]
        except KeyError:
            return 0.0

    def is_active(self, y_seq: tuple) -> bool:
        return y_seq in self._alphas

    def items(self) -> ItemsView[Tuple, float]:
        return self._alphas.items()

    def values(self) -> ValuesView[float]:
        return self._alphas.values()

    def keys(self) -> KeysView[Tuple]:
        return self._alphas.keys()

    def update(self, y_seq: tuple, gamma: float):
        """
        Update the value of a dual variable.

        :param y_seq: tuple of strings, sequence of candidate indices identifying the dual variable
        :param gamma: scalar, step-width used to update the dual variable value
        """
        for y_seq_active in self._alphas:
            if y_seq == y_seq_active:
                continue

            self._alphas[y_seq_active] -= (gamma * self._alphas[y_seq_active])

        try:
            # Update an active dual variable
            # self._alphas[y_seq] = (1 - gamma) * self._alphas[y_seq] + gamma * (self.C / self.N)
            self._alphas[y_seq] += gamma * ((self.C / self.N) - self._alphas[y_seq])
        except KeyError:
            # Add a new dual active dual variable
            self._alphas[y_seq] = gamma * (self.C / self.N)


class _StructuredSVM(object):
    def __init__(self, C=1.0, n_epochs=1000, label_loss="hamming", rs=None, stepsize="diminishing"):
        """
        Structured Support Vector Machine (SSVM) class.

        :param C: scalar, SVM regularization parameter. Must be > 0.
        :param n_epochs: scalar, Number of epochs, i.e. maximum number of iterations.
        :param label_loss: string, Which label loss should be used. Default is 'hamming' loss.
        """
        self.C = C
        self.max_n_epochs = n_epochs
        self.label_loss = label_loss

        if self.label_loss == "hamming":
            self.label_loss_fun = hamming_loss
        else:
            raise ValueError("Invalid label loss '%s'. Choices are 'hamming'.")

        self.stepsize = stepsize
        if self.stepsize not in ["diminishing", "linesearch"]:
            raise ValueError("Invalid stepsize method '%s'. Choices are 'diminishing' and 'linesearch'." %
                             self.stepsize)

        self.rs = check_random_state(rs)  # type: np.random.RandomState

    @staticmethod
    def _is_feasible(alphas: List[DualVariablesForExample], C: float) -> bool:
        """
        Check the feasibility of the dual variable.

        :param alphas:
        :param C:
        :return:
        """
        val = C / len(alphas)  # C / N
        for a_i in alphas:
            sum_a_i = 0
            for a_iy in a_i.values():
                # Alphas need to be positive
                if a_iy < 0:
                    print("Dual variable is smaller 0: %f" % a_iy)
                    return False

                sum_a_i += a_iy

            if not np.isclose(sum_a_i, val):
                print("Dual variable does not sum to C / N: (expected) %f - (actual) %f = %f" % (
                    val, sum_a_i, sum_a_i - val))
                return False

        return True

    @staticmethod
    def _get_step_size_diminishing(k: int, N: int) -> float:
        """
        Step-width calculation after [1].

        :param k:
        :param N:
        :return:
        """
        return (2 * N) / (k + 2 * N)


class StructuredSVMMetIdent(_StructuredSVM):
    def __init__(self, C=1.0, n_epochs=1000, label_loss="hamming", rs=None, batch_size=1, stepsize="diminishing"):
        self.batch_size = batch_size

        # States defining a fitted SSVM Model
        self.K_train = None  # type: np.ndarray
        self.y_train = None
        self.fps_active = None
        self.alphas = None  # type: DualVariables
        self.N = None

        super(StructuredSVMMetIdent, self).__init__(C=C, n_epochs=n_epochs, label_loss=label_loss, rs=rs,
                                                    stepsize=stepsize)

    @staticmethod
    def _sanitize_fit_args(idict, keys, bool_default):
        if idict is None:
            idict = {}

        for key in keys:
            if key not in idict:
                idict[key] = bool_default

        return idict

    def fit(self, X: np.ndarray, y: np.ndarray, candidates: CandidateSetMetIdent, num_init_active_vars_per_seq=1,
            debug_args: Optional[Dict] = None, pre_calc_args: Optional[Dict] = None):
        """
        Train the SSVM given a dataset.

        :param debug_args: dict or None, Dictionary containing parameters for the debugging. If it is None, no debugging
            or any other information, such as objective values, are tracked during the fitting process.

            Dictionary (key: value):

                "track_objectives": boolean, indicating whether primal and dual objective values as well as the duality
                    gap should be printed / written to tensorboard.

                "track_dual_variables": boolean, indicating whether information regarding the dual variables should be
                    tracked. Those include a dual value histogram, the number of active variables (a(i, y) > 0) and the
                    minimum dual value (min_(i, y) a(i, y)).

                "track_stepsize": boolean, indicating whether the step-size should be tracked.

                "track_topk_acc": boolean, indicating whether the top-k accuracies (1, 5, 10, 20) should be calculated
                    during the optimization. If True, the training set is split into training' (90%) and validation
                    (10%). The accuracy is calculated for both sets.

                "summary_writer": ResourceSummaryWriter, output the debugging information to Tensorboard. If this key is
                    missing, the debugging information will be only printed to the standard output.

        :param pre_calc_args: dict or None, Dictionary defining which data will be pre-computed, i.e. we trade
            memory against CPU. If needed, the pre_calculated data is updated, e.g. if the set of active variables
            changes. If it is None, no all data will be computed on the fly (should have the lowest memory footprint).

            Dictionary (key: value)

                "pre_calc_label_losses": boolean, indicating whether the label losses between the ground truth molecular
                    structure and their corresponding candidate structures (for all examples i) should be calculated.
                    This parameter does not effect the label losses between the gt structures and the active ones, as
                    this data is always pre-computed.

                "pre_calc_L_Ci_S_matrices": boolean, indicating whether the matrices between the candidate and active
                    molecular structures should be pre-calculated and updated. Those matrices can become quite large
                    and therefore consume a lot of memory. In total, all matrices consume the space of (M, |S|), where
                    M is practically the number of _all_ candidates.
                    WARN: UPDATE NEEDED WHEN ACTIVE SET CHANGES

                "pre_calc_L_Ci_matrices": boolean, indicating whether the matrices between training molecular structures
                    and the candidate structures should be pre-calculated (for all examples i). In total, all matrices
                    consume the space of (M, N).

                --- Related to debug information ---

                "pre_calc_L_matrices": boolean, indicating whether the L (n_train, n_train), L_S (|S|, n_train) and L_SS
                    (|S|, |S|) matrices should pre-computed. This are needed for the objective value tracking. This
                    parameter has an effect only, if 'track_objectives' is True.
                    WARN: UPDATE NEEDED WHEN ACTIVE SET CHANGES
        """
        # Simple check for kernelized inputs on the spectra side
        if not np.array_equal(X, X.T):
            raise ValueError("Currently we only except kernelized inputs on the spectra side. This is tested, by "
                             "checking X == X.T.")

        # Handle debug arguments
        debug_args = self._sanitize_fit_args(
            debug_args, ["track_objectives", "track_topk_acc", "track_stepsize", "track_dual_variables"], False)

        if debug_args["track_topk_acc"]:
            # Split full training dataset into training' (90%) and validation (10%)
            train_set, val_set = next(GroupKFold(n_splits=10).split(X, groups=y))

            # Here:
            #   n_train  ... original size of full training set
            #   n_train' ... size of 90% of the original training set
            #   n_val    ... size of 10% of the original training set, used as validation set
            data_for_topk_acc = {
                "X_val": X[np.ix_(val_set, train_set)],  # shape = (n_val, n_train')
                "y_val": y[val_set]                      # shape = (n_val, )
            }

            # New data for training: training'
            X = X[np.ix_(train_set, train_set)]  # shape = (n_train', n_train')
            y = y[train_set]                     # shape = (n_train', )
        else:
            data_for_topk_acc = {}

        # Assign the training inputs and labels (candidate identifier)
        self.K_train = X
        self.y_train = y
        self.N = len(self.K_train)  # Number of training examples: n_train or n_train' depending on the debug args.

        # Initialize dual variables
        print("Initialize dual variables: ...", end="")
        self.alphas = DualVariables(C=self.C, cand_ids=[[candidates.get_labelspace(y[i])] for i in range(self.N)],
                                    rs=self.rs, num_init_active_vars=num_init_active_vars_per_seq)
        assert self._is_feasible_matrix(self.alphas, self.C), "Initial dual variables must be feasible."
        print("100")

        # Collect fingerprints and label losses corresponding to the (initially) active dual variables
        self.fps_active, label_losses_active = self._get_active_fingerprints_and_losses(
            self.alphas, self.y_train, candidates, verbose=True)
        # fps_active with shape           (|S|, d_fps) DONE: Update after every step
        # label_losses_active with shape  (|S|, )      DONE: Update after every step

        # Pre-calculate data as requested
        pre_calc_args = self._sanitize_fit_args(
            pre_calc_args, ["pre_calc_label_losses", "pre_calc_L_Ci_S_matrices", "pre_calc_L_Ci_matrices",
                            "pre_calc_L_matrices"], False)

        label_losses = {}       # Memory: O(M)
        mol_kernel_L_Ci = {}    # Memory: O(N x M)
        mol_kernel_L_S_Ci = {}  # Memory: O(|S| x M)    DONE: Update for current batch before each step.
        #                       TODO: For all data every k'th step for debugging.
        #   NOTE(1): |S| grows with the iterations
        #   NOTE(2): The matrices are stored in transpose (|S| x |C_i|). This allows the usage
        #            of the "np.ndarray.resize" function. Matrices are transposed when accessed.
        for i in tqdm(range(self.N), desc="Pre-calculate data (Losses, L_Ci and L_Ci_S matrices"):
            y_i = self.y_train[i]

            if pre_calc_args["pre_calc_label_losses"] and y_i not in label_losses:
                # Label loss: Loss of the gt fingerprint to all corresponding candidate fingerprints of i
                label_losses[y_i] = self.label_loss_fun(candidates.get_gt_fp(y_i), candidates.get_candidates_fp(y_i))

            if pre_calc_args["pre_calc_L_Ci_matrices"] and y_i not in mol_kernel_L_Ci:
                # Kernels between the training examples and candidates
                mol_kernel_L_Ci[y_i] = candidates.getMolKernel_ExpVsCand(self.y_train, y_i).T  # shape = (|Sigma_i|, N)

            if pre_calc_args["pre_calc_L_Ci_S_matrices"] and y_i not in mol_kernel_L_S_Ci:
                # Kernels between active structures and all candidates of i
                mol_kernel_L_S_Ci[y_i] = candidates.get_kernel(self.fps_active, candidates.get_candidates_fp(y_i))
                # shape = (|S|, |C_i|)

        pre_calc_data = {"label_losses_active": label_losses_active,
                         "label_losses": label_losses,
                         "mol_kernel_L_Ci": mol_kernel_L_Ci,
                         "mol_kernel_L_S_Ci": mol_kernel_L_S_Ci}

        # Memory: O(N**2) + O(|S| x N) + O(|S| x |S|)
        if pre_calc_args["pre_calc_L_matrices"]:
            # Pre-calculate kernel molecule kernel matrices
            pre_calc_data["L"] = candidates.getMolKernel_ExpVsExp(self.y_train)
            pre_calc_data["L_S"] = candidates.get_kernel(self.fps_active, candidates.get_gt_fp(self.y_train))
            pre_calc_data["L_SS"] = candidates.get_kernel(self.fps_active)
            # L with shape    (N, N)
            # L_S with shape  (|S|, N)                  DONE: Update only every k'th step for debugging
            # L_SS with sahep (|S|, |S|)                DONE: Update only every k'th step for debugging

        k = 0
        self._write_debug_output(debug_args, epoch=0, step_batch=0, step_total=k, stepsize=np.nan,
                                 data_for_topk_acc=data_for_topk_acc, pre_calc_data=pre_calc_data,
                                 candidates=candidates)

        for epoch in range(self.max_n_epochs):  # Full pass over the data
            for step, batch in enumerate(
                    mit.chunked(np.random.RandomState(epoch).permutation(np.arange(self.N)), self.batch_size)):
                # Update mol_kernel_L_S_Ci for the current batch
                # ----------------------------------------------
                if pre_calc_args["pre_calc_L_Ci_S_matrices"]:
                    for i in batch:
                        y_i = self.y_train[i]
                        if pre_calc_data["mol_kernel_L_S_Ci"][y_i].shape[0] == self.fps_active.shape[0]:
                            # Nothing to update
                            continue

                        # example i requires an update for the L_S_Ci
                        _n_missing = self.fps_active.shape[0] - pre_calc_data["mol_kernel_L_S_Ci"][y_i].shape[0]
                        assert _n_missing > 0

                        # Change size by adding one row
                        pre_calc_data["mol_kernel_L_S_Ci"][y_i].resize(
                            (pre_calc_data["mol_kernel_L_S_Ci"][y_i].shape[0] + _n_missing,
                             pre_calc_data["mol_kernel_L_S_Ci"][y_i].shape[1]),
                            refcheck=False)

                        # Fill the missing values
                        pre_calc_data["mol_kernel_L_S_Ci"][y_i][-_n_missing:] = candidates.get_kernel(
                            self.fps_active[-_n_missing:], candidates.get_candidates_fp(y_i))
                        assert not np.any(np.isnan(pre_calc_data["mol_kernel_L_S_Ci"][y_i]))

                # -----------------
                # SOLVE SUB-PROBLEM
                # -----------------
                res_batch = [self._solve_sub_problem(i, candidates, pre_calc_data=pre_calc_data) for i in batch]

                # -----------------
                # GET THE STEP SIZE
                # -----------------
                if self.stepsize == "diminishing":
                    gamma = self._get_step_size_diminishing(k, self.N)
                elif self.stepsize == "linesearch":
                    gamma = self._get_step_size_linesearch(batch, res_batch, candidates)
                else:
                    raise ValueError(
                        "Invalid stepsize method '%s'. Choices are 'diminishing' and 'linesearch'." % self.stepsize)

                # -------------------------
                # UPDATE THE DUAL VARIABLES
                # -------------------------
                _new_fps = np.full((self.batch_size, self.fps_active.shape[1]), fill_value=np.nan)
                _new_losses = np.full((self.batch_size, ), fill_value=np.nan)
                _n_added = 0
                for idx, i in enumerate(batch):
                    y_i_hat = res_batch[idx][0]
                    # HINT: Here happens the actual update ...
                    if self.alphas.update(i, y_i_hat, gamma):
                        # Add the fingerprint belonging to the newly added active dual variable
                        _new_fps[_n_added, :] = candidates.get_candidates_fp(y[i], y_i_hat)

                        # Add the label loss belonging to the newly added active dual variable
                        _new_losses[_n_added] = self.label_loss_fun(_new_fps[_n_added], candidates.get_gt_fp(y[i]))

                        _n_added += 1

                if _n_added > 0:
                    # Update the 'fps_active' and 'label_losses_active'
                    _old_nrow = self.fps_active.shape[0]
                    self.fps_active.resize((self.fps_active.shape[0] + _n_added, self.fps_active.shape[1]),
                                           refcheck=False)
                    self.fps_active[_old_nrow:] = _new_fps[:_n_added]
                    assert not np.any(np.isnan(self.fps_active))

                    # Add the label loss belonging to the newly added active dual variable
                    assert _old_nrow == pre_calc_data["label_losses_active"].shape[0]
                    pre_calc_data["label_losses_active"].resize(
                        (pre_calc_data["label_losses_active"].shape[0] + _n_added, ), refcheck=False)
                    pre_calc_data["label_losses_active"][_old_nrow:] = _new_losses[:_n_added]
                    assert not np.any(np.isnan(pre_calc_data["label_losses_active"]))

                assert self._is_feasible_matrix(self.alphas, self.C), \
                    "Dual variables not feasible anymore after update."

                # Write out debug information
                # ---------------------------
                if (k % 10) == 0:  # TODO: write out debug information every 10'th iteration --> make it a parameter
                    if pre_calc_args["pre_calc_L_matrices"]:
                        _n_missing = self.fps_active.shape[0] - pre_calc_data["L_S"].shape[0]
                        if _n_missing > 0:
                            # Update the L_S and L_SS
                            # -----------------------
                            # L_S: Old shape (|S|, N) --> new shape (|S| + n_added, N)
                            pre_calc_data["L_S"].resize(
                                (pre_calc_data["L_S"].shape[0] + _n_missing, pre_calc_data["L_S"].shape[1]),
                                refcheck=False)
                            pre_calc_data["L_S"][-_n_missing:] = candidates.get_kernel(
                                self.fps_active[-_n_missing:], candidates.get_gt_fp(self.y_train))

                            # L_SS: Old shape (|S|, |S|) --> new shape (|S| + n_added, |S| + n_added)
                            new_entries = candidates.get_kernel(self.fps_active[-_n_missing:],
                                                                self.fps_active)  # shape = (n_added, |S| + n_added)
                            _L_SS = np.zeros((pre_calc_data["L_SS"].shape[0] + _n_missing,
                                              pre_calc_data["L_SS"].shape[1] + _n_missing))
                            _L_SS[:pre_calc_data["L_SS"].shape[0], :pre_calc_data["L_SS"].shape[1]] = pre_calc_data["L_SS"]
                            _L_SS[pre_calc_data["L_SS"].shape[0]:, :] = new_entries
                            _L_SS[:, pre_calc_data["L_SS"].shape[1]:] = new_entries.T
                            pre_calc_data["L_SS"] = _L_SS
                            assert np.all(np.equal(pre_calc_data["L_SS"], pre_calc_data["L_SS"].T))

                    # Update the L_S_Ci matrices
                    # --------------------------
                    if pre_calc_args["pre_calc_L_Ci_S_matrices"]:
                        for y_i in pre_calc_data["mol_kernel_L_S_Ci"]:
                            if pre_calc_data["mol_kernel_L_S_Ci"][y_i].shape[0] == self.fps_active.shape[0]:
                                # There is nothing to update here
                                continue

                            _n_missing = self.fps_active.shape[0] - pre_calc_data["mol_kernel_L_S_Ci"][y_i].shape[0]
                            assert _n_missing > 0

                            pre_calc_data["mol_kernel_L_S_Ci"][y_i].resize(
                                (pre_calc_data["mol_kernel_L_S_Ci"][y_i].shape[0] + _n_missing,
                                 pre_calc_data["mol_kernel_L_S_Ci"][y_i].shape[1]),
                                refcheck=False)
                            pre_calc_data["mol_kernel_L_S_Ci"][y_i][-_n_missing:] = candidates.get_kernel(
                                self.fps_active[-_n_missing:], candidates.get_candidates_fp(y_i))
                            assert not np.any(np.isnan(pre_calc_data["mol_kernel_L_S_Ci"][y_i]))

                    self._write_debug_output(debug_args, epoch=epoch, step_batch=step, step_total=k + 1,
                                             stepsize=gamma, data_for_topk_acc=data_for_topk_acc,
                                             pre_calc_data=pre_calc_data, candidates=candidates)

                k += 1

        return self

    def _write_debug_output(self, debug_args, epoch, step_batch, step_total, stepsize, pre_calc_data, data_for_topk_acc,
                            candidates):
        """

        :param step:
        :param epoch:
        :param stepsize:
        :param pre_calc_data:
        :param data_for_topk_acc:
        :param candidates:
        :return:
        """
        # Get the summary writer if provided
        try:
            summary_writer = debug_args["summary_writer"]
        except KeyError:
            summary_writer = None

        print("Epoch %d / %d; Step: %d / %d (per epoch); Step total: %d" % (
            epoch, self.max_n_epochs, step_batch, np.ceil(self.K_train.shape[0] / self.batch_size), step_total))

        if debug_args["track_objectives"]:
            prim_obj, dual_obj, rel_duality_gap, lin_duality_gap = self._evaluate_primal_and_dual_objective(
                candidates, pre_calc_data=pre_calc_data)
            print("\tf(w) = %.5f; g(a) = %.5f\n"
                  "\tRelative duality gap = %.5f; Linear duality gap = %.5f"
                  % (prim_obj, dual_obj, rel_duality_gap, lin_duality_gap))

            if summary_writer:
                with summary_writer.as_default():
                    with tf.name_scope("Objective Functions"):
                        tf.summary.scalar("Primal", prim_obj, step_total)
                        tf.summary.scalar("Dual", dual_obj, step_total)
                        tf.summary.scalar("Duality gap", rel_duality_gap, step_total)

        if debug_args["track_dual_variables"]:
            print("\tNumber of active variables = %d" % self.alphas.n_active())

            if summary_writer:
                with summary_writer.as_default():
                    with tf.name_scope("Optimizer"):
                        tf.summary.scalar("Number of active dual variables", self.alphas.n_active(), step_total)

                    tf.summary.histogram(
                        "Dual variable distribution",
                        data=np.array(np.sum(self.alphas.get_dual_variable_matrix(), axis=0)).flatten(),
                        buckets=100, step=step_total)

        if debug_args["track_stepsize"]:
            print("\tStep size = %.5f" % stepsize)

            if summary_writer:
                with summary_writer.as_default():
                    with tf.name_scope("Optimizer"):
                        tf.summary.scalar("Step-size", stepsize, step_total)

        if debug_args["track_topk_acc"]:
            acc_k_train = self.score(self.K_train, self.y_train, candidates)
            print("\tTop-1=%2.2f; Top-5=%2.2f; Top-10=%2.2f\tTraining" % (
                acc_k_train[0], acc_k_train[4], acc_k_train[9]))

            acc_k_val = self.score(data_for_topk_acc["X_val"], data_for_topk_acc["y_val"], candidates)
            print("\tTop-1=%2.2f; Top-5=%2.2f; Top-10=%2.2f\tValidation" % (acc_k_val[0], acc_k_val[4], acc_k_val[9]))

            if summary_writer:
                with summary_writer.as_default():
                    for tpk in [1, 5, 10, 20]:
                        with tf.name_scope("Metric (Training)"):
                            tf.summary.scalar("Top-%d (training)" % tpk, acc_k_train[tpk - 1], step_total)

                        with tf.name_scope("Metric (Validation)"):
                            tf.summary.scalar("Top-%d (validation)" % tpk, acc_k_val[tpk - 1], step_total)

    def _get_active_fingerprints_and_losses(self, alphas: DualVariables, y: np.ndarray,
                                            candidates: CandidateSetMetIdent, verbose=False):
        """
        Load the molecular fingerprints corresponding to the molecules of active dual variables. Furthermore, calculate
        the label losses between the "active fingerprints" and their corresponding ground truth fingerprints.

        :param alphas: DualVariables, dual variable information used to determine the active variable set and
            fingerprints

        :param y: array-like, shape = (n_train,), string identifier of the ground truth molecules, e.g. the training
            molecules.

        :param candidates: CandidateSetMetIdent, candidate set information to extract the active fingerprints from

        :param verbose: boolean, indicating whether a tqdm progress bar should be used to show the progress.

        :return: tuple (np.ndarray, np.ndarray)
            fps_active: shape = (n_active, d_fps), fingerprint vectors corresponding to the active dual variables
            lab_losses_active: shape = (n_active, ), label loss vector between the active fingerprints and corresponding
                ground truth fingerprints
        """
        fps_active = np.zeros((alphas.n_active(), candidates.n_fps()))
        label_losses_active = np.zeros((alphas.n_active(), ))

        itr = range(alphas.n_active())
        if verbose:
            itr = tqdm(itr, desc="Collect active fingerprints and losses")

        for c in itr:
            j, ybar = alphas.get_iy_for_col(c)
            # Get the fingerprints of the active candidates for example j
            fps_active[c] = candidates.get_candidates_fp(y[j], ybar)
            # Get label loss between the active fingerprint candidate and its corresponding gt fingerprint
            label_losses_active[c] = self.label_loss_fun(candidates.get_gt_fp(y[j]), fps_active[c])

        return fps_active, label_losses_active

    @staticmethod
    def _is_feasible_matrix(alphas: DualVariables, C: float) -> bool:
        """
        Check whether the given dual variables are feasible.

        :param alphas: DualVariables, dual variables to test.

        :param C: scalar, regularization parameter of the support vector machine

        :return: boolean, indicating whether the dual variables are feasible.
        """
        B = alphas.get_dual_variable_matrix()
        N = B.shape[0]

        if (B < 0).getnnz():
            return False

        if not np.all(np.isclose(B.sum(axis=1), C / N)):
            return False

        return True

    def _get_candidate_scores(self, i: int, candidates: CandidateSetMetIdent, pre_calc_data: Dict) -> np.ndarray:
        """
        Function to evaluate the following term for all candidates y in Sigma_i corresponding to example i:

            s(x_i, y) = sum_j sum_{y' in Sigma_j} alpha(j, y') < Psi(x_j, y_j) - Psi(x_j, y') , Psi(x_i, y) >

            for all y in Sigma_i

        Alternative expression using primal formulation:

            s(x_i, y) = < w , Psi(x_i, y) >

        :return: array-like, shape = (|Sigma_i|, )
        """
        # Get candidate fingerprints for all y in Sigma_i
        if "mol_kernel_L_S_Ci" not in pre_calc_data or self.y_train[i] not in pre_calc_data["mol_kernel_L_S_Ci"]:
            L_Ci_S_available = False
        else:
            L_Ci_S_available = True

        if "mol_kernel_l_y" not in pre_calc_data or self.y_train[i] not in pre_calc_data["mol_kernel_l_y"]:
            L_Ci_available = False
        else:
            L_Ci_available = True

        if  L_Ci_available and L_Ci_S_available:
            fps_Ci = None
        else:
            fps_Ci = candidates.get_candidates_fp(self.y_train[i])

        if L_Ci_S_available:
            L_Ci_S = pre_calc_data["mol_kernel_L_S_Ci"][self.y_train[i]].T
            assert L_Ci_S.shape[1] == self.fps_active.shape[0]
        else:
            L_Ci_S = candidates.get_kernel(fps_Ci, self.fps_active)
        # L_Ci_S with shape = (|Sigma_i|, |S|)

        if L_Ci_available:
            L_Ci = pre_calc_data["mol_kernel_l_y"][self.y_train[i]]
        else:
            L_Ci = candidates.get_kernel(fps_Ci, candidates.get_gt_fp(self.y_train))
        # L_Ci with shape = (|Sigma_i|, N)

        B_S = self.alphas.get_dual_variable_matrix(type="dense")  # shape = (N, |Sigma_i|)

        N = self.K_train.shape[0]

        scores = np.array(self.C / N * L_Ci @ self.K_train[i] - L_Ci_S @ (B_S.T @ self.K_train[i])).flatten()

        return scores

    def _solve_sub_problem(self, i: int, candidates: CandidateSetMetIdent, pre_calc_data: Dict) \
            -> Tuple[Tuple, float]:
        """
        Function to solve the sub-problem: Find the candidate molecular structure y in Sigma_i for example i that is the
        solution to the following optimization problem:

            argmax{y in Sigma_i} loss(y_i, y) + < w , Psi(x_i, y) >

        :return: tuple(
            tuple, label sequence of the candidate maximizing the above problem
            scalar, value of the optimization problem
        )
        """
        # HINT: Only one row in A has changed since last time. Eventually there was a new column.
        # HINT: 'loss' and 's_j_y' do not change if the set of active dual variable changes.
        cand_scores = self._get_candidate_scores(i, candidates, pre_calc_data)

        # Get the label loss for example i
        if self.y_train[i] in pre_calc_data["label_losses"]:
            loss = pre_calc_data["label_losses"][self.y_train[i]]
        else:
            fp_i = candidates.get_gt_fp(self.y_train[i]).flatten()
            loss = self.label_loss_fun(fp_i, candidates.get_candidates_fp(self.y_train[i]))

        score = np.array(loss + cand_scores).flatten()

        # Find the max-violator
        max_idx = np.argmax(score).item()

        # Get the corresponding candidate sequence label
        y_hat_i = (candidates.get_labelspace(self.y_train[i])[max_idx], )

        return y_hat_i, score[max_idx]

    def _evaluate_primal_and_dual_objective(self, candidates: CandidateSetMetIdent, pre_calc_data: Dict) \
            -> Tuple[float, float, float, float]:
        """
        Function to calculate the dual and primal objective values.

        :return:
        """
        # Pre-calculate some matrices
        if "L" in pre_calc_data:
            L  = pre_calc_data["L"]
            L_S = pre_calc_data["L_S"]  # shape = (|S|, N)
            L_SS = pre_calc_data["L_SS"]  # shape = (|S|, |S|)
        else:
            L = candidates.getMolKernel_ExpVsExp(self.y_train)
            L_S = candidates.get_kernel(self.fps_active, candidates.get_gt_fp(self.y_train))
            L_SS = candidates.get_kernel(self.fps_active)

        # Calculate the dual objective
        B_S = self.alphas.get_dual_variable_matrix(type="dense")  # shape = (N, |S|)
        N = self.K_train.shape[0]
        aTATAa = np.sum(self.K_train * ((self.C**2 / N**2 * L) + (B_S @ (L_SS @ B_S.T - 2 * self.C / N * L_S))))
        aTl = np.sum(B_S @ pre_calc_data["label_losses_active"])
        dual_obj = aTl - aTATAa / 2

        # Calculate the primal objective
        wtw = aTATAa
        assert wtw >= 0
        const = np.sum((self.C / N * L - L_S.T @ B_S.T) * self.K_train, axis=1)
        xi = 0
        for i in range(N):
            _, max_score = self._solve_sub_problem(i, candidates, pre_calc_data)
            xi += np.maximum(0, max_score - const[i])
        prim_obj = wtw / 2 + (self.C / N) * xi
        prim_obj = prim_obj.flatten().item()
        assert prim_obj >= dual_obj

        # Calculate the different duality gaps
        lin_duality_gap = prim_obj - dual_obj
        assert lin_duality_gap >= 0
        rel_duality_gap = lin_duality_gap / (np.abs(prim_obj) + 1)

        return prim_obj, dual_obj, rel_duality_gap, lin_duality_gap

    def predict(self, X: np.ndarray, y: np.ndarray, candidates: CandidateSetMetIdent) -> Dict[int, np.ndarray]:
        """
        Predict the scores for each candidate corresponding to the individual spectra.

        FIXME: Currently, we pass the molecule identifier as candidate set identifier for each spectrum.

        :param X: array-like, shape = (n_test, n_train), test-train spectra kernel

        :param y: array-like, shape = (n_test, ), candidate set identifier for each spectrum

        :param candidates: CandidateSetMetIdent, all needed information about the candidate sets

        :return: dict:
            keys are integer indices of each spectrum;
            values are array-like with shape = (|Sigma_i|,) containing predicted candidate scores
        """
        B_S = self.alphas.get_dual_variable_matrix(type="dense")
        N = B_S.shape[0]
        score = {}

        for i in range(X.shape[0]):
            # Calculate the kernels between the training examples and candidates
            mol_kernel_l_y_i = candidates.getMolKernel_ExpVsCand(self.y_train, y[i])

            # ...
            s_ybar_y = X[i] @ B_S @ candidates.get_kernel(self.fps_active, candidates.get_candidates_fp(y[i]))

            # molecule kernel between all candidates of example i and all other training examples
            s_j_y = (self.C / N) * X[i] @ mol_kernel_l_y_i  # shape = (|Sigma_i|, )

            score[i] = np.array(s_j_y - s_ybar_y).flatten()

        return score

    def score(self, X: np.ndarray, y: np.ndarray, candidates: CandidateSetMetIdent, score_type="predicted"):
        """
        Calculate the top-k accuracy scores for a given test spectrum set.

        :param X: array-like, shape = (n_test, n_train), test-train spectra kernel

        :param y: array-like, shape = (n_test, ), candidate set identifier for each spectrum

        :param candidates: CandidateSetMetIdent, all needed information about the candidate sets

        :param score_type: string, indicating which method should be used to get the scores for the candidates
            "predicted" used the predict candidate scores using the learning model
            "random" repeatedly assigns random candidate scores and averages the performance (10 repetitions)
            "first_candidate" always chooses the first candidate in die candidate list

        :return: array-like, with top-k accuracy at index (k - 1)
        """
        d_cand = {}
        for i in range(X.shape[0]):
            sigma_i = candidates.get_labelspace(y[i])
            d_cand[i] = {"n_cand": len(sigma_i), "index_of_correct_structure": sigma_i.index(y[i])}

        if score_type == "predicted":
            scores = self.predict(X, y, candidates)
            tp_k, acc_k = get_topk_performance_csifingerid(d_cand, scores)

        elif score_type == "random":
            n_rep = 10
            for rep in range(n_rep):
                scores = {i: np.random.RandomState((rep + 1) * i).rand(d_cand[i]["n_cand"]) for i in d_cand}
                if rep == 0:
                    tp_k, acc_k = get_topk_performance_csifingerid(d_cand, scores)
                else:
                    _tp_k, _acc_k = get_topk_performance_csifingerid(d_cand, scores)
                    tp_k += _tp_k
                    acc_k += _acc_k
                tp_k /= n_rep
                acc_k /= n_rep

        elif score_type == "first_candidate":
            scores = {i: np.arange(d_cand[i]["n_cand"], 0, -1) for i in d_cand}
            tp_k, acc_k = get_topk_performance_csifingerid(d_cand, scores)

        else:
            raise ValueError("Invalid score type: '%s'. Choices are 'predicted', 'random' and 'first_candidate'"
                             % score_type)

        return acc_k

    def _get_step_size_linesearch(self, is_k: List[int], res_k: List[Tuple[Tuple, float]],
                                  candidates: CandidateSetMetIdent) -> float:
        """
        Implementation of the line-search algorithm for the step-size determination.

        :param is_k: list, indices of the training examples of the batch in iteration k
        :param res_k:
        :param candidates:

        :return: scalar, step-size determined using line-search
        """
        alpha_iy, alpha_a = self.alphas.get_blocks(is_k)
        s_iy, s_a = [], self.C / len(self.y_train)
        for idx, i in enumerate(is_k):
            s_iy.append((i, res_k[idx][0]))
        s_minus_a = DualVariables(self.C, self.alphas.l_cand_ids, initialize=False).set_alphas(s_iy, s_a) \
            - DualVariables(self.C, self.alphas.l_cand_ids, initialize=False).set_alphas(alpha_iy, alpha_a)

        B_S = self.alphas.get_dual_variable_matrix(type="dense")
        bB_bS = s_minus_a.get_dual_variable_matrix(type="dense")  # shape = (N, |\bar{S}|)

        bS, bl = self._get_active_fingerprints_and_losses(s_minus_a, self.y_train, candidates, verbose=False)
        L_bS = candidates.get_kernel(bS, candidates.get_gt_fp(self.y_train))  # shape = (|\bar{S}|, N)
        L_bSS = candidates.get_kernel(bS, self.fps_active)  # shape = (|\bar{S}|, |S|)
        L_bSbS = candidates.get_kernel(bS)  # shape = (|\bar{S}|, |\bar{S}|)

        _ass = np.where((bB_bS != 0).sum(axis=1))[0]
        # FIXME: This assertions fail if a candidate set has only one element, and a - s is zero even for an i in I.
        # assert len(_ass) == len(is_k)
        assert np.all(np.isin(_ass, is_k))
        bB_bS = bB_bS[is_k]  # shape = (|I|, |\bar{S}|)

        s_minus_aTATAa = np.sum(self.K_train[is_k] * (bB_bS @ (- self.C / len(self.y_train) * L_bS + L_bSS @ B_S.T)))
        a_minus_sTbl = np.sum(bB_bS @ bl)

        nom = a_minus_sTbl - s_minus_aTATAa
        den = np.sum(self.K_train[np.ix_(is_k, is_k)] * (bB_bS @ L_bSbS @ bB_bS.T))

        return np.clip(nom / den, 0, 1).item()


class StructuredSVMSequences(_StructuredSVM):
    def __init__(self, C=1.0, n_epochs=1000, label_loss="hamming", rs=None):
        super(StructuredSVMSequences, self).__init__(C=C, n_epochs=n_epochs, label_loss=label_loss, rs=rs)

    def fit(self, data: SequenceSample, num_init_active_vars_per_seq=1):
        """
        Train the SSVM given a dataset.
        :return:
        """
        # Number of training sequences
        N = len(data)

        # Set up the dual initial dual vector
        alphas = [DualVariablesForExample(N=N, C=self.C, cand_ids=data.get_labelspace(i),
                                          rs=self.rs, num_init_active_vars=num_init_active_vars_per_seq)
                  for i in range(N)]
        assert self._is_feasible(alphas, self.C), "Initial dual variables must be feasible."

        k = 0
        while k < self.max_n_epochs:
            # Pick a random coordinate to update
            i = self.rs.choice(N)

            # Find the most violating example
            y_i_hat = self._solve_sub_problem(alphas, data, i)

            # Get step-width
            gamma = self._get_step_size_diminishing(k, N)

            # Update the dual variables
            alphas[i].update(y_i_hat, gamma)

            assert self._is_feasible(alphas, self.C)

        return self

    def _solve_sub_problem(self, alpha: List[DualVariablesForExample], data: SequenceSample, i: int) -> tuple:
        """
        Find the most violating example by solving the MAP problem. Forward-pass using max marginals.

        :param alphas:
        :param X:
        :return:
        """
        V = list(range(data.L))  # Number of sequence elements defines the set of nodes
        E = [(sigma, sigma + 1) for sigma in V[:-1]]  # Edge set of a linear graph (tree-like)
        labspace_i = data.get_labelspace(i)

        # Pre-calculate the Hamming-losses
        y_i = data.get_gt_labels(i)
        lloss = {sigma: np.array([hamming_loss(y_i, y_cand) for y_cand in labspace_i[sigma]]) for sigma in V}
        # TODO: The hamming loss can be globally pre-computed, as it does not depend on alpha.

        # Pre-calculate the node-potentials: sv_i
        sv_i = {}
        for sigma in V:
            # - A score-vector for each node sigma in V
            # - Each score-vector is has the length of the label space corresponding to this node Sigma_{i sigma}
            # - The label space corresponds candidates for spectrum sigma in sequence i.
            sv_i[sigma] = np.zeros_like(labspace_i[sigma])  # Sigma_{i\sigma}

            # Working with the dual representation requires us to go over all training sequences, here denoted with j.
            for j in range(data.N):
                # Get active dual variables for j: alpha(j, bar_y) > 0
                # y__j_ybar, a__j_ybar = zip(*alpha[j].items())
                # y__j_ybar ... list of tuples = list of active label sequences
                # a__j_ybar ... value of the dual variable associated with each active label sequence

                for y__j_ybar, a__j_ybar in alpha[j].items():
                    sv_i[sigma] += a__j_ybar * data.delta_jointKernelMS((j, None), (i, sigma), y__j_ybar)

            sv_i[sigma] += lloss[sigma]

        # Pre-calculate the edge-potentials: se_i
        pass