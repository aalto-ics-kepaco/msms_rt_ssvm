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
import networkx as nx

from collections import OrderedDict
from copy import deepcopy
from typing import List, ItemsView, Tuple, ValuesView, KeysView, Iterator, Dict, Union, Optional, TypeVar, Callable
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import GroupKFold
from scipy.sparse import csc_matrix, lil_matrix, csr_matrix
from tqdm import tqdm

from ssvm.data_structures import SequenceSample, CandidateSetMetIdent
from ssvm.loss_functions import hamming_loss, tanimoto_loss
from ssvm.evaluation_tools import get_topk_performance_csifingerid
from ssvm.factor_graphs import ChainFactorGraph, identity
from ssvm.kernel_utils import generalized_tanimoto_kernel, tanimoto_kernel


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
        self.random_state = check_random_state(random_state)

        self.num_init_active_vars = num_init_active_vars
        assert self.num_init_active_vars > 0

        # Store a shuffled version of the candidate sets for each example and sequence element
        self.label_space = [[self.random_state.permutation(_cand_ids).tolist() for _cand_ids in label_space[i]]
                            for i in range(self.N)]

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
            n_added = 0
            # Lazy generation of sample sequences, no problem with exponential space here
            # FIXME: Produces heavily biased sequences, as last indices is increased first, then the second last ...
            for y_seq in it.product(*self.label_space[i]):
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

        # TODO Faster verification of the label sequence correctness, we transform all candidate set lists in to sets
        # self.l_cand_ids = [for cand_ids_seq in (for cand_ids_exp in self.l_cand_ids)]

        return _alphas, _y2col, _iy

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
            alphas = np.full((len(iy), ), fill_value=alphas)
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

    def get_blocks(self, i: Optional[Union[List[int], np.ndarray, int]] = None) \
            -> Tuple[List[Tuple[int, Tuple]], List[float]]:
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
                iy.append((_i, y_seq))
                a.append(self.get_dual_variable(_i, y_seq))

        return iy, a

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


class _StructuredSVM(object):
    """
    Structured Support Vector Machine (SSVM) meta-class
    """
    def __init__(self, C: Union[int, float] = 1, n_epochs: int = 100, batch_size: int = 1, label_loss: str = "hamming",
                 step_size: str = "diminishing", random_state: Optional[Union[int, np.random.RandomState]] = None):
        """
        Structured Support Vector Machine (SSVM)

        :param C: scalar, SVM regularization parameter. Must be > 0.

        :param n_epochs: scalar, Number of epochs, i.e. passes over the complete dataset.

        :param batch_size: scalar, Batch size, i.e. number of examples updated in each iteration 

        :param label_loss: string, indicating which label-loss is used.
        
        :param step_size: string, indicating which strategy is used to determine the step-size. 

        :param random_state: None | int | instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.
        """
        self.C = C
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.label_loss = label_loss
        self.random_state = random_state
        self.step_size = step_size

        if self.label_loss == "hamming":
            self.label_loss_fun = hamming_loss
        elif self.label_loss == "tanimoto_loss":
            self.label_loss_fun = tanimoto_loss
        else:
            raise ValueError("Invalid label loss '%s'. Choices are 'hamming' and 'tanimoto_loss'.")

        if self.step_size not in ["diminishing", "linesearch"]:
            raise ValueError("Invalid stepsize method '%s'. Choices are 'diminishing' and 'linesearch'." %
                             self.step_size)

    @staticmethod
    def _is_feasible_matrix(alphas: DualVariables, C: Union[int, float]) -> bool:
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
    def __init__(self, C=1.0, n_epochs=1000, label_loss="hamming", random_state=None, batch_size=1,
                 step_size="diminishing"):
        self.batch_size = batch_size

        # States defining a fitted SSVM Model
        self.K_train = None  # type: np.ndarray
        self.y_train = None
        self.fps_active = None
        self.alphas = None  # type: DualVariables
        self.N = None

        super(StructuredSVMMetIdent, self).__init__(C=C, n_epochs=n_epochs, label_loss=label_loss,
                                                    random_state=random_state, step_size=step_size)


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
            self.train_set = None

        # Assign the training inputs and labels (candidate identifier)
        self.K_train = X
        self.y_train = y
        self.N = len(self.K_train)  # Number of training examples: n_train or n_train' depending on the debug args.

        # Initialize dual variables
        print("Initialize dual variables: ...", end="")
        self.alphas = DualVariables(C=self.C, label_space=[[candidates.get_labelspace(y[i])] for i in range(self.N)],
                                    random_state=self.random_state, num_init_active_vars=num_init_active_vars_per_seq)
        assert self._is_feasible_matrix(self.alphas, self.C), "Initial dual variables must be feasible."

        if self.use_aggregated_model:
            self.alphas_avg = deepcopy(self.alphas)
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
            if self.use_aggregated_model:
                raise ValueError("Cannot use pre-calculated L matrices when a averaged model is used. Set "
                                 "'pre_calc_L_matrices' to False.")

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

        for epoch in range(self.n_epochs):  # Full pass over the data
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
                        "Invalid stepsize method '%s'. Choices are 'diminishing' and 'linesearch'." % self.step_size)

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

                # Maintain the average model
                if self.use_aggregated_model:
                    rho = 2 / (k + 2)
                    self.alphas_avg = self.alphas_avg + rho * (self.alphas - self.alphas_avg)

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
            epoch, self.n_epochs, step_batch, np.ceil(self.K_train.shape[0] / self.batch_size), step_total))

        if debug_args["track_objectives"]:
            prim_obj, dual_obj, rel_duality_gap, lin_duality_gap = self._evaluate_primal_and_dual_objective(
                candidates, pre_calc_data=pre_calc_data)

            if step_total == 0:
                self.lin_duality_gap_0 = lin_duality_gap

            if self.conv_criteria == "rel_duality_gap_decay":
                convergence_value = lin_duality_gap / self.lin_duality_gap_0
            elif self.conv_criteria == "duality_gap":
                convergence_value = lin_duality_gap
            elif self.conv_criteria == "normalized_duality_gap":
                convergence_value = rel_duality_gap

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
            L_Ci_S_available = False
        else:
            L_Ci_S_available = False

        if self.y_train[i] in pre_calc_data["mol_kernel_L_Ci"]:
            L_Ci_available = True
        else:
            L_Ci_available = False

        if L_Ci_available and L_Ci_S_available:
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
        L = pre_calc_data["L"]
        B_S = self.alphas.get_dual_variable_matrix(type="dense")  # shape = (N, |S|)
        L_S = pre_calc_data["L_S"]  # shape = (|S|, N)
        L_SS = pre_calc_data["L_SS"]  # shape = (|S|, |S|)

        # Calculate the dual objective
        N = self.K_train.shape[0]
        aTATAa = np.sum(self.K_train * ((self.C**2 / N**2 * L) + (B_S @ (L_SS @ B_S.T - 2 * self.C / N * L_S))))
        aTl = np.sum(B_S @ pre_calc_data["lab_losses_active"])
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
        Prediction function for the model application to new data.

        :return:
        """
        if self.use_aggregated_model:
            B_S = self.alphas_avg.get_dual_variable_matrix("dense")
            fps_active, _ = self._get_active_fingerprints_and_losses(self.alphas_avg, self.y_train, candidates)
        else:
            B_S = self.alphas.get_dual_variable_matrix("dense")
            fps_active = self.fps_active

        _pre_calc_data = {"mol_kernel_L_Ci": {}, "mol_kernel_L_S_Ci": {}, "B_S": B_S, "fps_active": fps_active}

        return self._predict(X, y, candidates, pre_calc_data=_pre_calc_data, for_training=False)

    def _predict_on_training_data(self, X: np.ndarray, y: np.ndarray, candidates: CandidateSetMetIdent,
                                  pre_calc_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[int, np.ndarray]:
        """
        Prediction function used during the optimization, e.g. to access the training and by the sub-problem solver.
        """
        if self.use_aggregated_model:
            B_S = self.alphas_avg.get_dual_variable_matrix("dense")
            fps_active, _ = self._get_active_fingerprints_and_losses(self.alphas_avg, self.y_train, candidates)

            _pre_calc_data = {"mol_kernel_L_Ci": pre_calc_data["mol_kernel_L_Ci"],
                              "mol_kernel_L_S_Ci": {},  # forces local re-calculation
                              "B_S": B_S, "fps_active": fps_active}
        else:
            B_S = self.alphas.get_dual_variable_matrix("dense")
            fps_active = self.fps_active

            _pre_calc_data = {"mol_kernel_L_Ci": pre_calc_data["mol_kernel_L_Ci"],
                              "mol_kernel_L_S_Ci": pre_calc_data["mol_kernel_L_S_Ci"],
                              "B_S": B_S, "fps_active": fps_active}

        return self._predict(X, y, candidates, pre_calc_data=_pre_calc_data, for_training=True)

        pass

    def _predict(self, X: np.ndarray, y: np.ndarray, candidates: CandidateSetMetIdent, pre_calc_data, for_training) \
            -> Dict[int, np.ndarray]:
        """
        Predict the scores for each candidate corresponding to the individual spectra.

        FIXME: Currently, we pass the molecule identifier as candidate set identifier for each spectrum.

        :param X: array-like, shape = (n_test, n_train), test-train spectra kernel

        :param y: array-like, shape = (n_test, ), candidate set identifier for each spectrum

        :param candidates: CandidateSetMetIdent, all needed information about the candidate sets

        :param pre_calc_data: TODO

        :param for_training: boolean, indicating whether prediction are done for the training process, or the model is
            applied to a new set of data (test scenario).

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

    def _score_on_training_data(self, X: np.ndarray, y: np.ndarray, candidates: CandidateSetMetIdent,
                                pre_calc_data: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Calculate the top-k accuracy scores on the training set. This function is meant to be used during the model
        training.

        The function makes
        """
        d_cand = {}
        for i in range(X.shape[0]):
            sigma_i = candidates.get_labelspace(y[i], for_training=True)
            d_cand[i] = {"n_cand": len(sigma_i), "index_of_correct_structure": sigma_i.index(y[i])}

        scores = self._predict_on_training_data(X, y, candidates, pre_calc_data)
        _, acc_k = get_topk_performance_csifingerid(d_cand, scores)

        return acc_k

    def _get_step_size_linesearch(self, I_B: List[int], res_k: List[Tuple[Tuple, float]],
                                  candidates: CandidateSetMetIdent) -> float:
        """
        Implementation of the line-search algorithm for the step-size determination as described in Section B.7. 

        :param I_B: list, of training examples belong to the current mini-batch
        :param res_k: list of tuples, output of the sub-problem solver. Contains the label sequence of the most
            violating candidate per example.
        :param candidates:

        :return: scalar, step-size determined using line-search
        """
        # Get a dual variable object representing the difference dual values: s(i, y) - a(i, y)
        alpha_iy, alpha_a = self.alphas.get_blocks(I_B)
        s_iy, s_a = [], self.C / len(self.y_train)
        for idx, i in enumerate(I_B):
            s_iy.append((i, res_k[idx][0]))
        s_minus_a = DualVariables(self.C, self.alphas.label_space, initialize=False).set_alphas(s_iy, s_a) \
            - DualVariables(self.C, self.alphas.label_space, initialize=False).set_alphas(alpha_iy, alpha_a)

        # Get the B(S) matrix with shape: (N, |S|)
        B_S = self.alphas.get_dual_variable_matrix(type="dense")

        # Get the B(S_Delta) matrix with shape: (|I|, |S_Delta|)
        B_SDelta = s_minus_a.get_dual_variable_matrix(type="dense")
        _ass = np.where((B_SDelta != 0).sum(axis=1))[0]
        assert np.all(np.isin(_ass, I_B))
        B_SDelta = B_SDelta[I_B]

        # Get the fingerprints and label-losses for the active set S_Delta
        fps_S_Delta, label_losses_S_Delta = self._get_active_fingerprints_and_losses(
            s_minus_a, self.y_train, candidates, verbose=False)

        # Calculate the different molecule kernels (see Theorems 3 and 4)
        L_S_Delta = candidates.get_kernel(fps_S_Delta, candidates.get_gt_fp(self.y_train))  # shape = (|S_delta|, N)
        L_S_S_Delta = candidates.get_kernel(fps_S_Delta, self.fps_active)  # shape = (|S_delta|, |S|)
        L_S_Delta_S_Delta = candidates.get_kernel(fps_S_Delta)  # shape = (|S_delta|, |S_delta|)

        # Calculate the nominator (see Theorem 3)
        nom = np.sum(B_SDelta @ label_losses_S_Delta) + \
              np.sum(self.K_train[I_B] * (B_SDelta @ (self.C / len(self.y_train) * L_S_Delta - L_S_S_Delta @ B_S.T)))

        # Calculate the denominator (see Theorem 4)
        den = np.sum(self.K_train[np.ix_(I_B, I_B)] * (B_SDelta @ L_S_Delta_S_Delta @ B_SDelta.T))

        return np.clip(nom / (den + 1e-8), 0, 1).item()


class StructuredSVMSequencesFixedMS2(_StructuredSVM):
    """
    Structured Support Vector Machine (SSVM) for (MS, RT)-sequence classification.
    """
    def __init__(self, mol_feat_label_loss: str, mol_feat_retention_order: str,
                 mol_kernel: Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
                 *args, **kwargs):

        self.mol_feat_label_loss = mol_feat_label_loss
        self.mol_feat_retention_order = mol_feat_retention_order
        self.mol_kernel = self.get_mol_kernel(mol_kernel)

        super().__init__(*args, **kwargs)

    @staticmethod
    def get_mol_kernel(mol_kernel: Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]) \
            -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """

        :param mol_kernel:
        :return:
        """
        if callable(mol_kernel):
            return mol_kernel
        elif mol_kernel == "tanimoto":
            return tanimoto_kernel
        elif mol_kernel == "minmax":
            return generalized_tanimoto_kernel
        else:
            raise ValueError("Invalid molecule kernel")

    def fit(self, data: SequenceSample, n_init_per_example: int = 1, n_trees_per_sequence: int = 1):
        """
        Train the SSVM given a dataset.

        :param data: SequenceSample, set of training sequences. All needed information, such as features, labels and
            potential label sequences, are accessible through this object.

        :param n_init_per_example: scalar, number of initially active dual variables per example. That is, the number of
            active (potential) label sequences.

        :param n_trees_per_sequence: scalar, number of spanning trees per sequence.

        :return: reference to it self.
        """
        random_state = check_random_state(self.random_state)

        # Number of training sequences
        N = len(data)

        # Set up the dual initial dual vector
        alphas = DualVariables(C=self.C, label_space=data.get_labelspace(), num_init_active_vars=n_init_per_example,
                               random_state=self.random_state)
        assert self._is_feasible_matrix(alphas, self.C), "Initial dual variables must be feasible."

        # Initialize the graphs for each sequence
        graph_set = self._get_graph_set(data, n_trees_per_sequence)

        # Run over the data
        # - each epoch is a cycle through the full data
        # - each batch contains a subset of the full data
        n_iterations_total = 0
        for epoch in range(self.n_epochs):
            for step, batch in enumerate(mit.chunked(random_state.permutation(np.arange(N)), self.batch_size)):
                # Find the most violating examples for the current batch
                y_I_hat = [self._solve_sub_problem(alphas, data, graph_set, i) for i in batch]

                # Get step-width
                gamma = self._get_step_size_diminishing(n_iterations_total, N)

                # Update the dual variables
                is_new = [alphas.update(i, y_i_hat, gamma) for i, y_i_hat in zip(batch, y_I_hat)]

                assert self._is_feasible_matrix(alphas, self.C), "Dual variables after update are not feasible anymore."

        return self

    @staticmethod
    def _get_graph_set(data: SequenceSample, n_trees_per_sequence: int = 1) -> List[List[nx.Graph]]:
        """

        :param data:
        :param n_trees_per_sequence:
        :return:
        """
        if n_trees_per_sequence < 1:
            raise ValueError("Number of trees per sequence must >= 1.")

        if n_trees_per_sequence > 1:
            raise NotImplementedError("Currently only one tree per sequence allowed.")

        return [[ChainFactorGraph._get_chain_connectivity(data_i)] for data_i in data]

    def _solve_sub_problem(self, alphas: DualVariables, data: SequenceSample, graph_set: List[List[nx.Graph]], i: int) \
            -> Tuple:
        """
        Find the most violating example by solving the MAP problem. Forward-pass using max marginals.

        :param alphas: DualVariables, dual variables
        
        :param data: SequenceSample, set of training sequences. All needed information, such as features, labels and
            potential label sequences, are accessible through this object.

        :param i: scalar, example index for which the sub-problem should be solved.
        
        :return:
        """
        N = len(data)  # Number of training examples
        L_i = len(data[i])  # Length of the i'th sequence

        # Set up the candidate dictionary as needed for the 'msmsrt_scorer' package
        candidates = OrderedDict()
        for s in range(L_i):
            candidates[s] = {}

            # Get the candidate identifiers
            candidates[s]["structure"] = data[i].get_labelspace(s)

            # Get the index of the correct structure
            candidates[s]["index_of_correct_structure"] = data[i].get_index_of_correct_structure(s)

            # MS2 scores are fixed and assumed to be pre-computed
            candidates[s]["ms2_score"] = data[i].get_ms2_scores(s)
            candidates[s]["ms2_score_of_correct_structure"] = \
                candidates[s]["ms2_score"][candidates[s]["index_of_correct_structure"]]
            candidates[s]["ms2_score_loss"] = \
                candidates[s]["ms2_score_of_correct_structure"] - candidates[s]["ms2_score"]

            # Calculate the label loss for the current sequence
            candidates[s]["label_loss"] = data[i].get_label_loss(self.label_loss_fun, self.mol_feat_label_loss, s)

            # Calculate the node-score for 'msmsrt_scorer'
            # (1 / L) * (Loss(y_i, y_is) - (S(x_i, y_i) - S(x_i, y_is)))
            candidates[s]["log_score"] = (candidates[s]["label_loss"] - candidates[s]["ms2_score_loss"]) / L_i

            # Load the candidate molecule feature representations used for the retention order prediction
            candidates[s]["mol_feat_retention_order"] = data[i].get_molecule_features(self.mol_feat_retention_order, s)

        # Calculate the transition matrices encoding the SSVM's prediction on the candidate pairs.
        sign_delta = np.concatenate((data[j].get_sign_delta_t(graph_set[j][0].edges)) for j in range(len(data)))

        order_probs = dict()
        for s, t in graph_set[i][0].edges:
            order_probs[s] = {t: dict()}
            order_probs[s][t]["log_score"] = ...

            # -----------
            # Equation  I
            # -----------
            Y_candidates_s = candidates[s]["mol_feat_retention_order"]
            lambda_delta_s = np.vstack(
                (data[j].get_lambda_delta(
                    graph_set[j][0].edges, Y_candidates_s, self.mol_feat_retention_order, self.mol_kernel))
                for j in range(len(data))
            )

            Y_candidates_t = candidates[t]["mol_feat_retention_order"]
            lambda_delta_t = np.vstack(
                (data[j].get_lambda_delta(
                    graph_set[j][0].edges, Y_candidates_t, self.mol_feat_retention_order, self.mol_kernel))
                for j in range(len(data))
            )

            I = self.C * ((sign_delta @ lambda_delta_s)[:, np.newaxis] - (sign_delta @ lambda_delta_t)[np.newaxis, :]) / N

            # -----------
            # Equation II
            # -----------
            alphas.get_blocks(i)


        # Find the most violating example
        Z_max, _ = ChainFactorGraph(candidates=candidates, make_order_probs=identity, order_probs=order_probs) \
            .MAP_only()  # type: List[int]

        # MAP returns a list of candidate indices, we need to convert them back to actual molecules identifier
        label_space_i = data[i].get_labelspace()
        y_i_hat = tuple(label_space_i[s][Z_max[s]] for s in range(L_i))

        return y_i_hat
