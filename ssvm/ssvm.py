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
        self.l_cand_ids = []
        for i in range(self.N):
            self.l_cand_ids.append([self.rs.permutation(_cand_ids).tolist() for _cand_ids in cand_ids[i]])

        self.num_init_active_vars = num_init_active_vars
        assert self.num_init_active_vars > 0

        if initialize:
            # Initialize the dual variables
            self.initialize_alphas()
        else:
            self._alphas = None
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
        self.n_epochs = n_epochs
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
    def _get_diminishing_stepwidth(k: int, N: int) -> float:
        """
        Step-width calculation after [1].

        :param k:
        :param N:
        :return:
        """
        return (2 * N) / (k + 2 * N)


class StructuredSVMMetIdent(_StructuredSVM):
    def __init__(self, C=1.0, n_epochs=1000, label_loss="hamming", rs=None, batch_size=1):
        self.batch_size = batch_size

        # States defining a fitted SSVM Model
        self.K_train = None  # type: np.ndarray
        self.y_train = None
        self.fps_active = None
        self.alphas = None  # type: DualVariables
        self.train_set = None

        super(StructuredSVMMetIdent, self).__init__(C=C, n_epochs=n_epochs, label_loss=label_loss, rs=rs)

    def fit(self, X: np.ndarray, y: np.ndarray, candidates: CandidateSetMetIdent, num_init_active_vars_per_seq=1,
            train_summary_writer: Optional[ResourceSummaryWriter] = None):
        """
        Train the SSVM given a dataset.
        """
        # Split of some training data as a validation set, if Tensorboard debug information are written out.
        if train_summary_writer is not None:
            # take 10% of the data for validation
            self.train_set, val_set = next(GroupKFold(n_splits=10).split(X, groups=y))
            X_orig_train = np.array(X[self.train_set])
            X_val = X[val_set]  # shape = (n_val, n_original_train)
            X = X[np.ix_(self.train_set, self.train_set)]
            y_val = y[val_set]
            y = y[self.train_set]

        # Assign the training inputs and labels (candidate identifier)
        self.K_train = X
        self.y_train = y

        # Number of training sequences
        N = len(self.K_train)

        # To allow pre-calculation of some relevant data, we already draw the random sequence of examples we use for the
        # optimization.
        i_k = []
        for _ in range(self.n_epochs):
            i_k.append(self.rs.choice(N, size=self.batch_size, replace=False))

        # Pre-calculate some data needed for the sub-problem solving
        lab_losses = {}
        mol_kernel_l_y = {}
        for k, res_k in tqdm(list(it.product(range(self.n_epochs), range(self.batch_size))), desc="Pre-calculate data"):
            i = i_k[k][res_k]

            # Check whether the relevant stuff was already pre-computed
            if y[i] in lab_losses:
                continue

            # Pre-calculate the label loss: Loss of the gt fingerprint to all corresponding candidate fingerprints of i
            fp_i = candidates.get_gt_fp(y[i])
            lab_losses[y[i]] = self.label_loss_fun(fp_i, candidates.get_candidates_fp(y[i]))

            # Pre-calculate the kernels between the training examples and candidates
            mol_kernel_l_y[y[i]] = candidates.getMolKernel_ExpVsCand(y, y[i]).T  # shape = (|Sigma_i|, N)

        # Initialize dual variables
        print("Initialize dual variables: ...", end="")
        self.alphas = DualVariables(C=self.C, cand_ids=[[candidates.get_labelspace(y[i])] for i in range(N)],
                                    rs=self.rs, num_init_active_vars=num_init_active_vars_per_seq)
        assert self._is_feasible_matrix(self.alphas, self.C), "Initial dual variables must be feasible."
        print("100")

        # Collect active candidate fingerprints and losses
        self.fps_active, lab_losses_active = self._get_active_fingerprints_and_losses(
            self.alphas, self.y_train, candidates, verbose=True)

        k = 0
        if train_summary_writer is None:
            self._write_debug_output(0, lab_losses_active, candidates, np.nan)
        else:
            self._write_debug_output(0, lab_losses_active, candidates, np.nan, train_summary_writer, X_val,
                                     y_val, X_orig_train, self.y_train)

        while k < self.n_epochs:
            # Find the most violating example
            # TODO: Can we solve the sub-problem for a full batch at the same time?
            res_k = [self._solve_sub_problem(
                i, candidates, pre_calc_data={"lab_losses": lab_losses, "mol_kernel_l_y": mol_kernel_l_y})
                for i in i_k[k]]

            # Get step-width
            if self.stepsize == "diminishing":
                gamma = self._get_diminishing_stepwidth(k, N)
            elif self.stepsize == "linesearch":
                gamma = self._get_linesearch_stepwidth(i_k[k], res_k, candidates)

            # Update the dual variables
            for idx, i in enumerate(i_k[k]):
                y_i_hat = res_k[idx][0]
                if self.alphas.update(i, y_i_hat, gamma):
                    # Add the fingerprint belonging to the newly added active dual variable
                    self.fps_active.resize(self.fps_active.shape[0] + 1, self.fps_active.shape[1])
                    self.fps_active[self.fps_active.shape[0] - 1] = candidates.get_candidates_fp(y[i], y_i_hat)

                    # Add the label loss belonging to the newly added active dual variable
                    lab_losses_active.resize(lab_losses_active.shape[0] + 1)
                    lab_losses_active[lab_losses_active.shape[0] - 1] = self.label_loss_fun(
                        self.fps_active[self.fps_active.shape[0] - 1], candidates.get_gt_fp(y[i]))

            assert self._is_feasible_matrix(self.alphas, self.C), "Dual variables not feasible anymore after update."

            if (((k + 1) % 10) == 0) or ((k + 1) == self.n_epochs):
                if train_summary_writer is None:
                    self._write_debug_output(k + 1, lab_losses_active, candidates, gamma)
                else:
                    self._write_debug_output(k + 1, lab_losses_active, candidates, gamma, train_summary_writer, X_val,
                                             y_val, X_orig_train, self.y_train)

            k += 1

        return self

    def _write_debug_output(self, epoch, lab_losses_active, candidates, stepsize, train_summary_writer=None, X_val=None,
                            y_val=None, X_orig_train=None, y_orig_train=None):
        print("Epoch: %d / %d" % (epoch, self.n_epochs))

        prim_obj, dual_obj, rel_duality_gap, lin_duality_gap = self._evaluate_primal_and_dual_objective(
            candidates, pre_calc_data={"lab_losses_active": lab_losses_active})
        print("\tf(w) = %.5f; g(a) = %.5f\n"
              "\tstep-size = %.5f\n"
              "\tRelative duality gap = %.5f; Linearized duality gap = %.5f"
              % (prim_obj, dual_obj, stepsize, rel_duality_gap, lin_duality_gap))

        if train_summary_writer is not None:
            with train_summary_writer.as_default():
                with tf.name_scope("Objective Functions"):
                    tf.summary.scalar("Primal", prim_obj, epoch)
                    tf.summary.scalar("Dual", dual_obj, epoch)
                    tf.summary.scalar("Duality gap", rel_duality_gap, epoch)

                with tf.name_scope("Optimizer"):
                    tf.summary.scalar("Number of active dual variables", self.alphas.n_active(), epoch)
                    tf.summary.scalar("Step-size", stepsize, epoch)

                tf.summary.histogram(
                    "Dual variable distribution",
                    data=np.array(np.sum(self.alphas.get_dual_variable_matrix(), axis=0)).flatten(),
                    buckets=100, step=epoch)

                with tf.name_scope("Metric (Validation)"):
                    acc_k_val = self.score(X_val, y_val, candidates)
                    print("\tTop-1=%2.2f; Top-5=%2.2f; Top-10=%2.2f\tValidation" %
                          (acc_k_val[0], acc_k_val[4], acc_k_val[9]))
                    for tpk in [1, 5, 10, 20]:
                        tf.summary.scalar("Top-%d (validation)" % tpk, acc_k_val[tpk - 1], epoch)

                with tf.name_scope("Metric (Training)"):
                    acc_k_train = self.score(X_orig_train, y_orig_train, candidates)
                    print("\tTop-1=%2.2f; Top-5=%2.2f; Top-10=%2.2f\tTraining" %
                              (acc_k_train[0], acc_k_train[4], acc_k_train[9]))
                    for tpk in [1, 5, 10, 20]:
                        tf.summary.scalar("Top-%d (training)" % tpk, acc_k_train[tpk - 1], epoch)

    def _get_active_fingerprints_and_losses(self, alphas: DualVariables, y: np.ndarray,
                                            candidates: CandidateSetMetIdent, verbose=False):
        """
        :param alphas:
        :param y:
        :param candidates:
        :param verbose:
        :return:
        """
        fps_active = np.zeros((alphas.n_active(), candidates.n_fps()))
        lab_losses_active = np.zeros((alphas.n_active(), ))

        itr = range(alphas.n_active())
        if verbose:
            itr = tqdm(itr, desc="Collect active fingerprints and losses")

        for c in itr:
            j, ybar = alphas.get_iy_for_col(c)
            # Get the fingerprints of the active candidates for example j
            fps_active[c] = candidates.get_candidates_fp(y[j], ybar)
            # Get label loss between the active fingerprint candidate and its corresponding gt fingerprint
            lab_losses_active[c] = self.label_loss_fun(candidates.get_gt_fp(y[j]), fps_active[c])

        return fps_active, lab_losses_active

    @staticmethod
    def _is_feasible_matrix(alphas: DualVariables, C: float) -> bool:
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
        fps_Ci = candidates.get_candidates_fp(self.y_train[i])
        L_Ci_S = candidates.get_kernel(fps_Ci, self.fps_active)  # shape = (|Sigma_i|, |S|)

        if "mol_kernel_l_y" in pre_calc_data:
            L_Ci = pre_calc_data["mol_kernel_l_y"][self.y_train[i]]
        else:
            L_Ci = candidates.get_kernel(fps_Ci, candidates.get_gt_fp(self.y_train))
        # L_Ci with shape = (|Sigma_i|, N)

        B_S = self.alphas.get_dual_variable_matrix(type="csr")  # shape = (N, |Sigma_i|)

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
        if "lab_losses" in pre_calc_data:
            loss = pre_calc_data["lab_losses"][self.y_train[i]]
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
        L = candidates.getMolKernel_ExpVsExp(self.y_train)  # shape = (N, N)
        B_S = self.alphas.get_dual_variable_matrix(type="csr")  # shape = (N, |S|)
        L_S = candidates.get_kernel(self.fps_active, candidates.get_gt_fp(self.y_train))  # shape = (|S|, N)
        L_SS = candidates.get_kernel(self.fps_active)  # shape = (|S|, |S|)

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
        Predict the scores for each candidate corresponding to the individual spectra.

        FIXME: Currently, we pass the molecule identifier as candidate set identifier for each spectrum.

        :param X: array-like, shape = (n_test, n_train), test-train spectra kernel

        :param y: array-like, shape = (n_test, ), candidate set identifier for each spectrum

        :param candidates: CandidateSetMetIdent, all needed information about the candidate sets

        :return: dict:
            keys are integer indices of each spectrum;
            values are array-like with shape = (|Sigma_i|,) containing predicted candidate scores
        """
        if self.train_set is not None:
            # Remove those training examples, that where part of the validation set and therefore not used for fitting
            # the model.
            X = X[:, self.train_set]
        assert X.shape[1] == self.K_train.shape[0]

        B_S = self.alphas.get_dual_variable_matrix(type="csc")
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

    def _get_linesearch_stepwidth(self, is_k: List[int], res_k: List[Tuple[Tuple, float]],
                                  candidates: CandidateSetMetIdent) -> float:
        """

        :param is_k:
        :param res_k:
        :param candidates:
        :return:
        """
        alpha_iy, alpha_a = self.alphas.get_blocks(is_k)
        s_iy, s_a = [], self.C / len(self.y_train)
        for idx, i in enumerate(is_k):
            s_iy.append((i, res_k[idx][0]))
        s_minus_a = DualVariables(self.C, self.alphas.l_cand_ids, initialize=False).set_alphas(s_iy, s_a) \
            - DualVariables(self.C, self.alphas.l_cand_ids, initialize=False).set_alphas(alpha_iy, alpha_a)

        B_S = self.alphas.get_dual_variable_matrix(type="csr")
        bB_bS = s_minus_a.get_dual_variable_matrix(type="csr")  # shape = (N, |\bar{S}|)
        bS, bl = self._get_active_fingerprints_and_losses(s_minus_a, self.y_train, candidates, verbose=False)
        L_bS = candidates.get_kernel(bS, candidates.get_gt_fp(self.y_train))  # shape = (|\bar{S}|, N)
        L_bSS = candidates.get_kernel(bS, self.fps_active)  # shape = (|\bar{S}|, |S|)
        L_bSbS = candidates.get_kernel(bS)  # shape = (|\bar{S}|, |\bar{S}|)

        s_minus_aTATAa = np.sum(self.K_train * (bB_bS @ (- self.C / len(self.y_train) * L_bS + L_bSS @ B_S.T)))
        a_minus_sTbl = np.sum(bB_bS @ bl)

        nom = a_minus_sTbl - s_minus_aTATAa
        den = np.sum(self.K_train * (bB_bS @ L_bSbS @ bB_bS.T))

        return np.maximum(0, np.minimum(1, nom / den))


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
        while k < self.n_epochs:
            # Pick a random coordinate to update
            i = self.rs.choice(N)

            # Find the most violating example
            y_i_hat = self._solve_sub_problem(alphas, data, i)

            # Get step-width
            gamma = self._get_diminishing_stepwidth(k, N)

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