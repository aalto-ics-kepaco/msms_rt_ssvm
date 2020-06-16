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

from typing import List, ItemsView, Tuple, ValuesView, KeysView, Iterator, Dict, Union, Optional
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import GroupKFold
from scipy.sparse import csc_matrix, lil_matrix, csr_matrix
from tensorflow.python.ops.summary_ops_v2 import ResourceSummaryWriter
from tqdm import tqdm

from ssvm.data_structures import SequenceSample, CandidateSetMetIdent
from ssvm.loss_functions import hamming_loss
from ssvm.evaluation_tools import get_topk_performance_csifingerid


class DualVariables(object):
    def __init__(self, C: float, N: int, cand_ids: List[List[List[str]]], num_init_active_vars=1, rs=None,
                 index_of_correct_candidate_seq=None):
        """

        :param C: scalar, regularization parameter of the Structured SVM

        :param N: scalar, total number of sequences used for training

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
        """
        self.C = C
        assert self.C > 0, "The regularization parameter must be positive."
        self.N = N
        assert self.N > 0
        assert len(cand_ids) == self.N
        self.l_cand_ids = cand_ids
        self.num_init_active_vars = num_init_active_vars
        assert self.num_init_active_vars > 0
        self.rs = check_random_state(rs)

        # Initialize the dual variables
        self._alphas, self._y2col, self._iy = self._get_initial_alphas(index_of_correct_candidate_seq)

    def _get_initial_alphas(self, index_of_correct_candidate_seq=None) -> Tuple[lil_matrix, List[Dict], List[Tuple]]:
        """
        Return initialized dual variables.

        :return: dictionary: key = label sequence, value = dual variable value
        """
        _alphas = lil_matrix((self.N, self.N * self.num_init_active_vars))
        _y2col = [{} for _ in range(self.N)]
        _iy = []

        # Initial dual variable value ensuring feasibility
        init_var_val = self.C / (self.N * self.num_init_active_vars)  # (C / N) / num_act = C / (N * num_act)

        for col, (i, _) in enumerate(it.product(range(self.N), range(self.num_init_active_vars))):
            if index_of_correct_candidate_seq is not None:
                y_seq = tuple(cand_ids[idx] for cand_ids, idx in zip(self.l_cand_ids[i],
                                                                     index_of_correct_candidate_seq[i]))
            else:
                y_seq = tuple(self.rs.choice(cand_ids) for cand_ids in self.l_cand_ids[i])
            assert y_seq not in _y2col, "Oups, we sampled the same active dual variable again."
            _alphas[i, col] = init_var_val
            _y2col[i][y_seq] = col
            _iy.append((i, y_seq))

        return _alphas, _y2col, _iy

    def update(self, i: int, y_seq: tuple, gamma: float) -> bool:
        """
        Update the value of a dual variable.

        :param i, scalar, sequence example index
        :param y_seq: tuple of strings, sequence of candidate indices identifying the dual variable
        :param gamma: scalar, step-width used to update the dual variable value
        """
        try:
            # Update an active dual variable
            is_new = False
            col = self._y2col[i][y_seq]
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

    def get_dual_variable_matrix(self, type="csr") -> Union[csr_matrix, csc_matrix]:
        """
        Returns the dual variable matrix as a csr-sparse matrix.

        :return: csr_matrix, shape = (N, n_active_dual)
        """
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
        return self._iy[c]


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
    def __init__(self, C=1.0, n_epochs=1000, label_loss="hamming", rs=None):
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
        self.K_train = None
        self.y_train = None
        self.fps_active = None
        self.alphas = None
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
        for k, s in tqdm(list(it.product(range(self.n_epochs), range(self.batch_size))), desc="Pre-calculate data"):
            i = i_k[k][s]

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
        self.alphas = DualVariables(C=self.C, N=N, cand_ids=[[candidates.get_labelspace(y[i])] for i in range(N)],
                                    rs=self.rs, num_init_active_vars=num_init_active_vars_per_seq)
        assert self._is_feasible_matrix(self.alphas, self.C), "Initial dual variables must be feasible."
        print("100")

        # Collect active candidate fingerprints and losses
        self.fps_active, lab_losses_active = self._initialize_active_fingerprints_and_losses(candidates, verbose=True)

        print("Objective values:",
              self._evaluate_primal_and_dual_objective(candidates,
                                                       pre_calc_data={"lab_losses_active": lab_losses_active}))

        k = 0
        while k < self.n_epochs:
            # Find the most violating example
            # TODO: Can we solve the sub-problem for a full batch at the same time?
            res_k = [self._solve_sub_problem(i, candidates, pre_calc_data={"lab_losses": lab_losses,
                                                                           "mol_kernel_l_y": mol_kernel_l_y})
                     for i in i_k[k]]

            # Get step-width
            gamma = self._get_diminishing_stepwidth(k, N)

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
                print("Epoch: %d / %d" % (k + 1, self.n_epochs))

                prim_obj, dual_obj, duality_gap = self._evaluate_primal_and_dual_objective(
                    candidates, pre_calc_data={"lab_losses_active": lab_losses_active})
                print("\tf(w) = %.5f; g(a) = %.5f; step-size = %.5f" % (prim_obj, dual_obj, gamma))

                if train_summary_writer is not None:
                    with train_summary_writer.as_default():
                        tf.summary.scalar("Dual Objective", dual_obj, k + 1)
                        tf.summary.scalar("Step-size", gamma, k + 1)
                        tf.summary.scalar("Number of active dual variables", self.alphas.n_active(), k + 1)
                        tf.summary.scalar("Primal Objective", prim_obj, k + 1)
                        tf.summary.scalar("Duality gap", duality_gap, k + 1)

                        acc_k = self.score(X_val, y_val, candidates)
                        for tpk in [0, 4, 9, 19]:
                            tf.summary.scalar("Top-%d" % (tpk + 1), acc_k[tpk], k + 1)
                        print("\tTop-1=%.2f; Top-5=%.2f; Top-10=%.2f" % (acc_k[0], acc_k[4], acc_k[9]))

            k += 1

        return self

    def _initialize_active_fingerprints_and_losses(self, candidates: CandidateSetMetIdent, verbose=False):
        """
        :param alphas:
        :param y:
        :param candidates:
        :return:
        """
        fps_active = np.zeros((self.alphas.n_active(), candidates.n_fps()))
        lab_losses_active = np.zeros((self.alphas.n_active(), ))

        itr = range(self.alphas.n_active())
        if verbose:
            itr = tqdm(itr, desc="Collect active fingerprints and losses")

        for c in itr:
            j, ybar = self.alphas.get_iy_for_col(c)
            # Get the fingerprints of the active candidates for example j
            fps_active[c] = candidates.get_candidates_fp(self.y_train[j], ybar)
            # Get label loss between the active fingerprint candidate and its corresponding gt fingerprint
            lab_losses_active[c] = self.label_loss_fun(candidates.get_gt_fp(self.y_train[j]), fps_active[c])

        return fps_active, lab_losses_active

    @staticmethod
    def _is_feasible_matrix(alphas: DualVariables, C: float) -> bool:
        A = alphas.get_dual_variable_matrix()
        N = A.shape[0]

        if (A < 0).getnnz():
            return False

        if not np.all(np.isclose(A.sum(axis=1), C / N)):
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
            -> Tuple[float, float, float]:
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

        return prim_obj, dual_obj, (prim_obj - dual_obj) / (np.abs(prim_obj) + 1)

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