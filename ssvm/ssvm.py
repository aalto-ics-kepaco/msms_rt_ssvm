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

from typing import List, ItemsView, Tuple, ValuesView, KeysView, Iterator, Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_random_state
from sklearn.metrics import hamming_loss
from scipy.sparse import csc_matrix, coo_matrix, lil_matrix, csr_matrix

from ssvm.data_structures import SequenceSample, CandidateSetMetIdent


class DualVariables(object):
    def __init__(self, C: float, N: int, cand_ids: List[List[List[str]]], num_init_active_vars=1, rs=None):
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
        self._alphas, self._y2col, self._iy = self._get_initial_alphas()

    def _get_initial_alphas(self) -> Tuple[lil_matrix, List[Dict], List[Tuple]]:
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

    def get_dual_variable_matrix(self) -> csr_matrix:
        """
        Returns the dual variable matrix as a csr-sparse matrix.

        :return: csr_matrix, shape = (N, n_active_dual)
        """
        return csr_matrix(self._alphas)

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


# class _StructuredSVM(BaseEstimator, ClassifierMixin)
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

        self.rs = check_random_state(rs)  # Type: np.random.RandomState

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
        Step-width calculation by [1].

        :param k:
        :param N:
        :return:
        """
        return (2 * N) / (k + 2 * N)


class StructuredSVMMetIdent(_StructuredSVM):
    def __init__(self, C=1.0, n_epochs=1000, label_loss="hamming", rs=None, sub_problem_solver="version_01"):
        self.sub_problem_solver = sub_problem_solver

        super(StructuredSVMMetIdent, self).__init__(C=C, n_epochs=n_epochs, label_loss=label_loss, rs=rs)

    def fit(self, X: np.ndarray, y: np.ndarray, candidates: CandidateSetMetIdent, num_init_active_vars_per_seq=1):
        """
        Train the SSVM given a dataset.
        :return:
        """
        # Number of training sequences
        N = len(X)

        # To allow pre-calculation of some relevant data, we already draw the random sequence of examples we use for the
        # optimization.
        i_k = self.rs.choice(N, self.n_epochs)

        # Pre-calculate some data needed for the sub-problem solving
        print("Pre-calculate data: ", end="")
        lab_losses = {}
        mol_kernel_l_y = {}
        for idx, i in enumerate(i_k):
            if ((idx + 1) % int(np.floor(self.n_epochs / 10))) == 0:
                print("...%.0f" % ((idx + 1) / self.n_epochs * 100), end="")

            y_i = y[i]

            # Check whether the relevant stuff was already pre-computed
            if y_i in lab_losses:
                continue

            # Pre-calculate the label loss: Loss of the gt fingerprint to all corresponding candidate fingerprints of i
            lab_losses[y_i] = np.array([self.label_loss_fun(candidates.get_gt_fp(y[i]).flatten(), fp.flatten())
                                        for fp in candidates.get_candidates_fp(y_i)])

            # Pre-calculate the kernels between the training examples and candidates
            mol_kernel_l_y[y_i] = candidates.getMolKernel_ExpVsCand(y, y_i)  # shape = (N, |Sigma_i|)
        print("")

        # Initialize dual variables
        alphas = DualVariables(C=self.C, N=N, cand_ids=[[candidates.get_labelspace(y[i])] for i in range(N)],
                               rs=self.rs, num_init_active_vars=num_init_active_vars_per_seq)
        assert self._is_feasible_matrix(alphas, self.C), "Initial dual variables must be feasible."

        # Collect active candidate fingerprints
        fps_active = lil_matrix((alphas.n_active(), candidates.n_fps()))
        for c in range(alphas.n_active()):
            # Get the fingerprints of the active candidates for example j
            j, ybar = alphas.get_iy_for_col(c)
            fps_active[c] = candidates.get_candidates_fp(y[j], ybar, as_dense=False)

        k = 0
        while k < self.n_epochs:
            if (k % 10) == 0:
                print("Epoch: %d / %d" % (k + 1, self.n_epochs))

            # Pick a random coordinate to update
            i = i_k[k]
            print("Selected example: %d" % i)
            print("Number of candidates: %d" % len(candidates.get_labelspace(y[i])))

            # Find the most violating example
            y_i_hat = self._solve_sub_problem_v03(alphas, X, y, i, candidates,
                                                  pre_calc_data={"lab_losses": lab_losses,
                                                                 "mol_kernel_l_y": mol_kernel_l_y,
                                                                 "fps_active": fps_active})

            # Get step-width
            gamma = self._get_diminishing_stepwidth(k, N)

            # Update the dual variables
            if alphas.update(i, y_i_hat, gamma):
                fps_active.resize(fps_active.shape[0] + 1, fps_active.shape[1])
                fps_active[fps_active.shape[0] - 1, :] = candidates.get_candidates_fp(y[i], y_i_hat, as_dense=False)

            assert self._is_feasible_matrix(alphas, self.C), "Initial dual variables must be feasible."

            k += 1

        # Store relevant data for the prediction
        self.X_train = X
        self.alphas = alphas
        self.fps_active = fps_active

        return self

    def score(self, X: np.ndarray, y: np.ndarray, candidates: CandidateSetMetIdent):
        assert X.shape[1] == self.X_train.shape[0]



    @staticmethod
    def _is_feasible_matrix(alphas: DualVariables, C: float) -> bool:
        A = alphas.get_dual_variable_matrix()
        N = A.shape[0]

        if (A < 0).getnnz():
            return False

        if not np.all(np.isclose(A.sum(axis=1), C / N)):
            return False

        return True

    def _solve_sub_problem_v01(self, alphas, X, y, i, candidates: CandidateSetMetIdent) -> Tuple:
        """

        :param alphas:
        :param X:
        :param i:
        :param candidates:
        :return:
        """
        N = len(alphas)

        # Calculate label loss
        y_i = y[i]
        fp_i = candidates.get_gt_fp(y_i).flatten()

        if self.label_loss == "hamming":
            loss = np.array([hamming_loss(fp_i, fp.flatten()) for fp in candidates.get_candidates_fp(y_i)])
        else:
            raise ValueError("Invalid label loss '%s'. Choices are 'hamming'.")

        # spectra kernel similarity between example i and all other examples
        k_i = X[i, :]  # shape = (N, )

        # molecule kernel between all candidates of example i and all other training examples
        l_y = candidates.getMolKernel_ExpVsCand(y, y_i)  # shape = (N, |Sigma_i|)

        s_j_y = (self.C / N) * k_i @ l_y  # shape = (|Sigma_i|, )

        # Version 1:
        s_ybar_y = np.zeros_like(l_y)
        for j in range(N):
            ybars = [k[0] for k in alphas[j]]
            a_j_ybars = np.array(list(alphas[j].values()))
            s_ybar_y[j, :] = a_j_ybars @ candidates.getMolKernel_CandVsCand(y[j], y_i, ybars)  # shape = (|Sigma_i|, )

        s_ybar_y = k_i @ s_ybar_y  # shape = (|Sigma_i|, )

        return (candidates.get_labelspace(y_i)[np.argmax(loss + s_j_y - s_ybar_y).item()], )

    def _solve_sub_problem_v02(self, alphas, X, y, i, candidates: CandidateSetMetIdent) -> Tuple:
        """

        :param alphas:
        :param X:
        :param i:
        :param candidates:
        :return:
        """
        N = len(alphas)

        # Calculate label loss
        y_i = y[i]
        fp_i = candidates.get_gt_fp(y_i).flatten()

        if self.label_loss == "hamming":
            loss = np.array([hamming_loss(fp_i, fp.flatten()) for fp in candidates.get_candidates_fp(y_i)])
        else:
            raise ValueError("Invalid label loss '%s'. Choices are 'hamming'.")

        # spectra kernel similarity between example i and all other examples
        k_i = X[i, :]  # shape = (N, )

        # molecule kernel between all candidates of example i and all other training examples
        l_y = candidates.getMolKernel_ExpVsCand(y, y_i)  # shape = (N, |Sigma_i|)

        s_j_y = (self.C / N) * k_i @ l_y  # shape = (|Sigma_i|, )

        # Version 2:
        # Collect all "other" candidate fingerprints we need to calculate the similarity with. Those are basically the
        # ones corresponding to the active dual variables.
        n_active_vars = np.sum([len(alphas[j]) for j in range(N)]).item()
        a_ybars = np.zeros((n_active_vars, N))
        fps_ybars = np.full((n_active_vars, len(fp_i)), fill_value=np.nan)
        last_idx = 0
        for j in range(N):
            # Active dual variable values of example j
            a_ybars[last_idx:(last_idx + len(alphas[j])), j] = list(alphas[j].values())

            # Get the fingerprints of the active candidates for example j
            ybars = [k[0] for k in alphas[j]]
            fps_ybars[last_idx:(last_idx + len(alphas[j])), :] = candidates.get_candidates_fp(y[j], ybars)

            last_idx += len(alphas[j])

        assert not np.any(np.isnan(fps_ybars))

        s_ybar_y = candidates.get_kernel(candidates.get_candidates_fp(y_i), fps_ybars) @ csc_matrix(a_ybars)

        s_ybar_y = s_ybar_y @ k_i  # shape = (|Sigma_i|, )

        return (candidates.get_labelspace(y_i)[np.argmax(loss + s_j_y - s_ybar_y).item()], )

    def _solve_sub_problem_v03(self, alphas, X, y, i, candidates: CandidateSetMetIdent, pre_calc_data) -> Tuple:
        """

        :param alphas:
        :param X:
        :param i:
        :param candidates:
        :return:
        """
        A = alphas.get_dual_variable_matrix()
        N = A.shape[0]

        # Calculate label loss
        y_i = y[i]
        loss = pre_calc_data["lab_losses"][y_i]

        # spectra kernel similarity between example i and all other examples
        k_i = X[i, :]  # shape = (N, )

        # molecule kernel between all candidates of example i and all other training examples
        l_y = pre_calc_data["mol_kernel_l_y"][y_i]  # shape = (N, |Sigma_i|)

        s_j_y = (self.C / N) * k_i @ l_y  # shape = (|Sigma_i|, )

        s_ybar_y = A @ candidates.get_kernel(csr_matrix(pre_calc_data["fps_active"]),
                                             candidates.get_candidates_fp(y_i, as_dense=False))

        s_ybar_y = k_i @ s_ybar_y  # shape = (|Sigma_i|, )

        return (candidates.get_labelspace(y_i)[np.argmax(loss + s_j_y - s_ybar_y).item()], )


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