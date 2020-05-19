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

from collections import OrderedDict
from typing import List
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_random_state
from sklearn.metrics import hamming_loss

from ssvm.sequence import Sequence, SequenceSample


class SequenceDualVariables(object):
    def __init__(self, C: float, N: int, cand_ids: List[List[str]]):
        self.C = C  # SSVM regularization parameter
        self.N = N  # Total number of sequences
        self.l_cand_ids = cand_ids  # Candidate identifiers for each spectra in the sequence

    def initialize(self, rs=None, num_active_vars=1):
        """

        :param rs:
        :param num_active_vars: scalar, number of initially active dual variables.
        :return:
        """
        if self._is_initialized():
            raise AssertionError("Dual variable has been already initialized.")

        # Set up the random generator
        rs = check_random_state(rs)  # Type: np.random.RandomState

        # Active dual variables are stored in a dictionary: key = label sequence, value = variable value
        _alphas = OrderedDict()

        init_var_val = self.C / (self.N * num_active_vars)  # (C / N) / num_act = C / (N * num_act)
        for i in range(num_active_vars):
            y_seq = (rs.choice(cand_ids) for cand_ids in self.l_cand_ids)
            assert y_seq not in _alphas, "Oups, we sampled the same active dual variable again."
            # TODO: Ensure that the same variable is not sampled repeatedly. Should be extremely rare in application.

            _alphas[y_seq] = init_var_val

        self._alphas = _alphas

        return self

    def __len__(self) -> int:
        self._ensure_is_initialized()

        return len(self._alphas)

    def __iter__(self) -> OrderedDict:
        self._ensure_is_initialized()

        return self._alphas

    def _is_initialized(self) -> bool:
        return hasattr(self, "_alphas")

    def _ensure_is_initialized(self):
        if not self._is_initialized():
            raise AssertionError("Dual variables have not been initialized yet.")

    def items(self):
        return self._alphas.items()


class DualVariables(object):
    def __init__(self, N: int, X: List[Sequence], C: float):
        """
        :param N: scalar, Number of training examples
        """
        self.N = N
        self.X = X
        self.C = C

        self._alphas = [{} for _ in range(N)]  # Store only active dual variables
        self._k = 0  # Number of updates done
        
        super(DualVariables, self).__init__()

    def initialize(self, random_state=None):
        """

        :param random_state:
        :return:
        """
        random_state = check_random_state(random_state)
        for i in range(self.N):
            y_seq = (random_state.randint(n_cand) for n_cand in self.X[i].get_n_cand())  # draw a label sequence vector
            self._alphas[i][y_seq] = self.C / self.N
            # HINT: eventually we can draw multiple label sequence vectors trying to "as orthogonal as possible"

    def __len__(self):
        return len(self._alphas)

    def __iter__(self) -> List[dict]:
        """
        :return: list of dicts
        """
        return self._alphas

    def __getitem__(self, item):
        return self._alphas[item]

    def update(self, i: int, y_seq: tuple, gamma: float):
        try:
            self._alphas[i][y_seq] = (1 - gamma) * self._alphas[i][y_seq] + gamma * (self.N / self.C)
        except KeyError:
            self._alphas[i][y_seq] = gamma * (self.N / self.C)


class StructuredSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, n_epochs=1000, label_loss="hamming"):
        """
        Structured Support Vector Machine (SSVM) class.

        :param C: scalar, SVM regularization parameter. Must be > 0.
        :param n_epochs: scalar, Number of epochs, i.e. maximum number of iterations.
        :param label_loss: string, Which label loss should be used. Default is 'hamming' loss.
        """
        self.C = C
        self.n_epochs = n_epochs
        self.label_loss = label_loss

    def fit(self, X: List[Sequence], y: List[tuple]):
        """
        Train the SSVM given a dataset.

        :param X: list of tuples, of length N. Each element X_i, i.e. X[i], is a tuple (x_i, t_i, C_i) and represents
            a (MS, RT) training sequence:

            x_i: list of array-likes, of length L. Each element x_{is}, i.e. X[i][0][s], corresponds the either to the
                feature or kernel similarity vector of a sequence element s with the other training data
            t_i: array-like, of length L. Each element t_{is}, i.e. X[i][1][s], is the retention time of sequence
                element s. Therefore, it is a scalar value.
            C_i: list of array-likes. of length L. Each element C_{is}, i.e. X[i][2][s], corresponds to either to the
                feature or kernel similarity vector of the molecular candidates of the sequence element s
                TODO: Here we need to versions. (1) The embedding for the MS, and (2) for the Retention Order

        :param y: list of array-likes,
        :return:
        """
        # Number of training sequences
        N = len(X)

        self._X_train = X

        # Get the number of dual variables per sequence and in total
        n_dual = np.array([np.prod(seq.get_n_cand()) for seq in X])

        # Set up the dual initial dual vector
        alphas = DualVariables(N=N, C=self.C, X=X)
        assert self._is_feasible(alphas, self.C), "Initial dual variables must be feasible."

        k = 0
        while k < self.n_epochs:
            # Pick a random coordinate to update
            i = np.random.randint(N)

            # Find the most violating example
            y_i_hat = self._solve_sub_problem(alphas, X, y, i)

            # Get step-width
            gamma = self._get_diminishing_stepwidth(k, N)

            # Update the dual variables
            alphas.update(i, y_i_hat, gamma)

            assert self._is_feasible(alphas, self.C)

        return self

    def _solve_sub_problem(self, alpha: DualVariables, X: List[Sequence], y: List[tuple], i: int) -> tuple:
        """
        Find the most violating example by solving the MAP problem. Forward-pass using max marginals.

        :param alphas:
        :param X:
        :return:
        """
        # X_i = X[i]
        # y_i = y[i]
        #
        # V = list(range(len(X_i)))  # Number of sequence elements defines the set of nodes
        # E = [(sigma, sigma + 1) for sigma in V[:-1]]  # Edge set of a linear graph (tree-like)
        #
        # # Pre-calculate the Hamming-losses
        # lloss = {sigma: np.array([hamming_loss(y_i, cand) for cand in X_i[sigma].cand]) for sigma in V}
        # # TODO: The hamming loss can be globally pre-computed, as it does not depend on alpha.
        #
        # # Pre-calculate the node-potentials: sv_i
        # sv_i = {}
        # for sigma in V:
        #     sv_i[sigma] = np.zeros(shape=X_i.get_n_cand_sig(sigma))
        #
        #     for j, (X_j, y_j) in enumerate(zip(X, y)):
        #         # If no active dual variables
        #         if not len(alpha[j]):
        #             continue
        #
        #         # Get active dual variables for j: alpha(j, bar_y) > 0
        #         yj_active, aj_active = zip(*alpha[j].items())
        #         print(aj_active)
        #
        #         for bst_y in yj_active:
        #             sv_i[sigma] += data.jointKernel(X_j, y_j, X_i[sigma], None) - jointKernelMS(X_j, bst_y, X_i[sigma], None)
        #
        #
        # # Pre-calculate the edge-potentials: se_i


        pass

    @staticmethod
    def _is_feasible(alphas: DualVariables, C: float) -> bool:
        """
        Check the feasibility of the dual variable.

        :param alphas:
        :param C:
        :return:
        """
        val = C / len(alphas)  # C / N
        for a_i in alphas:
            sum_a_i = 0
            for a_iy in a_i:
                # Alphas need to be positive
                if a_iy < 0:
                    return False

                sum_a_i += a_iy

            if sum_a_i != val:  # TODO: Use "not np.isclose(sum_a_i, C / N)"
                return False

        raise True

    @staticmethod
    def _get_diminishing_stepwidth(k: int, N: int) -> float:
        """
        Step-width calculation by [1].

        :param k:
        :param N:
        :return:
        """
        return (2 * N) / (k + 2 * N)


class StructuredSVM2(BaseEstimator, ClassifierMixin):
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

        self.rs = check_random_state(rs)  # Type: np.random.RandomState

    def fit(self, data: SequenceSample, num_init_active_vars_per_seq=1):
        """
        Train the SSVM given a dataset.
        :return:
        """
        # Number of training sequences
        N = len(data)

        # Set up the dual initial dual vector
        alphas = [SequenceDualVariables(N=N, C=self.C, cand_ids=data.get_labelspace(i))
                      .initialize(rs=self.rs, num_active_vars=num_init_active_vars_per_seq)
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
            alphas.update(i, y_i_hat, gamma)

            assert self._is_feasible(alphas, self.C)

        return self

    def _solve_sub_problem(self, alpha: List[SequenceDualVariables], data: SequenceSample, i: int) -> tuple:
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

    @staticmethod
    def _is_feasible(alphas: List[SequenceDualVariables], C: float) -> bool:
        """
        Check the feasibility of the dual variable.

        :param alphas:
        :param C:
        :return:
        """
        val = C / len(alphas)  # C / N
        for a_i in alphas:
            sum_a_i = 0
            for a_iy in a_i:
                # Alphas need to be positive
                if a_iy < 0:
                    return False

                sum_a_i += a_iy

            if sum_a_i != val:  # TODO: Use "not np.isclose(sum_a_i, C / N)"
                return False

        raise True

    @staticmethod
    def _get_diminishing_stepwidth(k: int, N: int) -> float:
        """
        Step-width calculation by [1].

        :param k:
        :param N:
        :return:
        """
        return (2 * N) / (k + 2 * N)