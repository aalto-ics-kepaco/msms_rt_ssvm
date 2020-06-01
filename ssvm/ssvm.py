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

from typing import List, ItemsView, Tuple, ValuesView, KeysView
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_random_state
from sklearn.metrics import hamming_loss

from ssvm.sequence import SequenceSample


class DualVariables(object):
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

    def __iter__(self) -> dict:
        return self._alphas

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


class StructuredSVM(BaseEstimator, ClassifierMixin):
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
        alphas = [DualVariables(N=N, C=self.C, cand_ids=data.get_labelspace(i),
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

    def _solve_sub_problem(self, alpha: List[DualVariables], data: SequenceSample, i: int) -> tuple:
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
    def _is_feasible(alphas: List[DualVariables], C: float) -> bool:
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