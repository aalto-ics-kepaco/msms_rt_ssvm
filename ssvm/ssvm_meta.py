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

from typing import Optional, Union

from ssvm.loss_functions import hamming_loss, tanimoto_loss, minmax_loss, generalized_tanimoto_loss
from ssvm.dual_variables import DualVariables


class _StructuredSVM(object):
    """
    Structured Support Vector Machine (SSVM) meta-class
    """

    def __init__(self, C: Union[int, float] = 1, n_epochs: int = 100, batch_size: Union[int, None] = 1,
                 label_loss: str = "hamming", step_size_approach: str = "diminishing",
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        """
        Structured Support Vector Machine (SSVM)

        :param C: scalar, SVM regularization parameter. Must be > 0.

        :param n_epochs: scalar, Number of epochs, i.e. passes over the complete dataset.

        :param batch_size: scalar or None, Batch size, i.e. number of examples updated in each iteration. If None, the
            batch encompasses the complete dataset.

        :param label_loss: string, indicating which label-loss is used.

        :param step_size_approach: string, indicating which strategy is used to determine the step-size.

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
        self.step_size_approach = step_size_approach

        if self.label_loss == "hamming":
            self.label_loss_fun = hamming_loss
        elif self.label_loss == "tanimoto_loss":
            self.label_loss_fun = tanimoto_loss
        elif self.label_loss == "minmax_loss":
            self.label_loss_fun = minmax_loss
        elif self.label_loss == "generalized_tanimoto_loss":
            self.label_loss_fun = generalized_tanimoto_loss
        else:
            raise ValueError(
                "Invalid label loss '%s'. Choices are 'hamming', 'tanimoto_loss', 'minmax_loss' and "
                "'generalized_tanimoto_loss'."
            )

        if self.step_size_approach not in ["diminishing", "linesearch", "linesearch_parallel"]:
            raise ValueError("Invalid stepsize method '%s'. Choices are 'diminishing' and 'linesearch'." %
                             self.step_size_approach)

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
        Step-size calculation after [1].

        :param k: scalar, number of iterations so far

        :param N: scalar, number of training samples

        :return: scalar, step-size
        """
        return (2 * N) / (k + 2 * N)
