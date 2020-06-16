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
import unittest
import pickle
import gzip
import os
import itertools as it

from typing import Tuple

from ssvm.ssvm import DualVariablesForExample, _StructuredSVM, StructuredSVMMetIdent, DualVariables


class TestStructuredSVM(unittest.TestCase):
    def test_is_feasible(self):
        N = 3
        C = 2

        # ------------------------------------
        # Check a feasible alpha
        alphas = [{("M1", "M4", "M9"): C / N},
                  {("M5", "M1", "M10"): C / (2 * N), ("M6", "M2", "M1"): C / (2 * N)},
                  {("M2", "M1", "M3"): C / (3 * N), ("M5", "M2", "M1"): C / (3 * N), ("M10", "M1", "M7"): C / (3 * N)}]
        assert len(alphas) == N

        self.assertTrue(_StructuredSVM._is_feasible(alphas, C))
        self.assertFalse(_StructuredSVM._is_feasible(alphas, 1.9))
        self.assertFalse(_StructuredSVM._is_feasible(alphas, 2.1))

        # ------------------------------------
        # Check a infeasible alpha
        alphas = [{("M1", "M4", "M9"): C / N},
                  {("M5", "M1", "M10"): C / (2 * N), ("M6", "M2", "M1"): C / (3 * N)},
                  {("M2", "M1", "M3"): C / (3 * N), ("M5", "M2", "M1"): C / (2 * N), ("M10", "M1", "M7"): C / (3 * N)}]
        assert len(alphas) == N

        self.assertFalse(_StructuredSVM._is_feasible(alphas, C))

        alphas = [{("M1", "M4", "M9"): - 1},
                  {("M5", "M1", "M10"): C / (2 * N), ("M6", "M2", "M1"): C / (3 * N)},
                  {("M2", "M1", "M3"): C / (3 * N), ("M5", "M2", "M1"): C / (2 * N), ("M10", "M1", "M7"): C / (3 * N)}]
        assert len(alphas) == N

        self.assertFalse(_StructuredSVM._is_feasible(alphas, C))


class TestStructuredSVMMetIdent(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up a small example data for the metabolite identification tests.
        """
        if not os.path.exists("small_metabolite_data.pkl.gz"):
            # Create a small example containing metabolite identification data from the ISMB 2016 paper
            from sklearn.model_selection import ShuffleSplit
            from ssvm.examples.metabolite_identification import read_data
            from ssvm.data_structures import CandidateSetMetIdent

            idir = "/home/bach/Documents/doctoral/data/metindent_ismb2016"
            X, fps, mols, mols2cand = read_data(idir)

            # Get a smaller subset
            _, subset = next(ShuffleSplit(n_splits=1, test_size=0.025, random_state=1989).split(X))
            X = X[np.ix_(subset, subset)]
            fps = fps[subset]
            mols = mols[subset]
            print("N samples:", len(mols))

            # Wrap the candidate sets for easier access
            cand = CandidateSetMetIdent(mols, fps, mols2cand, idir=os.path.join(idir, "candidates"), preload_data=False)

            with gzip.open(os.path.join("small_metabolite_data.pkl.gz"), "wb") as gz_file:
                pickle.dump({"X": X, "mols": mols, "cand": cand}, gz_file)

        with gzip.open("small_metabolite_data.pkl.gz", "rb") as gz_file:
            data = pickle.load(gz_file)

        self.ssvm = StructuredSVMMetIdent(C=2)
        self.ssvm.K_train = data["X"]
        self.ssvm.y_train = data["mols"]
        self.cand = data["cand"]  # type: CandidateSetMetIdent
        self.N = self.ssvm.K_train.shape[0]
        self.rs = np.random.RandomState(18221)
        self.ssvm.alphas = DualVariables(C=self.ssvm.C, rs=self.rs, num_init_active_vars=2,
                                         cand_ids=[[self.cand.get_labelspace(self.ssvm.y_train[i])]
                                                   for i in range(self.N)])

    def test_get_candidate_scores(self):
        """
        Test that the scores for the candidates corresponding to a particular spectrum x_i are correctly calculated
        given an SSVM model (w or alphas). The score is calculated like:

            s(x_i, y) = < w , Psi(x_i, y) >     for all y in Sigma_i
        """
        i = 10

        self.ssvm.fps_active, lab_losses_active = self.ssvm._initialize_active_fingerprints_and_losses(
            self.cand, verbose=True)
        scores = self.ssvm._get_candidate_scores(i, self.cand, {"lab_losses_active": lab_losses_active})

        fps_gt_i = self.cand.get_gt_fp(self.ssvm.y_train[i])[np.newaxis, :]
        fps_gt = self.cand.get_gt_fp(self.ssvm.y_train)
        B = self.ssvm.alphas.get_dual_variable_matrix(type="csr")

        # Part which is constant for all y in Sigma_i
        s1 = self.ssvm.C / self.N * self.ssvm.K_train[i] @ self.cand.get_kernel(fps_gt_i, fps_gt).flatten()
        self.assertTrue(np.isscalar(s1))

        s2 = (self.cand.get_kernel(fps_gt_i, self.ssvm.fps_active) @ B.T @ self.ssvm.K_train[i]).item()
        self.assertTrue(np.isscalar(s2))

        # Part that is specific to each candidate y in Sigma_i
        s3 = self.ssvm.C / self.N * self.cand.get_kernel(
            self.cand.get_candidates_fp(self.ssvm.y_train[i]), fps_gt) @ self.ssvm.K_train[i]
        self.assertEqual((len(self.cand.get_labelspace(self.ssvm.y_train[i])),), s3.shape)

        s4 = self.cand.get_kernel(self.cand.get_candidates_fp(self.ssvm.y_train[i]),
                                  self.ssvm.fps_active) @ B.T @ self.ssvm.K_train[i]
        self.assertEqual((len(self.cand.get_labelspace(self.ssvm.y_train[i])),), s4.shape)

        np.testing.assert_allclose(scores, (s3 - s4))


class TestDualVariables(unittest.TestCase):
    def test_initialization(self):
        # ----------------------------------------------------
        cand_ids = [
            [
                ["M1", "M2"],
                ["M1", "M2", "M19", "M10"],
                ["M20", "M4", "M5"],
                ["M2"],
                ["M3", "M56", "M8"]
            ],
            [
                ["M1"],
                ["M1", "M2", "M3", "M10", "M20"],
                ["M4", "M5"],
                ["M2", "M4"],
                ["M7", "M9", "M8"]
            ],
            [
                ["M1", "M2", "M3"],
                ["M1", "M2", "M9", "M10", "M22"],
                ["M20", "M7"],
                ["M2", "M99"],
                ["M72", "M8"]
            ]
        ]
        N = len(cand_ids)

        for C in [0.5, 1.0, 2.0]:
            for num_init_active_vars in range(1, 3):
                n_active = N * num_init_active_vars
                alphas = DualVariables(C=C, cand_ids=cand_ids, num_init_active_vars=num_init_active_vars, rs=10910)

                self.assertEqual(n_active, alphas.n_active())

                B = alphas.get_dual_variable_matrix()
                self.assertEqual((N, n_active), B.shape)
                np.testing.assert_equal(np.full(N, fill_value=num_init_active_vars),
                                        np.array(np.sum(B > 0, axis=1)).flatten())

                # FIXME: We rely for the test on a not tested function
                self.assertTrue(StructuredSVMMetIdent._is_feasible_matrix(alphas, C))

                for i in range(N):
                    for k in range(num_init_active_vars):
                        col = i * num_init_active_vars + k
                        idx, y = alphas.get_iy_for_col(col)  # type: Tuple[int, Tuple]
                        self.assertEqual(i, idx)
                        self.assertIn(y, list(it.product(*cand_ids[i])))

    def test_initialization_singleton_sequences(self):
        # ----------------------------------------------------
        cand_ids = [
            [
                ["M1", "M2", "M19", "M10"]
            ],
            [
                ["M7", "M9", "M8"]
            ],
            [
                ["M72", "M11"]
            ],
            [
                ["M12", "M13", "M3", "M4", "M22"]
            ]
        ]
        N = len(cand_ids)

        for C in [0.5, 1.0, 2.0]:
            for num_init_active_vars in range(1, 3):
                n_active = N * num_init_active_vars
                alphas = DualVariables(C=C, cand_ids=cand_ids, num_init_active_vars=num_init_active_vars, rs=10910)

                self.assertEqual(n_active, alphas.n_active())

                B = alphas.get_dual_variable_matrix()
                self.assertEqual((N, n_active), B.shape)

                np.testing.assert_equal(np.full(N, fill_value=num_init_active_vars),
                                        np.array(np.sum(B > 0, axis=1)).flatten())

                # FIXME: We rely for the test on a not tested function
                self.assertTrue(StructuredSVMMetIdent._is_feasible_matrix(alphas, C))

                for i in range(N):
                    for k in range(num_init_active_vars):
                        col = i * num_init_active_vars + k
                        idx, y = alphas.get_iy_for_col(col)  # type: Tuple[int, Tuple]
                        self.assertEqual(i, idx)
                        self.assertIn(y, list(it.product(*cand_ids[i])))

    def test_update(self):
        # ----------------------------------------------------
        cand_ids = [
            [
                ["M1", "M2", "M19", "M10"]
            ],
            [
                ["M7", "M9", "M8"]
            ],
            [
                ["M72", "M11"]
            ],
            [
                ["M12", "M13", "M3", "M4", "M22"]
            ]
        ]
        N = len(cand_ids)
        C = 2.0
        gamma = 0.46

        alphas = DualVariables(C=C, cand_ids=cand_ids, num_init_active_vars=2, rs=10910)
        # Active variables
        # [(0, ('M19',)), (0, ('M10',)),
        #  (1, ('M8',)),  (1, ('M7',)),
        #  (2, ('M72',)), (2, ('M11',)),
        #  (3, ('M22',)), (3, ('M12',))]

        # ---- Update an active dual variable ----
        B_old = alphas.get_dual_variable_matrix().todense()
        val_old_M8 = alphas.get_dual_variable(1, ("M8",))
        val_old_M7 = alphas.get_dual_variable(1, ("M7",))
        val_old_M9 = alphas.get_dual_variable(1, ("M9",))

        self.assertEqual(C / (N * 2), val_old_M8)
        self.assertEqual(C / (N * 2), val_old_M7)
        self.assertEqual(0, val_old_M9)

        self.assertFalse(alphas.update(1, ("M8",), gamma))
        self.assertEqual(8, alphas.n_active())  # Number of active variables must not change
        self.assertEqual((1 - gamma) * val_old_M8 + gamma * C / N, alphas.get_dual_variable(1, ("M8",)))
        self.assertEqual((1 - gamma) * val_old_M7 + gamma * 0,     alphas.get_dual_variable(1, ("M7",)))
        self.assertEqual((1 - gamma) * val_old_M9 + gamma * 0,     alphas.get_dual_variable(1, ("M9",)))

        B_new = alphas.get_dual_variable_matrix().todense()
        np.testing.assert_equal(B_old[[0, 2, 3], :], B_new[[0, 2, 3], :])  # No changes to other variables

        # ---- Update an inactive dual variable ----
        val_old_M12 = alphas.get_dual_variable(3, ("M12",))
        val_old_M13 = alphas.get_dual_variable(3, ("M13",))
        val_old_M3 = alphas.get_dual_variable(3, ("M3",))
        val_old_M4 = alphas.get_dual_variable(3, ("M4",))
        val_old_M22 = alphas.get_dual_variable(3, ("M22",))

        self.assertEqual(C / (N * 2), val_old_M12)
        self.assertEqual(0, val_old_M13)
        self.assertEqual(0, val_old_M3)
        self.assertEqual(0, val_old_M4)
        self.assertEqual(C / (N * 2), val_old_M22)

        self.assertTrue(alphas.update(3, ("M3",), gamma))
        self.assertEqual(9, alphas.n_active())
        self.assertEqual((1 - gamma) * val_old_M12 + gamma * 0, alphas.get_dual_variable(3, ("M12",)))
        self.assertEqual((1 - gamma) * val_old_M13 + gamma * 0, alphas.get_dual_variable(3, ("M13",)))
        self.assertEqual((1 - gamma) * val_old_M3 + gamma * C / N, alphas.get_dual_variable(3, ("M3",)))
        self.assertEqual((1 - gamma) * val_old_M4 + gamma * 0, alphas.get_dual_variable(3, ("M4",)))
        self.assertEqual((1 - gamma) * val_old_M22 + gamma * 0, alphas.get_dual_variable(3, ("M22",)))

        self.assertEqual((3, ("M3",)), alphas.get_iy_for_col(8))


class TestDualVariablesForExamples(unittest.TestCase):
    def test_initialization(self):
        # ------------------------------------
        cand_ids = [["M1"], ["M1", "M2", "M3", "M10", "M20"], ["M4", "M5"], ["M2", "M4"], ["M7", "M9", "M8"]]

        for C, N in [(1, 10), (0.5, 2), (2, 1)]:
            for n in range(1, 11):
                alphas = DualVariablesForExample(C=C, N=N, cand_ids=cand_ids, rs=11, num_init_active_vars=n)
                self.assertEqual(n, len(alphas))
                for key, a in alphas.items():
                    self.assertIsInstance(key, tuple)
                    self.assertEqual(C / (N * n), a)

        # ------------------------------------
        cand_ids = [["M1", "M2", "M3", "M10", "M20"]]

        for C, N in [(1, 10), (0.5, 2), (2, 1)]:
            for n in range(1, 3):
                alphas = DualVariablesForExample(C=C, N=N, cand_ids=cand_ids, rs=11, num_init_active_vars=n)
                self.assertEqual(n, len(alphas))
                for key, a in alphas.items():
                    self.assertIsInstance(key, tuple)
                    self.assertEqual(C / (N * n), a)

        # ------------------------------------
        cand_ids = [["M1"], ["M1", "M2", "M3", "M10", "M20"], ["M4", "M5"], ["M2", "M4"], ["M7", "M9", "M8"]]
        alphas = [DualVariablesForExample(C=1.5, N=3, cand_ids=cand_ids, rs=12, num_init_active_vars=2),
                  DualVariablesForExample(C=1.5, N=3, cand_ids=cand_ids, rs=12, num_init_active_vars=1),
                  DualVariablesForExample(C=1.5, N=3, cand_ids=cand_ids, rs=12, num_init_active_vars=3)]
        self.assertTrue(_StructuredSVM._is_feasible(alphas, 1.5))

    def test_update(self):
        cand_ids_1 = [["M1"], ["M1", "M2", "M3", "M10", "M20"], ["M4", "M5"], ["M2", "M4"], ["M7", "M9", "M8"]]

        # ------------------------------------
        gamma = 0.75
        C = 2
        N = 1
        alphas = DualVariablesForExample(C=C, N=N, cand_ids=cand_ids_1, rs=123, num_init_active_vars=2)

        # ('M1', 'M3', 'M4', 'M2', 'M8') and ('M1', 'M2', 'M5', 'M2', 'M9') are active.

        # Update an active dual variable
        y_seq_active = ('M1', 'M3', 'M4', 'M2', 'M8')
        a_old = alphas[y_seq_active]
        alphas.update(y_seq_active, gamma=gamma)
        self.assertTrue(alphas[y_seq_active] >= 0)
        self.assertTrue(alphas[y_seq_active] <= (C / N))
        self.assertEqual((1 - gamma) * a_old + gamma * (C / N), alphas[y_seq_active])

        # Update an inactive dual variable
        y_seq_inactive = ('M1', 'M20', 'M5', 'M2', 'M7')
        a_old = alphas[y_seq_inactive]
        self.assertEqual(0, a_old)
        alphas.update(y_seq_inactive, gamma=gamma)
        self.assertTrue(alphas[y_seq_inactive] >= 0)
        self.assertTrue(alphas[y_seq_inactive] <= (C / N))
        self.assertEqual((1 - gamma) * a_old + gamma * (C / N), alphas[y_seq_inactive])

        # ------------------------------------
        cand_ids_2 = [["M1", "M2"], ["M1", "M2", "M19", "M10"], ["M20", "M4", "M5"], ["M2"], ["M3", "M56", "M8"]]
        cand_ids_3 = [["M1", "M2", "M3"], ["M1", "M2", "M9", "M10", "M22"], ["M20", "M7"], ["M2", "M99"], ["M72", "M8"]]

        C = 2
        N = 3
        alphas = [DualVariablesForExample(C=C, N=N, cand_ids=cand_ids_1, rs=123, num_init_active_vars=2),
                  DualVariablesForExample(C=C, N=N, cand_ids=cand_ids_2, rs=121, num_init_active_vars=1),
                  DualVariablesForExample(C=C, N=N, cand_ids=cand_ids_3, rs=321, num_init_active_vars=4)]

        self.assertTrue(_StructuredSVM._is_feasible(alphas, C))
        print(alphas[0].items())

        # Dual variable set belonging to example 0
        y_seq_active = ('M1', 'M3', 'M4', 'M2', 'M8')
        y_seq_inactive = ('M1', 'M20', 'M5', 'M2', 'M7')
        alphas[0].update(y_seq_active, gamma)
        print(alphas[0].items())
        self.assertTrue(_StructuredSVM._is_feasible(alphas, C))
        alphas[0].update(y_seq_inactive, gamma)
        self.assertTrue(_StructuredSVM._is_feasible(alphas, C))


if __name__ == '__main__':
    unittest.main()
