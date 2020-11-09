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
from scipy.sparse import csr_matrix

from ssvm.ssvm import _StructuredSVM, StructuredSVMMetIdent, DualVariables


class TestStructuredSVM(unittest.TestCase):
    def test_is_feasible_matrix(self):
        class DummyDualVariables(object):
            def __init__(self, B):
                self.B = B

            def get_dual_variable_matrix(self):
                return self.B

        C = 2.2
        N = 4

        # ---- Feasible dual variables ----
        alphas = DummyDualVariables(
            csr_matrix([
                [C / (N * 2), C / (N * 2), 0, 0, 0, 0, 0, 0],
                [0, 0, C / (N * 2), C / (N * 2), 0, 0, 0, 0],
                [0, 0, 0, 0, C / (N * 3), C / (N * 3), C / (N * 3), 0],
                [0, 0, 0, 0, 0, 0, 0, C / N]
            ])
        )
        self.assertTrue(_StructuredSVM._is_feasible_matrix(alphas, C))
        self.assertFalse(_StructuredSVM._is_feasible_matrix(alphas, C + 1))

        # ---- Infeasible dual variables ----
        alphas = DummyDualVariables(
            csr_matrix([
                [C / (N * 3), C / (N * 2), 0, 0, 0, 0, 0, 0],
                [0, 0, C / (N * 2), C / (N * 2), 0, 0, 0, 0],
                [0, 0, 0, 0, C / (N * 2), C / (N * 2), C / (N * 3), 0],
                [0, 0, 0, 0, 0, 0, 0, C / N]
            ])
        )
        self.assertFalse(_StructuredSVM._is_feasible_matrix(alphas, C))
        self.assertFalse(_StructuredSVM._is_feasible_matrix(alphas, C + 1))


# class TestStructuredSVMMetIdent(unittest.TestCase):
#     def setUp(self) -> None:
#         """
#         Set up a small example data for the metabolite identification tests.
#         """
#         if not os.path.exists("small_metabolite_data.pkl.gz"):
#             # Create a small example containing metabolite identification data from the ISMB 2016 paper
#             from sklearn.model_selection import ShuffleSplit
#             from ssvm.development.ssvm_metident__conv_params import read_data
#             from ssvm.data_structures import CandidateSetMetIdent
#
#             idir = "/home/bach/Documents/doctoral/data/metindent_ismb2016"
#             X, fps, mols, mols2cand = read_data(idir)
#
#             # Get a smaller subset
#             _, subset = next(ShuffleSplit(n_splits=1, test_size=0.025, random_state=1989).split(X))
#             X = X[np.ix_(subset, subset)]
#             fps = fps[subset]
#             mols = mols[subset]
#             print("N samples:", len(mols))
#
#             # Wrap the candidate sets for easier access
#             cand = CandidateSetMetIdent(mols, fps, mols2cand, idir=os.path.join(idir, "candidates"), preload_data=False)
#
#             with gzip.open(os.path.join("small_metabolite_data.pkl.gz"), "wb") as gz_file:
#                 pickle.dump({"X": X, "mols": mols, "cand": cand}, gz_file)
#
#         with gzip.open("small_metabolite_data.pkl.gz", "rb") as gz_file:
#             data = pickle.load(gz_file)
#
#         self.ssvm = StructuredSVMMetIdent(C=2)
#         self.ssvm.K_train = data["X"]
#         self.ssvm.y_train = data["mols"]
#         self.cand = data["cand"]  # type: CandidateSetMetIdent
#         self.N = self.ssvm.K_train.shape[0]
#         self.rs = np.random.RandomState(18221)
#         self.ssvm.alphas = DualVariables(C=self.ssvm.C, random_state=self.rs, num_init_active_vars=2,
#                                          label_space=[[self.cand.get_labelspace(self.ssvm.y_train[i])]
#                                                       for i in range(self.N)])
#
#     def test_get_candidate_scores(self):
#         """
#         Test that the scores for the candidates corresponding to a particular spectrum x_i are correctly calculated
#         given an SSVM model (w or alphas). The score is calculated like:
#
#             s(x_i, y) = < w , Psi(x_i, y) >     for all y in Sigma_i
#         """
#         i = 10
#
#         self.ssvm.fps_active, lab_losses_active = self.ssvm._get_active_fingerprints_and_losses(
#             self.ssvm.alphas, self.ssvm.y_train, self.cand, verbose=True)
#         scores = self.ssvm._get_candidate_scores(
#             self.ssvm.K_train[i], self.ssvm.y_train[i], self.cand,
#             {"lab_losses_active": lab_losses_active, "mol_kernel_L_S_Ci": {}, "mol_kernel_L_Ci": {},
#              "fps_active": self.ssvm.fps_active, "B_S": self.ssvm.alphas.get_dual_variable_matrix("dense")},
#             for_training=True)
#
#         fps_gt_i = self.cand.get_gt_fp(self.ssvm.y_train[i])[np.newaxis, :]
#         fps_gt = self.cand.get_gt_fp(self.ssvm.y_train)
#         B = self.ssvm.alphas.get_dual_variable_matrix(type="csr")
#
#         # Part which is constant for all y in Sigma_i
#         s1 = self.ssvm.C / self.N * self.ssvm.K_train[i] @ self.cand.get_kernel(fps_gt_i, fps_gt).flatten()
#         self.assertTrue(np.isscalar(s1))
#
#         s2 = (self.cand.get_kernel(fps_gt_i, self.ssvm.fps_active) @ B.T @ self.ssvm.K_train[i]).item()
#         self.assertTrue(np.isscalar(s2))
#
#         # Part that is specific to each candidate y in Sigma_i
#         L_Ci = self.cand.get_kernel(self.cand.get_candidate_fps(self.ssvm.y_train[i]), fps_gt)
#         s3 = self.ssvm.C / self.N * L_Ci @ self.ssvm.K_train[i]
#         self.assertEqual((len(self.cand.get_labelspace(self.ssvm.y_train[i])),), s3.shape)
#
#         L_Ci_S = self.cand.get_kernel(self.cand.get_candidate_fps(self.ssvm.y_train[i]), self.ssvm.fps_active)
#         s4 = L_Ci_S @ (B.T @ self.ssvm.K_train[i])
#         self.assertEqual((len(self.cand.get_labelspace(self.ssvm.y_train[i])),), s4.shape)
#
#         np.testing.assert_allclose(scores, (s3 - s4))


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
            for num_init_active_vars in range(1, 202, 20):
                alphas = DualVariables(C=C, label_space=cand_ids, num_init_active_vars=num_init_active_vars,
                                       random_state=num_init_active_vars)

                _n_max_possible_vars = [np.minimum(num_init_active_vars, len(list(it.product(*cand_ids[i]))))
                                        for i in range(N)]
                n_active = np.sum(_n_max_possible_vars)
                self.assertEqual(n_active, alphas.n_active())

                B = alphas.get_dual_variable_matrix()
                self.assertEqual((N, n_active), B.shape)
                np.testing.assert_equal(np.array(_n_max_possible_vars),
                                        np.array(np.sum(B > 0, axis=1)).flatten())

                self.assertTrue(StructuredSVMMetIdent._is_feasible_matrix(alphas, C))

                col = 0
                for i in range(N):
                    sampled_y_seqs = set()
                    for k in range(_n_max_possible_vars[i]):
                        idx, y_seq = alphas.get_iy_for_col(col)  # type: Tuple[int, Tuple]
                        self.assertNotIn(y_seq, sampled_y_seqs)
                        sampled_y_seqs.add(y_seq)
                        self.assertEqual(i, idx)
                        self.assertIn(y_seq, list(it.product(*cand_ids[i])))
                        col += 1

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
            for num_init_active_vars in range(1, 5):  # 1, 2, 3, 4
                alphas = DualVariables(C=C, label_space=cand_ids, num_init_active_vars=num_init_active_vars, random_state=10910)

                _n_max_possible_vars = [np.minimum(num_init_active_vars, len(cand_ids[i][0])) for i in range(N)]
                n_active = np.sum(_n_max_possible_vars)
                self.assertEqual(n_active, alphas.n_active())

                B = alphas.get_dual_variable_matrix()
                self.assertEqual((N, n_active), B.shape)

                np.testing.assert_equal(np.array(_n_max_possible_vars),
                                        np.array(np.sum(B > 0, axis=1)).flatten())

                self.assertTrue(StructuredSVMMetIdent._is_feasible_matrix(alphas, C))

                col = 0
                for i in range(N):
                    sampled_y_seqs = set()
                    for k in range(_n_max_possible_vars[i]):
                        idx, y_seq = alphas.get_iy_for_col(col)  # type: Tuple[int, Tuple]
                        self.assertNotIn(y_seq, sampled_y_seqs)
                        sampled_y_seqs.add(y_seq)
                        self.assertEqual(i, idx)
                        self.assertIn(y_seq, list(it.product(*cand_ids[i])))
                        col += 1

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

        alphas = DualVariables(C=C, label_space=cand_ids, num_init_active_vars=2, random_state=10910)
        # print(alphas._iy)
        # Active variables
        # [(0, ('M2',)),  (0, ('M1',)),
        #  (1, ('M8',)),  (1, ('M9',)),
        #  (2, ('M72',)), (2, ('M11',)),
        #  (3, ('M13',)), (3, ('M4',))]

        # ---- Update an active dual variable ----
        B_old = alphas.get_dual_variable_matrix().todense()
        val_old_M8 = alphas.get_dual_variable(1, ("M8",))
        val_old_M7 = alphas.get_dual_variable(1, ("M7",))
        val_old_M9 = alphas.get_dual_variable(1, ("M9",))

        self.assertEqual(C / (N * 2), val_old_M8)
        self.assertEqual(0,           val_old_M7)
        self.assertEqual(C / (N * 2), val_old_M9)

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

        self.assertEqual(0,           val_old_M12)
        self.assertEqual(C / (N * 2), val_old_M13)
        self.assertEqual(0,           val_old_M3)
        self.assertEqual(C / (N * 2), val_old_M4)
        self.assertEqual(0,           val_old_M22)

        self.assertTrue(alphas.update(3, ("M3",), gamma))
        self.assertEqual(9, alphas.n_active())
        self.assertEqual((1 - gamma) * val_old_M12 + gamma * 0, alphas.get_dual_variable(3, ("M12",)))
        self.assertEqual((1 - gamma) * val_old_M13 + gamma * 0, alphas.get_dual_variable(3, ("M13",)))
        self.assertEqual((1 - gamma) * val_old_M3 + gamma * C / N, alphas.get_dual_variable(3, ("M3",)))
        self.assertEqual((1 - gamma) * val_old_M4 + gamma * 0, alphas.get_dual_variable(3, ("M4",)))
        self.assertEqual((1 - gamma) * val_old_M22 + gamma * 0, alphas.get_dual_variable(3, ("M22",)))

        self.assertEqual((3, ("M3",)), alphas.get_iy_for_col(8))

    def test_eq_dual_domain(self):
        cand_ids_1a = [
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
        cand_ids_1b = [
            [
                ["M1", "M2"],
                ["M1", "M2", "M19", "M10"],
                ["M20", "M4", "M5"],
                ["M2"],
                ["M3", "M56", "M8"]
            ],
            [
                ["M1"],
                ["M1", "M3", "M2", "M10", "M20"],
                ["M4", "M5"],
                ["M2", "M4"],
                ["M7", "M9", "M8"]
            ],
            [
                ["M1", "M2", "M3"],
                ["M1", "M2", "M9", "M10", "M22"],
                ["M20", "M7"],
                ["M2", "M99"],
                ["M8", "M72"]
            ]
        ]
        cand_ids_2a = [
            [
                ["M1", "M2"],
                ["M1", "M2", "M20", "M10"],
                ["M20", "M4", "M5"],
                ["M2"],
                ["M3", "M56", "M8"]
            ],
            [
                ["M1"],
                ["M1", "M3", "M2", "M10", "M20"],
                ["M4", "M5"],
                ["M2", "M3"],
                ["M7", "M9", "M8"]
            ],
            [
                ["M1", "M2", "M3"],
                ["M1", "M2", "M9", "M10", "M22"],
                ["M20", "M7"],
                ["M2", "M99"],
                ["M8", "M72"]
            ]
        ]
        cand_ids_2b = [
            [
                ["M1", "M2"],
                ["M1", "M2", "M19", "M10"],
                ["M2"],
                ["M3", "M56", "M8"]
            ],
            [
                ["M1"],
                ["M1", "M3", "M2", "M10", "M20"],
                ["M4", "M5"],
                ["M2", "M4"],
                ["M7", "M9", "M8"]
            ],
            [
                ["M1", "M2", "M3"],
                ["M1", "M2", "M9", "M10", "M22"],
                ["M20", "M7"],
                ["M2", "M99"],
                ["M8", "M72"]
            ]
        ]

        alpha_1a = DualVariables(C=2, label_space=cand_ids_1a, initialize=False, random_state=120)
        alpha_1b = DualVariables(C=2, label_space=cand_ids_1b, initialize=False, random_state=120)
        alpha_2a = DualVariables(C=2, label_space=cand_ids_2a, initialize=False, random_state=120)
        alpha_2b = DualVariables(C=2, label_space=cand_ids_2b, initialize=False, random_state=120)
        self.assertTrue(DualVariables._eq_dual_domain(alpha_1a, alpha_1a))
        self.assertTrue(DualVariables._eq_dual_domain(alpha_1a, alpha_1b))
        self.assertFalse(DualVariables._eq_dual_domain(alpha_1a, alpha_2a))
        self.assertFalse(DualVariables._eq_dual_domain(alpha_1a, alpha_2b))
        self.assertFalse(DualVariables._eq_dual_domain(alpha_2a, alpha_2b))

    def test_deepcopy(self):
        from copy import deepcopy

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
        alpha = DualVariables(C=2, label_space=cand_ids, random_state=120)
        alpha_cp = deepcopy(alpha)
        self.assertEqual(alpha.get_dual_variable_matrix().shape,
                         alpha_cp.get_dual_variable_matrix().shape)

        alpha.update(0, ("M1", "M2", "M20", "M2", "M3"), gamma=0.25)
        self.assertEqual(4, alpha.n_active())
        self.assertEqual(3, alpha_cp.n_active())
        self.assertNotEqual(alpha.get_dual_variable_matrix().shape,
                            alpha_cp.get_dual_variable_matrix().shape)

    def test_multiplication(self):
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
        C = 1.5
        N = len(cand_ids)

        # factor != 0
        for i, fac in enumerate([-1, -0.78, 0.64, 1, 2]):
            alphas = DualVariables(C=C, label_space=cand_ids, random_state=i)

            fac_mul_alphas = fac * alphas
            alphas_fac_mul = alphas * fac

            self.assertIsInstance(fac_mul_alphas, DualVariables)
            self.assertIsInstance(alphas_fac_mul, DualVariables)

            self.assertTrue(DualVariables._eq_dual_domain(alphas, alphas_fac_mul))
            self.assertTrue(DualVariables._eq_dual_domain(alphas, fac_mul_alphas))

            for i, y_seq in alphas._iy:
                self.assertEqual((C / N), alphas.get_dual_variable(i, y_seq))
                self.assertEqual((C / N) * fac, fac_mul_alphas.get_dual_variable(i, y_seq))
                self.assertEqual((C / N) * fac, alphas_fac_mul.get_dual_variable(i, y_seq))

        # factor == 0
        alphas = DualVariables(C=C, label_space=cand_ids, random_state=1)

        fac_mul_alphas = 0 * alphas
        alphas_fac_mul = alphas * 0

        self.assertIsInstance(fac_mul_alphas, DualVariables)
        self.assertIsInstance(alphas_fac_mul, DualVariables)

        self.assertEqual([], fac_mul_alphas._iy)
        self.assertEqual((alphas.N, 0), fac_mul_alphas.get_dual_variable_matrix().shape)
        self.assertEqual([{} for _ in range(N)], fac_mul_alphas._y2col)

        self.assertEqual([], alphas_fac_mul._iy)
        self.assertEqual((alphas.N, 0), alphas_fac_mul.get_dual_variable_matrix().shape)
        self.assertEqual([{} for _ in range(N)], alphas_fac_mul._y2col)

    def test_addition(self):
        # TODO: Implement
        self.skipTest("Not implemented yet.")

    def test_subtraction(self):
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

        alphas = DualVariables(C=1.5, label_space=cand_ids, random_state=1)

        # Test subtracting a dual variable class from itself ==> empty dual variable set
        sub_alphas = alphas - alphas
        self.assertTrue(DualVariables._eq_dual_domain(alphas, sub_alphas))
        self.assertEqual([], sub_alphas._iy)
        self.assertEqual((alphas.N, 0), sub_alphas.get_dual_variable_matrix().shape)
        self.assertEqual([{} for _ in range(alphas.N)], sub_alphas._y2col)

        # Test subtracting an empty dual variable set ==> no changes
        sub_alphas = alphas - (alphas - alphas)
        self.assertTrue(DualVariables._eq_dual_domain(alphas, sub_alphas))
        self.assertEqual(len(alphas._iy), len(sub_alphas._iy))
        for i, y_seq in alphas._iy:
            self.assertIn((i, y_seq), sub_alphas._iy)
            self.assertEqual(alphas.get_dual_variable(i, y_seq), sub_alphas.get_dual_variable(i, y_seq))

        # Test subtracting two non-empty dual variable sets
        alphas_2 = DualVariables(C=1.5, label_space=cand_ids, random_state=2)
        sub_alphas = alphas - alphas_2
        print(alphas._iy)
        # [(0, ('M10',)), (1, ('M7',)), (2, ('M72',)), (3, ('M3',))]
        print(alphas_2._iy)
        # [(0, ('M19',)), (1, ('M7',)), (2, ('M11',)), (3, ('M22',))]
        self.assertTrue(DualVariables._eq_dual_domain(alphas, sub_alphas))
        self.assertTrue(DualVariables._eq_dual_domain(alphas_2, sub_alphas))
        self.assertEqual(6, sub_alphas.n_active())
        for (i, y_seq), a in [((0, ("M10", )), alphas.C / alphas.N), ((0, ("M19", )), - alphas.C / alphas.N),
                              ((2, ("M72", )), alphas.C / alphas.N), ((2, ("M11", )), - alphas.C / alphas.N),
                              ((3, ("M3", )),  alphas.C / alphas.N), ((3, ("M22", )), - alphas.C / alphas.N)]:
            self.assertIn((i, y_seq), sub_alphas._iy)
            self.assertEqual(a, sub_alphas.get_dual_variable(i, y_seq))


if __name__ == '__main__':
    unittest.main()
