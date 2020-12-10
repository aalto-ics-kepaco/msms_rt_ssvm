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
import sqlite3
import numpy as np
import unittest
import time
import itertools as it
import networkx as nx
import pandas as pd

from matchms.Spectrum import Spectrum
from typing import Tuple
from scipy.sparse import csr_matrix
from copy import deepcopy

from ssvm.ssvm import _StructuredSVM, StructuredSVMMetIdent, DualVariables, StructuredSVMSequencesFixedMS2
from ssvm.data_structures import CandidateSQLiteDB, SequenceSample, RandomSubsetCandidateSQLiteDB, SpanningTrees

DB_FN = "/home/bach/Documents/doctoral/projects/local_casmi_db/db/use_inchis/DB_LATEST.db"


class Test_StructuredSVM(unittest.TestCase):
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


class TestStructuredSVMSequencesFixedMS2(unittest.TestCase):
    def setUp(self) -> None:
        # ===================
        # Get list of Spectra
        # ===================
        self.db = sqlite3.connect("file:" + DB_FN + "?mode=ro", uri=True)

        # Read in spectra and labels
        res = pd.read_sql_query("SELECT spectrum, molecule, rt, challenge FROM challenges_spectra "
                                "   INNER JOIN spectra s ON s.name = challenges_spectra.spectrum", con=self.db)
        self.spectra = [Spectrum(np.array([]), np.array([]),
                                 {"spectrum_id": spec_id, "retention_time": rt, "dataset": chlg, "molecule_id": mol})
                        for (spec_id, rt, chlg, mol) in zip(res["spectrum"], res["rt"], res["challenge"],
                                                            res["molecule"])]
        self.labels = res["molecule"].to_list()

        self.db.close()

        # ===================
        # Setup a SSVM
        # ===================
        self.ssvm = StructuredSVMSequencesFixedMS2(
            mol_feat_label_loss="iokr_fps__positive", mol_feat_retention_order="substructure_count",
            mol_kernel="minmax", C=2)

        self.N = 50
        self.ssvm.training_data_ = SequenceSample(
            self.spectra, self.labels,
            RandomSubsetCandidateSQLiteDB(db_fn=DB_FN, molecule_identifier="inchi", random_state=192,
                                          number_of_candidates=50, include_correct_candidate=True),
            N=self.N, L_min=10,
            L_max=15, random_state=19, ms2scorer="MetFrag_2.4.5__8afe4a14")
        self.ssvm.alphas_ = DualVariables(
            self.ssvm.C, label_space=self.ssvm.training_data_.get_labelspace(), num_init_active_vars=3)
        self.ssvm.training_graphs_ = [SpanningTrees(sequence, random_state=i)
                                      for i, sequence in enumerate(self.ssvm.training_data_)]

    def test_get_lambda_delta(self):
        def mol_kernel(x, y):
            return x @ y.T

        rt_loop = 0.0
        rt_vec = 0.0

        n_rep = 15
        for rep in range(n_rep):
            n_features = 231
            n_molecules = 501
            L = 101
            G = nx.generators.trees.random_tree(L, seed=(rep + 3))

            Y_sequence = np.random.RandomState(rep + 5).rand(L, n_features)
            Y_candidates = np.random.RandomState(rep + 6).rand(n_molecules, n_features)

            start = time.time()
            lambda_delta_ref = np.empty((L - 1, n_molecules))
            for idx, (s, t) in enumerate(G.edges):
                lambda_delta_ref[idx] = mol_kernel(Y_sequence[s], Y_candidates) - mol_kernel(Y_sequence[t], Y_candidates)
            rt_loop += time.time() - start

            start = time.time()
            lambda_delta = StructuredSVMSequencesFixedMS2._get_lambda_delta(Y_sequence, Y_candidates, G, mol_kernel)
            rt_vec += time.time() - start

            self.assertEqual(lambda_delta_ref.shape, lambda_delta.shape)
            np.testing.assert_almost_equal(lambda_delta_ref, lambda_delta)

        print("== get_lambda_delta ==")
        print("Loop: %.3fs" % (rt_loop / n_rep))
        print("Vec: %.3fs" % (rt_vec / n_rep))

    def test_I_rsvm_jfeat(self):
        N_E = np.sum([len(self.ssvm.training_graphs_[j][0].edges) for j in range(self.N)])  # total number of edges

        rt_loop = 0.0
        rt_vec = 0.0

        for i in range(5):  # inspect only the first 5 label sequences
            Y_candidates = self.ssvm.training_data_[i].get_molecule_features_for_candidates("substructure_count", 2)

            start = time.time()
            I_ref = np.zeros((len(Y_candidates), ))
            for j in range(self.N):
                sign_delta_j = self.ssvm.training_data_[j].get_sign_delta_t(self.ssvm.training_graphs_[j][0])
                lambda_delta_j = self.ssvm._get_lambda_delta(
                    Y_sequence=self.ssvm.training_data_[j].get_molecule_features_for_labels(
                        self.ssvm.mol_feat_retention_order),
                    Y_candidates=Y_candidates,
                    G=self.ssvm.training_graphs_[j][0],
                    mol_kernel=self.ssvm.mol_kernel
                )
                I_ref += self.ssvm.C / len(self.ssvm.training_data_) * (sign_delta_j @ lambda_delta_j)

                # self.assertEqual((len(ssvm.training_graphs_[j][0].edges),), sign_delta_j.shape)
                # self.assertEqual((len(ssvm.training_graphs_[j][0].edges), len(Y_candidates)), lambda_delta_j.shape)

            rt_loop += time.time() - start

            start = time.time()
            I = self.ssvm._I_jfeat_rsvm(Y_candidates)
            rt_vec += time.time() - start

            self.assertEqual((len(Y_candidates), ), I.shape)
            np.testing.assert_almost_equal(I_ref, I)
            self.assertTrue(np.all((I / self.ssvm.C * self.N) >= -N_E))
            self.assertTrue(np.all((I / self.ssvm.C * self.N) <= N_E))

        print("== I_rsvm_jfeat ==")
        print("Loop: %.3fs" % (rt_loop / 5))
        print("Vec: %.3fs" % (rt_vec / 5))

    def test_predict_molecule_preference_values(self):
        for i in range(5):  # inspect only the first 5 label sequences
            Y_candidates = self.ssvm.training_data_[i].get_molecule_features_for_candidates("substructure_count", 2)
            pref = self.ssvm.predict_molecule_preference_values(Y_candidates)
            self.assertEqual((len(Y_candidates), ), pref.shape)

    def test_get_node_and_edge_potentials(self):
        i = 8
        npot, epot = self.ssvm._get_node_and_edge_potentials(self.ssvm.training_data_[i],
                                                             self.ssvm.training_graphs_[i][0])

        self.assertEqual(len(self.ssvm.training_graphs_[i][0]), len(npot))

        for s, t in self.ssvm.training_graphs_[i][0].edges:
            self.assertIn(s, epot)
            self.assertIn(t, epot[s])

            # Reverse direction
            self.assertIn(t, epot)
            self.assertIn(s, epot[t])

            # Check shape of transition matrices
            n_cand_s = len(self.ssvm.training_data_[i].get_molecule_features_for_candidates(
                self.ssvm.mol_feat_retention_order, s))
            n_cand_t = len(self.ssvm.training_data_[i].get_molecule_features_for_candidates(
                self.ssvm.mol_feat_retention_order, t))
            self.assertEqual((n_cand_s, n_cand_t), epot[s][t]["log_score"].shape)
            self.assertEqual((n_cand_t, n_cand_s), epot[t][s]["log_score"].shape)

    def test_inference(self):
        for i in [0, 8, 3]:
            # =========================
            # With loss augmentation
            # =========================
            y_i_hat__la = self.ssvm.inference(
                self.ssvm.training_data_[i], self.ssvm.training_graphs_[i],
                loss_augmented=True)

            # =========================
            # Without loss augmentation
            # =========================
            y_i_hat__wola = self.ssvm.inference(
                self.ssvm.training_data_[i], self.ssvm.training_graphs_[i],
                loss_augmented=False)

            self.assertEqual(len(self.ssvm.training_data_[i]), len(y_i_hat__la))
            self.assertEqual(len(self.ssvm.training_data_[i]), len(y_i_hat__wola))

            for s in range(len(self.ssvm.training_data_[i])):
                self.assertIn(y_i_hat__la[s], self.ssvm.training_data_[i].get_labelspace(s))
                self.assertIn(y_i_hat__wola[s], self.ssvm.training_data_[i].get_labelspace(s))

    def test_inference_without_tree_given(self):
        for i in [0, 8, 3]:
            # =========================
            # With loss augmentation
            # =========================
            y_i_hat__la = self.ssvm.inference(self.ssvm.training_data_[i], Gs=None, loss_augmented=True)

            # =========================
            # Without loss augmentation
            # =========================
            y_i_hat__wola = self.ssvm.inference(self.ssvm.training_data_[i], Gs=None, loss_augmented=False)

            self.assertEqual(len(self.ssvm.training_data_[i]), len(y_i_hat__la))
            self.assertEqual(len(self.ssvm.training_data_[i]), len(y_i_hat__wola))

            for s in range(len(self.ssvm.training_data_[i])):
                self.assertIn(y_i_hat__la[s], self.ssvm.training_data_[i].get_labelspace(s))
                self.assertIn(y_i_hat__wola[s], self.ssvm.training_data_[i].get_labelspace(s))

    def test_max_marginals(self):
        for i in [0, 8, 3]:
            marg = self.ssvm.max_marginals(self.ssvm.training_data_[i], Gs=None)

            self.assertEqual(len(self.ssvm.training_data_[i]), len(marg))

            for s in range(len(self.ssvm.training_data_[i])):
                self.assertEqual(self.ssvm.training_data_[i].get_labelspace(s), marg[s]["label"])
                self.assertEqual(len(self.ssvm.training_data_[i].get_labelspace(s)), len(marg[s]["score"]))

    # ------------------------------------------------------------
    # FOR THE SCORING WE CURRENTLY ONLY TEST THE OUTPUT DIMENSIONS
    # ------------------------------------------------------------
    def test_topk_score(self):
        topk_acc = self.ssvm.topk_score(self.ssvm.training_data_[3], Gs=None, max_k=100)
        self.assertTrue(len(topk_acc) < 100)

        topk_acc = self.ssvm.topk_score(self.ssvm.training_data_[3], Gs=None, max_k=100, pad_output=True)
        self.assertEqual(100, len(topk_acc))

    def test_top1_score(self):
        top1_acc = self.ssvm.top1_score(self.ssvm.training_data_[3], Gs=None)
        self.assertTrue(np.isscalar(top1_acc))

        top1_acc = self.ssvm.top1_score(self.ssvm.training_data_[3], Gs=None, map=True)
        self.assertTrue(np.isscalar(top1_acc))

    def test_ndcg_score(self):
        ndcg_ll = self.ssvm.ndcg_score(self.ssvm.training_data_[2], use_label_loss=True)
        self.assertTrue(np.isscalar(ndcg_ll))

        ndcg_ohc = self.ssvm.ndcg_score(self.ssvm.training_data_[2], use_label_loss=False)
        self.assertTrue(np.isscalar(ndcg_ohc))

    def test_score(self):
        # Top-1 (averaged)
        score = self.ssvm.score(
            [self.ssvm.training_data_[0], self.ssvm.training_data_[1], self.ssvm.training_data_[3]],
            stype="top1_mm")
        self.assertTrue(np.isscalar(score))

        # Top-1
        score = self.ssvm.score(
            [self.ssvm.training_data_[0], self.ssvm.training_data_[1], self.ssvm.training_data_[3]],
            stype="top1_mm", average=False)
        self.assertEqual((3, ), score.shape)

        # Top-k (averaged)
        score = self.ssvm.score(
            [self.ssvm.training_data_[0], self.ssvm.training_data_[1], self.ssvm.training_data_[3]],
            stype="topk_mm", max_k=100)
        self.assertEqual((100, ), score.shape)

        # Top-k
        score = self.ssvm.score(
            [self.ssvm.training_data_[0], self.ssvm.training_data_[1], self.ssvm.training_data_[3]],
            stype="topk_mm", average=False, max_k=100)
        self.assertEqual((3, 100), score.shape)


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
                alphas = DualVariables(C=C, label_space=cand_ids, num_init_active_vars=num_init_active_vars,
                                       random_state=10910)

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
        print(alphas._iy)
        # Active variables
        # [(0, ('M19',)), (0, ('M10',)),
        #  (1, ('M8',)), (1, ('M7',)),
        #  (2, ('M72',)), (2, ('M11',)),
        #  (3, ('M4',)), (3, ('M3',))]

        # ---- Update an active dual variable ----
        B_old = alphas.get_dual_variable_matrix().todense()
        val_old_M8 = alphas.get_dual_variable(1, ("M8",))
        val_old_M7 = alphas.get_dual_variable(1, ("M7",))
        val_old_M9 = alphas.get_dual_variable(1, ("M9",))

        self.assertEqual(C / (N * 2), val_old_M8)
        self.assertEqual(0,           val_old_M9)
        self.assertEqual(C / (N * 2), val_old_M7)

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
        self.assertEqual(0,           val_old_M13)
        self.assertEqual(C / (N * 2), val_old_M3)
        self.assertEqual(C / (N * 2), val_old_M4)
        self.assertEqual(0,           val_old_M22)

        self.assertTrue(alphas.update(3, ("M12",), gamma))
        self.assertEqual(9, alphas.n_active())
        self.assertEqual((1 - gamma) * val_old_M12 + gamma * C / N, alphas.get_dual_variable(3, ("M12",)))
        self.assertEqual((1 - gamma) * val_old_M13 + gamma * 0, alphas.get_dual_variable(3, ("M13",)))
        self.assertEqual((1 - gamma) * val_old_M3 + gamma * 0, alphas.get_dual_variable(3, ("M3",)))
        self.assertEqual((1 - gamma) * val_old_M4 + gamma * 0, alphas.get_dual_variable(3, ("M4",)))
        self.assertEqual((1 - gamma) * val_old_M22 + gamma * 0, alphas.get_dual_variable(3, ("M22",)))

        self.assertEqual((3, ("M12",)), alphas.get_iy_for_col(8))

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

        self.assertTrue(alpha.update(0, ("M1", "M2", "M20", "M2", "M3"), gamma=0.25))
        self.assertEqual(4, alpha.n_active())
        self.assertEqual(3, alpha_cp.n_active())
        self.assertNotEqual(alpha.get_dual_variable_matrix().shape,
                            alpha_cp.get_dual_variable_matrix().shape)

    def test_iter(self):
        cand_ids = [
            [
                ["MA1", "MA2"],
                ["MA1", "MA2", "MA19", "MA10"],
                ["MA20", "MA4", "MA5"],
                ["MA2"],
                ["MA3", "MA56", "MA8"]
            ],
            [
                ["MB1"],
                ["MB1", "MB2", "MB3", "MB10", "MB20"],
                ["MB4", "MB5"],
                ["MB2", "MB4"],
                ["MB7", "MB9", "MB8"]
            ],
            [
                ["MC1", "MC2", "MC3"],
                ["MC1", "MC2", "MC9", "MC10", "MC22"],
                ["MC20", "MC7"],
                ["MC2", "MC99"],
                ["MC72", "MC8"]
            ]
        ]

        for rep in range(10):
            alpha = DualVariables(C=2, label_space=cand_ids, random_state=rep, num_init_active_vars=5)

            for i, (pref, _) in enumerate(zip(["A", "B", "C"], cand_ids)):
                for y, a in alpha.iter(i):
                    for ys in y:
                        self.assertTrue(ys.startswith("M" + pref))

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
        # print(alphas._iy)
        # [(0, ('M2',)), (1, ('M9',)), (2, ('M11',)), (3, ('M4',))]
        # print(alphas_2._iy)
        # [(0, ('M1',)), (1, ('M7',)), (2, ('M72',)), (3, ('M12',))]
        self.assertTrue(DualVariables._eq_dual_domain(alphas, sub_alphas))
        self.assertTrue(DualVariables._eq_dual_domain(alphas_2, sub_alphas))
        self.assertEqual(8, sub_alphas.n_active())
        for (i, y_seq), a in [((0, ("M2", )), alphas.C / alphas.N), ((0, ("M1", )), - alphas.C / alphas.N),
                              ((1, ("M9",)), alphas.C / alphas.N), ((1, ("M7",)), - alphas.C / alphas.N),
                              ((2, ("M11", )), alphas.C / alphas.N), ((2, ("M72", )), - alphas.C / alphas.N),
                              ((3, ("M4", )),  alphas.C / alphas.N), ((3, ("M12", )), - alphas.C / alphas.N)]:
            self.assertIn((i, y_seq), sub_alphas._iy)
            self.assertEqual(a, sub_alphas.get_dual_variable(i, y_seq))

        # Test subtracting two non-empty dual variable sets
        alphas_2 = DualVariables(C=1.5, label_space=cand_ids, random_state=5)
        sub_alphas = alphas - alphas_2
        # print(alphas._iy)
        # [(0, ('M2',)), (1, ('M9',)), (2, ('M11',)), (3, ('M4',))]
        # print(alphas_2._iy)
        # [(0, ('M10',)), (1, ('M8',)), (2, ('M11',)), (3, ('M4',))]
        self.assertTrue(DualVariables._eq_dual_domain(alphas, sub_alphas))
        self.assertTrue(DualVariables._eq_dual_domain(alphas_2, sub_alphas))
        self.assertEqual(4, sub_alphas.n_active())
        for (i, y_seq), a in [((0, ("M2", )), alphas.C / alphas.N), ((0, ("M10", )), - alphas.C / alphas.N),
                              ((1, ("M9",)), alphas.C / alphas.N), ((1, ("M8",)), - alphas.C / alphas.N),
                              ((2, ("M11", )), 0), ((2, ("M11", )), 0),
                              ((3, ("M4", )),  0), ((3, ("M4", )), 0)]:
            if a != 0:
                self.assertIn((i, y_seq), sub_alphas._iy)
            else:
                self.assertNotIn((i, y_seq), sub_alphas._iy)
            self.assertEqual(a, sub_alphas.get_dual_variable(i, y_seq))


if __name__ == '__main__':
    unittest.main()
