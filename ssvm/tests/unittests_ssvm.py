####
#
# The MIT License (MIT)
#
# Copyright 2020, 2021 Eric Bach <eric.bach@aalto.fi>
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
import networkx as nx
import pandas as pd

from matchms.Spectrum import Spectrum
from scipy.sparse import csr_matrix

from ssvm.ssvm import _StructuredSVM, StructuredSVMSequencesFixedMS2
from ssvm.data_structures import SequenceSample, RandomSubsetCandSQLiteDB_Bach2020, SpanningTrees
from ssvm.kernel_utils import generalized_tanimoto_kernel_FAST
from ssvm.dual_variables import DualVariables

DB_FN = "Bach2020_test_db.sqlite"


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
            RandomSubsetCandSQLiteDB_Bach2020(db_fn=DB_FN, molecule_identifier="inchi", random_state=192,
                                              number_of_candidates=50, include_correct_candidate=True),
            N=self.N, L_min=10,
            L_max=15, random_state=19, ms2scorer="MetFrag_2.4.5__8afe4a14")
        self.ssvm.alphas_ = DualVariables(
            self.ssvm.C, label_space=self.ssvm.training_data_.get_labelspace(), num_init_active_vars=3)
        self.ssvm.training_graphs_ = [SpanningTrees(sequence, random_state=i)
                                      for i, sequence in enumerate(self.ssvm.training_data_)]

    def test_get_lambda_delta(self):
        rt_loop = 0.0
        rt_vec = 0.0

        n_rep = 15
        for rep in range(n_rep):
            n_features = 301
            n_molecules = 501
            L = 200
            G = nx.generators.trees.random_tree(L, seed=(rep + 3))

            Y_sequence = np.random.RandomState(rep + 5).randn(L, n_features)
            Y_candidates = np.random.RandomState(rep + 6).randn(n_molecules, n_features)

            start = time.time()
            lambda_delta_ref = np.empty((L - 1, n_molecules))
            for idx, (s, t) in enumerate(G.edges):
                lambda_delta_ref[idx] = generalized_tanimoto_kernel_FAST(Y_sequence[s][np.newaxis, :], Y_candidates) \
                                        - generalized_tanimoto_kernel_FAST(Y_sequence[t][np.newaxis, :], Y_candidates)
            rt_loop += time.time() - start

            start = time.time()
            lambda_delta = StructuredSVMSequencesFixedMS2._get_lambda_delta_OLD(
                Y_sequence, Y_candidates, G, generalized_tanimoto_kernel_FAST)
            rt_vec += time.time() - start

            self.assertEqual(lambda_delta_ref.shape, lambda_delta.shape)
            np.testing.assert_almost_equal(lambda_delta_ref, lambda_delta)

        print("== get_lambda_delta ==")
        print("Loop: %.3fs" % (rt_loop / n_rep))
        print("Vec: %.3fs" % (rt_vec / n_rep))

    def test_get_lambda_delta_NEW(self):
        rt_old = 0.0
        rt_new = 0.0

        n_rep = 15
        for rep in range(n_rep):
            n_features = 301
            n_molecules = 501
            L = 200
            G = nx.generators.trees.random_tree(L, seed=(rep + 3))

            Y_candidates = np.random.RandomState(rep + 6).randn(n_molecules, n_features)

            l_Y_sequence = []
            for nS in [1, 2, 3, 4, 5]:
                l_Y_sequence.append(np.random.RandomState((nS * rep) + 5).randn(L, n_features))

                start = time.time()
                lambda_delta_ref = np.dstack(
                    [
                        StructuredSVMSequencesFixedMS2._get_lambda_delta_OLD(
                            Y_sequence, Y_candidates, G, generalized_tanimoto_kernel_FAST)
                        for Y_sequence in l_Y_sequence
                    ])
                rt_old += time.time() - start
                self.assertEqual((L - 1, len(Y_candidates), nS), lambda_delta_ref.shape)

                start = time.time()
                lambda_delta = StructuredSVMSequencesFixedMS2._get_lambda_delta(
                    np.vstack(l_Y_sequence), Y_candidates, G, generalized_tanimoto_kernel_FAST)
                self.assertEqual(((L - 1) * nS, len(Y_candidates)), lambda_delta.shape)

                # Bring output to shape = (|E_j|, |S_j|, n_molecules)
                lambda_delta = lambda_delta.reshape(
                    (
                        nS,
                        L - 1,
                        len(Y_candidates),
                    ))
                self.assertEqual((nS, L - 1, len(Y_candidates)), lambda_delta.shape)
                rt_new += time.time() - start

                for nSi in range(nS):
                    np.testing.assert_allclose(lambda_delta_ref[:, :, nSi], lambda_delta[nSi, :, :])

        print("== get_lambda_delta ==")
        print("Old: %.3fs" % (rt_old / n_rep))
        print("New: %.3fs" % (rt_new / n_rep))

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

    def test_I_rsvm_jfeat__FOR_LARGE_MEMORY(self):
        N_E = np.sum([len(self.ssvm.training_graphs_[j][0].edges) for j in range(self.N)])  # total number of edges

        for i in range(10):  # inspect only the first 10 label sequences
            Y_candidates = self.ssvm.training_data_[i].get_molecule_features_for_candidates("substructure_count", 2)

            I = self.ssvm._I_jfeat_rsvm__FOR_LARGE_MEMORY(Y_candidates)
            I_ref = self.ssvm._I_jfeat_rsvm(Y_candidates)

            self.assertEqual((len(Y_candidates), ), I.shape)
            np.testing.assert_allclose(I_ref, I)
            self.assertTrue(np.all((I / self.ssvm.C * self.N) >= -N_E))
            self.assertTrue(np.all((I / self.ssvm.C * self.N) <= N_E))

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


if __name__ == '__main__':
    unittest.main()
