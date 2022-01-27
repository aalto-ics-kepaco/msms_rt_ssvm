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
import os

from matchms.Spectrum import Spectrum
from scipy.sparse import csr_matrix

from ssvm.ssvm import _StructuredSVM, StructuredSVMSequencesFixedMS2
from ssvm.data_structures import SequenceSample, RandomSubsetCandSQLiteDB_Bach2020, SpanningTrees
from ssvm.kernel_utils import generalized_tanimoto_kernel_FAST
from ssvm.dual_variables import DualVariables

DB_FN = os.path.join(os.path.dirname(__file__), "Bach2020_test_db.sqlite")


class TestStructuredSVM(unittest.TestCase):
    class DummyDualVariables(object):
        def __init__(self, B):
            self.B = B

        def get_dual_variable_matrix(self):
            return self.B

    def test_is_feasible_matrix(self):
        C = 2.2
        N = 4

        # ---- Feasible dual variables ----
        alphas = self.DummyDualVariables(
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
        alphas = self.DummyDualVariables(
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
    def get_lambda_delta_ref(self, Y_sequence, Y_candidates, G, mol_kernel):
        lambda_delta = np.zeros((len(G.edges), len(Y_candidates)))

        # Sum over the edges (s, t) in E_i
        for idx, (s, t) in enumerate(G.edges):
            # Feature vectors associated with labels of the edge (s, t)
            y_s = Y_sequence[s][np.newaxis, :]
            y_t = Y_sequence[t][np.newaxis, :]
            # Kernel value difference
            lambda_delta[idx] = mol_kernel(y_s, Y_candidates) - mol_kernel(y_t, Y_candidates)

        return lambda_delta

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
            mol_kernel="minmax", C=2
        )

        self.N = 25
        self.ssvm.training_data_ = SequenceSample(
            self.spectra, self.labels,
            candidates=RandomSubsetCandSQLiteDB_Bach2020(
                db_fn=DB_FN, molecule_identifier="inchi", random_state=192, number_of_candidates=15,
                include_correct_candidate=True, init_with_open_db_conn=False
            ),
            N=self.N, L_min=4, L_max=32, random_state=19, ms_scorer="MetFrag_2.4.5__8afe4a14")

        # Initialize the SSVM alpha values
        with self.ssvm.training_data_.candidates:
            self.ssvm.training_label_space_ = self.ssvm.training_data_.get_labelspace()
            self.ssvm.alphas_ = DualVariables(
                self.ssvm.C, label_space=self.ssvm.training_label_space_, num_init_active_vars=3,
                random_state=90210
            )
        # Generate the training graphs
        self.ssvm.training_graphs_ = [
            SpanningTrees(sequence, random_state=i) for i, sequence in enumerate(self.ssvm.training_data_)
        ]

    def test_linesearch(self):
        self.skipTest("Needs to be implemented")

        n_jobs_old = self.ssvm.n_jobs

        n_iter = 25
        t_ref = 0.0
        t_opt = 0.0

        for k in range(n_iter):
            I_batch = np.random.RandomState(k).choice(range(self.N), size=self.N, replace=False)
            y_I_hat = []
            TFG_I = []
            for i in I_batch:
                # Do the inference step
                y_hat, TFG = self.ssvm.inference(
                    self.ssvm.training_data_[i], self.ssvm.training_graphs_[i], loss_augmented=True,
                    return_graphical_model=True, update_direction=self.ssvm.update_direction
                )
                y_I_hat.append(y_hat)
                TFG_I.append(TFG)

            # -------------------------------
            # Calculate line-search step-size
            # -------------------------------

            # Using reference implementation
            start = time.time()
            gamma = self.ssvm._get_step_size_linesearch(I_batch, y_I_hat, TFG_I)
            print(gamma)
            t_ref += (time.time() - start)

            # Update the dual variables
            for idx, i in enumerate(I_batch):
                self.ssvm.alphas_.update(i, y_I_hat[idx], gamma)

        print("== _get_step_size_linesearch ==")
        print("Reference: %.3fs" % (t_ref / n_iter))
        print("Parallel: %.3fs" % (t_opt / n_iter))

        self.ssvm.n_jobs = n_jobs_old

    def test_get_sign_delta(self):
        sign_delta = self.ssvm._get_sign_delta(tree_index=0)

        self.assertEqual(
            (sum(len(Gs_i[0].edges) for Gs_i in self.ssvm.training_graphs_), ),
            sign_delta.shape
        )

        idx = 0
        for i in range(self.N):
            nE_i = len(self.ssvm.training_graphs_[i][0].edges)

            sign_delta_i = sign_delta[idx:(idx + nE_i)]

            self.assertEqual((nE_i,), sign_delta_i.shape)
            np.testing.assert_equal(
                self.ssvm.training_data_[i].get_sign_delta_t(self.ssvm.training_graphs_[i][0]) / nE_i,
                sign_delta_i
            )

            # Test getting sign delta only for a single example
            self.assertEqual((nE_i, ), self.ssvm._get_sign_delta(tree_index=0, example_index=i).shape)
            np.testing.assert_equal(
                sign_delta_i,
                self.ssvm._get_sign_delta(tree_index=0, example_index=i)
            )

            idx += nE_i

    def test_get_lambda_delta__single_sequence(self):
        # We measure the run-time over 15 repetitions
        n_rep = 15
        rt_loop = 0.0
        rt_vec = 0.0

        # Dimensions of the sequence and candidate feature matrices
        n_features = 301
        n_candidates = 120

        for rep in range(n_rep):
            # Generate a random sequence length
            L = np.random.RandomState(rep + 7).randint(1, 16)

            # Generate a random spanning tree to super-impose it on the sequence
            G = nx.generators.trees.random_tree(L, seed=(rep + 3))

            # Generate the random feature matrices for the sequence and the candidates
            Y_sequence = np.random.RandomState(rep + 5).randn(L, n_features)
            Y_candidates = np.random.RandomState(rep + 6).randn(n_candidates, n_features)

            # Generate the reference solution by implemented in the looped version
            start = time.time()
            lambda_delta_ref = self.get_lambda_delta_ref(Y_sequence, Y_candidates, G, generalized_tanimoto_kernel_FAST)
            rt_loop += time.time() - start

            # Use the vectorized implementation
            start = time.time()
            lambda_delta = StructuredSVMSequencesFixedMS2._get_lambda_delta(
                Y_sequence, Y_candidates, G, generalized_tanimoto_kernel_FAST
            )
            rt_vec += time.time() - start

            self.assertEqual(lambda_delta_ref.shape, lambda_delta.shape)
            np.testing.assert_almost_equal(lambda_delta_ref, lambda_delta)

        # print("== get_lambda_delta ==")
        # print("Loop: %.3fs" % (rt_loop / n_rep))
        # print("Vec: %.3fs" % (rt_vec / n_rep))

    def test_get_lambda_delta__multiple_sequences(self):
        # Dimensions of the sequence and candidate feature matrices
        n_features = 301
        n_candidates = 120

        for rep in range(25):
            # Generate a random sequence length
            L = np.random.RandomState(rep + 7).randint(1, 16)

            # Generate a random spanning tree to super-impose it on the sequence
            G = nx.generators.trees.random_tree(L, seed=(rep + 3))

            # Generate the random feature matrices for the sequence and the candidates
            Y_candidates = np.random.RandomState(rep + 6).randn(n_candidates, n_features)

            # Generate a random number of sequences
            nS = np.random.RandomState(rep + 4).randint(1, 7)

            # Generate 'nS' many random sequences (associated with active dual variables)
            l_Y_sequence = [
                np.random.RandomState((nS * rep) + 5).randn(L, n_features)
                for _ in range(nS)
            ]

            # Compute the lambda delta vectors for each active sequence and stack them
            lambda_delta_ref = np.vstack(
                [
                    self.get_lambda_delta_ref(Y_sequence, Y_candidates, G, generalized_tanimoto_kernel_FAST)
                    for Y_sequence in l_Y_sequence
                ]
            )

            # Pass all sequences at ones to the delta lambda computation and compare to the reference
            lambda_delta = StructuredSVMSequencesFixedMS2._get_lambda_delta(
                np.vstack(l_Y_sequence), Y_candidates, G, generalized_tanimoto_kernel_FAST
            )

            self.assertEqual(((L - 1) * nS, len(Y_candidates)), lambda_delta_ref.shape)
            self.assertEqual(lambda_delta_ref.shape, lambda_delta.shape)
            np.testing.assert_almost_equal(lambda_delta_ref, lambda_delta)

    def test_I_rsvm_jfeat(self):
        N_E = sum(len(self.ssvm.training_graphs_[j][0].edges) for j in range(self.N))  # total number of edges

        rt_loop = 0.0
        rt_vec = 0.0

        with self.ssvm.training_data_.candidates:
            for i in range(self.N):
                Y_candidates = np.vstack(
                    self.ssvm.training_data_[i].get_molecule_features_for_candidates("substructure_count")
                )

                start = time.time()
                I_ref = np.zeros((len(Y_candidates), ))
                for j in range(self.N):
                    # Get sign-delta / |E_j|
                    sign_delta_j = self.ssvm.training_data_[j].get_sign_delta_t(self.ssvm.training_graphs_[j][0])
                    sign_delta_j /= len(self.ssvm.training_graphs_[j][0].edges)  # divide by the number of edges

                    #
                    lambda_delta_j = self.ssvm._get_lambda_delta(
                        Y_sequence=self.ssvm.training_data_[j].get_molecule_features_for_labels(
                            self.ssvm.mol_feat_retention_order),
                        Y_candidates=Y_candidates,
                        G=self.ssvm.training_graphs_[j][0],
                        mol_kernel=self.ssvm.mol_kernel
                    )
                    I_ref += (self.ssvm.C * (sign_delta_j @ lambda_delta_j) / len(self.ssvm.training_data_))

                rt_loop += time.time() - start

                start = time.time()
                I = self.ssvm._I_jfeat_rsvm(Y_candidates)
                rt_vec += time.time() - start

                self.assertEqual((len(Y_candidates), ), I.shape)
                np.testing.assert_almost_equal(I_ref, I)
                self.assertTrue(np.all((I / self.ssvm.C * self.N) >= -N_E))
                self.assertTrue(np.all((I / self.ssvm.C * self.N) <= N_E))

        # print("== I_rsvm_jfeat ==")
        # print("Loop: %.3fs" % (rt_loop / self.N))
        # print("Vec: %.3fs" % (rt_vec / self.N))

    def test_I_rsvm_jfeat__FOR_LARGE_MEMORY(self):
        N_E = np.sum([len(self.ssvm.training_graphs_[j][0].edges) for j in range(self.N)])  # total number of edges

        with self.ssvm.training_data_.candidates:
            for i in range(self.N):
                Y_candidates = np.vstack(
                    self.ssvm.training_data_[i].get_molecule_features_for_candidates("substructure_count")
                )

                I = self.ssvm._I_jfeat_rsvm__FOR_LARGE_MEMORY(Y_candidates)
                I_ref = self.ssvm._I_jfeat_rsvm(Y_candidates)

                self.assertEqual((len(Y_candidates), ), I.shape)
                np.testing.assert_allclose(I_ref, I)
                self.assertTrue(np.all((I / self.ssvm.C * self.N) >= -N_E))
                self.assertTrue(np.all((I / self.ssvm.C * self.N) <= N_E))

    def test_II_rsvm_jfeat(self):
        t_loop = 0.0
        t_vec = 0.0

        with self.ssvm.training_data_.candidates:
            for i in range(self.N):
                Y_candidates = np.vstack(
                    self.ssvm.training_data_[i].get_molecule_features_for_candidates("substructure_count")
                )

                # Use implementation in the SSVM class
                start = time.time()
                II = self.ssvm._II_jfeat_rsvm(Y_candidates)
                t_vec += (time.time() - start)

                # -----------------
                # Use implementation completely based on loops
                start = time.time()
                II_ref = np.zeros((len(Y_candidates), ))

                for j in range(self.N):
                    sign_delta = self.ssvm._get_sign_delta(0, j)

                    for (_, sj), aj in zip(*self.ssvm.alphas_.get_blocks(j)):  # SUM {y in Sigma}
                        Y_sequence = self.ssvm.training_data_.candidates.get_molecule_features_by_molecule_id(sj, "substructure_count")

                        lambda_delta = self.ssvm._get_lambda_delta(
                            Y_sequence, Y_candidates, self.ssvm.training_graphs_[j][0], self.ssvm.mol_kernel
                        )

                        for idx, y in enumerate(Y_candidates):
                            II_ref[idx] += (aj * sign_delta @ lambda_delta[:, idx])
                t_loop += (time.time() - start)

                np.testing.assert_almost_equal(II, II_ref)

        # print("== II_rsvm_jfeat ==")
        # print("Loop: %.3fs" % (t_loop / self.N))
        # print("Vec: %.3fs" % (t_vec / self.N))

    def test_predict_molecule_preference_values(self):
        with self.ssvm.training_data_.candidates:
            for i in range(5):  # inspect only the first 5 label sequences
                Y_candidates = self.ssvm.training_data_[i].get_molecule_features_for_candidates("substructure_count", 2)
                pref = self.ssvm.predict_molecule_preference_values(Y_candidates)
                self.assertEqual((len(Y_candidates), ), pref.shape)

    def test_get_node_and_edge_potentials(self):
        with self.ssvm.training_data_.candidates:
            for i in [0, 8, 5]:
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
