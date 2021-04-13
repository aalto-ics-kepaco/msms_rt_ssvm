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
import more_itertools as mit
import tensorflow as tf
import networkx as nx
import logging
import time

from functools import lru_cache
from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Optional, TypeVar, Callable
from sklearn.utils.validation import check_random_state
from sklearn.metrics import ndcg_score
from joblib.parallel import Parallel, delayed
from joblib.memory import Memory

from msmsrt_scorer.lib.exact_solvers import TreeFactorGraph
from msmsrt_scorer.lib.evaluation_tools import get_topk_performance_from_scores as get_topk_performance_from_marginals
from msmsrt_scorer.lib.cindex_measure import cindex

import ssvm.cfg
from ssvm.data_structures import SequenceSample, Sequence, LabeledSequence, SpanningTrees
from ssvm.data_structures import CandidateSQLiteDB
from ssvm.factor_graphs import get_random_spanning_tree
from ssvm.kernel_utils import tanimoto_kernel, _min_max_dense_jit, generalized_tanimoto_kernel_FAST
from ssvm.ssvm_meta import _StructuredSVM
from ssvm.dual_variables import DualVariables


DUALVARIABLES_T = TypeVar('DUALVARIABLES_T', bound='DualVariables')

JOBLIB_CACHE = Memory(ssvm.cfg.JOBLIB_MEMORY_CACHE_LOCATION, verbose=0)


# ================
# Setup the Logger
SSVM_LOGGER = logging.getLogger("ssvm_fixedms2__seq_spec_cand_db")
SSVM_LOGGER.setLevel(logging.INFO)
SSVM_LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.INFO)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

SSVM_LOGGER.addHandler(CH)
# ================


@JOBLIB_CACHE.cache(ignore=["candidates"], mmap_mode="r")
def get_molecule_features_by_molecule_id_CACHED(candidates: CandidateSQLiteDB, molecule_ids: Tuple[str, ...],
                                                features: str) -> np.ndarray:
    """
    Function to load molecular features by molecule IDs from the candidate database. The method of the CandidateDB class
    is wrapped here to allow for caching.
    """
    return candidates.get_molecule_features_by_molecule_id(molecule_ids, features)


class StructuredSVMSequencesFixedMS2SeqSpecCandDB(_StructuredSVM):
    """
    Structured Support Vector Machine (SSVM) for (MS, RT)-sequence classification.
    """
    def __init__(self, mol_feat_label_loss: str, mol_feat_retention_order: str,
                 mol_kernel: Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],  n_trees_per_sequence: int = 1,
                 retention_order_weight=None, n_jobs: int = 1, log_transform_node_potentials: bool = True,
                 average_node_and_edge_potentials: bool = False, update_direction: str = "map", *args, **kwargs):
        """
        :param mol_feat_label_loss:
        :param mol_feat_retention_order:
        :param mol_kernel:
        :param n_trees_per_sequence: scalar, number of spanning trees per sequence.
        :param args:
        :param kwargs:
        """

        self.mol_feat_label_loss = mol_feat_label_loss
        self.mol_feat_retention_order = mol_feat_retention_order
        self.mol_kernel = self.get_mol_kernel(mol_kernel)
        self.n_trees_per_sequence = n_trees_per_sequence
        self.retention_order_weight = retention_order_weight
        self.n_jobs = n_jobs
        self.log_transform_node_potentials = log_transform_node_potentials
        self.average_node_and_edge_potentials = average_node_and_edge_potentials
        self.update_direction = update_direction

        if self.retention_order_weight is None:
            self.D_rt = 1.0
            self.D_ms = 1.0
        elif np.isscalar(self.retention_order_weight):
            if not (0 <= self.retention_order_weight <= 1):
                raise ValueError("The retention order weight must be in [0, 1].")

            # Define the weight on the retention order and mass spectra scores
            self.D_rt = self.retention_order_weight
            self.D_ms = 1.0 - self.D_rt
        else:
            raise ValueError("The retention order weight must be either a scalar or None.")

        if self.n_trees_per_sequence > 1:
            raise ValueError("Currently only a single spanning tree per sequence is supported.")

        super().__init__(*args, **kwargs)

    # ==============
    # MODEL FITTING
    # ==============
    def fit(self, data: SequenceSample, n_init_per_example: int = 1,
            summary_writer: Optional[tf.summary.SummaryWriter] = None,
            validation_data: Optional[Union[SequenceSample, LabeledSequence]] = None):
        """
        Train the SSVM given a dataset.

        :param data: SequenceSample, set of training sequences. All needed information, such as features, labels and
            potential label sequences, are accessible through this object.

        :param n_init_per_example: scalar, number of initially active dual variables per example. That is, the number of
            active (potential) label sequences.

        :param summary_writer: Tensorflow SummaryWriter

        :param validation_data: validation sequence sample or a single labeled sequence

        :return: reference to it self.
        """
        self.training_data_ = data
        self.validation_data_ = validation_data

        # Open the connection to the candidate DB
        label_space = []
        for i in range(len(self.training_data_)):
            with self.training_data_[i].candidates:
                label_space.append(self.training_data_[i].get_labelspace())

        # Set up the dual initial dual vector
        self.alphas_ = DualVariables(C=self.C, label_space=label_space, num_init_active_vars=n_init_per_example,
                                     random_state=self.random_state)
        assert self._is_feasible_matrix(self.alphas_, self.C), "Initial dual variables must be feasible."

        # Initialize the graphs for each sequence
        self.training_graphs_ = [SpanningTrees(sequence, self.n_trees_per_sequence, i)
                                 for i, sequence in enumerate(self.training_data_)]

        self.training_graphs_info_ = {}
        self.training_graphs_info_["n_edges"] = [sptree.n_edges for sptree in self.training_graphs_]
        self.training_graphs_info_["n_edges_total"] = sum(self.training_graphs_info_["n_edges"])

        if self.validation_data_ is not None:
            self.validation_graphs_ = [SpanningTrees(sequence, self.n_trees_per_sequence, i)
                                       for i, sequence in enumerate(self.validation_data_)]
        else:
            self.validation_graphs_ = None

        # Run over the data
        # - each epoch is a cycle through the full data
        # - each batch contains a subset of the full data

        # Number of training sequences
        N = len(self.training_data_)

        random_state = check_random_state(self.random_state)
        n_iterations_total = 0

        if summary_writer is not None:
            _data_to_log = {}

            # Ranking performance scores and duality gap
            _start_time = time.time()
            _data_to_log["duality_gap"], _data_to_log["training__ndcg_ohc"] = self._get_ndcg_score_and_duality_gap()
            SSVM_LOGGER.info("Train data NDCG score = %.3f; Duality gap = %f (time = %.3fs)" %
                             (_data_to_log["training__ndcg_ohc"], _data_to_log["duality_gap"], time.time() - _start_time))
            self.duality_gap_0_ = _data_to_log["duality_gap"]
            _data_to_log["rel_duality_gap_decay"] = _data_to_log["duality_gap"] / self.duality_gap_0_

            if validation_data is not None:
                # Ranking performance on validation set
                _start_time = time.time()
                _data_to_log["validation__ndcg_ohc"] = \
                    self.score(self.validation_data_, self.validation_graphs_, stype="ndcg_ohc",
                               spanning_tree_random_state=len(self.validation_data_))
                SSVM_LOGGER.info("Validation data NDCG score = %.3f (time = %.3fs)" %
                                 (_data_to_log["validation__ndcg_ohc"], time.time() - _start_time))

            self.write_log(n_iterations_total, _data_to_log, summary_writer)

        SSVM_LOGGER.info("=== Start training ===")
        SSVM_LOGGER.info("batch_size = {}".format(self.batch_size))

        t_epoch, n_epochs_ran = 0, 0
        t_batch, n_batches_ran = 0, 0

        for epoch in range(self.n_epochs):
            SSVM_LOGGER.info("Epoch: %d / %d" % (epoch + 1, self.n_epochs))
            _start_time_epoch = time.time()

            # Setting the batch-size to None will result in a single batch per epoch --> dense update
            if self.batch_size is None:
                _batches = [list(range(N))]
            else:
                _batches = list(mit.chunked(random_state.permutation(np.arange(N)), self.batch_size))

            updates_made = False
            for step, I_batch in enumerate(_batches):
                SSVM_LOGGER.info("Step: %d / %d" % (step + 1, len(_batches)))
                _start_time_batch = time.time()

                # Find the most violating examples for the current batch
                _start_time = time.time()
                res = Parallel(n_jobs=self.n_jobs)(delayed(self.inference)(
                    self.training_data_[i], self.training_graphs_[i], loss_augmented=True, return_graphical_model=True,
                    update_direction=self.update_direction)
                                                   for i in I_batch)
                y_I_hat = [y_i_hat for y_i_hat, _ in res]
                SSVM_LOGGER.info("Finding the most violating examples took %.3fs" % (time.time() - _start_time))

                # Get step-width
                _start_time = time.time()
                if self.step_size_approach == "linesearch":
                    step_size = self._get_step_size_linesearch(I_batch, y_I_hat, [TFG for _, TFG in res])
                elif self.step_size_approach == "linesearch_parallel":
                    step_size = self._get_step_size_linesearch_parallel(I_batch, y_I_hat, [TFG for _, TFG in res])
                else:
                    step_size = self._get_step_size_diminishing(n_iterations_total, N)
                SSVM_LOGGER.info("Step-size determination using '%s' (took %.3fs): %.4f"
                                 % (self.step_size_approach, time.time() - _start_time, step_size))

                # Update the dual variables
                n_iterations_total += 1
                SSVM_LOGGER.info("Number of active dual variables (a_iy > 0)")
                SSVM_LOGGER.info("\tBefore update: %d" % self.alphas_.n_active())

                if step_size > 0:
                    is_new = [self.alphas_.update(i, y_i_hat, step_size) for i, y_i_hat in zip(I_batch, y_I_hat)]
                    SSVM_LOGGER.info("\tAfter update: %d" % self.alphas_.n_active())
                    SSVM_LOGGER.info("Which active sequences are new:")
                    for _i, _is_new in zip(I_batch, is_new):
                        SSVM_LOGGER.info("\tExample {:>4} new? {}".format(_i, _is_new))
                    updates_made = True

                    assert self._is_feasible_matrix(self.alphas_, self.C), \
                        "Dual variables after update are not feasible anymore."

                    if summary_writer is not None:
                        _data_to_log = {
                            "step_size": step_size,
                            "active_variables": self.alphas_.n_active(),
                            "training__avg_label_loss_batch": self._get_batch_label_loss(I_batch, y_I_hat),
                            "training__cindex_pref_values": self.score(self.training_data_, stype="cindex"),
                        }

                        if self.validation_data_ is not None:
                            _data_to_log["validation__cindex_pref_values"] = self.score(
                                self.validation_data_, stype="cindex")

                        self.write_log(n_iterations_total, _data_to_log, summary_writer)

                n_batches_ran += 1
                t_batch += (time.time() - _start_time_batch)
                SSVM_LOGGER.info("Batch run-time (avg): %.3fs" % (t_batch / n_batches_ran))
                SSVM_LOGGER.handlers[0].flush()

            n_epochs_ran += 1
            t_epoch += (time.time() - _start_time_epoch)
            SSVM_LOGGER.info("Epoch run-time (avg): %.3fs" % (t_epoch / n_epochs_ran))
            SSVM_LOGGER.handlers[0].flush()

            if summary_writer is not None:
                _data_to_log = {}

                # Ranking performance scores and duality gap
                _start_time = time.time()
                _data_to_log["duality_gap"], _data_to_log["training__ndcg_ohc"] = self._get_ndcg_score_and_duality_gap()
                SSVM_LOGGER.info("Train data NDCG score = %.3f; Duality gap = %f (time = %.3fs)" %
                                 (_data_to_log["training__ndcg_ohc"], _data_to_log["duality_gap"],
                                  time.time() - _start_time))
                _data_to_log["rel_duality_gap_decay"] = _data_to_log["duality_gap"] / self.duality_gap_0_

                if validation_data is not None:
                    # Ranking performance on validation set
                    _start_time = time.time()
                    _data_to_log["validation__ndcg_ohc"] = \
                        self.score(self.validation_data_, self.validation_graphs_, stype="ndcg_ohc",
                                   spanning_tree_random_state=len(self.validation_data_))
                    SSVM_LOGGER.info("Validation data NDCG score = %.3f (time = %.3fs)" %
                                     (_data_to_log["validation__ndcg_ohc"], time.time() - _start_time))

                self.write_log(n_iterations_total, _data_to_log, summary_writer)

            if not updates_made:
                break

        return self

    def _joblib_wrapper(self, i: int) -> Tuple[TreeFactorGraph, Tuple[str, ...], float]:
        with self.training_data_[i].candidates:
            edge_potentials = self._get_edge_potentials(self.training_data_[i], self.training_graphs_[i][0])

            # --------------------------------------------
            # Run the inference needed for the duality gap
            node_potentials_la = self._get_node_potentials(self.training_data_[i], self.training_graphs_[i][0], True)

            TFG, Z_max = self._inference(node_potentials_la, edge_potentials, self.training_graphs_[i][0])

            # MAP returns a list of candidate indices, we need to convert them back to actual molecules identifier
            y_hat = tuple(self.training_data_[i].get_labelspace(s)[Z_max[s]] for s in self.training_graphs_[i][0].nodes)
            # --------------------------------------------

            # --------------------------------------------
            # Run the NDCG scoring
            node_potentials = self._get_node_potentials(self.training_data_[i], self.training_graphs_[i][0], False)
            max_marginals = self._max_marginals(
                self.training_data_[i], node_potentials, edge_potentials, self.training_graphs_[i][0])
            ndcg_score = self.ndcg_score(self.training_data_[i], use_label_loss=False, marginals=max_marginals)
            # --------------------------------------------

        return TFG, y_hat, ndcg_score

    def _get_ndcg_score_and_duality_gap(self) -> Tuple[float, float]:
        """

        """
        N = len(self.training_data_)

        res = Parallel(n_jobs=self.n_jobs)(delayed(self._joblib_wrapper)(i) for i in range(N))
        TFG_I, y_I_hat, ndcg_scores = map(list, zip(*res))

        # Calculate (s - a)
        s_minus_a = DualVariables.get_s(self.alphas_, y_I_hat) - self.alphas_

        # Evaluate <s - a, \nabla g(a)>
        gap = np.sum(
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._duality_gap_joblib_wrapper)(i, s_minus_a, TFG_I) for i in range(N))
        ).item()

        return gap, np.mean(ndcg_scores).item()

    def _get_batch_label_loss(self, I_batch: List[int], y_I_hat: List[Tuple[str, ...]]) -> float:
        """
        Function to calculate the label loss for a set of most-violating label sequences.
        """
        lloss = 0

        for idx, i in enumerate(I_batch):
            with self.training_data_[i].candidates:
                # Load the features of the inferred candidate sequence
                Y_i_hat = self.training_data_[i].candidates.get_molecule_features_by_molecule_id(
                    y_I_hat[idx], features=self.mol_feat_label_loss)

                # Load the features of the true candidate sequence
                Y_i = self.training_data_[i].get_molecule_features_for_labels(features=self.mol_feat_label_loss)

                # Calculate the label-loss
                lloss += np.mean([self.label_loss_fun(Y_i[s], Y_i_hat[s]) for s in range(len(self.training_data_[i]))])

        return lloss / len(I_batch)

    def _duality_gap_joblib_wrapper(self, i: int, s_minus_a: DualVariables, TFG_I: List[TreeFactorGraph]) -> float:
        gap_i = 0

        if s_minus_a.n_active(i) == 0:
            return gap_i

        with self.training_data_[i].candidates:
            # Go over the active "dual" variables: s(i, y) - a(i, y) != 0
            for y, dual_value in s_minus_a.iter(i):
                Z = self.label_sequence_to_Z(y, self.training_data_[i].get_labelspace())
                gap_i += dual_value * TFG_I[i].likelihood(Z, log=True)  # dual_value = s(i, y) - a(i, y)

        return gap_i

    def _duality_gap(self) -> float:
        """
        Function to calculate the duality gap.
        """
        N = len(self.training_data_)

        # Solve the sub-problem for all training examples
        res = Parallel(n_jobs=self.n_jobs)(
            delayed(self.inference)(
                self.training_data_[i], self.training_graphs_[i], loss_augmented=True, return_graphical_model=True)
            for i in range(N))
        y_I_hat, TFG_I = map(list, zip(*res))

        # Calculate (s - a)
        s_minus_a = DualVariables.get_s(self.alphas_, y_I_hat) - self.alphas_

        # Evaluate <s - a, \nabla g(a)>
        gap = np.sum(
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._duality_gap_joblib_wrapper)(i, s_minus_a, TFG_I) for i in range(N))
        ).item()

        return gap

    # ==============
    # PREDICTION
    # ==============
    def predict(self, sequence: Sequence, map: bool = False, Gs: Optional[SpanningTrees] = None,
                n_jobs: Optional[int] = None) \
            -> Union[Tuple[str, ...], Dict[int, Dict[str, Union[int, np.ndarray, List[str]]]]]:
        """
        Function to predict the marginal distribution of the candidate sets along the given (MS, RT)-sequence.

        TODO: We should allow to pass in a SequenceSample to allow making predictions for multiple sequences at a time.
        TODO: Number of jobs per tree is currently not used by the MAP estimate function.

        :param sequence: Sequence or LabeledSequence, (MS, RT)-sequence and associated data, e.g. candidate sets.

        :param map: boolean, indicating whether only the most likely candidate sequence should be returned instead of
            the marginals.

        :param Gs: SpanningTrees or None, spanning trees (nx.Graphs) used to approximate the MRT associated with
            the (MS, RT)-sequence. If None, a set of spanning trees is generated (see also 'self.n_trees_per_sequence')

        :param n_jobs: scalar or None, number of parallel jobs to process the random-spanning-trees associated
            with the sequence in parallel. If None, no parallelization is used.

        :return:
            Either a dictionary of dictionaries containing the marginals for the candidate sets of each sequence
            (MS, RT)-tuple (map = False, see 'max_marginals')

                or

            A tuple of strings containing the most likely label sequence for the given input (map = True, see 'inference')
        """
        if map:
            return self.inference(sequence, Gs=Gs, loss_augmented=False)
        else:
            return self.max_marginals(sequence, Gs=Gs, n_jobs=n_jobs)

    def inference(self, sequence: Union[Sequence, LabeledSequence], Gs: Optional[SpanningTrees] = None,
                  loss_augmented: bool = False, return_graphical_model: bool = False, update_direction: str = "map") \
            -> Union[Tuple[str, ...], Tuple[Tuple[str, ...], TreeFactorGraph]]:
        """
        Infer the most likely label sequence for the given (MS, RT)-sequence.

        :param sequence: Sequence or LabeledSequence, (MS, RT)-sequence and associated data, e.g. candidate sets.

        :param Gs: SpanningTrees or None, spanning trees (nx.Graphs) used to approximate the MRT associated with
            the (MS, RT)-sequence. If None, a set of spanning trees is generated (see also 'self.n_trees_per_sequence')

        :param loss_augmented: boolean, indicating whether the node potentials should be augmented with the label loss.
            Set this option to True during model parameter estimation.

        :param return_graphical_model: boolean, indicating whether the graphical model, e.g. the Tree-Factor-Graph, used
            to solve performance the inference should be returned.

        :param update_direction: string, specifying which update direction should be returned:
            "map" (default): solves the (loss-augmented) decoding problem ~ most violating example
            "random": returns a random label sequence as update direction (intended for debugging purposes)

        :return: tuple, length = L, most likely label sequence
        """
        if Gs is None:
            Gs = SpanningTrees(sequence, self.n_trees_per_sequence)

        # Calculate the node- and edge-potentials
        with sequence.candidates:
            node_potentials, edge_potentials = self._get_node_and_edge_potentials(sequence, Gs[0], loss_augmented)

            # Find the MAP estimate
            TFG, Z_max = self._inference(node_potentials, edge_potentials, Gs[0], update_direction)

            # MAP returns a list of candidate indices, we need to convert them back to actual molecules identifier
            y_hat = tuple(sequence.get_labelspace(s)[Z_max[s]] for s in Gs[0].nodes)

        if return_graphical_model:
            return y_hat, TFG
        else:
            return y_hat

    def max_marginals(self, sequence: Union[Sequence, LabeledSequence], Gs: Optional[SpanningTrees] = None,
                      n_jobs: Optional[int] = None) -> Dict[int, Dict]:
        """
        Calculate the max-marginals for all possible label assignments of the given (MS, RT)-sequence.

        :param sequence: Sequence or LabeledSequence, (MS, RT)-sequence and associated data, e.g. candidate sets.

        :param Gs: SpanningTrees or None, spanning trees (nx.Graphs) used to approximate the MRT associated with
            the (MS, RT)-sequence. If None, a set of spanning trees is generated (see also 'self.n_trees_per_sequence')

        :return: dictionary = {
            sequence_index: dictionary = {
                labels: list of strings, length = m, candidate labels
                score: array-like, shape = (m, ), candidate marginals scores
                n_cand: scalar, number of candidate labels
            }
        } the dictionary is of length
        """
        if Gs is None:
            Gs = SpanningTrees(sequence, self.n_trees_per_sequence)

        if n_jobs is None:
            n_jobs = 1

        mms = Parallel(n_jobs=n_jobs)(delayed(self._max_marginals_joblib_wrapper)(sequence, G) for G in Gs)
        mm_agg = mms[0]

        # Aggregate max-marginals by simply averaging them
        if len(Gs) > 1:
            for mm in mms[1:]:
                for node in Gs.get_nodes():
                    assert mm_agg[node]["n_cand"] == mm[node]["n_cand"]
                    mm_agg[node]["score"] += mm[node]["score"]

                for node in Gs.get_nodes():
                    mm_agg[node]["score"] /= len(Gs)

        return mm_agg

    def _max_marginals_joblib_wrapper(self, sequence: Union[Sequence, LabeledSequence], G: nx.Graph):
        """
        Wrapper needed to calculate the max-marginals in parallel using joblib.
        """
        with sequence.candidates:
            node_potentials, edge_potentials = self._get_node_and_edge_potentials(sequence, G)

            # Calculate the max-marginals
            return self._max_marginals(sequence, node_potentials, edge_potentials, G, normalize=True)

    def predict_molecule_preference_values(self, Y: np.ndarray) -> np.ndarray:
        """
        :param Y: array-like, shape = (n_molecules, n_features), molecular feature vectors to calculate the preference
            values for.

        :return: array-like, shape = (n_molecules, ), preference values for all molecules.
        """
        I = self._I_jfeat_rsvm(Y)

        # Note: L_i = |E_j|
        N = len(self.training_data_)

        # List of the molecular structures belonging to the active examples, i.e. a(i, y) > 0
        l_Y_act_sequence = []  # length = (N, )
        l_A_Sj = []
        for j in range(N):
            Sj, A_Sj = self.alphas_.get_blocks(j)
            _, Sj = zip(*Sj)  # type: Tuple[Tuple[str, ...]]  # length = number of active sequences for example j
            l_A_Sj.append(np.array(A_Sj))

            # Sj: active labels, i.e. all y in Sigma_j for which a(j, y) > 0
            # A_Sj: dual values, array-like with shape = (|S_j|, )

            with self.training_data_[j].candidates:
                l_Y_act_sequence.append(get_molecule_features_by_molecule_id_CACHED(
                    self.training_data_[j].candidates, tuple(it.chain(*Sj)), self.mol_feat_retention_order))

        II = np.zeros((len(Y), N))  # shape = (n_molecules, )
        for j in range(N):
            lambda_delta = StructuredSVMSequencesFixedMS2SeqSpecCandDB._get_lambda_delta(
                l_Y_act_sequence[j], Y, self.training_graphs_[j][0], self.mol_kernel)

            # Bring output to shape = (|S_j|, |E_j|, n_molecules)
            lambda_delta = lambda_delta.reshape(
                (
                    self.alphas_.n_active(j),
                    self.training_graphs_[j].get_n_edges(),
                    len(Y)
                ))

            II[:, j] = np.einsum("j,ijk,i",
                                 self.training_data_[j].get_sign_delta_t(self.training_graphs_[j][0]),
                                 lambda_delta,
                                 l_A_Sj[j])

        return I - np.sum(II, axis=1)

    @lru_cache(maxsize=ssvm.cfg.LRU_CACHE_MAX_SIZE)
    def _get_sign_delta(self, tree_index: int):
        N = len(self.training_data_)

        return np.concatenate([
            # sign(delta t_j)
            self.training_data_[j].get_sign_delta_t(self.training_graphs_[j][tree_index])
            for j in range(N)  # shape = (|E_j|, )
        ])

    def _I_jfeat_rsvm(self, Y: np.array) -> np.ndarray:
        """
        :param Y: array-like, shape = (n_molecules, n_features), molecular feature vectors to calculate the preference
            values for.

        :return: array-like, shape = (n_molecules, )
        """
        # Note: L_i = |E_j|
        N = len(self.training_data_)

        # List of the retention time differences for all examples j over the spanning trees
        sign_delta = self._get_sign_delta(0)

        # List of the ground truth molecular structures for all examples j
        l_Y_gt_sequence = []
        for j in range(N):
            with self.training_data_[j].candidates:
                l_Y_gt_sequence.append(
                    self._get_molecule_features_for_labels(self.training_data_[j], self.mol_feat_retention_order)
                )

        lambda_delta = np.vstack([
            self._get_lambda_delta(l_Y_gt_sequence[j], Y, self.training_graphs_[j][0], self.mol_kernel)
            for j in range(N)  # shape = (|E_j|, n_mol), with n_mol = Y.shape[0]
        ])

        # TODO: This term is constant regardless of the dual variables.
        # C / N * < sign_delta , lambda_delta >, shape = (n_molecules, )
        return self.C * (sign_delta @ lambda_delta) / N

    def _get_node_and_edge_potentials(self, sequence: Union[Sequence, LabeledSequence], G: nx.Graph,
                                      loss_augmented: bool = False) -> Tuple[OrderedDict, OrderedDict]:
        """

        :param sequence: Sequence or LabeledSequence, (MS, RT)-sequence and associated data, e.g. candidate sets.

        :param G: networkx.Graph, random spanning tree of the MRF defined over the (MS, RT)-sequence.

        :param loss_augmented: boolean, indicating whether the node potentials should be augmented with the label loss.
            Set this option to True during model parameter estimation.

        :return: tuple (
            OrderedDict: key = nodes of G, values {"log_score": node scores, "n_cand": size of the label space},
            nested dictionary: key = nodes of G, (nested) key = nodes of G (<- edges): (nested) values: transition
                matrices
        )
        """
        return self._get_node_potentials(sequence, G, loss_augmented), self._get_edge_potentials(sequence, G)

    def _get_node_potentials(self, sequence: Union[Sequence, LabeledSequence], G: nx.Graph,
                             loss_augmented: bool = False) -> OrderedDict:
        """
        Calculate the Node potentials
        """
        # Calculate the node potentials. If needed, augment the MS scores with the label loss
        node_potentials = OrderedDict()

        # First load all MS2 scores
        _raw_scores = []
        for s in G.nodes:  # V
            # Load the raw scores
            _raw_scores.append(sequence.get_ms2_scores(s, scale_scores_to_range=False, return_as_ndarray=True))
            node_potentials[s] = {"n_cand": len(_raw_scores[-1])}

        # Calculate the normalization parameter based on ALL candidates
        _c1, _c2 = CandidateSQLiteDB.get_normalization_parameters_c1_and_c2(np.hstack(_raw_scores))

        for idx, s in enumerate(G.nodes):  # V
            # Normalize the raw scores using the global parameters to (0, 1] (and transform to log-scores)
            _score = CandidateSQLiteDB.normalize_scores(_raw_scores[idx], _c1, _c2)

            if self.log_transform_node_potentials:
                _score = np.log(_score)

            node_potentials[s]["log_score"] = self.D_ms * _score

            # Add label loss is loss-augmented scores are requested
            if loss_augmented:
                node_potentials[s]["log_score"] += \
                    sequence.get_label_loss(self.label_loss_fun, self.mol_feat_label_loss, s)  # Delta_i(y)

            if self.average_node_and_edge_potentials:
                node_potentials[s]["log_score"] /= len(G.nodes)

        return node_potentials

    def _get_edge_potentials(self, sequence: Union[Sequence, LabeledSequence], G: nx.Graph) -> OrderedDict:
        """

        """
        # Load the candidate features for all nodes
        l_Y = [sequence.get_molecule_features_for_candidates(self.mol_feat_retention_order, s) for s in G.nodes]
        pref_scores = self.predict_molecule_preference_values(np.vstack(l_Y))

        # Get a map from the node to the corresponding candidate feature vector indices
        cs_n_Y = np.cumsum([0] + [len(Y) for Y in l_Y])
        node_2_idc = [slice(cs_n_Y[l], cs_n_Y[l + 1]) for l in range(len(l_Y))]

        # Calculate the edge potentials
        edge_potentials = OrderedDict()
        for s, t in G.edges:
            if s not in edge_potentials:
                edge_potentials[s] = OrderedDict()

            edge_potentials[s][t] = {
                "log_score": self.D_rt * self._get_edge_potentials_rsvm(
                    # Retention times
                    sequence.get_retention_time(s),
                    sequence.get_retention_time(t),
                    # Pre-computed preference scores
                    pref_scores_s=pref_scores[node_2_idc[s]],
                    pref_scores_t=pref_scores[node_2_idc[t]]
                )
            }

            if self.average_node_and_edge_potentials:
                edge_potentials[s][t]["log_score"] /= len(G.edges)

            # As we do not know in which direction the edges are traversed during the message passing, we need to add
            # the transition matrices for both directions, i.e. s -> t and t -> s.
            if t not in edge_potentials:
                edge_potentials[t] = OrderedDict()

            edge_potentials[t][s] = {"log_score": edge_potentials[s][t]["log_score"].T}

        return edge_potentials

    def _get_edge_potentials_rsvm(self, rt_s: float, rt_t: float,
                                  Y_s: Optional[np.ndarray] = None, Y_t: Optional[np.ndarray] = None,
                                  pref_scores_s: Optional[np.ndarray] = None, pref_scores_t: Optional[np.ndarray] = None):
        """
        Predict the transition matrix M (n_cand_s x n_cand_t) for the edge = (s, t) with:

            M[y_s, y_t] = < w, Psi((rt_s, rt_t), (y_s, y_t)) for all y_s in Y_s and y_t in Y_t

        :param rt_s: scalar, retention time of node s, i.e. RT_s

        :param rt_t: scalar, retention time of node t, i.e. RT_t

        :param Y_s: array-like, shape = (n_cand_s, n_features), row-wise feature matrix for all labels associated with
            node s

        :param Y_t: array-like, shape = (n_cand_t, n_features), row-wise feature matrix for all labels associated with
            node t

        :return: array-like, shape = (n_cand_s, n_cand_t), transition matrix for edge (s, t)
        """
        if pref_scores_s is None:
            if Y_s is None:
                raise ValueError("Candidate features must be provided.")

            pref_scores_s = self.predict_molecule_preference_values(Y_s)
            # shape = (n_mol_s, )

        if pref_scores_t is None:
            if Y_t is None:
                raise ValueError("Candidate features must be provided.")

            pref_scores_t = self.predict_molecule_preference_values(Y_t)
            # shape = (n_mol_t, )

        return np.sign(rt_s - rt_t) * (pref_scores_s[:, np.newaxis] - pref_scores_t[np.newaxis, :])

    # ==============
    # SCORING
    # ==============
    def score(self, data: SequenceSample, l_Gs: Optional[List[SpanningTrees]] = None, stype: str = "top1_mm",
              average: bool = True, n_trees_per_sequence: Optional[int] = None, **kwargs) -> Union[float, np.ndarray]:
        """
        :param data: SequenceSample, Sample of (MS, RT)-sequence and associated data, e.g. candidate sets.

        :param l_Gs: list of SpanningTrees or None, spanning tree sets for all example sequences. Used to approximate the
            MRFs. If None, a set of spanning trees is generated (see also 'self.n_trees_per_sequence') for each sequence.

        :param stype: string, indicating which scoring type to use:

            "top1_mm" - top1 accuracy using max-marginals
            "top1_map" - top1 accuracy of the MAP estimate
            "topk_mm" - topk accuracy using max-marginals (k = 50)
            "ndcg_ll" - Normalized Discounted Cumulative Gain calculated from the label-loss and max-marginals ranking.
            "ndcg_ohc" - Normalized Discounted Cumulative Gain calculated from the one-hot-coding and max-marginals ranking.

        :param average: boolean, indicating whether the score should be averaged across the sample

        :param n_trees_per_sequence: scalar or None, number of random-spanning-trees per sequence to score. If None,
            the number of trees is equal the to the number used for training. This parameter is ignored when
            spanning trees are provided for the sequences (see l_Gs).

        :param kwargs: keyword arguments passed to the scoring functions

        :return: scalar or array-like, requested score
        """
        if len(data) <= 1:
            n_jobs_for_data = 1
            n_jobs_for_trees = self.n_jobs
        else:
            n_jobs_for_data = self.n_jobs
            n_jobs_for_trees = 1

        if n_trees_per_sequence is None:
            n_trees_per_sequence = self.n_trees_per_sequence

        if l_Gs is None:
            l_Gs = [SpanningTrees(seq, n_trees=n_trees_per_sequence,
                                  random_state=kwargs.get("spanning_tree_random_state", None)) for seq in data]

        if stype == "top1_mm":
            scores = Parallel(n_jobs=n_jobs_for_data)(
                delayed(self.top1_score)(sequence, Gs=Gs) for sequence, Gs in zip(data, l_Gs))
        elif stype == "top1_map":
            scores = Parallel(n_jobs=n_jobs_for_data)(
                delayed(self.top1_score)(sequence, Gs=Gs, map=True) for sequence, Gs in zip(data, l_Gs))
        elif stype == "topk_mm":
            return_percentage = kwargs.get("return_percentage", True)
            max_k = kwargs.get("max_k", 50)
            topk_method = kwargs.get("topk_method", "casmi")
            scores = Parallel(n_jobs=n_jobs_for_data)(
                delayed(self.topk_score)(
                    sequence, Gs=Gs, return_percentage=return_percentage, max_k=max_k,  pad_output=True,
                    n_jobs_for_trees=n_jobs_for_trees, topk_method=topk_method) for sequence, Gs in zip(data, l_Gs))
        elif stype == "ndcg_ll":
            scores = Parallel(n_jobs=n_jobs_for_data)(
                delayed(self.ndcg_score)(sequence, Gs=Gs, use_label_loss=True) for sequence, Gs in zip(data, l_Gs))
        elif stype == "ndcg_ohc":
            scores = Parallel(n_jobs=n_jobs_for_data)(
                delayed(self.ndcg_score)(sequence, Gs=Gs, use_label_loss=False) for sequence, Gs in zip(data, l_Gs))
        elif stype == "cindex":
            scores = Parallel(n_jobs=n_jobs_for_data)(delayed(self.cindex)(sequence) for sequence in data)
        else:
            raise ValueError("Invalid scoring type: '%s'." % stype)

        scores = np.array(scores)  # shape = (n_samples, ) or (n_samples, max_k)

        # Average the scores across the samples if desired
        if average:
            scores = np.mean(scores, axis=0)

        return scores

    def cindex(self, sequence: LabeledSequence) -> float:
        """
        Function to calculate the concordance index (cindex) for a labeled (MS, RT)-sequence. For that, first the
        preference values for the ground truth molecular structure are predicted. Subsequently, cindex is calculated
        by comparing the predicted retention order based on the preference values with the observed order based on the
        retention times.

        :param sequence: LabeledSequence, (MS, RT)-sequence for which the cindex should be calculated.

        :return: scalar, cindex
        """
        with sequence.candidates:
            # Predict preference scores for the ground truth labels
            pref_score = self.predict_molecule_preference_values(
                sequence.get_molecule_features_for_labels(self.mol_feat_retention_order))

            # Get the retention times
            rts = sequence.get_retention_time()

        return cindex(rts, pref_score)

    def topk_score(self, sequence: LabeledSequence, Gs: Optional[SpanningTrees] = None, return_percentage: bool = True,
                   max_k: Optional[int] = None, pad_output: bool = False, n_jobs_for_trees: Optional[int] = None,
                   topk_method="casmi2016") \
            -> Union[int, float, np.ndarray]:
        """
        Calculate top-k accuracy of the ranked candidate lists based on the max-marginals.

        TODO: For sklearn compatibility we need to separate the sequence data from the labels.
        TODO: Add a logger event here.

        :param sequence: Sequence or LabeledSequence, (MS, RT)-sequence and associated data, e.g. candidate sets.

        :param Gs: SpanningTrees or None, spanning trees (nx.Graphs) used to approximate the MRT associated with
            the (MS, RT)-sequence. If None, a set of spanning trees is generated (see also 'self.n_trees_per_sequence')

        :param return_percentage: boolean, indicating whether the percentage of correctly ranked candidates in top-k
            should be returned (True) or the absolute number (False).

        :param max_k: scalar, up to with k the top-k performance should be returned. If None, k is set to infinite.

        :param pad_output: boolean, indicating whether the top-k accuracy output array should be padded to the length
            equal 'max_k'. The value at the end of the array is repeated.

        :param n_jobs_for_trees: scalar or None, number of jobs to parallelize the marginal computation for the
            individual spanning trees.

        :param topk_method: string, which method to use for the top-k accuracy calculation

        :return: array-like, shape = (max_k, ), list of top-k, e.g. top-1, top-2, ..., accuracies. Note, the array is
            zero-based, i.e. at index 0 we have top-1. If max_k == 1, the output is a scalar.
        """
        # Calculate the marginals
        marginals = self.predict(sequence, Gs=Gs, map=False, n_jobs=n_jobs_for_trees)

        return self._topk_score(sequence, marginals, return_percentage, max_k, pad_output, topk_method)

    def top1_score(self, sequence: LabeledSequence, Gs: Optional[SpanningTrees] = None, map: bool = False) -> float:
        """
        Calculate top-1 accuracy of the ranked candidate lists based on the max-marginals.

        :param sequence: Sequence or LabeledSequence, (MS, RT)-sequence and associated data, e.g. candidate sets.

        :param Gs: SpanningTrees or None, spanning trees (nx.Graphs) used to approximate the MRT associated with
            the (MS, RT)-sequence. If None, a set of spanning trees is generated (see also 'self.n_trees_per_sequence')
            
        :param map: boolean, indicating whether the top-1 accuracy should be calculated from the most likely candidate
            sequence (MAP estimate) (True) or the using the highest ranked candidates based on the marginals (False).

        :return: scalar, top-1 accuracy
        """
        if map:
            # Use most likely candidate assignment (MAP)
            map_seq = self.predict(sequence, map=True, Gs=Gs)
            top1 = np.sum([y_map_s == sequence.get_labels(s) for s, y_map_s in enumerate(map_seq)]) * 100 / len(map_seq)
        else:
            # Use top ranked candidate based on the marginals
            top1 = self.topk_score(sequence, Gs=Gs, max_k=1, return_percentage=True)

        return top1

    def ndcg_score(self, sequence: LabeledSequence, Gs: Optional[SpanningTrees] = None, use_label_loss: bool = True,
                   marginals: Optional[Dict] = None) -> float:
        """
        Compute the Normalized Discounted Cumulative Gain (NDCG) for the given (MS, RT)-sequence.

        ''
            Sum the true scores ranked in the order induced by the predicted scores, after applying a logarithmic
            discount. Then divide by the best possible score (Ideal DCG, obtained for a perfect ranking) to obtain a
            score between 0 and 1.
        ''
        From: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html

        The NDCG score will be high, if the predicted ranking is close to the ground truth (GT) ranking.

        :param sequence: Sequence or LabeledSequence, (MS, RT)-sequence and associated data, e.g. candidate sets.

        :param Gs: SpanningTrees or None, spanning trees (nx.Graphs) used to approximate the MRT associated with
            the (MS, RT)-sequence. If None, a set of spanning trees is generated (see also 'self.n_trees_per_sequence')
            
        :param use_label_loss: boolean, indicating whether the label-loss should be used to define the GT ranking, be
            using 1 - label_loss for each candidate as GT relevance value (True). If set to False, a one-hot-encoding
            vector with a one at the correct candidate and zero otherwise is used as GT relevance vector.

        :return: scalar, NDCG value averaged across the sequence
        """
        def _one_hot_vector(idx, length):
            v = np.zeros((length, ))
            v[idx] = 1
            return v

        with sequence.candidates:
            if marginals is None:
                # Calculate the marginals
                marginals = self.predict(sequence, Gs=Gs, map=False)
            else:
                if Gs is not None:
                    raise ValueError("When marginals are provided, than no spanning trees should be provided. It cannot "
                                     "be ensured that the provided spanning trees have been used to calculate the marginals.")

            # Get the ground-truth relevance for the ranked lists
            if use_label_loss:
                # 1 - label_loss
                gt_relevance = [
                    1 - sequence.get_label_loss(self.label_loss_fun, self.mol_feat_label_loss, s)
                    for s in marginals
                ]
            else:
                # One-hot-encoding of the positive of the correct molecular structure
                gt_relevance = [
                    _one_hot_vector(marginals[s]["label"].index(sequence.get_labels(s)), marginals[s]["n_cand"])
                    for s in marginals
                ]

            # Calculate the NDCG averaged over the sequence
            scores = np.ones(len(marginals))
            for s in marginals:
                if len(marginals[s]["score"]) > 1:
                    scores[s] = ndcg_score(gt_relevance[s][np.newaxis, :], marginals[s]["score"][np.newaxis, :])

        return scores.mean().item()

    # ================
    # STEP SIZE
    # ================
    def _get_step_size_linesearch_parallel(self, I_batch: List[int], y_I_hat: List[Tuple[str, ...]],
                                           TFG_I: List[TreeFactorGraph]) -> float:
        """
        Determine step-size using line-search. The calculation is distributed over multiple workers. The number of jobs
        is equal to the batch-size (see self.batch_size).

        :param I_batch: list, training example indices in the current batch

        :param y_I_hat: list of tuples, list of most-violating label sequences

        :param TFG_I: list of TreeFactorGraph, list of graphical models associated with the training examples. The TFGs
            must be generated using loss-augmentation.

        :return: scalar, line search step-size
        """
        N = len(self.training_data_)

        # Update direction s
        s = DualVariables(self.C, self.alphas_.label_space, initialize=False).set_alphas(
            [(i, y_i_hat) for i, y_i_hat in zip(I_batch, y_I_hat)], self.C / N)

        # Difference of the dual vectors: s - a
        s_minus_a = s - self.alphas_  # type: DualVariables

        if s_minus_a.get_blocks(I_batch) == ([], []):
            # The best update direction found is equal to the current solution (model). No update needed.
            return -1

        # Pre-compute some stuff
        l_sign_delta_t = [self.training_data_[i].get_sign_delta_t(self.training_graphs_[i][0]) for i in I_batch]
        l_P = [self._get_P(self.training_graphs_[i][0]) for i in I_batch]

        # Calculate the nominator and denominator using a helper function in parallel.
        nom, den = zip(*Parallel(n_jobs=self.n_jobs)(
            delayed(self._linesearch_joblib_wrapper)(s_minus_a, TFG_I[idx], I_batch, i, idx, l_sign_delta_t, l_P)
            for idx, i in enumerate(I_batch)))

        return np.maximum(0.0, np.minimum(1.0, np.sum(nom) / np.sum(den)))

    def _linesearch_joblib_wrapper(self, s_minus_a, TFG_i, I_batch, i, idx_i, l_sign_delta_t, l_P):
        """
        Helper function to calculate the line-search step-size in parallel.
        """
        nom = 0.0
        den = 0.0

        l_sign_delta_t_i_T_P_i = {}
        l_mol_features_i = {}

        with self.training_data_[i].candidates:
            for y, fac in s_minus_a.iter(i):
                Z = self.label_sequence_to_Z(y, s_minus_a.label_space[i])
                nom += fac * TFG_i.likelihood(Z, log=True)  # fac = s(i, y) - a(i, y)

                l_mol_features_i[y] = get_molecule_features_by_molecule_id_CACHED(
                    self.training_data_[i].candidates, tuple(y), self.mol_feat_retention_order)

                # Go over all i, j for which i = j
                l_sign_delta_t_i_T_P_i[y] = l_sign_delta_t[idx_i] @ l_P[idx_i]
                L_yy = self.mol_kernel(l_mol_features_i[y], l_mol_features_i[y])
                den += fac ** 2 * l_sign_delta_t_i_T_P_i[y] @ L_yy @ l_sign_delta_t_i_T_P_i[y]

            for idx_j, j in enumerate(I_batch[idx_i + 1:], idx_i + 1):
                sign_delta_t_j_T_P_j = l_sign_delta_t[idx_j] @ l_P[idx_j]

                # Go over the active "dual" variables: s(i, y) - a(i, y) != 0
                with self.training_data_[j].candidates:
                    for by, fac_j in s_minus_a.iter(j):
                        mol_features_by = get_molecule_features_by_molecule_id_CACHED(
                            self.training_data_[j].candidates, tuple(by), self.mol_feat_retention_order)

                        for y, fac_i in s_minus_a.iter(i):
                            L_yby = self.mol_kernel(l_mol_features_i[y], mol_features_by)
                            den += 2 * fac_i * fac_j * l_sign_delta_t_i_T_P_i[y] @ L_yby @ sign_delta_t_j_T_P_j

        return nom, den

    def _get_step_size_linesearch(self, I_batch: List[int], y_I_hat: List[Tuple[str, ...]],
                                  TFG_I: List[TreeFactorGraph]) -> float:
        """
        Determine step-size using line-search.

        :param I_batch: list, training example indices in the current batch

        :param y_I_hat: list of tuples, list of most-violating label sequences

        :param TFG_I: list of TreeFactorGraph, list of graphical models associated with the training examples. The TFGs
            must be generated using loss-augmentation.

        :return: scalar, line search step-size
        """

        N = len(self.training_data_)

        # Update direction s
        s = DualVariables(self.C, self.training_data_.get_labelspace(), initialize=False).set_alphas(
            [(i, y_i_hat) for i, y_i_hat in zip(I_batch, y_I_hat)], self.C / N)

        # Difference of the dual vectors: s - a
        s_minus_a = s - self.alphas_  # type: DualVariables

        if s_minus_a.get_blocks(I_batch) == ([], []):
            # The best update direction found is equal to the current solution (model). No update needed.
            return -1

        # ----------------
        # Nominator
        # ----------------
        nom = 0.0

        # Tracking some variables we need later for the denominator
        l_mol_features = [{} for _ in I_batch]

        for idx, i in enumerate(I_batch):
            TFG = TFG_I[idx]

            # Go over the active "dual" variables: s(i, y) - a(i, y) != 0
            for y, fac in s_minus_a.iter(i):
                Z = self.label_sequence_to_Z(y, self.training_data_[i].get_labelspace())
                nom += fac * TFG.likelihood(Z, log=True)  # fac = s(i, y) - a(i, y)

                # Load the molecular features for the active sequence y
                with self.training_data_[i].candidates:
                    l_mol_features[idx][y] = get_molecule_features_by_molecule_id_CACHED(
                        self.training_data_[i].candidates, tuple(y), self.mol_feat_retention_order)

        # ----------------
        # Denominator
        # ----------------
        den = 0.0

        l_sign_delta_t = [self.training_data_[i].get_sign_delta_t(self.training_graphs_[i][0]) for i in I_batch]
        l_P = [self._get_P(self.training_graphs_[i][0]) for i in I_batch]

        # Go over all i, j for which i = j
        for idx, i in enumerate(I_batch):
            sign_delta_t_i_T_P_i = l_sign_delta_t[idx] @ l_P[idx]

            for y, fac in s_minus_a.iter(i):
                L_yy = self.mol_kernel(l_mol_features[idx][y], l_mol_features[idx][y])
                den += fac**2 * sign_delta_t_i_T_P_i @ L_yy @ sign_delta_t_i_T_P_i

        # Go over all (i, j) for which i != j. Note (i, j) gets the same values as (j, i)
        for idx_i, i in enumerate(I_batch):
            sign_delta_t_i_T_P_i = l_sign_delta_t[idx_i] @ l_P[idx_i]

            for idx_j, j in enumerate(I_batch[idx_i + 1:], idx_i + 1):
                sign_delta_t_j_T_P_j = l_sign_delta_t[idx_j] @ l_P[idx_j]

                for y, fac_i in s_minus_a.iter(i):
                    for by, fac_j in s_minus_a.iter(j):
                        L_yby = self.mol_kernel(l_mol_features[idx_i][y], l_mol_features[idx_j][by])
                        den += 2 * fac_i * fac_j * sign_delta_t_i_T_P_i @ L_yby @ sign_delta_t_j_T_P_j

        return np.maximum(0.0, np.minimum(1.0, nom / den))

    # ================
    # STATIC METHODS
    # ================
    @staticmethod
    def _topk_score(
            sequence: LabeledSequence,
            marginals: Union[Tuple[str, ...], Dict[int, Dict[str, Union[int, np.ndarray, List[str]]]]],
            return_percentage: bool = True,
            max_k: Optional[int] = None,
            pad_output: bool = False,
            topk_method="casmi2016") -> Union[int, float, np.ndarray]:
        """
        Calculate top-k accuracy of the ranked candidate lists based on the max-marginals.

        TODO: For sklearn compatibility we need to separate the sequence data from the labels.
        TODO: Add a logger event here.

        :param sequence: Sequence or LabeledSequence, (MS, RT)-sequence and associated data, e.g. candidate sets.

        :param marginals: Dictionary, pre-computed candidate marginals for the given sequence

        :param return_percentage: boolean, indicating whether the percentage of correctly ranked candidates in top-k
            should be returned (True) or the absolute number (False).

        :param max_k: scalar, up to with k the top-k performance should be returned. If None, k is set to infinite.

        :param pad_output: boolean, indicating whether the top-k accuracy output array should be padded to the length
            equal 'max_k'. The value at the end of the array is repeated.

        :param topk_method: string, which method to use for the top-k accuracy calculation

        :return: array-like, shape = (max_k, ), list of top-k, e.g. top-1, top-2, ..., accuracies. Note, the array is
            zero-based, i.e. at index 0 we have top-1. If max_k == 1, the output is a scalar.
        """
        # Need to find the index of the correct label / molecular structure in the candidate sets to determine the top-k
        # accuracy.
        for s in marginals:
            marginals[s]["index_of_correct_structure"] = marginals[s]["label"].index(sequence.get_labels(s))

        # Calculate the top-k performance
        if topk_method.lower().startswith("casmi"):
            topk_method = "casmi2016"
        elif topk_method.lower().startswith("csi"):
            topk_method = "csifingerid"
        else:
            raise ValueError("Invalid topk-method '%s'." % topk_method)
        topk = get_topk_performance_from_marginals(marginals, method=topk_method)[return_percentage]  # type: np.ndarray

        # Restrict the output the maximum k requested
        if max_k is not None:
            assert max_k >= 1

            topk = topk[:max_k]

            # Pad the output to match the length with max_k,
            #   e.g. let max_k = 7 then [12, 56, 97, 100] --> [12, 56, 97, 100, 100, 100, 100]
            if pad_output:
                _n_to_pad = max_k - len(topk)
                topk = np.pad(topk, (0, _n_to_pad), mode="edge")
                assert len(topk) == max_k

        # Output scalar of only top-1 is requested.
        if len(topk) == 1:
            topk = topk[0].item()

        return topk

    @staticmethod
    def write_log(n_iter: int, data_to_log: dict, summary_writer: tf.summary.SummaryWriter):
        """
        :param n_iter:
        :param data_to_log:
        :param summary_writer:
        :return:
        """
        for k, v in data_to_log.items():
            with summary_writer.as_default():
                if k in ["active_variables", "duality_gap", "rel_duality_gap_decay"]:
                    scope = "Model"
                elif k in ["step_size"]:
                    scope = "Optimizer"
                elif k.startswith("training__"):
                    scope = "Metric (Training)"
                elif k.startswith("validation__"):
                    scope = "Metric (Validation)"
                else:
                    raise ValueError("Invalid score: %s" % k)

                with tf.name_scope(scope):
                    tf.summary.scalar(k, data=v, step=n_iter)

    @staticmethod
    def get_mol_kernel(mol_kernel: Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]) \
            -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """

        :param mol_kernel:
        :return:
        """
        if callable(mol_kernel):
            return mol_kernel
        elif mol_kernel == "tanimoto":
            return tanimoto_kernel
        elif mol_kernel == "minmax":
            return generalized_tanimoto_kernel_FAST
        elif mol_kernel == "minmax_numba":
            return _min_max_dense_jit
        else:
            raise ValueError("Invalid molecule kernel")

    @staticmethod
    def label_sequence_to_Z(y: Tuple[str, ...], labelspace: List[List[str]]) -> List[int]:
        """
        :param y: tuple, label sequence

        :param labelspace: list of lists, label sequence spaces

        :return: list, indices of the labels along the provided sequence in the label space
        """
        return [labelspace_s.index(y_s) for y_s, labelspace_s in zip(y, labelspace)]

    @staticmethod
    def _get_P(G: nx.Graph) -> np.ndarray:
        """

        :param G:
        :return:
        """
        P = np.zeros((len(G.edges), len(G.nodes)))
        for e_idx, (s, t) in enumerate(G.edges):
            P[e_idx, s] = 1
            P[e_idx, t] = -1

        return P

    @staticmethod
    def _get_graph_set(data: SequenceSample, n_trees_per_sequence: int = 1) -> List[List[nx.Graph]]:
        """
        Generate a list of random spanning trees (RST) for the MRFs defined over each training label sequence.

        :param data: SequenceSample, labeled training sequences

        :param n_trees_per_sequence: scalar, number of spanning trees per label sequence

        :return: list of lists of networkx.Graph, length = n_trees_per_sequence, length[i] = number of sequences
        """
        if n_trees_per_sequence < 1:
            raise ValueError("Number of trees per sequence must >= 1.")

        if n_trees_per_sequence > 1:
            raise NotImplementedError("Currently only one tree per sequence allowed.")

        return [[get_random_spanning_tree(y_seq, random_state=tree) for y_seq in data]
                for tree in range(n_trees_per_sequence)]

    @staticmethod
    def _max_marginals(sequence: Union[Sequence, LabeledSequence], node_potentials: OrderedDict,
                       edge_potentials: OrderedDict, G: nx.Graph, normalize=True) -> Dict:
        """
        Max-max_marginals
        """
        # Calculate the normalized max-max_marginals
        marg = TreeFactorGraph(candidates=node_potentials, var_conn_graph=G, make_order_probs=None,
                               order_probs=edge_potentials, D=None) \
            .max_product() \
            .get_max_marginals(normalize)

        return {s: {"label": sequence.get_labelspace(s), "score": marg[s], "n_cand": len(marg[s])} for s in G.nodes}

    @staticmethod
    def _inference(node_potentials: OrderedDict, edge_potentials: OrderedDict, G: nx.Graph,
                   update_direction: str = "map") -> Tuple[TreeFactorGraph, List[int]]:
        """
        MAP inference

        :param update_direction: string, specifying which update direction should be returned:
            "map" (default): solves the (loss-augmented) decoding problem ~ most violating example
            "random": returns a random label sequence as update direction (intended for debugging purposes)
        """
        # MAP inference (find most likely label sequence) for the given node- and edge-potentials
        TFG = TreeFactorGraph(candidates=node_potentials, var_conn_graph=G, make_order_probs=None,
                              order_probs=edge_potentials, D=None)

        if update_direction == "random":
            Z_max = [np.random.choice(range(node_potentials[s]["n_cand"])) for s in node_potentials.keys()]
        elif update_direction == "map":
            Z_max, _ = TFG.MAP_only()
        else:
            raise ValueError("Invalid update direction: '%s'. Choices are 'random' and 'map'." % update_direction)

        return TFG, Z_max

    @staticmethod
    def _get_lambda_delta_OLD(Y_sequence: np.ndarray, Y_candidates: np.ndarray, G: nx.Graph,
                              mol_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Calculate the 'Lambda_Delta(y, y_s)' term, with shape = (L - 1, ) for all y_s in Y_candidates.

        :param Y_sequence: array-like, shape = (L, n_features), molecule features associated with the

        :param Y_candidates: array-like, shape = (n_candidates, n_features), feature matrix

        :param G: networkx.Graph, spanning tree defined over the MRF associated with the label sequence.

        :param mol_kernel: callable, function that takes in two feature row-matrices as input and outputs the kernel
            similarity matrix.

        :return: array-shape = (L - 1, n_molecules)
        """
        assert len(G.nodes) > 1, "There must be at least two nodes in the graph."
        assert len(G.edges) > 0, "There must be at least one edge in the graph."

        Ky = mol_kernel(Y_sequence, Y_candidates)  # shape = (L, n_candidates)

        bS, bT = zip(*G.edges)  # each of length |E| = L - 1

        return Ky[list(bS)] - Ky[list(bT)]

    @staticmethod
    def _get_lambda_delta(Y_sequence: np.ndarray, Y_candidates: np.ndarray, G: nx.Graph,
                          mol_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Calculate the 'Lambda_Delta(y, y_s)' term, with shape = ((L - 1) * |S|, ) for all y_s in Y_candidates.

        :param Y_sequence: array-like, shape = (L * |S|, n_features), molecule features associated with the

        :param Y_candidates: array-like, shape = (n_candidates, n_features), feature matrix

        :param G: networkx.Graph, spanning tree defined over the MRF associated with the label sequence.

        :param mol_kernel: callable, function that takes in two feature row-matrices as input and outputs the kernel
            similarity matrix.

        :return: array-shape = ((L - 1) * |S|, n_molecules)
        """
        L = len(G.nodes)
        n_S = len(Y_sequence) // L

        Ky = mol_kernel(Y_sequence, Y_candidates)  # shape = (L * |S|, n_candidates)

        bS, bT = zip(*G.edges)  # each of length |E| = L - 1

        if n_S > 1:
            corr = (np.arange(n_S) * L)[:, np.newaxis]
            out = Ky[(corr + np.array(bS)).flatten()] - Ky[(corr + np.array(bT)).flatten()]
        else:
            out = Ky[list(bS)] - Ky[list(bT)]

        return out

    @staticmethod
    def _get_molecule_features_for_labels(sequence: LabeledSequence, features: str) -> np.ndarray:
        """
        TODO
        """
        return get_molecule_features_by_molecule_id_CACHED(sequence.candidates, tuple(sequence.labels), features)
