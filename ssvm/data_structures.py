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
import os
import numpy as np
import pandas as pd

from scipy.io import loadmat
from typing import List, Tuple, Union, Dict, Optional
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_random_state, check_is_fitted

from ssvm.kernel_utils import tanimoto_kernel


class CandidateSetMetIdent(object):
    def __init__(self, mols: np.ndarray, fps: np.ndarray, mols2cand: Dict, idir: str, preload_data=False,
                 max_n_train_candidates=np.inf):
        self.mols = mols
        self.fps = fps
        self.mols2cand = mols2cand
        self.idir = idir
        self.mol2idx = {mol: i for i, mol in enumerate(self.mols)}
        self.preload_data = preload_data
        self.max_n_train_candidates = max_n_train_candidates

        if self.preload_data:
            self._cand_sets = self._preload_candidate_sets()
        else:
            self._cand_sets = None

    def _preload_candidate_sets(self):
        cand_sets = {}
        for idx, mol in enumerate(self.mols):
            cand_sets[mol] = self._load_candidate_set(mol)

        return cand_sets

    def _load_candidate_set(self, mol: str) -> Dict:
        cand = loadmat(os.path.join(self.idir, "candidate_set_" + self.mols2cand[mol] + ".mat"))

        # Clean up some Matlab specific stuff
        del cand["__header__"]
        del cand["__version__"]
        del cand["__globals__"]

        cand["fp"] = cand["fp"].toarray().T  # shape = (n_samples, n_fps)
        cand["inchi"] = [inchi[0][0] for inchi in cand["inchi"]]  # molecule identifier
        cand["n_cand"] = len(cand["inchi"])  # number of candidates
        # dict: mol identifier -> index (row) in fingerprint matrix for example
        cand["mol2idx"] = {_cand_mol: i for i, _cand_mol in enumerate(cand["inchi"])}

        try:
            cand["index_of_correct_structure"] = cand["mol2idx"][mol]
        except KeyError:
            raise KeyError("Cannot find correct molecular structure '%s' in candidate set '%s'." % (
                mol, self.mols2cand[mol]))

        if not np.isinf(self.max_n_train_candidates):
            # Sample a sub-set for the training
            subset = np.random.RandomState(cand["n_cand"]).choice(
                cand["n_cand"], int(np.minimum(self.max_n_train_candidates, cand["n_cand"])), replace=False)

            # Ensure that the current candidate is in the sub-set
            if cand["index_of_correct_structure"] not in subset:
                subset[0] = cand["index_of_correct_structure"]

            cand["training_subset"] = subset
        else:
            cand["training_subset"] = np.arange(cand["n_cand"])

        return cand

    def n_fps(self) -> int:
        return self.fps.shape[1]

    def get_labelspace(self, mol: str, for_training=False) -> List[str]:
        # TODO: We do not really need to load all candidates just to get the label space.
        if self.preload_data:
            cand = self._cand_sets[mol]
        else:
            cand = self._load_candidate_set(mol)

        labspace = cand["inchi"]

        if for_training:
            labspace = [labspace[idx] for idx in cand["training_subset"]]

        return labspace

    def get_gt_fp(self, exp_mols: Optional[Union[str, np.ndarray]] = None) -> np.ndarray:
        if exp_mols is not None:
            exp_mols = np.atleast_1d(exp_mols)
            idc = [self.mol2idx[_mol] for _mol in exp_mols]

            if len(idc) == 1:
                idc = idc[0]

            fps = self.fps[idc]
        else:
            fps = self.fps

        return fps

    def get_candidate_fps(self, mol: str, mol_sel=None, for_training=False) -> np.ndarray:
        if self.preload_data:
            cand = self._cand_sets[mol]
        else:
            cand = self._load_candidate_set(mol)

        fps = cand["fp"]

        if for_training:
            if mol_sel is not None:
                mol_sel = np.atleast_1d(mol_sel)
                assert len(mol_sel) == 1
                idc = cand["mol2idx"][mol_sel[0]]
                assert idc in cand["training_subset"]
                fps = fps[[idc]]
            else:
                fps = fps[cand["training_subset"]]
        else:
            if mol_sel is not None:
                mol_sel = np.atleast_1d(mol_sel)
                idc = [cand["mol2idx"][_mol] for _mol in mol_sel]
                fps = fps[idc]

        return fps

    def getMolKernel_ExpVsCand(self, exp_mols: np.ndarray, mol: str, kernel="tanimoto", for_training=False) -> \
            np.ndarray:
        if self.preload_data:
            cand = self._cand_sets[mol]
        else:
            cand = self._load_candidate_set(mol)

        cand_fps = cand["fp"]

        if for_training:
            cand_fps = cand_fps[cand["training_subset"]]

        idc = [self.mol2idx[_mol] for _mol in exp_mols]
        K = self.get_kernel(self.fps[idc], cand_fps, kernel)

        if for_training:
            assert K.shape == (len(exp_mols), len(cand["training_subset"]))
        else:
            assert K.shape == (len(exp_mols), cand["n_cand"])

        return K

    def getMolKernel_ExpVsExp(self, exp_mols: np.ndarray, kernel="tanimoto"):
        idc = [self.mol2idx[_mol] for _mol in exp_mols]
        K = self.get_kernel(self.fps[idc], self.fps[idc], kernel)
        assert K.shape == (len(exp_mols), len(exp_mols))

        return K

    def get_kernel(self, fps_A: np.ndarray, fps_B: Optional[np.ndarray] = None, kernel="tanimoto") -> np.ndarray:
        if kernel == "tanimoto":
            K = tanimoto_kernel(fps_A, fps_B, shallow_input_check=True)
        else:
            raise ValueError("Invalid kernel '%s'. Choices are 'tanimoto'." % kernel)
        return K


class Sequence(object):
    def __init__(self, spec_ids: List[str]):
        self.spec_ids = spec_ids
        self.L = len(self.spec_ids)

        self.cand = None  # stores a reference to the candidates of each sequence element

    def __len__(self) -> int:
        """
        Return the length of the sequence.

        :return: scalar, length of the sequence
        """
        return self.L

    def get_n_cand_sig(self, sig) -> int:
        """
        :param sig: scalar, index of the sequence element to get the number of candidates.
        :return: scalar, number of candidates
        """
        return len(self.cand[sig])

    def get_n_cand(self) -> List[int]:
        return [self.get_n_cand_sig(sig) for sig in range(len(self))]

    def initalize_kernels(self):
        pass


class LabeledSequence(Sequence):
    """
    Class representing the a _labeled_ (MS, RT)-sequence (x, t, y) with associated molecular candidate set C.
    """
    def __init__(self, spec_ids: List[str], labels: List[str]):
        """
        :param spec_ids: list of strings, spectrum-ids belonging sequence
        :param labels: list of strings, ground truth molecule identifiers belonging to the spectra of the sequence
        """
        self.labels = labels

        super(LabeledSequence, self).__init__(spec_ids=spec_ids)

    def as_Xy_input(self) -> Tuple[List[Sequence], List[str]]:
        """
        Return the (MS, RT)-sequence and ground truth label separately as input for the sklearn interface.

        Usage: sklearn.fit(*LabeledSequence(...).as_Xy_input)

        :return:
        """
        return self.spec_ids, self.labels


class SequenceSample(object):
    """
    Class representing a sequence sample.
    """
    def __init__(self, spec_ids: List[str], kappa_ms: np.ndarray, spec_db: Union[str, pd.DataFrame],
                 cand_db: Union[str, pd.DataFrame], cand_def="mass"):
        """
        :param data: list of strings, of length N, spectra ids that should be used for the sampling the sequences
        :param kappa_ms: array-like, shape=(N, N), kernel matrix encoding the similarity of the spectra. The order of the
            rows must correspond to the spectra ids, i.e. the i'th row K[i] corresponds to the spec_id[i].
        :param spec_db: string or pandas.DataFrame, Database storing information about the spectra (x_sigma), such as
            its retention time (t_sigma), its ground-truth label (y_sigma) and chromatographic configuration.
            string: filename of an SQLite DB storing the information
            DataFrame: pandas table storing the information
        :param cand_db: string or pandas.DataFrame, Database storing molecular candidates and their feature
            representation. The candidates are queried for each spectrum according to "cand_def".
        :param cand_def: string, which method should be used to define the candidate set for each sequence spectrum.
            "mass", by mass-window
            "mf", by molecular formula
            "fixed", loaded a pre-defined set
        """
        self.spec_ids = spec_ids
        assert pd.Series(self.spec_ids).is_unique, "Spectra IDs must be unique."
        self.specid_2_ind = {}
        for s, id in enumerate(self.spec_ids):
            self.specid_2_ind[s] = id

        self.kappa_ms = kappa_ms
        self.cand_def = cand_def
        assert self.cand_def in ["mass", "mf", "fixed"]

        self.spec_db = spec_db
        self.cand_db = cand_db

    def get_train_test_split(self):
        """
        75% training and 25% test split.
        """
        return next(self.get_train_test_generator(n_splits=4))

    def get_train_test_generator(self, n_splits=4):
        """
        Split the spectra ids into training and test sets. Thereby, all spectra belonging to the same molecular
        structure (regardless of their Massbank sub-dataset membership) are either in the test or training.

        Internally the scikit-learn function "GroupKFold" is used to split the data. As groups we use the molecular
        identifiers.

        :param n_splits: scalar, number of cross-validation splits.

        :yields:
            train : ndarray
                The training set indices for that split.

            test : ndarray
                The testing set indices for that split.
        """

        for train, test in GroupKFold(n_splits=n_splits).split(self.spec_ids, groups=self._load_mol_identifier()):
            # Get training and test subsets of the spectra ids. Spectra belonging to the same molecular structure are
            # either in the training or the test set.
            spec_ids_train = [self.spec_ids[i] for i in train]
            spec_ids_test = [self.spec_ids[i] for i in test]

            yield (
                SequenceSample(spec_ids_train, kappa_ms=self.kappa_ms[np.ix_(train, train)],
                               spec_db=self.spec_db, cand_db=self.cand_db, cand_def=self.cand_def),
                SequenceSample(spec_ids_test, kappa_ms=self.kappa_ms[np.ix_(test, train)],
                               spec_db=self.spec_db, cand_db=self.cand_db, cand_def=self.cand_def)
            )

    def get_n_samples(self):
        """
        Return the number of (spectrum, rt) examples. This refers to the number of unique spectra in the sample.

        :return: scalar, Number of (spectrum, rt) examples.
        """
        return len(self.spec_ids)

    def generate_sequences(self, N, L, rs=None, sort_sequence_by_rt=True):
        """
        Generate (spectra, rts, labels) sequences to train the StructuredSVM. In the main document we refer to this
        samples as:

            (x_i, t_i, y_i),

        with:

            x_i = (x_i1, ..., x_iL) ... being a list of spectra
            t_i = (t_i1, ..., t_iL) ... being the list of corresponding retention times
            y_i = (y_i1, ..., y_iL) ... being the list of corresponding ground-truth labels

        :param N: scalar, number of sequences
        :param L: scalar, length of the individual sequences
        :param rs:
        :param sort_sequence_by_rt: boolean, indicating, whether the sequence elements, i.e. x_i1, ..., should be
            sorted by their retention time.
        :return:
        """
        if hasattr(self, "spl_seqs"):
            # FIXME: Could we call the sample generation straight in the constructor?
            raise AssertionError("Sample sequences have already been generated and should be generated again.")

        # TODO: How to include the candidates here?
        data = pd.DataFrame({"spec_id": self.spec_ids,
                             "rt": self._load_retention_times(),
                             "mol_id": self._load_mol_identifiers(),
                             "mb_ds": self._load_dataset()})

        spl_seqs = []
        rs = check_random_state(rs)  # Type: np.random.RandomState
        for mb_ds in rs.choice(data["mb_ds"], N):
            seq = data[data["mb_ds"] == mb_ds].sample(L, random_state=rs)

            # Sort the sequence elements by their retention time
            if sort_sequence_by_rt:
                seq.sort_values(by="rt", inplace=True, ascending=True)

            spl_seqs.append([seq["spec_id"].to_list(), seq["rt"].to_list(), seq["mol_id"].to_list()])
            # FIXME: Here we can have multiple times the same molecule in the sample, e.g. due to different adducts.

            assert seq["mol_id"].is_unique, "Each molecule should appear only ones in the set of molecules."

        self.N = N
        self.L = L
        self.spl_seqs = spl_seqs

    def __len__(self):
        """
        :return: scalar, number of sample sequences
        """
        check_is_fitted(self, "spl_seqs")
        return len(self.spl_seqs)

    def as_Xy_input(self):
        check_is_fitted(self, "spl_seqs")
        x, rt, y = zip(*self.spl_seqs)
        return x, y

    def get_labelspace(self, i: int) -> List[List[str]]:
        cand_ids = []

        for sigma in range(self.L):
            cand_ids.append(self._get_cand_ids(self.spl_seqs[i][0][sigma]))

        return cand_ids

    def get_gt_labels(self, i: int) -> tuple:
        return tuple(self.spl_seqs[i][2][sigma] for sigma in range(self.L))

    def jointKernelMS(self, j_tau, i_sigma, y_j, y_i):
        """

        :param j:
        :param i:
        :param y_j:
        :param y_i:
        :return:
        """
        j, tau = j_tau
        assert tau is None  # HINT: We handle all tau at ones.
        i, sigma = i_sigma

        # TODO: Build fast index data-structure here.
        spec_ids_j = self.spl_seqs[j][0]  # spectra ids belonging to sequence j
        k_idx_j = [self.specid_2_ind[spec_id] for spec_id in spec_ids_j]  # spectra kernel columns belonging to seq. j

        spec_id_i_sigma = self.spl_seqs[i][0][sigma]
        k_idx_i_sigma = self.specid_2_ind[spec_id_i_sigma]

        K_ms = self.kappa_ms[k_idx_i_sigma][k_idx_j]  # shape=(1, L)

        self._get_mol_rep_MS(self.spl_seqs[i][2][sigma])  # y_sigma

    def delta_jointKernelMS(self, j_tau, i_sigma, y__j_ybar):
        """

        :param j_tau:
        :param i_sigma:
        :param y__j_ybar:
        :return:
        """
        j, tau = j_tau
        i, sigma = i_sigma

        # TODO: Build fast index data-structure here.
        spec_ids_j = self.spl_seqs[j][0]  # spectra ids belonging to sequence j
        k_idx_j = [self.specid_2_ind[spec_id] for spec_id in spec_ids_j]  # spectra kernel columns belonging to seq. j

        spec_ids_i = self.spl_seqs[i][0][sigma]
        k_ids_i = self.specid_2_ind[spec_ids_i]

        K_ms = self.kappa_ms[k_ids_i][k_idx_j]  # shape=(1, L)

        y__j_tau = self.get_gt_labels(j)[tau]
        psi_y__j_tau = self._get_mol_rep_MS(y__j_tau)
        psi_y__j_ybar = self._get_mol_rep_MS(y__j_ybar)

        Psi_y__y = self._get_mol_rep_MS(self.get_labelspace(i)[sigma])

        L_ms__j_tau = self.lambda_ms(Psi_y__y, psi_y__j_tau)  # shape=(L, 1)
        L_ms__j_ybar = self.lambda_ms(Psi_y__y, psi_y__j_ybar)  # shape=(L, 1)

        return K_ms @ (L_ms__j_tau - L_ms__j_ybar)


    # def __getitem__(self, item):
    #     """
    #
    #     :param item:
    #     :return:
    #     """
    #     check_is_fitted(self, "spl_seqs")
    #     return self.spec_ids[item]


if __name__ == "__main__":
    ss = SequenceSample(["A", "B", "C"], np.random.rand(3, 3), "", "")
    ss_train, ss_test = ss.get_train_test_split()
    ss_train.generate_sequences(100, 20)
