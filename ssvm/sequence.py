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
import pandas as pd

from typing import List, Tuple, Union
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_random_state, check_is_fitted


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
    def __init__(self, spec_ids: List[str], K_ms: np.ndarray, spec_db: Union[str, pd.DataFrame],
                 cand_db: Union[str, pd.DataFrame], cand_def="mass"):
        """
        :param data: list of strings, of length N, spectra ids that should be used for the sampling the sequences
        :param K_ms: array-like, shape=(N, N), kernel matrix encoding the similarity of the spectra. The order of the
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

        self.K_ms = K_ms
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
                SequenceSample(spec_ids_train, K_ms=self.K_ms[np.ix_(train, train)],
                               spec_db=self.spec_db, cand_db=self.cand_db, cand_def=self.cand_def),
                SequenceSample(spec_ids_test, K_ms=self.K_ms[np.ix_(test, train)],
                               spec_db=self.spec_db, cand_db=self.cand_db, cand_def=self.cand_def)
            )

    def get_n_samples(self):
        """
        Return the number of (spectrum, rt) examples. This refers to the number of unique spectra in the sample.

        :return: scalar, Number of (spectrum, rt) examples.
        """
        return len(self.spec_ids)

    def generate_sequences(self, N, L, rs=None):
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
            spl_seqs.append(zip(seq["spec_id"].to_list(), seq["rt"].to_list(), seq["mol_id"].to_list()))
            # FIXME: Here we can have multiple times the same molecule in the sample, e.g. due to different adducts.

            assert seq["mol_id"].is_unique, "Each molecule should appear only ones in the set of molecules."

        self.spl_seqs = spl_seqs

    def __len__(self):
        """
        :return: scalar, number of sample sequences
        """
        check_is_fitted(self, "spl_seqs")
        return len(self.spl_seqs)



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
