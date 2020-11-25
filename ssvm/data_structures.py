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
import sqlite3
import logging
import numpy as np
import pandas as pd
import networkx as nx

from collections import OrderedDict
from scipy.io import loadmat
from typing import List, Tuple, Union, Dict, Optional, Callable, Iterable
from networkx.classes.graph import EdgeView

from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import MiniBatchKMeans

from ssvm.kernel_utils import tanimoto_kernel, generalized_tanimoto_kernel

from matchms.Spectrum import Spectrum

# Setup the Logger
LOGGER = logging.getLogger(__name__)
CH = logging.StreamHandler()
FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)
LOGGER.addHandler(CH)


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


class Molecule(object):
    def __init__(self, inchi: str, inchikey: str, smiles_iso: str, smiles_can: str,
                 metadata: Optional[dict] = None, features: Optional[dict] = None, identifier="inchikey"):
        """

        :param inchi: string, InChI identifier of the molecule.

        :param inchikey: string, InChIKey identifier

        :param smiles_iso: string, Isomeric SMILES representation

        :param smiles_can: string, Canonical SMILES

        :param metadata: dictionary, any meta data provided with the molecule, e.g. PubChem ID etc.

        :param features: dictionary, feature calculated from the molecular representation. The keys should indicate the
            feature name and the values should be np.ndarray.
        """
        self.inchi = inchi
        self.inchikey = inchikey
        self.smiles_iso = smiles_iso
        self.smiles_can = smiles_can
        self.metadata = metadata
        self.features = features
        self.identifier = identifier

        self.inchikey1, self.inchikey2, self.inchikey3 = inchikey.split("-")

        if self.identifier not in ['inchi', 'inchikey', 'inchikey1']:
            raise ValueError("Invalid molecule identifier: '%s'. Choices are 'inchi', 'inchikey' and 'inchikey1'.")

    def get_identifier(self):
        """
        Return identifier string of the molecule.
        """
        return self.__getattribute__(self.identifier)


class CandidateSQLiteDB(object):
    def __init__(self, db_fn: str, cand_def: str = "fixed", molecule_identifier: str = "inchikey"):
        """
        :param db_fn:

        :param cand_def:

        :param molecule_identifier:
        """
        self.db_fn = db_fn
        self.cand_def = cand_def
        self.molecule_identifier = molecule_identifier

        # Open read-only database connection
        self.db = sqlite3.connect("file:" + self.db_fn + "?mode=ro", uri=True)

        if self.cand_def != "fixed":
            raise NotImplementedError("Currently only fixed candidate set definition supported.")

        # TODO: Do we need to close the connection here?
        # self._ensure_molecule_identifier_is_available(self.molecule_identifier)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit function for the context manager. Closes the connection to the candidate database.
        """
        self.close()

    def _get_labelspace_query(self, spectrum: Spectrum, candidate_subset: Optional[List] = None) -> str:
        """

        :param spectrum:
        :return:
        """
        query = "SELECT m.%s as identifier FROM candidates_spectra " \
                "   INNER JOIN molecules m ON m.inchi = candidates_spectra.candidate" \
                "   WHERE spectrum IS '%s'" % (self.molecule_identifier, spectrum.get("spectrum_id"))

        if candidate_subset is not None:
            query += " AND identifier IN %s" % self._in_sql(candidate_subset)

        query += "  GROUP BY identifier" \
                 "  ORDER BY identifier"

        return query

    def _get_molecule_feature_query(self, spectrum: Spectrum, feature: str, candidate_subset: Optional[List] = None) \
            -> str:
        """

        :param spectrum:
        :param feature:
        :param candidate_subset:
        :return:
        """
        query = "SELECT m.%s AS identifier, %s AS molecular_feature FROM candidates_spectra" \
                "   INNER JOIN molecules m ON m.inchi = candidates_spectra.candidate" \
                "   INNER JOIN fingerprints_data fd ON fd.molecule = candidates_spectra.candidate" \
                "   WHERE spectrum IS '%s'" % (self.molecule_identifier, feature, spectrum.get("spectrum_id"))

        if candidate_subset is not None:
            query += " AND identifier IN %s" % self._in_sql(candidate_subset)

        query += "  GROUP BY identifier" \
                 "  ORDER BY identifier"

        return query

    def _get_ms2_score_query(self, spectrum: Spectrum, ms2scorer: str, candidate_subset: Optional[List] = None) -> str:
        """

        :param spectrum:
        :param ms2scorer:
        :param candidate_subset:
        :return:
        """
        query = "SELECT m.%s AS identifier, score AS ms2_score" \
                "   FROM (SELECT * FROM candidates_spectra WHERE candidates_spectra.spectrum IS '%s') cs" \
                "   INNER JOIN molecules m ON m.inchi = cs.candidate" \
                "   LEFT OUTER JOIN (" \
                "       SELECT candidate, score FROM spectra_candidate_scores" \
                "       WHERE participant IS '%s' AND spectrum IS '%s') scs ON cs.candidate = scs.candidate" \
                % (self.molecule_identifier, spectrum.get("spectrum_id"), ms2scorer, spectrum.get("spectrum_id"))

        # Restrict candidate subset if needed
        if candidate_subset is not None:
            query += "   WHERE identifier in %s" % self._in_sql(candidate_subset)

        query += "  GROUP BY identifier" \
                 "  ORDER BY identifier"

        return query

    def _get_candidate_subset(self, spectrum: Spectrum) -> None:
        """
        The baseclass does not restrict the candidate set.
        """
        return None

    def _get_feature_dimension(self, feature: str) -> int:
        """
        :param feature: string, identifier of the requested molecule feature.

        :return: scalar, dimensionality of the feature
        """
        return self.db.execute("SELECT length FROM fingerprints_meta WHERE name IS '%s'" % (feature,)).fetchall()[0][0]

    def _ensure_feature_is_available(self, feature: str) -> None:
        """
        Raises an ValueError, if the requested molecular feature is not in the database.

        :param feature: string, identifier of the requested molecule feature.
        """
        if feature not in pd.read_sql_query("SELECT name FROM fingerprints_meta", self.db)["name"].to_list():
            raise ValueError("Requested feature is not in the database: '%s'." % feature)

    def _ensure_molecule_identifier_is_available(self, molecule_identifier: str):
        """
        Raises an ValueError, if the requested molecule identifier is not a column in the molecule table of the database.

        :param molecule_identifier: string, column name of the molecule identifier
        """
        if molecule_identifier not in pd.read_sql_query("SELECT * FROM molecules LIMIT 1", self.db).columns:
            raise ValueError("Molecule identifier '%s' is not available." % molecule_identifier)

    def _get_molecule_feature_matrix(self, df_features: pd.DataFrame, features: str) -> np.ndarray:
        """
        Function to parse the molecular features stored as strings in the candidate database and load them into a numpy
        array.

        :param df_features: pandas.DataFrame, table with two columns (identifier, molecular_feature).

        :param features: string, identifier of the molecular feature. This information is needed to properly parse the
            feature string representation and store it in a numpy array.

        :return:
        """
        # Set up an output array
        n = len(df_features)
        d = self._get_feature_dimension(features)
        X = np.zeros((n, d))

        if features == "substructure_count":
            for i, row in enumerate(df_features["molecular_feature"]):
                _fp = eval("{" + row + "}")
                X[i, list(_fp.keys())] = list(_fp.values())
        else:
            for i, row in enumerate(df_features["molecular_feature"]):
                _ids = list(map(int, row.split(",")))
                X[i, _ids] = 1

        return X

    def close(self) -> None:
        """
        Closes the database connection.
        """
        self.db.close()

    def get_n_cand(self, spectrum: Spectrum) -> int:
        """
        Return the number of available candidates for the given spectrum.

        :param spectrum: matchms.Spectrum, of the sequence element to get the number of candidates.

        :return: scalar, number of candidates
        """
        return len(self.get_labelspace(spectrum))

    def get_molecule_features(self, spectrum: Spectrum, features: str, return_dataframe: bool = False) \
            -> Union[np.ndarray, pd.DataFrame]:
        """
        :param spectrum: matchms.Spectrum, of the sequence element to get the molecular features.

        :param features: string, identifier of the feature to load from the database.

        :param return_dataframe: boolean, indicating whether the score should be returned in a pandas.DataFrame storing
            the molecule ids and molecule features (shape = (n_samples, d_features + 1)).

        :return: array-like, shape = (n_cand, feature_dimension), feature matrix
        """
        self._ensure_feature_is_available(features)

        df_features = pd.read_sql_query(
            self._get_molecule_feature_query(spectrum, features, self._get_candidate_subset(spectrum)), self.db)

        feature_matrix = self._get_molecule_feature_matrix(df_features, features)

        if return_dataframe:
            df_features = pd.concat((df_features["identifier"], pd.DataFrame(feature_matrix)), axis=1)
        else:
            df_features = feature_matrix

        return df_features

    def get_molecule_features_by_molecule_id(self, molecule_ids: Union[List[str], Tuple[str, ...]], features: str,
                                             return_dataframe: bool = False) -> Union[pd.DataFrame, np.ndarray]:
        """
        Load the molecular features for the specified molecule identifiers from the database.

        :param molecule_ids: list or tuple of strings, molecule identifier

        :param features: string, molecular feature identifier

        :param return_dataframe: boolean, indicating whether the score should be returned in a pandas.DataFrame storing
            the molecule ids and molecule features (shape = (n_samples, d_features + 1)).

        :return: array-like, shape = (n_mol, feature_dimension), feature matrix. Each row i corresponds to molecule i.
        """
        self._ensure_feature_is_available(features)

        df_features = pd.read_sql_query(
            "SELECT %s AS identifier, %s AS molecular_feature FROM molecules"
            "   INNER JOIN fingerprints_data fd ON fd.molecule = molecules.inchi"
            "   WHERE identifier IN %s" % (self.molecule_identifier, features, self._in_sql(molecule_ids)),
            self.db, index_col="identifier")

        # Ensure feature rows have the same order as the _input_ molecule identifier
        df_features = df_features.loc[molecule_ids].reset_index()

        feature_matrix = self._get_molecule_feature_matrix(df_features, features)

        if return_dataframe:
            df_features = pd.concat((df_features["identifier"], pd.DataFrame(feature_matrix)), axis=1)
        else:
            df_features = feature_matrix

        return df_features

    def get_labelspace(self, spectrum: Spectrum, candidate_subset: Optional[List] = None) -> List[str]:
        """
        Returns the label-space for the given spectrum.

        :param spectrum: matchms.Spectrum, of the sequence element to get the label space.

        :param candidate_subset:

        :return: list of strings, molecule identifiers for the given spectrum.
        """
        return [row[0] for row in self.db.execute(self._get_labelspace_query(spectrum, candidate_subset))]

    def get_ms2_scores(self, spectrum: Spectrum, ms2scorer: str, min_score_value: float = 0.0,
                       max_score_value: float = 1.0, return_dataframe: bool = False) \
            -> Union[pd.DataFrame, List[float]]:
        """

        :param spectrum: matchms.Spectrum, of the sequence element to get the MS2 scores.

        :param ms2scorer: string, identifier of the MS2 scoring method for which the scores should be loaded from the
            database.

        :param min_score_value: scalar, minimum output value of the scaled scores.

        :param return_dataframe: boolean, indicating whether the score should be returned in a two-column
            pandas.DataFrame storing the molecule ids and MS2 scores.

        :return: list of MS2 scores, from the range [0, 1] (default) or [min_score_value, 1]
        """
        # Load the MS2 scores
        df_scores = pd.read_sql_query(
            self._get_ms2_score_query(spectrum, ms2scorer, self._get_candidate_subset(spectrum)), self.db)

        # Fill missing MS2 scores with the minimum score
        min_score = df_scores["ms2_score"].min(skipna=True)
        df_scores["ms2_score"] = df_scores["ms2_score"].fillna(value=min_score)

        # Scale scores to [min_score_value, 1]
        df_scores["ms2_score"] = MinMaxScaler(feature_range=(min_score_value, max_score_value)) \
            .fit_transform(df_scores["ms2_score"][:, np.newaxis])

        if not return_dataframe:
            df_scores = df_scores["ms2_score"].tolist()

        return df_scores

    @staticmethod
    def _in_sql(li: Union[List[str], Tuple[str, ...]]) -> str:
        """
        Concatenates a list of strings to a SQLite ready string that can be used in combination
        with the 'in' statement.

        E.g.:
            ["house", "boat", "donkey"] --> "('house', 'boat', 'donkey')"


        :param li: list of strings

        :return: SQLite ready string for 'in' statement
        """
        return "(" + ",".join(["'%s'" % li for li in np.atleast_1d(li)]) + ")"


class RandomSubsetCandidateSQLiteDB(CandidateSQLiteDB):
    def __init__(self, number_of_candidates: Union[int, float], include_correct_candidate: bool = False,
                 random_state: Optional[Union[int, np.random.RandomState]] = None, *args, **kwargs):
        """
        :param number_of_candidates: scalar, If integer: minimum number of candidates per spectrum. If float, the
            scalar represents the fraction of candidates per spectrum.

        :param include_correct_candidate: boolean, indicating whether the correct candidate should be kept in the
            candidate list.

        :param random_state:

        :param kwargs: dict, arguments passed to CandidateSQLiteDB
        """
        self.number_of_candidates = number_of_candidates
        self.include_correct_candidate = include_correct_candidate
        self.random_state = random_state

        self._labelspace_subset = {}

        if isinstance(self.number_of_candidates, float):
            if (self.number_of_candidates <= 0) or (self.number_of_candidates >= 1.0):
                raise ValueError("If the number of candidates is given as fraction it must lay in the range (0, 1).")

        super().__init__(*args, **kwargs)

    def _get_candidate_subset(self, spectrum: Spectrum) -> List[str]:
        """
        :param spectrum: matchms.Spectrum, of the sequence element to get the number of candidates.

        :return: list of strings, molecule identifier for the candidate subset of the given spectrum (not sorted !).
        """
        spectrum_id = spectrum.get("spectrum_id")

        try:
            candidates_sub = self._labelspace_subset[spectrum_id]
        except KeyError:
            # Load all candidate identifiers
            candidates_all = super().get_labelspace(spectrum, candidate_subset=None)

            if isinstance(self.number_of_candidates, float):
                n_cand = np.round(self.number_of_candidates * len(candidates_all))
            else:
                n_cand = np.minimum(len(candidates_all), self.number_of_candidates)

            # Get a random subset
            candidates_sub = check_random_state(self.random_state).choice(candidates_all, n_cand.astype(int), replace=False)

            if self.include_correct_candidate:
                molecule_id = spectrum.get("molecule_id")

                if molecule_id is None:
                    raise ValueError("Cannot ensure that the ground truth structure is included in the candidate set, "
                                     "as no 'molecule_id' is specified in the Spectrum object.")

                if molecule_id not in candidates_all:
                    raise ValueError("The molecule id '%s' is not in the candidate set." % molecule_id)

                if molecule_id not in candidates_sub:
                    candidates_sub[0] = molecule_id

            # Store the subset for later use
            self._labelspace_subset[spectrum_id] = candidates_sub.tolist()

        return candidates_sub

    def get_n_cand(self, spectrum: Spectrum) -> int:
        """
        Return the number of candidates in the random label space subset for the given spectrum.
        """
        return len(self.get_labelspace(spectrum))

    def get_labelspace(self, spectrum: Spectrum, candidate_subset: Optional[List] = None) -> List[str]:
        """
        Return the label space of the random subset.
        """
        if candidate_subset is not None:
            raise RuntimeError("Candidate subset cannot be requested in any sub-class of 'CandidateSQLiteDB'.")

        return super().get_labelspace(spectrum, self._get_candidate_subset(spectrum))


# class CentroidCandidateSQLiteDB(CandidateSQLiteDB):
#     def __init__(self, feature: str, number_of_candidates: Union[int, float], random_state=None, *args, **kwargs):
#         self.feature = feature
#         self.number_of_candidates = number_of_candidates
#         self.random_state = random_state
#
#         self.candidate_subset = {}
#
#         if isinstance(self.number_of_candidates, float):
#             if (self.number_of_candidates <= 0) or (self.number_of_candidates >= 1.0):
#                 raise ValueError("If the number of candidates is given as fraction it must lay in the range (0, 1).")
#
#         super().__init__(*args, **kwargs)
#
#         self._ensure_feature_is_available(self.feature)
#
#     def _get_candidate_subset(self, spectrum: Spectrum) -> List[str]:
#         """
#         :param spectrum: matchms.Spectrum, of the sequence element to get the number of candidates.
#
#         :return: list of strings, molecule identifier for the candidate subset of the given spectrum.
#         """
#         spectrum_id = spectrum.get("spectrum_id")
#
#         try:
#             candidates = self.candidate_subset[spectrum_id]
#         except KeyError:
#             # Load all candidate identifiers
#             candidates = [row[0] for row in self.db.execute(self._get_labelspace_query(spectrum))]
#
#             if isinstance(self.number_of_candidates, float):
#                 n_cand_out = np.round(len(candidates) * self.number_of_candidates)
#             else:
#                 n_cand_out = np.minimum(len(candidates), self.number_of_candidates)
#
#             # Load candidate features
#             rows = self.db.execute(self._get_molecule_feature_query(spectrum, self.feature))
#             n_cand_all = super().get_n_cand(spectrum)
#             d = self._get_feature_dimension(self.feature)
#             KX = generalized_tanimoto_kernel(
#                 CandidateSQLiteDB._get_molecule_feature_matrix(rows, (n_cand_all, d), self.feature))
#
#             # Cluster the candidates
#             cls = MiniBatchKMeans(
#                 n_clusters=n_cand_out.astype(int), n_init=10, random_state=self.random_state, init='k-means++') \
#                 .fit_predict(KX)
#
#             centroids = []
#             idc = np.arange(len(cls))
#             for c in np.unique(cls):
#                 centroids.append(idc[cls == c][np.argsort(np.median(KX[np.ix_(cls == c, cls == c)], axis=0))[0]])
#             candidates = [candidates[i] for i in centroids]
#
#             # Store the subset for later use
#             self.candidate_subset[spectrum_id] = candidates
#
#         return candidates
#
#     def get_n_cand(self, spectrum: Spectrum) -> int:
#         """
#
#         :param spectrum:
#         :return:
#         """
#         return len(self.candidate_subset[spectrum.get("spectrum_id")])


class Sequence(object):
    def __init__(self, spectra: List[Spectrum], candidates: CandidateSQLiteDB, ms2scorer: Optional[str] = None):
        self.spectra = spectra
        self.candidates = candidates
        self.ms2scorer = ms2scorer

        self.L = len(self.spectra)

        # Extract retention time (RT) information from spectra and pre-calculate RT differences and signs
        self._rts = np.array([spectrum.get("retention_time") for spectrum in self.spectra])
        self._rt_differences = self._rts[:, np.newaxis] - self._rts[np.newaxis, :]
        self._rt_diff_signs = np.sign(self._rt_differences)

    def __len__(self) -> int:
        """
        Return the length of the sequence.

        :return: scalar, length of the sequence
        """
        return self.L

    def get_retention_time(self, s: Optional[int] = None):
        if s is None:
            return [self.get_retention_time(s) for s in range(self.__len__())]
        else:
            return self._rts[s]

    def get_molecule_features_for_candidates(self, features: str, s: Optional[int] = None) \
            -> Union[List[np.ndarray], np.ndarray]:
        """
        Returns the molecular features for all candidates of the spectra in the sequence.

        :param s: scalar, index of the spectrum for which candidate features should be loaded. If None, features for all
            candidates are loaded.

        :param features: string, identifier of the feature to load from the database.

        :return: array-like or list of array-likes, (list of) feature matrix (matrices)
            with shape = (n_cand_s, feature_dimension)
        """
        if s is None:
            return [self.get_molecule_features_for_candidates(features, s) for s in range(self.__len__())]
        else:
            return self.candidates.get_molecule_features(self.spectra[s], features)

    def get_n_cand(self, s: Optional[int] = None) -> Union[int, List[int]]:
        """
        Get the number of candidates for each spectrum in the sequence as list.

        :param s: scalar, sequence index for which the MS2 scores should be returned. If None, scores are returned for
            all spectra in the sequence.
        """
        if s is None:
            n_cand = [self.get_n_cand(s) for s in range(self.__len__())]
        else:
            n_cand = self.candidates.get_n_cand(self.spectra[s])

        return n_cand

    def get_labelspace(self, s: Optional[int] = None) -> Union[List[List[str]], List[str]]:
        """
        Get the label space, i.e. candidate molecule identifiers, for each spectrum as nested list.

        :param s: scalar, sequence index for which the MS2 scores should be returned. If None, scores are returned for
            all spectra in the sequence.
        """
        if s is None:
            labelspace = [self.get_labelspace(s) for s in range(self.__len__())]
        else:
            labelspace = self.candidates.get_labelspace(self.spectra[s])

        return labelspace

    def get_ms2_scores(self, s: Optional[int] = None) -> Union[List[List[float]], List[float]]:
        """
        Get the MS2 scores for the given index

        :param s: scalar, sequence index for which the MS2 scores should be returned. If None, scores are returned for
            all spectra in the sequence.
        """
        if self.ms2scorer is None:
            raise ValueError("No MS2 scorer specified!")

        if s is None:
            ms2_scores = [self.get_ms2_scores(s) for s in range(self.__len__())]
        else:
            ms2_scores = self.candidates.get_ms2_scores(self.spectra[s], self.ms2scorer)

        return ms2_scores

    def get_sign_delta_t(self, G: nx.Graph) -> np.ndarray:
        """
        :param G: networkx.Graph, representing the MRF or an tree-like approximation of it.

        :return: array-like, shape = (|E|,), sign of the retention time differences for all edges.
        """
        bS, bT = zip(*G.edges)
        return self._rt_diff_signs[list(bS), list(bT)]


class LabeledSequence(Sequence):
    """
    Class representing the a _labeled_ (MS, RT)-sequence (x, t, y) with associated molecular candidate set C.
    """
    def __init__(self, spectra: List[Spectrum], labels: List[str], candidates: CandidateSQLiteDB,
                 ms2scorer: Optional[str] = None):
        """
        :param spectra: list of strings, spectrum-ids belonging sequence
        :param labels: list of strings, ground truth molecule identifiers belonging to the spectra of the sequence
        """
        self.labels = labels

        super(LabeledSequence, self).__init__(spectra=spectra, candidates=candidates, ms2scorer=ms2scorer)

    def as_Xy_input(self) -> Tuple[List[Spectrum], List[str]]:
        """
        Return the (MS, RT)-sequence and ground truth label separately as input for the sklearn interface.

        Usage: sklearn.fit(*LabeledSequence(...).as_Xy_input)

        :return:
        """
        return self.spectra, self.labels

    def get_index_of_correct_structure(self, s: Optional[int] = None) -> Union[List[int], int]:
        """
        Return the index of candidate those molecule identifier matches the label of the label

        :param s: scalar, sequence index for which the index of scores should be returned. If None, scores are returned
            for all spectra in the sequence.

        :return: scalar, first index of the label, i.e. correct molecular structure, in the label-space.

        :raises: ValueError if the label could not be found in the label space
        """
        if s is None:
            return [self.get_index_of_correct_structure(s) for s in range(self.__len__())]
        else:
            return self.get_labelspace(s).index(self.labels[s])

    def get_label_loss(self, label_loss_fun: Callable[[np.ndarray, np.ndarray], np.ndarray], features: str,
                       s: Optional[int] = None) \
            -> Union[List[np.ndarray], np.ndarray]:
        """


        :param s: scalar, sequence index for which the label loss should be calculate. If None, losses are calculated
            for all spectra in the sequence.

        :param features: string, identifier of the molecular feature used to calculate the label loss. The feature must
            be available in the candidate database.

        :param label_loss_fun: Callable,

        :return:
        """
        if s is None:
            return [self.get_label_loss(label_loss_fun, features, s) for s in range(self.__len__())]
        else:
            Y = self.get_molecule_features_for_candidates(features, s)
            return label_loss_fun(Y[self.get_index_of_correct_structure(s)], Y)

    def get_molecule_features_for_labels(self, features: str) -> np.ndarray:
        """

        :param features:
        :return:
        """
        return self.candidates.get_molecule_features_by_molecule_id(self.labels, features)


class SequenceSample(object):
    """
    Class representing a sequence sample.
    """
    def __init__(self, spectra: List[Spectrum], labels: List[str], candidates: CandidateSQLiteDB, N: int, L_min: int,
                 L_max: Optional[int] = None, random_state: Optional[int] = None, sort_sequence_by_rt: bool = False, 
                 ms2scorer: Optional[str] = None):
        """
        :param data: list of matchms.Spectrum, spectra to sample sequences from

        :param labels: list of strings, labels of each spectrum identifying the ground truth molecule.

        :param candidates:

        :param N: scalar, number of sequences to sample.

        :param L_min: scalar, minimum length of the individual sequences

        :param L_max: scalar, maximum length of the individual sequences

        :param random_state:
        """
        self.spectra = spectra
        self.labels = labels
        self.candidates = candidates
        self.N = N
        self.L_min = L_min
        self.L_max = L_max
        self.random_state = random_state
        self.sort_sequence_by_rt = sort_sequence_by_rt
        self.ms2scorer = ms2scorer

        assert pd.Series([spectrum.get("spectrum_id") for spectrum in self.spectra]).is_unique, \
            "Spectra IDs must be unique."
        assert self.L_min > 0

        if self.L_max is None:
            self._L = np.full(self.N, fill_value=self.L_min)
        else:
            assert self.L_min < self.L_max
            self._L = check_random_state(self.random_state).randint(self.L_min, self.L_max + 1, self.N)

        # Extract information from the spectra to which dataset they each belong
        self._datasets = []
        self._dataset2idx = OrderedDict()
        for idx, spectrum in enumerate(self.spectra):
            self._datasets.append(spectrum.get("dataset"))
            try:
                self._dataset2idx[self._datasets[-1]].append(idx)
            except KeyError:
                self._dataset2idx[self._datasets[-1]] = [idx]
        self._n_spec_per_dataset = {ds: len(self._dataset2idx[ds]) for ds in self._dataset2idx}

        LOGGER.info("Number of datasets: %d" % len(self._dataset2idx))
        for k, v in self._n_spec_per_dataset.items():
            LOGGER.info("Dataset '%s' contains '%d' spectra." % (k, v))

        # Generate (MS, RT)-sequences
        self._sampled_sequences = self._sample_sequences()

    def __len__(self):
        """
        :return: scalar, number of sample sequences
        """
        return len(self._sampled_sequences)

    def __iter__(self):
        return iter(self._sampled_sequences)

    def __getitem__(self, item):
        return self._sampled_sequences[item]

    def _sample_sequences(self):
        """
        Generate (spectra, rts, labels) sequences to train the StructuredSVM. In the main document we refer to this
        samples as:

            (x_i, t_i, y_i),

        with:

            x_i = (x_i1, ..., x_iL) ... being a list of spectra
            t_i = (t_i1, ..., t_iL) ... being the list of corresponding retention times
            y_i = (y_i1, ..., y_iL) ... being the list of corresponding ground-truth labels

        :return:
        """
        rs = check_random_state(self.random_state)  # type: np.random.RandomState

        spl_seqs = []
        for i, ds in enumerate(rs.choice(self._datasets, self.N)):
            seq_idc = rs.choice(self._dataset2idx[ds], self._L[i], replace=False)
            seq_spectra = [self.spectra[sig] for sig in seq_idc]
            seq_labels = [self.labels[sig] for sig in seq_idc]

            # FIXME: Here we can have multiple times the same molecule in the sample, e.g. due to different adducts.
            assert pd.Series(seq_labels).is_unique, "Each molecule should appear only ones in the set of molecules."

            # Sort the sequence elements by their retention time
            if self.sort_sequence_by_rt:
                seq_spectra, seq_labels = zip(*sorted(zip(seq_spectra, seq_labels),
                                                      key=lambda s: s[0].get("retention_time")))

            spl_seqs.append(LabeledSequence(seq_spectra, seq_labels, candidates=self.candidates,
                                            ms2scorer=self.ms2scorer))

        return spl_seqs

    def get_train_test_split(self, n_splits=4):
        """
        75% training and 25% test split.
        """
        return next(self.get_train_test_generator(n_splits=n_splits))

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
        for train, test in GroupKFold(n_splits=n_splits).split(self.spectra, groups=self.labels):
            N_test = self.N // n_splits
            N_train = self.N - N_test

            # Get training and test subsets of the spectra ids. Spectra belonging to the same molecular structure are
            # either in the training or the test set.
            yield (
                SequenceSample([self.spectra[i] for i in train], [self.labels[i] for i in train],
                               candidates=self.candidates, N=N_train, L_min=self.L_min, L_max=self.L_max,
                               random_state=self.random_state, ms2scorer=self.ms2scorer),
                SequenceSample([self.spectra[i] for i in test], [self.labels[i] for i in test],
                               candidates=self.candidates, N=N_test, L_min=self.L_min, L_max=self.L_max,
                               random_state=self.random_state, ms2scorer=self.ms2scorer)
            )

    def get_n_samples(self):
        """
        Return the number of (spectrum, rt) examples. This refers to the number of unique spectra in the sample.

        :return: scalar, Number of (spectrum, rt) examples.
        """
        return len(self.spectra)

    def get_labelspace(self, idx: Optional[int] = None) -> Union[List[List[str]], List[List[List[str]]]]:
        """
        Returns the label space for all sequence samples or the sequence at index 'i'.

        :param idx: scalar, index of the sequence to get the label space for. If None, all label spaces are returned.
        """
        if idx is None:
            labelspace = []
            for seq in self.__iter__():
                labelspace.append(seq.get_labelspace())
        else:
            labelspace = self.__getitem__(idx).get_labelspace()

        return labelspace

    def as_Xy_input(self):
        x, rt, y = zip(*self._sampled_sequences)
        return x, y


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


if __name__ == "__main__":
    ss = SequenceSample(["A", "B", "C"], np.random.rand(3, 3), "", "")
    ss_train, ss_test = ss.get_train_test_split()
    ss_train._sample_sequences(100, 20)
