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
import logging
import numpy as np
import pandas as pd
import networkx as nx

from abc import ABC, abstractmethod
from functools import lru_cache
from copy import deepcopy
from collections import OrderedDict
from typing import List, Tuple, Union, Dict, Optional, Callable, Iterator, TypeVar

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.utils.validation import check_random_state
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from matchms.Spectrum import Spectrum

import ssvm.cfg
from ssvm.factor_graphs import get_random_spanning_tree

# Setup the Logger
LOGGER = logging.getLogger(__name__)
CH = logging.StreamHandler()
FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)
LOGGER.addHandler(CH)

SEQUENCE_SAMPLE_T = TypeVar('SEQUENCE_SAMPLE_T', bound='SequenceSample')


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


class ABCCandSQLiteDB(ABC):
    """
    Wrapper class around the candidate SQLite database (DB).

    The database is assumed to hold the following information:

        - (some kind of) in-silico matching scores for the (MS/MS-spectrum, molecular-candidate)-tuples
        - Retention time (RT) information for MS/MS spectra
        - Grouping information for the MS/MS spectra, e.g. their association with a dataset (= measurement setup)

    The wrapper class provides high-level accessing / query functions to retrieve information from DB needed for the
    Structure Support Vector Machine (SSVM) training respectively running the evaluation.
    """
    def __init__(
            self, db_fn: str, cand_def: str = "fixed", molecule_identifier: str = "inchikey",
            init_with_open_db_conn: bool = True,
            feature_transformer: Union[Dict[str, Union[Pipeline, BaseEstimator]], Union[Pipeline, BaseEstimator]] = None
    ):
        """
        :param db_fn: string, pathname of the candidate database.

        :param cand_def: string, specifying the candidate set definition:

            'fixed': Candidate sets are pre-defined. This requires the presence of a DB table establishing the relation-
                ship between the MS/MS-spectra and the their corresponding candidates. This association table is
                assumed to be called 'candidates_spectra'.

        :param molecule_identifier: string, specifying the name of the identifier used to distinguish between two
            different molecules (~ candidates) in the DB. For example, if 'inchikey's are used, than two molecules are
            considered to be different, if there 'inchikey' differs. If otherwise, the identifier is specified to be
            'inchikey1' than only the first part of the inchikey of two molecules is considered, which for example
            results in two stereo-isomers being considered equal. If multiple molecules (~ candidates) in the DB have
            the same identifier, than for example their MS/MS scores are aggregated (maximum score is returned), when
            the MS/MS scores are queried from the DB.

        :param init_with_open_db_conn: boolean, indicating whether a connection to the database should be established
            (i.e. an SQLite connection is opened) when the DB class is instantiated. If False, the connections needs to
            be established whenever any information is queried from the DB (e.g. using the implemented context-manager).
            This is useful, if the DB is shared across parallel processes using joblib.Parallel (with loky backend),
            where the data to be shared across the processes needs to be pickleable (which sqlite3.connections are not).

        :param feature_transformer: Dict or Transformer (Pipeline),
        """
        self.db_fn = db_fn
        self.cand_def = cand_def
        self.molecule_identifier = molecule_identifier
        self.init_with_open_db_conn = init_with_open_db_conn
        self.feature_transformer = deepcopy(feature_transformer)

        if self.init_with_open_db_conn:
            self.db = self.connect_to_db()
        else:
            self.db = None

        if self.cand_def != "fixed":
            raise NotImplementedError("Currently only fixed candidate set definition supported.")

        # Get the set of available "molecule identifiers" and "molecular features" from the DB
        with self:
            # --------------------
            # Molecule identifiers
            self.available_molecule_identifier = set(
                row[1] for row in self.db.execute("PRAGMA table_info(molecules)")
            )
            # --------------------

            # ------------------
            # Molecular features
            self.available_molecular_features = dict()

            # Get the available tables in the database
            table_names, = zip(*self.db.execute("SELECT name FROM sqlite_schema").fetchall())

            for feature_table in ["fingerprints_meta", "descriptors_meta"]:
                if feature_table not in table_names:
                    continue

                for row in self.db.execute("SELECT name FROM %s" % feature_table):
                    self.available_molecular_features[row[0]] = feature_table
            # --------------------

            # -------------------
            # MS2 scoring methods
            self.available_ms2_scores = self._get_available_ms2scorer()
            # -------------------

        # Make sure the requested molecular identifier is in the DB
        with self:
            self._ensure_molecule_identifier_is_available(self.molecule_identifier)

    @abstractmethod
    def _get_labelspace_query(self, spectrum: Spectrum, candidate_subset: Optional[List] = None) -> str:
        """
        Function to construct the SQLite query to load the label-space (= molecule identifiers of the candidates)
        associated with the given MS/MS spectrum.

        Note: The query always request the result to be "ORDER BY identifier", which means that the label-space is
            ordered by the molecule identifier.

        :param spectrum: matchms.Spectrum, MS/MS spectrum for which the label-space should be loaded.

        :return: string, SQLite query
        """
        pass

    @abstractmethod
    def _get_molecule_feature_query(
            self, spectrum: Spectrum, feature: str, feature_table: str, candidate_subset: Optional[List] = None
    ) -> str:
        """
        Function to construct the SQLite query to load the molecule features (e.g. fingerprint vectors) of the
        candidates associated with the given MS/MS spectrum.

        Note: The query always request the result to be "ORDER BY identifier", which means that the label-space is
            ordered by the molecule identifier.

        Note (2): We use the "GROUP BY identifier" clause to group the features associated with different candidates
            which have the same molecule identifier. The SQLite implementation does not ensure any particular order of
            the results and _any_ of the feature vector within a group could be returned. Keep this in mind, when your
            features encode properties of the molecules within a group differently (e.g. stereo-chemistry).

        :param spectrum: matchms.Spectrum, MS/MS spectrum for which the molecule features should be loaded.

        :param feature: string, name of the molecule feature to load.

        :param feature_table: string, name of the meta-table in which the feature was found.

        :param candidate_subset: list of strings, molecule identifiers to restrict the candidate set for which the
            features are returned.

        :return: string, SQLite query
        """
        pass

    @abstractmethod
    def _get_ms2_score_query(self, spectrum: Spectrum, ms2scorer: str, candidate_subset: Optional[List[str]] = None) \
            -> str:
        """
        Function to construct the SQLite query to load the MS/MS spectra in-silico scores of the candidates associated
        with the given MS/MS spectrum.

        Note: The query always request the result to be "ORDER BY identifier", which means that the label-space is
            ordered by the molecule identifier.

        Note (2): We use the "GROUP BY identifier" clause to group the MS/MS scores associated with different candidates
            which have the same molecule identifier. We return the maximum MS/MS score within each group.

        Note (3): If a candidate is associated with a spectrum, but has not been scored by the specified in-silico
            method, its MS/MS score is returned as None.

        :param spectrum: matchms.Spectrum, MS/MS spectrum for which the MS/MS scores should be loaded.

        :param ms2scorer: string, identifier of the in-silico method used to produce the MS/MS scores.

        :param candidate_subset: list of strings, molecule identifiers to restrict the candidate set for which the
            features are returned.

        :return: string, SQLite query
        """
        pass

    @abstractmethod
    def _get_molecule_feature_by_id_query(
            self, molecule_ids: Union[List[str], Tuple[str, ...]], features: str, feature_table: str
    ) -> str:
        """
        Function to construct the SQLite query to load the molecule features (e.g. fingerprint vectors) for the given
        molecule identifier.

        Note: The rows will be ordered as given in the molecule ids.

        Note (2): We use the "GROUP BY identifier" clause to group the features associated with different candidates
            which have the same molecule identifier. The SQLite implementation does not ensure any particular order of
            the results and _any_ of the feature vector within a group could be returned. Keep this in mind, when your
            features encode properties of the molecules within a group differently (e.g. stereo-chemistry).
        """
        pass

    @abstractmethod
    def _get_candidate_subset(self, spectrum: Spectrum) -> Optional[List[str]]:
        """
        Returns the candidate subset for the given spectrum. In an actual implementation this could be for example a
        random candidate subset.
        """
        pass

    def set_feature_transformer(
            self,
            feature_transformer: Union[Dict[str, Union[Pipeline, BaseEstimator]], Union[Pipeline, BaseEstimator]] = None
    ):
        """
        Function to modify the feature transformer of the candidate DB wrapper.
        """
        self.feature_transformer = deepcopy(feature_transformer)

    def get_feature_transformer(self) -> Union[Dict[str, Union[Pipeline, BaseEstimator]], Union[Pipeline, BaseEstimator]]:
        """
        Function to access the feature transformer of the candidate DB wrapper.
        """
        return deepcopy(self.feature_transformer)

    def _get_d_and_mode_feature(self, feature: str, feature_table: str) -> Tuple[int, str]:
        """
        :param feature: string, identifier of the requested molecule feature.

        :param feature_table: string, name of the meta-table in which the feature was found.

        :return: Tuple = (scalar, string), dimensionality of the feature and feature mode
        """
        return self.db.execute(
            "SELECT length, mode FROM %s WHERE name IS ?" % feature_table, (feature, )
        ).fetchone()

    def _get_d_feature(self, feature: str, feature_table: str) -> int:
        """
        :param feature: string, identifier of the requested molecule feature.

        :param feature_table: string, name of the meta-table in which the feature was found.

        :return: scalar, dimensionality of the feature
        """
        return self._get_d_and_mode_feature(feature, feature_table)[0]

    def _get_mode_feature(self, feature: str, feature_table: str) -> str:
        """
        :param feature: string, identifier of the requested molecule feature.

         :param feature_table: string, name of the meta-table in which the feature was found.

        :return: string, data-type mode of the feature
        """
        return self._get_d_and_mode_feature(feature, feature_table)[1]

    @abstractmethod
    def _get_available_ms2scorer(self) -> set:
        """
        Function returning all MS2 scorer available in the database.
        """
        pass

    def _ensure_ms2scorer_is_available(self, ms2scorer: str) -> None:
        """
        Raises an ValueError, if the requested MS2 scorer is not in the database.

        :param ms2scorer: string, identifier of the requested MS2 scorer
        :raises: ValueError
        """
        if ms2scorer not in self.available_ms2_scores:
            raise ValueError("Requested MS2 scorer is not in the database: '%s'." % ms2scorer)

    def _ensure_feature_is_available(self, feature: str) -> str:
        """
        Raises an ValueError, if the requested molecular feature is not in the database.

        :param feature: string, identifier of the requested molecule feature.

        :return: string, table in which the feature has been found

        :raises: ValueError
        """
        try:
            return self.available_molecular_features[feature]
        except KeyError:
            raise ValueError("Requested feature is not in the database: '%s'." % feature)

    def _ensure_molecule_identifier_is_available(self, molecule_identifier: str):
        """
        Raises an ValueError, if the requested molecule identifier is not a column in the molecule table of the database.

        :param molecule_identifier: string, column name of the molecule identifier
        """
        if molecule_identifier not in self.available_molecule_identifier:
            raise ValueError("Molecule identifier '%s' is not available." % molecule_identifier)

    @abstractmethod
    def _get_molecule_feature_matrix(self, data: Union[List[Tuple], Tuple[Tuple]], features: str, feature_table: str) \
            -> np.ndarray:
        """
        Function to parse the molecular features stored as strings in the candidate database and load them into a numpy
        array.

        :param data: List[Tuple] | Tuple[Tuple], length = n_molecules, molecule features represented as strings as
            stored in the candidate DB.

            Examples:
                [("3:32,12:1,16:2,...", ), ("1:3,2:1,6:2,...", ), ...]              --- Bach2020
                [("3,12,16,...", "32,1,2,..."), ("1,2,6", "3,1,2"), ...]            --- MassBank

        :param features: string, identifier of the molecular feature. This information is needed to properly parse the
            feature string representation and store it in a numpy array.

        :param feature_table: string, name of the meta-table in which the feature was found.

        :return: array-like, shape = (n_molecules, d_features)
        """
        pass

    def _get_connection_uri(self) -> str:
        """
        Returns the sqlite3 connection URI to open the candidate DB in read-only mode.
        """
        return "file:" + self.db_fn + "?mode=ro"

    def connect_to_db(self) -> sqlite3.Connection:
        """
        Connects to the candidate database in read-only mode.
        """
        return sqlite3.connect(self._get_connection_uri(), uri=True)

    def __enter__(self):
        if self.db is None:
            # Connection is not open --> open it
            self.db = self.connect_to_db()
            self._db_connected_in_context_manager = True
        else:
            # Connection is already open --> do nothing
            self._db_connected_in_context_manager = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit function for the context manager. Closes the connection to the candidate database.
        """
        if self._db_connected_in_context_manager:
            self.db.close()
            self.db = None

    def close(self) -> None:
        """
        Closes the database connection.
        """
        if self.db is not None:
            self.db.close()

    def get_n_cand(self, spectrum: Spectrum) -> int:
        """
        Return the number of available candidates for the given spectrum.

        :param spectrum: matchms.Spectrum, of the sequence element to get the number of candidates.

        :return: scalar, number of candidates
        """
        return len(self.get_labelspace(spectrum))

    def get_n_total_cand(self, spectrum: Spectrum) -> int:
        """
        Return the total number of available candidates for the given spectrum regardless of any candidate sub-set
        selection.

        :param spectrum: matchms.Spectrum, of the sequence element to get the number of candidates.

        :return: scalar, total number of candidates
        """
        return self.get_n_cand(spectrum)

    def get_molecule_features(self, spectrum: Spectrum, features: str, return_dataframe: bool = False) \
            -> Union[np.ndarray, pd.DataFrame]:
        """
        Load the specified molecule features for the candidates of the given spectrum.

        :param spectrum: matchms.Spectrum, of the sequence element to get the molecular features.

        :param features: string, identifier of the feature to load from the database.

        :param return_dataframe: boolean, indicating whether the score should be returned in a pandas.DataFrame storing
            the molecule ids and molecule features (shape = (n_samples, d_features + 1)).

        :return: array-like, shape = (n_cand, d_feature), feature matrix
        """
        feature_table = self._ensure_feature_is_available(features)

        # Query data from DB
        identifier, feature_rows = self._unpack_molecule_features(
            self.db.execute(
                self._get_molecule_feature_query(
                    spectrum, features, feature_table, self._get_candidate_subset(spectrum)
                )
            )
        )

        # Parse the feature strings into a feature matrix
        feature_matrix = self._get_molecule_feature_matrix(feature_rows, features, feature_table)

        # Apply feature transformation if a transformer is provided for the requested feature
        feature_matrix = self._transform_feature_matrix(feature_matrix, features)

        # Prepare output format
        if return_dataframe:
            df_features = pd.concat((pd.DataFrame({"identifier": identifier}), pd.DataFrame(feature_matrix)), axis=1)
        else:
            df_features = feature_matrix

        return df_features

    def _transform_feature_matrix(self, feature_matrix: np.ndarray, features: str) -> np.ndarray:
        """
        Apply the specified feature transformation to the feature matrix.
        """
        if isinstance(self.feature_transformer, dict):
            feature_transformer = self.feature_transformer.get(features, None)  # type: Union[BaseEstimator, Pipeline]
        else:
            feature_transformer = self.feature_transformer  # type: Union[BaseEstimator, Pipeline, None]

        if feature_transformer is not None:
            feature_matrix = feature_transformer.transform(feature_matrix)

        return feature_matrix

    @staticmethod
    def _unpack_molecule_features(res: sqlite3.Cursor):
        """
        :param res: sqlite3.Cursor, result of the '_get_molecule_feature_query

        :return:
        """
        return zip(*((row[0], row[1:]) for row in res))

    @lru_cache(maxsize=ssvm.cfg.LRU_CACHE_MAX_SIZE)
    def get_molecule_features_by_molecule_id(self, molecule_ids: Union[List[str], Tuple[str, ...]], features: str,
                                             return_dataframe: bool = False) -> Union[pd.DataFrame, np.ndarray]:
        """
        Load the molecular features for the specified molecule identifiers from the database.

        Note: We use the "GROUP BY identifier" clause to group the features associated with different candidates
            which have the same molecule identifier. The SQLite implementation does not ensure any particular order of
            the results and _any_ of the feature vector within a group could be returned. Keep this in mind, when your
            features encode properties of the molecules within a group differently (e.g. stereo-chemistry).

        :param molecule_ids: list or tuple of strings, molecule identifier

        :param features: string, molecular feature identifier

        :param return_dataframe: boolean, indicating whether the score should be returned in a pandas.DataFrame storing
            the molecule ids and molecule features (shape = (n_samples, d_features + 1)).

        :return: array-like, shape = (n_mol, d_feature), feature matrix. Row i corresponds to molecule i.
        """
        feature_table = self._ensure_feature_is_available(features)

        # Query data from DB
        identifiers, feature_rows = self._unpack_molecule_features(
            self.db.execute(self._get_molecule_feature_by_id_query(molecule_ids, features, feature_table))
        )

        # Parse feature strings into a matrix
        feature_matrix = self._get_molecule_feature_matrix(feature_rows, features, feature_table)

        # Apply feature transformation if a transformer is provided for the requested feature
        feature_matrix = self._transform_feature_matrix(feature_matrix, features)

        # Prepare output format
        if return_dataframe:
            df_features = pd.DataFrame(feature_matrix, index=identifiers) \
                .rename_axis("identifier") \
                .loc[list(molecule_ids)] \
                .reset_index()
        else:
            # Use heuristic to check whether there are repeated elements in the 'molecule_ids' list. In such a case, the
            # SQLite query only returns the features for unique molecules. We can use a pandas dataframe to fix that.
            if len(feature_matrix) < len(molecule_ids):
                try:
                    df_features = pd.DataFrame(feature_matrix, index=identifiers).loc[list(molecule_ids)].values
                except KeyError as e:
                    raise KeyError("The feature of the molecule with id = '%s' could not be loaded. " % e.args[0])
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

    def get_ms_scores(
            self, spectrum: Spectrum, ms_scorer: Union[str, List[str]], scale_scores_to_range: bool = True,
            return_dataframe: bool = False, return_as_ndarray: bool = False, score_fill_value: float = 1e-6,
            ms_scorer_weights: Optional[List[float]] = None, **kwargs
    ) -> Union[pd.DataFrame, List[float]]:
        """

        :param spectrum: matchms.Spectrum, of the sequence element to get the MS(2) scores.

        :param ms_scorer: string | List of strings, identifier(s) of the MS(2) scoring method(s) for which the scores
            should be loaded from the database. If multiple scores are requested, they

        :param scale_scores_to_range: boolean, indicating whether the scores should be normalized to the range (0, 1].

        :param return_dataframe: boolean, indicating whether the score should be returned in a two-column
            pandas.DataFrame storing the molecule ids and MS2 scores.

        :param return_as_ndarray: boolean, indicating whether the scores (if return_dataframe = False) should be
            returned as a numpy array.

        :param score_fill_value: float, value used to replace the missing MS2 scores, if _no_ candidate was assigned a
            scores. That can happen if the in-silico tool failed for the particular spectrum.

        :param ms_scorer_weights: List of floats, weights for each MS2 scorer when combining using weighted sum.

        :return: list of scalars or pandas.DataFrame, (normalized, if requested) MS2 scores
        """
        # FIXME: This is a nasty hack to allow calling this function from a sub-class when no candidate subset has been
        #        defined there.
        if "__candidate_subset" in kwargs:
            candidate_subset = kwargs["__candidate_subset"]
        else:
            candidate_subset = self._get_candidate_subset(spectrum)

        # Ensure MS2 scorers are provided as list
        if isinstance(ms_scorer, str):
            ms_scorer = [ms_scorer]
            ms_scorer_weights = [1.0]
        else:
            if ms_scorer_weights is None:
                ms_scorer_weights = [1.0 / len(ms_scorer) for _ in range(len(ms_scorer))]

            if len(ms_scorer) != len(ms_scorer_weights):
                raise ValueError(
                    "Number of MS2 scorer and scorer weights must be equal: %d vs. %d"
                    % (len(ms_scorer), len(ms_scorer_weights))
                )

        # Load the MS2 scores for each requested scorer separately
        df_scores = []
        for idx, mss in enumerate(ms_scorer):
            # Load the MS scores: The molecule identifier is used as index
            if mss in self.available_ms2_scores:
                # -- MS2 --
                df_scores.append(
                    pd.read_sql_query(
                        self._get_ms2_score_query(spectrum, mss, candidate_subset), self.db, index_col="identifier"
                    )
                )

                # Fill missing MS2 scores with the minimum score
                _fill_value = df_scores[idx]["ms_score"].min(skipna=True)
                if np.isnan(_fill_value):
                    # If no MS2 scores for the candidates are available (scoring failed), than we use a small positive constant.
                    _fill_value = score_fill_value
                df_scores[idx]["ms_score"] = df_scores[idx]["ms_score"].fillna(value=_fill_value)

                if scale_scores_to_range:
                    # Scale scores to (0, 1]
                    c1, c2 = self.get_normalization_parameters_c1_and_c2(df_scores[idx]["ms_score"].values)
                    df_scores[idx]["ms_score"] = self.normalize_scores(df_scores[idx]["ms_score"].values, c1, c2)
                # ---------
            elif mss == "CONST_MS_SCORE":
                # -- Ignore MS information --
                df_scores.append(
                    pd.DataFrame(
                        {"identifier": self.get_labelspace(spectrum), "ms_score": 1.0}
                    ).set_index("identifier")
                )
                # ---------------------------
            else:
                raise ValueError("Unsupported MS scorer: '%s'" % mss)

        # Combine the results
        df_out = ms_scorer_weights[0] * df_scores[0]  # type: pd.DataFrame
        for w, d in zip(ms_scorer_weights[1:], df_scores[1:]):
            assert len(d) == len(df_out), "UPS: When combining MS scores all score-tables must have the same size!"
            df_out += (w * d)

        if scale_scores_to_range:
            # Scale scores to (0, 1]
            df_out["ms_score"] /= df_out["ms_score"].max()

        # Prepare output
        if return_dataframe:
            df_out = df_out.reset_index()
        else:
            if return_as_ndarray:
                df_out = df_out["ms_score"].values
            else:
                df_out = df_out["ms_score"].tolist()

        return df_out

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

    @staticmethod
    def get_normalization_parameters_c1_and_c2(scores: np.ndarray) -> Tuple[float, float]:
        """
        Calculate two scalar score normalization parameters (c1 and c2) following [1]. Using these parameters the MS2
        scores are normalized as follows:

            scores = np.maximum(c2, scores + c1)

        [1] "Probabilistic framework for integration of mass spectrum and retention time information in small molecule
             identification", Bach et al. 2020

        :param scores: array-like, shape = (n_samples, ), raw MS2 scores from the in-silico method. NaN scores are
            not allowed.

        :return: tuple of scalars (c1, c2)
            - 'c1' is the abs of the smallest score if any score < 0 else 0
            - 'c2' is thousand-times smaller than the smallest positive score
        """
        if np.any(np.isnan(scores)):
            raise ValueError("NaN scores are not allowed. Cannot compute the regularization parameters.")

        # Parameter to lift scores to values >= 0
        min_score = np.min(scores)
        c1 = np.abs(min_score) if min_score < 0 else 0.0

        if np.all(scores == scores[0]):  # all scores are equal
            return c1, 1e-6

        # Parameter to avoid zero entries
        c2 = (np.min(scores[(scores + c1) > 0]) + c1) / 1000

        return c1, c2

    @staticmethod
    def normalize_scores(scores: Union[List, np.ndarray], c1: float, c2: float) -> np.ndarray:
        """
        :param scores: array-like, shape = (n_samples, ), raw MS2 scores from the in-silico method
        """
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)

        scores = np.maximum(c2, scores + c1)  # all scores are >= c2 > 0
        scores /= np.max(scores)  # maximum score is one

        return scores


class ABCCandSQLiteDB_Bach2020(ABCCandSQLiteDB):
    """
    Specific instance of the candidate set SQLite DB wrapper for the DB layout used by Bach et al. 2020

    Bach et al. 2020: https://doi.org/10.1093/bioinformatics/btaa998
    """
    def __init__(self, *args, **kwargs):
        """
        No additional parameters.
        """
        super().__init__(*args, **kwargs)

    def _get_available_ms2scorer(self) -> set:
        """
        See base-class
        """
        return set((p for p, in self.db.execute("SELECT name FROM participants")))

    def _get_labelspace_query(self, spectrum: Spectrum, candidate_subset: Optional[List] = None) -> str:
        """
        See base-class.
        """
        query = "SELECT m.%s as identifier FROM candidates_spectra " \
                "   INNER JOIN molecules m ON m.inchi = candidates_spectra.candidate" \
                "   WHERE spectrum IS '%s'" % (self.molecule_identifier, spectrum.get("spectrum_id"))

        if candidate_subset is not None:
            query += " AND identifier IN %s" % self._in_sql(candidate_subset)

        query += "  GROUP BY identifier" \
                 "  ORDER BY identifier"

        return query

    def _get_molecule_feature_query(
            self, spectrum: Spectrum, feature: str, feature_table: str, candidate_subset: Optional[List] = None
    ) -> str:
        """
        See base-class.
        """
        assert feature_table == "fingerprints_meta", "Ups."

        query = "SELECT m.%s AS identifier, %s AS molecular_feature FROM candidates_spectra" \
                "   INNER JOIN molecules m ON m.inchi = candidates_spectra.candidate" \
                "   INNER JOIN fingerprints_data fd ON fd.molecule = candidates_spectra.candidate" \
                "   WHERE spectrum IS '%s'" % (self.molecule_identifier, feature, spectrum.get("spectrum_id"))

        if candidate_subset is not None:
            query += " AND identifier IN %s" % self._in_sql(candidate_subset)

        query += "  GROUP BY identifier" \
                 "  ORDER BY identifier"

        return query

    def _get_ms2_score_query(self, spectrum: Spectrum, ms2scorer: str, candidate_subset: Optional[List[str]] = None) \
            -> str:
        """
        See base-class.
        """
        query = "SELECT m.%s AS identifier, MAX(score) AS ms_score" \
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

    def _get_molecule_feature_by_id_query(
            self, molecule_ids: Union[List[str], Tuple[str, ...]], features: str, feature_table: str
    ) -> str:
        """
        See base-class.
        """
        # Build string to order SQLite output as defined by the given molecule-ids.
        # Note: The order in which we provide the molecule ids is preserved in the output. But, if a molecule id appears
        #       multiple times, than its first appearance defines its position in the output.
        # Read: https://www.sqlite.org/lang_expr.html#case
        assert feature_table == "fingerprints_meta", "Ups"

        _order_query_str = "\n".join(["WHEN '%s' THEN %d" % (mid, i) for i, mid in enumerate(molecule_ids, start=1)])

        query = "SELECT %s AS identifier, %s AS molecular_feature FROM molecules" \
                "   INNER JOIN fingerprints_data fd ON fd.molecule = molecules.inchi" \
                "   WHERE identifier IN %s" \
                "   GROUP BY identifier" \
                "   ORDER BY CASE identifier" \
                "         %s" \
                "      END" % (self.molecule_identifier, features, self._in_sql(molecule_ids), _order_query_str)

        return query

    def _get_molecule_feature_matrix(self, data: Union[pd.Series, List, Tuple], features: str, feature_table: str) \
            -> np.ndarray:
        """
        See base-class.
        """
        assert feature_table == "fingerprints_meta", "Ups"

        # Determine number of samples and feature dimension
        n = len(data)
        d, mode = self._get_d_and_mode_feature(features, feature_table)

        if mode == "count":
            # Set up an output array
            X = np.zeros((n, d), dtype=int)
            for i, (bits_vals, ) in enumerate(data):
                for _fp_str in bits_vals.split(","):
                    _idx, _cnt = _fp_str.split(":")
                    X[i, int(_idx)] = int(_cnt)
        elif mode in ["binary", "binarized"]:
            # Set up an output array
            X = np.zeros((n, d))
            for i, (bits, ) in enumerate(data):
                X[i, list(map(int, bits.split(",")))] = 1
        else:
            raise ValueError("Invalid fingerprint mode: %s" % mode)

        return X


class ABCCandSQLiteDB_Massbank(ABCCandSQLiteDB):
    """
    Specific instance of the candidate set SQLite DB wrapper for the DB layout by massbank2db [1].

    [1] https://github.com/bachi55/massbank2db_FILES
    """
    def __init__(self, *args, **kwargs):
        """
        No additional parameters.
        """
        super().__init__(*args, **kwargs)

    def _get_available_ms2scorer(self) -> set:
        """
        See base-class
        """
        return set((p for p, in self.db.execute("SELECT name FROM scoring_methods")))

    def _get_labelspace_query(self, spectrum: Spectrum, candidate_subset: Optional[List] = None) -> str:
        """
        See base-class.
        """
        query = "SELECT m.%s as identifier FROM candidates_spectra " \
                "   INNER JOIN molecules m ON m.cid = candidates_spectra.candidate" \
                "   WHERE spectrum IS '%s'" % (self.molecule_identifier, spectrum.get("spectrum_id"))

        if candidate_subset is not None:
            query += " AND identifier IN %s" % self._in_sql(candidate_subset)

        query += "  GROUP BY identifier" \
                 "  ORDER BY identifier"

        return query

    def _get_molecule_feature_query(
            self, spectrum: Spectrum, feature: str, feature_table: str, candidate_subset: Optional[List] = None
    ) -> str:
        """
        See base-class.
        """
        if feature_table.startswith("fingerprints"):
            # Fingerprints
            query = "SELECT m.%s AS identifier, fd.bits, fd.vals FROM candidates_spectra" \
                    "   INNER JOIN molecules m ON m.cid = candidates_spectra.candidate" \
                    "   INNER JOIN fingerprints_data__%s fd ON fd.molecule = candidates_spectra.candidate" \
                    "   WHERE spectrum IS '%s'" \
                    % (self.molecule_identifier, feature, spectrum.get("spectrum_id"))
        elif feature_table.startswith("descriptors"):
            # Descriptors
            query = "SELECT m.%s AS identifier, dd.desc_vals FROM candidates_spectra" \
                    "   INNER JOIN molecules m ON m.cid = candidates_spectra.candidate" \
                    "   INNER JOIN descriptors_data__%s dd ON dd.molecule = candidates_spectra.candidate" \
                    "   WHERE spectrum IS '%s'" \
                    % (self.molecule_identifier, feature, spectrum.get("spectrum_id"))
        else:
            raise AssertionError(
                "Ups, that should not happen: feature_table = '%s', feature = '%s'" % (feature_table, feature)
            )

        if candidate_subset is not None:
            query += " AND identifier IN %s" % self._in_sql(candidate_subset)

        query += "  GROUP BY identifier" \
                 "  ORDER BY identifier"

        return query

    def _get_ms2_score_query(self, spectrum: Spectrum, ms2scorer: str, candidate_subset: Optional[List[str]] = None) \
            -> str:
        """
        See base-class.
        """
        query = "SELECT m.%s AS identifier, MAX(score) AS ms_score" \
                "   FROM (SELECT * FROM candidates_spectra WHERE candidates_spectra.spectrum IS '%s') cs" \
                "   INNER JOIN molecules m ON m.cid = cs.candidate" \
                "   LEFT OUTER JOIN (" \
                "       SELECT candidate, score FROM spectra_candidate_scores" \
                "       WHERE scoring_method IS '%s' AND spectrum IS '%s') scs ON cs.candidate = scs.candidate" \
                % (self.molecule_identifier, spectrum.get("spectrum_id"), ms2scorer, spectrum.get("spectrum_id"))

        # Restrict candidate subset if needed
        if candidate_subset is not None:
            query += "   WHERE identifier in %s" % self._in_sql(candidate_subset)

        query += "  GROUP BY identifier" \
                 "  ORDER BY identifier"

        return query

    def _get_molecule_feature_by_id_query(
            self, molecule_ids: Union[List[str], Tuple[str, ...]], features: str, feature_table: str
    ) -> str:
        """
        See base-class.
        """
        # Build string to order SQLite output as defined by the given molecule-ids.
        # Note: The order in which we provide the molecule ids is preserved in the output. But, if a molecule id appears
        #       multiple times, than its first appearance defines its position in the output.
        # Read: https://www.sqlite.org/lang_expr.html#case
        _order_query_str = "\n".join(["WHEN '%s' THEN %d" % (mid, i) for i, mid in enumerate(molecule_ids, start=1)])

        if feature_table.startswith("fingerprints"):
            # Fingerprints
            query = "SELECT %s AS identifier, bits, vals FROM fingerprints_data__%s fd" \
                    "   INNER JOIN main.molecules mols ON mols.cid = fd.molecule" \
                    "   WHERE identifier IN %s" \
                    "   GROUP BY identifier" \
                    "   ORDER BY CASE identifier" \
                    "         %s" \
                    "      END" % (self.molecule_identifier, features, self._in_sql(molecule_ids), _order_query_str)
        elif feature_table.startswith("descriptors"):
            # Descriptors
            query = "SELECT %s AS identifier, desc_vals FROM descriptors_data__%s dd" \
                    "   INNER JOIN main.molecules mols ON mols.cid = dd.molecule" \
                    "   WHERE identifier IN %s" \
                    "   GROUP BY identifier" \
                    "   ORDER BY CASE identifier" \
                    "         %s" \
                    "      END" % (self.molecule_identifier, features, self._in_sql(molecule_ids), _order_query_str)
        else:
            raise AssertionError(
                "Ups, that should not happen: feature_table = '%s', feature = '%s'" % (feature_table, features)
            )

        return query

    def _get_molecule_feature_matrix(self, data: Union[pd.Series, List, Tuple], features: str, feature_table: str) \
            -> np.ndarray:
        """
        See base-class.
        """
        # Determine number of samples and feature dimension
        n = len(data)
        d, mode = self._get_d_and_mode_feature(features, feature_table)

        if mode == "binary":
            dtype = float  # tanimoto kernel fastest with float-type
        elif mode == "count":
            dtype = int  # minmax kernel performs the best with int-type
        elif mode == "real":
            dtype = float
        else:
            raise ValueError("Invalid fingerprint mode: %s" % mode)

        # Set up an output array
        X = np.zeros((n, d), dtype=dtype)

        if mode in ["count", "real"]:
            if feature_table.startswith("fingerprints"):
                for i, (bits, vals) in enumerate(data):
                    X[i, list(map(int, bits.split(",")))] = list(map(dtype, vals.split(",")))
            elif feature_table.startswith("descriptors"):
                for i, (vals, ) in enumerate(data):
                    X[i] = list(map(dtype, vals.split(",")))
            else:
                raise AssertionError(
                    "Ups, that should not happen: feature_table = '%s', feature = '%s'" % (feature_table, features)
                )
        elif mode == "binary":
            for i, (bits, _) in enumerate(data):
                X[i, list(map(int, bits.split(",")))] = 1
        else:
            raise ValueError("Invalid fingerprint mode: %s" % mode)

        return X


class CandSQLiteDB_Bach2020(ABCCandSQLiteDB_Bach2020):
    def __init__(self, *args, **kwargs):
        """
        No additional parameters.
        """
        super().__init__(*args, **kwargs)

    def _get_candidate_subset(self, spectrum: Spectrum) -> Optional[List[str]]:
        """
        In the base-setting the candidates are not restricted. This is expressed by the return value None.
        """
        return None


class CandSQLiteDB_Massbank(ABCCandSQLiteDB_Massbank):
    def __init__(self, *args, **kwargs):
        """
        No additional parameters.
        """
        super().__init__(*args, **kwargs)

    def _get_candidate_subset(self, spectrum: Spectrum) -> Optional[List[str]]:
        """
        In the base-setting the candidates are not restricted. This is expressed by the return value None.
        """
        return None


class FixedSubsetCandSQLiteDB_Bach2020(ABCCandSQLiteDB_Bach2020):
    """
    This class allows to specify a fixed candidate subset for a given set of spectra.
    """

    def __init__(self, labelspace_subset: Dict[str, List[str]], assert_correct_candidate: bool = True, *args, **kwargs):
        """
        :param labelspace_subset: dictionary (spectrum_id: candidate_subset), storing the candidates subsets as list of
            strings for each spectrum identified by its spectrum id.

        :parma assert_correct_candidate: boolean, indicating whether it should be ensured that the ground-truth (gt)
            molecular structure, associated with each spectrum, is present in the candidate subsets. For that, the
            gt structure must be specified (see '_get_candidate_subset').
        """
        self.assert_correct_candidate = assert_correct_candidate
        self._labelspace_subset = labelspace_subset

        super().__init__(*args, **kwargs)

    def _get_candidate_subset(self, spectrum: Spectrum) -> List[str]:
        """
        :param spectrum: matchms.Spectrum, of the sequence element to get the number of candidates.

        :return: list of strings, molecule identifier for the candidate subset of the given spectrum (not sorted !).
        """
        spectrum_id = spectrum.get("spectrum_id")

        try:
            candidates_sub = self._labelspace_subset[spectrum_id]

            if self.assert_correct_candidate:
                molecule_id = spectrum.get("molecule_id")

                if molecule_id is None:
                    raise ValueError("Cannot ensure that the ground truth structure is included in the candidate set, "
                                     "as no 'molecule_id' is specified in the Spectrum object.")

                if molecule_id not in candidates_sub:
                    raise ValueError("The molecule id '%s' is not in the candidate set." % molecule_id)
        except KeyError:
            raise KeyError("Candidate set for the specified spectrum was not provided: spectrum-id = %s" % spectrum_id)

        return candidates_sub

    def get_labelspace(self, spectrum: Spectrum, candidate_subset: Optional[List] = None) -> List[str]:
        """
        Return the fixed label space (candidate subset)
        """
        if candidate_subset is not None:
            raise RuntimeError("Candidate subset cannot be requested in any sub-class of 'CandidateSQLiteDB'.")

        return super().get_labelspace(spectrum, self._get_candidate_subset(spectrum))


class ABCRandomSubsetCandSQLiteDB(ABCCandSQLiteDB):
    """
    This class allows to generate (and fix) a random candidate subset for each spectrum. For that, the first time a
    subset is requested for a spectrum (identified by its id) is sampled. The subset size can be specified in as
    absolute size or fraction of the available candidates.
    """
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
            candidates_sub = check_random_state(self.random_state).choice(candidates_all, n_cand.astype(int),
                                                                          replace=False)

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

    def get_n_total_cand(self, spectrum: Spectrum) -> int:
        """
        Returns the total number of candidates without sub-set selection.
        """
        return len(super().get_labelspace(spectrum))

    def get_labelspace(self, spectrum: Spectrum, candidate_subset: Optional[List] = None) -> List[str]:
        """
        Return the label space of the random subset.
        """
        if candidate_subset is not None:
            raise RuntimeError("Candidate subset cannot be requested in any sub-class of 'CandidateSQLiteDB'.")

        return super().get_labelspace(spectrum, self._get_candidate_subset(spectrum))


class RandomSubsetCandSQLiteDB_Bach2020(ABCRandomSubsetCandSQLiteDB, ABCCandSQLiteDB_Bach2020):
    """
    This class allows to generate (and fix) a random candidate subset for each spectrum. For that, the first time a
    subset is requested for a spectrum (identified by its id) is sampled. The subset size can be specified in as
    absolute size or fraction of the available candidates.
    """
    def __init__(self, *args, **kwargs):
        """
        No additional parameters.
        """
        super().__init__(*args, **kwargs)


class RandomSubsetCandSQLiteDB_Massbank(ABCRandomSubsetCandSQLiteDB, ABCCandSQLiteDB_Massbank):
    """
    This class allows to generate (and fix) a random candidate subset for each spectrum. For that, the first time a
    subset is requested for a spectrum (identified by its id) is sampled. The subset size can be specified in as
    absolute size or fraction of the available candidates.
    """
    def __init__(self, *args, **kwargs):
        """
        No additional parameters.
        """
        super().__init__(*args, **kwargs)


class Sequence(object):
    def __init__(
            self, spectra: List[Spectrum], candidates: ABCCandSQLiteDB,
            ms_scorer: Optional[Union[str, List[str]]] = None
    ):
        self.spectra = spectra
        self.candidates = candidates
        self.ms_scorer = ms_scorer

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

    def get_dataset(self) -> str:
        return self.spectra[0].get("dataset")

    def get_retention_time(self, s: Optional[int] = None) -> Union[float, List[float]]:
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
            with shape = (n_cand_s, d_feature)
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

    def get_n_total_cand(self, s: Optional[int] = None) -> Union[int, List[int]]:
        """
        Get the total number of candidates for each spectrum in the sequence as list.

        :param s: scalar, sequence index for which the MS2 scores should be returned. If None, scores are returned for
            all spectra in the sequence.
        """
        if s is None:
            n_cand = [self.get_n_total_cand(s) for s in range(self.__len__())]
        else:
            n_cand = self.candidates.get_n_total_cand(self.spectra[s])

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

    def get_ms_scores(
            self, s: Optional[int] = None, scale_scores_to_range: bool = True, return_as_ndarray: bool = False,
            score_fill_value: float = 1e-6
    ) -> Union[List[List[float]], List[float], List[np.ndarray], np.ndarray]:
        """
        Get the MS2 scores for the given sequence index

        :param s: scalar, sequence index for which the MS2 scores should be returned. If None, scores are returned for
            all spectra in the sequence.

        :param return_as_ndarray: boolean, indicating whether the MS2 scores should be returned as numpy array

        :param scale_scores_to_range: boolean, indicating whether the output scores should be scaled to (0, 1]

        :param score_fill_value: float, value used to replace the missing MS2 scores, if _no_ candidate was assigned a
            scores. That can happen if the in-silico tool failed for the particular spectrum.
        """
        if self.ms_scorer is None:
            raise ValueError("No MS2 scorer specified!")

        if s is None:
            ms2_scores = [
                self.get_ms_scores(s, scale_scores_to_range, return_as_ndarray, score_fill_value)
                for s in range(self.__len__())
            ]
        else:
            ms2_scores = self.candidates.get_ms_scores(
                self.spectra[s], self.ms_scorer, scale_scores_to_range=scale_scores_to_range,
                return_as_ndarray=return_as_ndarray, score_fill_value=score_fill_value
            )

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
    def __init__(self, spectra: List[Spectrum], candidates: ABCCandSQLiteDB, ms_scorer: Optional[str] = None,
                 labels: Optional[List[str]] = None, label_key: str = "molecule_id"):
        """
        :param spectra: list of strings, spectrum-ids belonging sequence

        :param candidates: ABCCandSQLiteDB, candidate set wrapper

        :param ms_scorer: string, of the MS2 scoring method for which the MS2 scores should be loaded, e.g. during the
            Structured SVM training.

        :param labels: list of strings, ground truth molecule identifiers belonging to the spectra of the sequence

        :param label_key: string, dictionary key used to load the label information from the spectra.
        """
        if labels is None:
            self.labels = []
            for spectrum in spectra:
                label = spectrum.get(label_key)
                if label is None:
                    raise ValueError(
                        "Spectrum '%s' has no label information (key = '%s')." % (
                            spectrum.get("spectrum_id"), label_key
                        )
                    )
                self.labels.append(label)
        else:
            if len(labels) != len(spectra):
                raise ValueError("Number of labels does not match the number of spectra: %d != %d" % (
                    len(labels), len(spectra)
                ))

            self.labels = labels

        super(LabeledSequence, self).__init__(spectra=spectra, candidates=candidates, ms_scorer=ms_scorer)

    def get_labels(self, s: Optional[int] = None) -> Union[str, List[str]]:
        """
        Get the sequence labels.

        :param s: scalar, sequence index for which the label should be returned. If None, labels are returned for all
            spectra in the sequence.

        :return: list of strings or string, label(s) of the spectra sequence
        """
        if s is None:
            return self.labels
        else:
            return self.labels[s]

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

    def get_label_loss(
            self,
            label_loss_fun: Union[Callable[[np.ndarray, np.ndarray], np.ndarray], Callable[[str, List[str]], np.ndarray]],
            features: str, s: Optional[int] = None
    ) -> Union[List[np.ndarray], np.ndarray]:
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
            if features in self.candidates.available_molecular_features:
                Y = self.get_molecule_features_for_candidates(features, s)
                y_gt = Y[self.get_index_of_correct_structure(s)]
            elif features == "MOL_ID":
                Y = self.get_labelspace(s)
                y_gt = self.labels[s]
            else:
                raise ValueError("")

            return label_loss_fun(y_gt, Y)

    def get_molecule_features_for_labels(self, features: str) -> np.ndarray:
        """

        :param features:
        :return:
        """
        return self.candidates.get_molecule_features_by_molecule_id(tuple(self.labels), features)


class SequenceSample(object):
    """
    Class representing a sequence sample.
    """
    def __init__(self, spectra: List[Spectrum], labels: List[str], candidates: ABCCandSQLiteDB, N: int,
                 L_min: int, L_max: Optional[int] = None, random_state: Optional[int] = None,
                 sort_sequence_by_rt: bool = False, ms_scorer: Optional[Union[List[str], str]] = None,
                 use_sequence_specific_candidates: bool = False):
        """
        :param data: list of matchms.Spectrum, spectra to sample sequences from

        :param labels: list of strings, labels of each spectrum identifying the ground truth molecule.

        :param candidates:

        :param N: scalar, number of sequences to sample.

        :param L_min: scalar, minimum length of the individual sequences

        :param L_max: scalar, maximum length of the individual sequences

        :param use_sequence_specific_candidates: boolean indicating whether for each training

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
        self.ms_scorer = ms_scorer
        self.use_sequence_specific_candidates = use_sequence_specific_candidates
        self.spectra_ids = [spectrum.get("spectrum_id") for spectrum in self.spectra]

        assert pd.Series(self.spectra_ids).is_unique, "Spectra IDs must be unique."
        assert self.L_min > 0

        if self.L_max is None:
            self._L = np.full(self.N, fill_value=self.L_min)
        else:
            assert self.L_min <= self.L_max
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
        self._sampled_sequences, self._sampled_datasets = self._sample_sequences()

    def __len__(self):
        """
        :return: scalar, number of sample sequences
        """
        return len(self._sampled_sequences)

    def __iter__(self) -> Iterator[LabeledSequence]:
        return iter(self._sampled_sequences)

    def __getitem__(self, item):
        return self._sampled_sequences[item]

    def __eq__(self, other):
        if self.labels != other.labels:
            return False

        if self.spectra != other.spectra:
            return False

        return True

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

        sequences = []
        datasets = []
        for i, ds in enumerate(rs.choice(self._datasets, self.N)):
            seq_idc = rs.choice(self._dataset2idx[ds], np.minimum(self._L[i], self._n_spec_per_dataset[ds]),
                                replace=False)
            seq_spectra = [self.spectra[sig] for sig in seq_idc]
            seq_labels = [self.labels[sig] for sig in seq_idc]

            # FIXME: Here we can have multiple times the same molecule in the sample, e.g. due to different adducts.
            # assert pd.Series(seq_labels).is_unique, "Each molecule should appear only ones in the set of molecules."

            # Sort the sequence elements by their retention time
            if self.sort_sequence_by_rt:
                seq_spectra, seq_labels = zip(*sorted(zip(seq_spectra, seq_labels),
                                                      key=lambda s: s[0].get("retention_time")))

            if self.use_sequence_specific_candidates:
                if not isinstance(self.candidates, ABCRandomSubsetCandSQLiteDB):
                    raise ValueError("Sequence specific candidate sets are only supported for random candidate subset "
                                     "candidate databases.")

                # Copy the candidate class
                seq_candidates = deepcopy(self.candidates)

                # Replace the random seed to be sequence specific
                seq_rs = seq_candidates.__getattribute__("random_state")
                assert isinstance(seq_rs, int)
                seq_candidates.__setattr__("random_state", seq_rs + i)
            else:
                seq_candidates = self.candidates

            sequences.append(
                LabeledSequence(seq_spectra, candidates=seq_candidates, ms_scorer=self.ms_scorer, labels=seq_labels)
            )
            datasets.append(ds)

        return sequences, datasets

    def get_train_test_split(self, spectra_cv: Union[GroupKFold, GroupShuffleSplit, int] = 5,
                             N_train: Optional[int] = None, N_test: Optional[int] = None,
                             L_min_test: Optional[int] = None, L_max_test: Optional[int] = None,
                             candidates_test: Optional[ABCCandSQLiteDB] = None,
                             use_sequence_specific_candidates_for_training: bool = False) \
            -> Tuple[SEQUENCE_SAMPLE_T, SEQUENCE_SAMPLE_T]:
        """
        Get a single training and test split
        """
        return next(self.get_train_test_generator(
            spectra_cv, N_train, N_test, L_min_test, L_max_test, candidates_test,
            use_sequence_specific_candidates_for_training=use_sequence_specific_candidates_for_training))

    def get_train_test_generator(self, spectra_cv: Union[GroupKFold, GroupShuffleSplit, int] = 5,
                                 N_train: Optional[int] = None, N_test: Optional[int] = None,
                                 L_min_test: Optional[int] = None, L_max_test: Optional[int] = None,
                                 candidates_test: Optional[ABCCandSQLiteDB] = None,
                                 use_sequence_specific_candidates_for_training: bool = False) \
            -> Tuple[SEQUENCE_SAMPLE_T, SEQUENCE_SAMPLE_T]:
        """
        Split the spectra ids into training and test sets. Thereby, all spectra belonging to the same molecular
        structure (regardless of their Massbank sub-dataset membership) are either in the test or training.

        Internally the scikit-learn function "GroupKFold" is used to split the data. As groups we use the molecular
        identifiers.

        :param spectra_cv: scalar | GroupKFold | GroupShuffleSplit
            If scalar, a GroupKFold splitter is initialized with the scalar being the number of splits.
            Otherwise, the provided cross-validation splitter is directly used.

        :param N_train: scalar | None
            If scalar, number of training sequences
            Otherwise, the number of training sequences is 80% of the self.N

        :param N_test: scalar | None
            If scalar, number of test sequences
            Otherwise, the number of test sequences is 20% of the self.N

        :yields:
            train : ndarray
                The training set indices for that split.

            test : ndarray
                The testing set indices for that split.
        """
        if isinstance(spectra_cv, int):
            spectra_cv = GroupKFold(n_splits=spectra_cv)
        else:
            assert isinstance(spectra_cv, (GroupKFold, GroupShuffleSplit))

        # Handle number of training and test sequences
        if N_train is None:
            N_train = np.round(self.N * 0.8).astype(int)

        if N_test is None:
            N_test = np.round(self.N * 0.2).astype(int)

        # Handle test sequence length specification
        if L_min_test is None:
            L_min_test = self.L_min

        if L_max_test is None:
            L_max_test = self.L_max

        if candidates_test is None:
            candidates_test = self.candidates

        for train, test in spectra_cv.split(self.spectra, groups=self.labels):
            # Get training and test subsets of the spectra ids. Spectra belonging to the same molecular structure are
            # either in the training or the test set.
            yield (
                SequenceSample([self.spectra[i] for i in train], [self.labels[i] for i in train],
                               candidates=self.candidates, N=N_train, L_min=self.L_min, L_max=self.L_max,
                               random_state=self.random_state, ms_scorer=self.ms_scorer,
                               use_sequence_specific_candidates=use_sequence_specific_candidates_for_training),
                SequenceSample([self.spectra[i] for i in test], [self.labels[i] for i in test],
                               candidates=candidates_test, N=N_test, L_min=L_min_test, L_max=L_max_test,
                               random_state=self.random_state, ms_scorer=self.ms_scorer,
                               use_sequence_specific_candidates=False)
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

    def get_spectra_ids(self) -> List[str]:
        """
        Returns all spectra ids associated with the sequence sample

        :return: list of strings, spectrum ids
        """
        return self.spectra_ids

    def get_spectra(self) -> List[Spectrum]:
        return self.spectra


class SpanningTrees(object):
    def __init__(self, sequence: Sequence, n_trees: int = 1,
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        """
        :param sequence:
        :param n_trees:
        :param random_state:
        """
        self.n_tress = n_trees
        self.random_state = check_random_state(random_state)  # type: np.random.RandomState
        self.n_nodes = len(sequence)
        self.n_edges = self.n_nodes - 1

        self.trees = [get_random_spanning_tree(sequence, random_state=self.random_state) for _ in range(self.n_tress)]

        for T in self.trees:
            assert self.trees[0].nodes == T.nodes

        self.nodes = self.trees[0].nodes

    def __getitem__(self, item) -> nx.Graph:
        assert isinstance(item, int)
        return self.trees[item]

    def __iter__(self):
        return self.trees.__iter__()

    def __len__(self):
        return self.n_tress

    def get_n_nodes(self):
        return self.n_nodes

    def get_n_edges(self):
        return self.n_edges

    def get_nodes(self):
        return self.nodes
