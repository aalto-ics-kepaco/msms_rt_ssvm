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
import os
import unittest
import pickle
import pandas as pd
import numpy as np
import itertools as it

from matchms.Spectrum import Spectrum

from ssvm.data_structures import CandidateSetMetIdent, SequenceSample, CandidateSQLiteDB, RandomSubsetCandidateSQLiteDB
from ssvm.data_structures import Sequence
from ssvm.examples.ssvm_metident import read_data

DB_FN = "/home/bach/Documents/doctoral/projects/local_casmi_db/db/use_inchis/DB_LATEST.db"


class TestCandidateSQLiteDB(unittest.TestCase):
    def test_get_number_of_candidates(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016"})

        # Molecule identifier: inchikey
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey")
        self.assertEqual(2233, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(1918, candidates.get_n_cand(spectrum))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "EAX030601"})

        # Molecule identifier: inchikey
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey")
        self.assertEqual(16, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(16, candidates.get_n_cand(spectrum))

    def test_get_ms2_scores(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016"})
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey")

        for min_score_value in [0, 0.01, 0.1]:
            # MS2 Scorer is IOKR
            scores = candidates.get_ms2_scores(spectrum, ms2scorer="IOKR__696a17f3", min_score_value=min_score_value)
            self.assertEqual(2233, len(scores))
            self.assertAlmostEqual(min_score_value, np.min(scores))
            self.assertAlmostEqual(1.0, np.max(scores))

            # MS2 Scorer is MetFrag
            scores = candidates.get_ms2_scores(spectrum, ms2scorer="MetFrag_2.4.5__8afe4a14",
                                               min_score_value=min_score_value)
            self.assertEqual(2233, len(scores))
            self.assertAlmostEqual(min_score_value, np.min(scores))
            self.assertAlmostEqual(1.0, np.max(scores))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "EAX030601"})
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey")

        # MS2 Scorer is IOKR
        scores = candidates.get_ms2_scores(spectrum, ms2scorer="IOKR__696a17f3")
        self.assertEqual(16, len(scores))
        self.assertEqual(0.0, np.min(scores))
        self.assertEqual(1.0, np.max(scores))

        # MS2 Scorer is MetFrag
        scores = candidates.get_ms2_scores(spectrum, ms2scorer="MetFrag_2.4.5__8afe4a14")
        self.assertEqual(16, len(scores))
        self.assertEqual(0.0, np.min(scores))
        self.assertEqual(1.0, np.max(scores))

    def test_get_molecule_features(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-019"})
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey1")

        # IOKR features
        fps = candidates.get_molecule_features(spectrum, feature="iokr_fps__positive")
        self.assertEqual((5103, 7936), fps.shape)
        self.assertTrue(np.all(np.isin(fps, [0, 1])))

        # Substructure features
        fps = candidates.get_molecule_features(spectrum, feature="substructure_count")
        self.assertEqual((5103, 307), fps.shape)
        self.assertTrue(np.all(fps >= 0))

    def test_all_outputs_are_sorted_equally(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL-UHFFFAOYSA-N"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = CandidateSQLiteDB(db_fn=DB_FN, molecule_identifier="inchikey")

        scores = candidates.get_ms2_scores(spectrum, "MetFrag_2.4.5__8afe4a14", return_dataframe=True)
        fps = candidates.get_molecule_features(spectrum, "iokr_fps__positive", return_dataframe=True)
        labspace = candidates.get_labelspace(spectrum)

        self.assertEqual(sorted(labspace), labspace)
        self.assertEqual(labspace, scores["identifier"].to_list())
        self.assertEqual(labspace, fps["identifier"].to_list())


class TestRandomSubsetCandidateSQLiteDB(unittest.TestCase):
    def test_get_labelspace(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL-UHFFFAOYSA-N"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey", include_correct_candidate=False)
        self.assertEqual(102, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))

        # Enforce correct sturcture to be present
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey", include_correct_candidate=True)
        self.assertEqual(102, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))
        self.assertIn(spectrum.metadata["molecule_id"], candidates.get_labelspace(spectrum))

        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey1", include_correct_candidate=False)
        self.assertEqual(102, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))

        # Enforce correct sturcture to be present
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey1", include_correct_candidate=True)
        self.assertEqual(102, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))
        self.assertIn(spectrum.metadata["molecule_id"], candidates.get_labelspace(spectrum))

    def test_all_outputs_are_sorted_equally(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL-UHFFFAOYSA-N"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey", include_correct_candidate=True)

        scores = candidates.get_ms2_scores(spectrum, "MetFrag_2.4.5__8afe4a14", return_dataframe=True)
        fps = candidates.get_molecule_features(spectrum, "iokr_fps__positive", return_dataframe=True)
        labspace = candidates.get_labelspace(spectrum)

        self.assertEqual(sorted(labspace), labspace)
        self.assertEqual(labspace, scores["identifier"].to_list())
        self.assertEqual(labspace, fps["identifier"].to_list())

    def test_get_number_of_candidates(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016"})

        # Molecule identifier: inchikey
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey")
        self.assertEqual(102, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(102, candidates.get_n_cand(spectrum))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "EAX030601"})

        # Molecule identifier: inchikey
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey")
        self.assertEqual(16, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(16, candidates.get_n_cand(spectrum))


class TestSequence(unittest.TestCase):
    def test_get_number_of_candidates(self):
        spectra_ids = ["Challenge-016", "Challenge-017", "Challenge-018", "Challenge-019"]
        spectra = [Spectrum(np.array([]), np.array([]), {"spectrum_id": spectrum_id}) for spectrum_id in spectra_ids]
        sequence = Sequence(spectra=spectra, candidates=CandidateSQLiteDB(DB_FN))

        n_cand = sequence.get_n_cand()
        self.assertEqual(2233, n_cand[0])
        self.assertEqual(1130, n_cand[1])
        self.assertEqual(62,   n_cand[2])
        self.assertEqual(5784, n_cand[3])

    def test_get_label_space(self):
        # Create a spectrum sequence
        spectra_ids = ["Challenge-016", "Challenge-022", "Challenge-018", "Challenge-019", "Challenge-001"]
        spectra = [Spectrum(np.array([]), np.array([]), {"spectrum_id": spectrum_id}) for spectrum_id in spectra_ids]
        sequence = Sequence(spectra=spectra, candidates=CandidateSQLiteDB(DB_FN))

        # Get the label space
        labelspace = sequence.get_labelspace()
        self.assertEqual(len(spectra_ids), len(labelspace))

        # Compare number of candidates
        for ls, n in zip(labelspace, sequence.get_n_cand()):
            self.assertEqual(n, len(ls))


class TestSequenceSample(unittest.TestCase):
    def setUp(self) -> None:
        self.db = sqlite3.connect("file:" + DB_FN + "?mode=ro", uri=True)

        # Read in spectra and labels
        res = pd.read_sql_query("SELECT spectrum, molecule, rt, challenge FROM challenges_spectra "
                                "   INNER JOIN spectra s ON s.name = challenges_spectra.spectrum", con=self.db)
        self.spectra = [Spectrum(np.array([]), np.array([]),
                                 {"spectrum_id": spec_id, "retention_time": rt, "dataset": chlg})
                        for (spec_id, rt, chlg) in zip(res["spectrum"], res["rt"], res["challenge"])]
        self.labels = res["molecule"].to_list()

    def tearDown(self) -> None:
        self.db.close()

    def test_sequence_generation(self):
        # Generate sequence sample
        N = 100
        L_min = 10
        seq_sample = SequenceSample(self.spectra, self.labels, None, N=N, L_min=L_min, random_state=201)
        self.assertEqual(N, len(seq_sample))
        self.assertTrue(all([len(ss) == L_min for ss in seq_sample]))

    def test_train_test_splitting(self):
        # Generate sequence sample
        seq_sample = SequenceSample(self.spectra, self.labels, None, N=31, L_min=20, random_state=789)

        # Get a train test split
        train_seq, test_seq = seq_sample.get_train_test_split()

        # Check length
        self.assertEqual(7, len(test_seq))
        self.assertEqual(24, len(train_seq))

        # No intersection of molecules between training and test sequences
        for i, j in it.product(test_seq, train_seq):
            self.assertTrue(len(set(i.labels) & set(j.labels)) == 0)


# class TestCandidateSetMetIdent(unittest.TestCase):
#     def test_is_pickleable(self):
#         idir = "/home/bach/Documents/doctoral/data/metindent_ismb2016"
#         X, fps, mols, mols2cand = read_data(idir)
#         cand = CandidateSetMetIdent(mols, fps, mols2cand, idir=os.path.join(idir, "candidates"),
#                                     preload_data=False)
#
#         cand_pkl = pickle.loads(pickle.dumps(cand))


if __name__ == '__main__':
    unittest.main()
