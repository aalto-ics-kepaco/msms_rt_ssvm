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
from ssvm.examples.metabolite_identification import read_data


DB_FN = "/home/bach/Documents/doctoral/projects/local_casmi_db/db/use_inchis/DB_LATEST.db"


class TestSequence(unittest.TestCase):
    def setUp(self) -> None:
        self.db = sqlite3.connect("file:" + DB_FN + "?mode=ro", uri=True)

    def tearDown(self) -> None:
        self.db.close()

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


class TestCandidateSetMetIdent(unittest.TestCase):
    def test_is_pickleable(self):
        idir = "/home/bach/Documents/doctoral/data/metindent_ismb2016"
        X, fps, mols, mols2cand = read_data(idir)
        cand = CandidateSetMetIdent(mols, fps, mols2cand, idir=os.path.join(idir, "candidates"),
                                    preload_data=False)

        cand_pkl = pickle.loads(pickle.dumps(cand))


if __name__ == '__main__':
    unittest.main()
