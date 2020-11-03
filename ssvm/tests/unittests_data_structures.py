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

from ssvm.data_structures import CandidateSetMetIdent, SequenceSample
from ssvm.development.ssvm_metident__conv_params import read_data


class TestSequenceSample(unittest.TestCase):
    def setUp(self) -> None:
        db_fn = "/home/bach/Documents/doctoral/projects/local_casmi_db/db/use_inchis/DB_LATEST.db"
        self.db = self.db = sqlite3.connect("file:" + db_fn + "?mode=ro", uri=True)

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

        # for i, (spectrum, label) in enumerate(seq_sample[11]):
        #     print(spectrum.get("spectrum_id"), spectrum.get("retention_time"), label)
        #     if i > 9:
        #         break

    def test_train_test_splitting(self):
        # Generate sequence sample
        N = 31
        L_min = 20
        seq_sample = SequenceSample(self.spectra, self.labels, None, N=N, L_min=L_min, random_state=789)

        train_seq, test_seq = seq_sample.get_train_test_split()

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
