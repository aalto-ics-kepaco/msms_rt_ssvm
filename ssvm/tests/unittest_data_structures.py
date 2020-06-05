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
import unittest
import pickle

from ssvm.data_structures import CandidateSetMetIdent
from ssvm.examples.metabolite_identification import read_data


class TestCandidateSetMetIdent(unittest.TestCase):
    def test_is_pickleable(self):
        idir = "/home/bach/Documents/doctoral/data/metindent_ismb2016"
        X, fps, mols, mols2cand = read_data(idir)
        cand = CandidateSetMetIdent(mols, fps, mols2cand, idir=os.path.join(idir, "candidates"),
                                    preload_data=False)

        cand_pkl = pickle.loads(pickle.dumps(cand))


if __name__ == '__main__':
    unittest.main()
