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
from typing import List, Tuple


class Sequence(object):
    def __init__(self, spec_ids: List[str], cand_def="mass"):
        self.spec_ids = spec_ids
        self.cand_def = cand_def
        assert self.cand_def in ["mass", "mf", "fixed"]

        self.L = len(self.spec_ids)

        self.cand = None  # stores a reference to the candidates of each sequence element

    def __len__(self) -> int:
        """
        Return the length of the sequence.

        :return: scalar, length of the sequence
        """
        return self.L

    def _get_n_cand_sig(self, sig) -> int:
        """
        :param sig: scalar, index of the sequence element to get the number of candidates.
        :return: scalar, number of candidates
        """
        return len(self.cand[sig])

    def get_n_cand(self) -> List[int]:
        return [self._get_n_cand_sig(sig) for sig in range(len(self))]


class LabeledSequence(Sequence):
    """
    Class representing the a _labeled_ (MS, RT)-sequence (x, t, y) with associated molecular candidate set C.
    """
    def __init__(self, spec_ids: List[str], labels: List[str], cand_def="mass"):
        """
        :param spec_ids: list of strings, spectrum-ids belonging sequence
        :param labels: list of strings, ground truth molecule identifiers belonging to the spectra of the sequence
        :param cand_def: string, which method should be used to define the candidate set for each sequence spectrum.
            "mass", by mass-window
            "mf", by molecular formula
            "fixed", loaded a pre-defined set
        """
        self.labels = labels

        super(LabeledSequence, self).__init__(spec_ids=spec_ids, cand_def=cand_def)

    def as_Xy_input(self) -> Tuple[List[Sequence], List[str]]:
        """
        Return the (MS, RT)-sequence and ground truth label separately as input for the sklearn interface.

        Usage: sklearn.fit(*LabeledSequence(...).as_Xy_input)

        :return:
        """
        return self.spec_ids, self.labels


