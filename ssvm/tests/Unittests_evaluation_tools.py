####
#
# The MIT License (MIT)
#
# Copyright 2019, 2020 Eric Bach <eric.bach@aalto.fi>
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
import unittest

from ssvm.evaluation_tools import get_topk_performance_csifingerid


class TestTopKPerformanceCSIFingerID(unittest.TestCase):
    def test_simple_cases(self):
        candidates = {
            0: {"score": np.array([7, 5, 5, 5, 3, 2]), "n_cand": 6, "index_of_correct_structure": 2},
            1: {"score": np.array([8, 8, 4, 3, 2, 2, 1]), "n_cand": 7, "index_of_correct_structure": 4},
            2: {"score": np.array([1, 1, 1]), "n_cand": 3, "index_of_correct_structure": np.nan},
            3: {"score": np.array([9, 8, 8, 8, 8, 1, 1]), "n_cand": 7, "index_of_correct_structure": 4},
            4: {"score": np.array([8, 7, 4, 3, 1]), "n_cand": 5, "index_of_correct_structure": 0},
            5: {"score": np.array([8, 8, 4, 3, 1, 0]), "n_cand": 6, "index_of_correct_structure": 1}
        }

        topk_cts, topk_acc = get_topk_performance_csifingerid(candidates)

        np.testing.assert_equal(topk_cts[0], 3 / 2)
        np.testing.assert_allclose(topk_cts[1], 31 / 12)
        np.testing.assert_allclose(topk_cts[2], 19 / 6)
        np.testing.assert_allclose(topk_cts[3], 15 / 4)
        np.testing.assert_allclose(topk_cts[4], 4.5)
        np.testing.assert_allclose(topk_cts[5], 5)
        np.testing.assert_allclose(topk_cts[6], 5)
