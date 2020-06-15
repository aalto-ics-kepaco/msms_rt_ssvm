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

from typing import Dict, Tuple, Optional
from scipy.stats import rankdata


def get_topk_performance_csifingerid(candidates: Dict, scores: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Topk performance calculation after [1]. (see 'get_topk_performance_from_scores')
    """
    cscnt = np.zeros((np.max([cnd["n_cand"] for cnd in candidates.values()]) + 2,))

    for i in candidates:
        # If the correct candidate is not in the set
        if np.isnan(candidates[i]["index_of_correct_structure"]):
            continue

        if scores is None:
            # Use ranking based on the candidate scores
            _scores = - candidates[i]["score"]
        else:
            # Use ranking based on the marginal scores after MS and RT integration
            _scores = - scores[i]

        # Calculate ranks
        _ranks = rankdata(_scores, method="ordinal") - 1

        # Get the contribution of the correct candidate
        _s = _scores[candidates[i]["index_of_correct_structure"]]
        _c = (1. / np.sum(_scores == _s)).item()
        _r = _ranks[_scores == _s]

        # For all candidate with the same score, we update their corresponding ranks with their contribution
        cscnt[_r] += _c

    cscnt = np.cumsum(cscnt)

    return cscnt, cscnt / len(candidates) * 100