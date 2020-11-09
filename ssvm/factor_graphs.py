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
import networkx as nx

from collections import OrderedDict
from typing import Union, Dict, Callable, Optional
from msmsrt_scorer.lib.exact_solvers import TreeFactorGraph

from ssvm.data_structures import Sequence


def identity(x):
    """
    :param x: array-like, with any shape

    :return: array-like, with same shape
    """
    return x


class ChainFactorGraph(TreeFactorGraph):
    def __init__(self, candidates: Union[OrderedDict, Dict], make_order_probs: Callable, D: float = 0.5,
                 order_probs: Optional[Dict] = None, use_log_space: bool = True, norm_order_scores: bool = False):
        """

        :param candidates:
        :param make_order_probs:
        :param order_probs:
        :param use_log_space:
        :param D:
        :param norm_order_scores:
        """
        super().__init__(candidates=candidates, make_order_probs=make_order_probs, order_probs=order_probs,
                         use_log_space=use_log_space, D=D, norm_order_scores=norm_order_scores,
                         var_conn_graph=self._get_chain_connectivity(candidates))

    @staticmethod
    def _get_chain_connectivity(candidates: Union[OrderedDict, Dict, Sequence]) -> nx.Graph:
        """

        :param candidates:

        :return: networkx.Graph
        """
        var_conn_graph = nx.Graph()

        # Add variable nodes
        if isinstance(candidates, dict) or isinstance(candidates, OrderedDict):
            var = list(candidates.keys())
        elif isinstance(candidates, Sequence):
            var = list(range(len(candidates)))
        else:
            raise ValueError("Invalid input type.")
        for i in var:
            var_conn_graph.add_node(i)

        # Add edges connecting the variable nodes, i.e. pairs considered for the score integration
        for idx in range(len(var) - 1):
            var_conn_graph.add_edge(var[idx], var[idx + 1])

        return var_conn_graph
