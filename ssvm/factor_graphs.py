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
import numpy as np
import itertools as it

from typing import Union, Optional
from sklearn.utils import check_random_state


def get_random_spanning_tree(y, random_state: Optional[Union[int, np.random.RandomState]] = None,
                             remove_edges_with_zero_rt_diff: bool = True):
    """
    Sample a random spanning tree from the full MRF.

    :param y: Sequence, label sequence over which the MRF is defined

    :param random_state: None | int | instance of RandomState used as seed for the random tree sampling
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    :param remove_edges_with_zero_rt_diff: boolean, indicating whether edges with zero RT difference should be
        included in the random spanning tree.

    :return: networkx graph
    """
    random_state = check_random_state(random_state)

    # Output graph
    var_conn_graph = nx.Graph()

    # Add variable nodes and edges with random weight
    var = list(range(len(y)))
    for s in var:
        var_conn_graph.add_node(s, retention_time=y.get_retention_time(s))

    for s, t in it.combinations(var, 2):
        rt_s, rt_t = var_conn_graph.nodes[s]["retention_time"], var_conn_graph.nodes[t]["retention_time"]
        rt_diff_st = rt_t - rt_s

        if remove_edges_with_zero_rt_diff and (rt_s == rt_t):
            edge_weight = np.inf  # Such edges will not be chosen in the MST
        else:
            edge_weight = random_state.rand()

        var_conn_graph.add_edge(s, t, weight=edge_weight, rt_diff=rt_diff_st)

    # Get minimum spanning tree
    var_conn_graph = nx.algorithms.tree.minimum_spanning_tree(var_conn_graph)

    return var_conn_graph


def identity(x):
    """
    :param x: array-like, with any shape

    :return: array-like, with same shape
    """
    return x
