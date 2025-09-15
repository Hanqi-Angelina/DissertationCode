from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Edge import Edge
from causallearn.utils.DAG2CPDAG import dag2cpdag
from GraphDrawer import AdjToGraph
import pydot
import pandas as pd

import numpy as np

def convert_to_GeneralGraph(m, var_names): 
    """
    Convert the adjacency matrix to a causallearn object.
    """
    nodes = [GraphNode(x) for x in var_names]
    G = GeneralGraph(nodes)
    for i in range(0, m.shape[0]):
        for j in range(i+1, m.shape[0]):
            if m[j,i] == 1 and m[i,j] == -1: 
                G.add_directed_edge(nodes[i], nodes[j])
            elif m[j,i] == -1 and m[i,j] == 1: 
                G.add_directed_edge(nodes[j], nodes[i])
    return(G)

def signed_to_CPDAG(m, var_names): 
    """
    Convert the signed adjacency matrix to CPDAG.
    """
    if np.any((m == 1) & (m.T == 1)) or np.any((m == -1) & (m.T == -1)): # check for any notations that does not match a directed edge
        return m
    else: 
        G = convert_to_GeneralGraph(m, var_names)
        CPDAG = dag2cpdag(G)
        return(CPDAG.graph)

def toGraph(M, name, var_names):
    """
    Convert the adjancency matrix to graphs. 
    """
    AdjToGraph(M, var_names).toDot(f'true_DAGs/{name}.dot')
    g, = pydot.graph_from_dot_file(f'true_DAGs/{name}.dot')
    g.write_png(f'true_DAGs/{name}.png', prog='dot')

def adj_to_dataframe(dgm):
    """
    Gets the adjacency matrix and saves it to a CSV file
    with variable names from self.vars as headers.
    """
    # Get the unlabeled NumPy adjacency matrix
    adj_numpy = dgm.get_causallearn_adj()

    # Get the list of all variable names from the class instance
    all_vars = dgm.vars

    # Convert the NumPy array to a pandas DataFrame and add labels
    adj_df = pd.DataFrame(adj_numpy, index=all_vars, columns=all_vars)

    return adj_df
