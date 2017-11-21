''' Connected Components in Datasets '''
import sys
import os
import pandas as pd
import networkx as nx
# from core.load_edgelist_from_dataframe import Pandas_DataFrame_From_Edgelist
from td_rndGraphs import load_edgelist

if len(sys.argv) <2:
    print ("usage: python .py path/to/daataset")
    exit()

G = load_edgelist(sys.argv[1])
Gc = max(nx.connected_component_subgraphs(G), key=len) # largest connected component
total =  len(list(nx.connected_component_subgraphs(G))) # [g.number_of_nodes() for g in sorted(list(nx.connected_component_subgraphs(G)),reverse=True)]
# print nx.info(Gc)
'''LCC (in terms of nbr of nodes, from a total nbr of connected components in a graph)'''
print ("LCC, |V|: {} in {} total: {}".format(Gc.number_of_nodes(), total, G.name))
