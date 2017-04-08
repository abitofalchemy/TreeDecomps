import networkx as nx
import numpy as np 

G = nx.karate_club_graph()
edges = list(G.edges())
n = G.number_of_nodes()

import igraph as ig
G = ig.Graph()
G.add_vertices(n)
G.add_edges(edges)
print (G)

