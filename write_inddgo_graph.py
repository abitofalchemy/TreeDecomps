import os
import networkx as nx
from datetime import datetime

def nx_edges_to_nddgo_graph(G):
	if G.name is None:
		t_str = datetime.now().strftime("%Y-%m-%d_%H_%M")
	edges = G.edges()	
	ofname = '{}.graph'.format(G.name)
	with open (ofname, 'w') as f:
		f.write('c {}\n'.format(G.name))
		f.write('p edge\t{}\t{}\n'.format(G.number_of_nodes(), G.number_of_edges()))
		for e in edges:
			f.write("e\t{}\t{}\n".format(e[0]+1,e[1]+1))
