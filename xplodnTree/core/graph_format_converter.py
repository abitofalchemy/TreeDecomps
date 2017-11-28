''' graph format converter ''' 
import networkx as nx
import pandas as pd
from utils import graph_name, Info
from os import path

def edgelist_to_dimacs(fname):
	g =nx.read_edgelist(fname, comments="%", data=False, nodetype=int)
	g.name = graph_name(fname)
	dimacsFiles = convert_nx_gObjs_to_dimacs_gObjs([g])
	return dimacsFiles#convert_nx_gObjs_to_dimacs_gObjs([g])


def edgelist_in_dimacs_out(ifname):
	'''
		args: graph is the input nx graph
		returns: output filename
	'''
	g = nx.read_edgelist(ifname, data=False)
	g.name = graph_name(ifname)
	ofname = '../datasets/{}.dimacs'.format(g.name)
	if path.exists(ofname):
		return None
	n = g.number_of_nodes()
	m = g.number_of_edges()
	edges = g.edges()
	edges = [(int(e[0]), int(e[1])) for e in edges]
	df = pd.DataFrame(edges)
	df.sort_values(by=[0], inplace=True)
	with open(ofname, 'w') as f:
		f.write('c {}\n'.format(g.name))
		f.write('p edge\t{}\t{}\n'.format(n+1,m))
		output_edges = lambda x: f.write("e\t{}\t{}\n".format(x[0]+1, x[1]+1))
		df.apply(output_edges, axis=1)
	if path.exists(ofname):
		Info("Wrote: %s"% ofname)
	# ToDo: a few extra checks could be added
	return ofname


