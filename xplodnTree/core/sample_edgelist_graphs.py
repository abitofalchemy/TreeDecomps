from utils import load_edgelist, Info
from sample_edgelist_graphs import *
from td_rndGraphs import *
import networkx as nx

def get_hrg_production_rules (fname):
	import graph_sampler as gs
	G = load_edgelist(fname)
	G.remove_edges_from(G.selfloop_edges())
	giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
	G = nx.subgraph(G, giant_nodes)
	Info(str(G.number_of_nodes()))
	if G.number_of_nodes() >= 500:
		Info('Grande')
		for Gprime in gs.rwr_sample(G, 2, 300):
			td([Gprime])
#	else:
#		T = td.quickbb(G)
#		root = list(T)[0]
#		T = td.make_rooted(T, root)
#		T = phrg.binarize(T)
#		root = list(T)[0]
#		root, children = T
#			# td.new_visit(T, G, prod_rules, TD)
#			td.new_visit(T, G, prod_rules)
#			
#			# print_treewidth(T) # TODO: needs to be fixed
#			exit()

import sys
get_hrg_production_rules(sys.argv[1])
