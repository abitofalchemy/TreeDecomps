import networkx as nx
import numpy as np
import pandas as pd
import net_metrics as met
import PHRG
import probabilistic_cfg as pcfg

# StarLog
# [-] generate a synthetic graph when GCD is requested
# - build compute_net_properties_modularity using igraph and networkx

#def compute_degree_distribution(G):

#def compute_hopplot(G):

#def compute_modularity(G):
def compute_net_properties_modularity(nx_graph):
	print "compute_net_properties_modularity"

def synthetic_graph_generator (ref, graph_model):
	G = ref
	synth_graph = None
	n = ref.number_of_nodes()

	if 'hrg' in graph_model:
		prod_rules = PHRG.probabilistic_hrg_learning(G) # derive rules
		g = pcfg.Grammar('S')
		for (id, lhs, rhs, prob) in prod_rules:
			g.add_rule(pcfg.Rule(id, lhs, rhs, prob))
		num_nodes = G.number_of_nodes()
		# print "Starting max size",'n=',num_nodes
		g.set_max_size(num_nodes)
		# print "Done with max size"
		Hstars = []
		rule_list = g.sample(num_nodes)
		synth_graph = PHRG.grow(rule_list, g)[0]

	return synth_graph



def compute_net_properties_gcd( graph, graphgen ):
	'''
	graphgen graph generator such as HRG, KRON, ChungLu, or BTER.
	'''

	# synthic graph
	sG = synthetic_graph_generator (ref = graph , graph_model = graphgen)

	# computer GCD
	print 'now compute gcd'

def compute_net_properties( edgelist_graph, net_properties_2compute ):
	'''
	edgelist_graph is a networkx graph object
	'''
	if not net_properties_2compute:
		return

	fname = edgelist_graph
	G = nx.read_edgelist( fname, comments="%", nodetype=np.int64)

	# print net_properties_2compute

	# results
	results = []
	if 'degree' in net_properties_2compute:
		results.append(met.degree_distribution_multiples([G]))
	if 'hopplot' in net_properties_2compute:
		results.append(met.hop_plot_multiples([G]))
	if 'modularity' in net_properties_2compute:
		print 'compute modularity'
		results.append(compute_net_properties_modularity(G))

	if 'gcd' in net_properties_2compute:
		results.append(compute_net_properties_gcd( G, 'hrg' ))

	return results
