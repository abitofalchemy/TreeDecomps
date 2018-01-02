#!/usr/bin/env python
__version__ = "0.1.0"

import os
import re
import sys

import networkx as nx
import pandas as pd

from core.PHRG import binarize, graph_checks
from core.graph_sampler import rwr_sample
from core.tree_decomposition import quickbb, make_rooted, new_visit
from core.utils import load_edgelist, Info, graph_name


def get_sampled_gpickled_graphs(G):
	G.remove_edges_from(G.selfloop_edges())
	print ([x.number_of_nodes() for x in sorted(nx.connected_component_subgraphs(G), key=len)])
	# print ([x.number_of_nodes() for x in list(nx.connected_component_subgraphs(G))])
	giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
	G = nx.subgraph(G, giant_nodes)
	num_nodes = G.number_of_nodes()
	graph_checks(G)

	prod_rules = {}
	K = 2
	n = 300

	j = 0
	if G.number_of_nodes() >500:
		for Gprime in rwr_sample(G, K, n):
			nx.write_gpickle(Gprime, "../datasets/{}_{}.p".format(gn,str(j)))
			T = quickbb(Gprime)
			root = list(T)[0]
			T = make_rooted(T, root)
			T = binarize(T)
			root = list(T)[0]
			root, children = T
			# td.new_visit(T, G, prod_rules, TD)
			new_visit(T, G, prod_rules)
			j += 1
	else:
		nx.write_gpickle (G, "../datasets/{}.p".format (gn))
		T = quickbb (G)
		root = list (T)[0]
		T = make_rooted (T, root)
		T = binarize (T)
		root = list (T)[0]
		root, children = T
		# td.new_visit(T, G, prod_rules, TD)
		new_visit (T, G, prod_rules)
	## 
	return prod_rules


if __name__ == '__main__':
	if len (sys.argv) < 2:
		Info ("Usage:")
		Info ("python xplotree_subgraphs_prs.py path/to/orig_net_edgelist")
		sys.exit (1)
	elif sys.argv[1] == "-ut":
		fname = "/Users/sal.aguinaga/KynKon/datasets/out.as20000102"
	else:
		fname = sys.argv[1]

	if not os.path.exists (fname):
		Info ("Path to edgeslits does not exists.")
		sys.exit (1)
	gn = graph_name (fname)
	prsfname = '../ProdRules/{}.tsv.phrg.prs'.format(gn)
	if os.path.exists(prsfname):
		Info('{} already exists'.format(prsfname))
		sys.exit(0)
	og = load_edgelist (fname)
	og.name = gn
	# sgp = glob("../datasets/"+ gn + "*.p" )

	print ("--")
	print ("-- derive subgraphs")
	print ("--")

	Info ("sample 2 subg of 300 nodes and derive the set of production rules")

	prod_rules = get_sampled_gpickled_graphs(og)
	# pp.pprint(pr)
	DBG = False

	for k in prod_rules.iterkeys():
		if DBG: print (k)
		s = 0
		for d in prod_rules[k]:
			s += prod_rules[k][d]
		for d in prod_rules[k]:
			prod_rules[k][d] = float(prod_rules[k][d]) / float(s)  # normailization step to create probs not counts.
			if DBG: print ('\t -> ', d, prod_rules[k][d])
	rules = []
	id = 0
	for k, v in prod_rules.iteritems():
		sid = 0
		for x in prod_rules[k]:
			rhs = re.findall("[^()]+", x)
			rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
			if DBG: print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])
			sid += 1
		id += 1

	df = pd.DataFrame(rules)
	# pp.pprint(df.values.tolist()); exit()

	df.to_csv('../ProdRules/{}.tsv.phrg.prs'.format(gn), header=False, index=False, sep="\t")
	if os.path.exists('../ProdRules/{}.tsv.phrg.prs'.format(gn)):
		print ('Saved', '../ProdRules/{}.tsv.phrg.prs'.format(gn))
	else:
		print ("Trouble saving")