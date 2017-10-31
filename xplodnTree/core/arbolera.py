__version__="0.1.0"

import pandas as pd #import DataFrame
from os import path
import networkx as nx
from collections import defaultdict
from itertools import combinations
from isomorph_interxn import label_match

DBG = False

def rhs_tomultigraph (rhs_clean):
	'''
	Parse the RHS of each rule into a graph fragment
	:param x:
	:return:
	'''
	import re
	import networkx as nx
	rhs_clean= [f[1:-1] for f in re.findall("'.+?'", rhs_clean)]

	# rhs_clean = [f[1:-1] for f in re.findall("[^()]+", x)]
	G1 = nx.MultiGraph()
	for he in rhs_clean:
		epair, ewt = he.split(':')
		if ewt is "T":
			if len(epair.split(",")) == 1:
				[G1.add_node(epair, label=ewt)]
			else:
				[G1.add_edge(epair.split(",")[0], epair.split(",")[1], label=ewt)]
		elif ewt is "N":
			if len(epair.split(",")) == 1:
				[G1.add_node(epair, label=ewt)]
			else:
				[G1.add_edges_from(list(combinations(epair.split(","), 2)), label=ewt)]

	return G1

def jacc_dist_for_pair_dfrms(df1, df2):
	"""
	df1 and df2 are each a dataframe (sets) to use for comparison
	returns: jaccard similarity score
	"""
	slen = len(df1)
	tlen = len(df2)
	# +++
	conc_df = pd.concat([df1, df2])
	#	print ">>>", conc_df.shape
	# ---
	seen_rules = defaultdict(list)
	ruleprob2sum = defaultdict(list)
	cnrules = []
	cntr = 0
#	DBG = True
	for r in conc_df.iterrows(): #/* for each rule in the stack */
		if r[1]['lhs'] not in seen_rules.keys():
#			print r[1]['rnbr'],
			seen_rules[r[1]['lhs']].append(r[1]['rnbr'])
			cnrules.append(r[1]['rnbr'])
			if DBG: print "+"
			cntr += 1
		else:	# lhs already seen
#			print r[1]['rnbr'],
			# print df1[df1['rnbr']==seen_rules[r[1]['lhs']][0]]['rhs'].values
			# check the current rhs if the lhs matches to something already seen and check for an isomorphic match
			# rhs1 = listify_rhs(r[1]['rhs'])
			rhs1 = r[1]['rhs']
			rhs2 = conc_df[conc_df['rnbr'] == seen_rules[r[1]['lhs']][0]]['rhs'].values[0]
#			rhs2 = conc_df[conc_df['rnbr'] == seen_rules[r[1]['lhs']][0]]['rhs']
			G1 = rhs_tomultigraph(rhs1)
			G2 = rhs_tomultigraph(rhs2)
#			for rl in rhs2.values:
#				G2 = rhs_tomultigraph(rl)
        #
			# if nx.is_isomorphic(G1, G2, edge_match=label_match):
			if nx.faster_could_be_isomorphic(G1, G2):
				if DBG: print ' <-curr', seen_rules[r[1]['lhs']][0], ':', conc_df[conc_df['rnbr'] == seen_rules[r[1]['lhs']][0]]['rnbr'].values, conc_df[conc_df['rnbr'] == seen_rules[r[1]['lhs']][0]]['cate'].values
				ruleprob2sum[seen_rules[r[1]['lhs']][0]].append(r[1]['rnbr'])
				seen_rules[r[1]['lhs']].append(r[1]['rnbr'])
			else:
				seen_rules[r[1]['lhs']].append(r[1]['rnbr'])
				cnrules.append(r[1]['rnbr'])
				if DBG: print "+"
				cntr += 1


	if DBG: print "len(ruleprob2sum)", len(ruleprob2sum)
	from json import dumps
	if DBG: print	dumps(ruleprob2sum, indent=4, sort_keys=True)
	# print ruleprob2sum
	if DBG: print "  Overlapping rules	", len(ruleprob2sum.keys()), sum([len(x) for x in ruleprob2sum.values()])
	if DBG: print "  Jaccard Sim:\t", (len(ruleprob2sum.keys())+sum([len(x) for x in ruleprob2sum.values()]))/ float(len(df1) + len(df2))

	print df1.groupby(['cate']).groups.keys()[0].split('_prules')[0], df2.groupby(['cate']).groups.keys()[0].rstrip('_prules'),

	return (len(ruleprob2sum.keys())+sum([len(x) for x in ruleprob2sum.values()]))/ float(len(df1) + len(df2))



def nx_edges_to_nddgo_graph(G,n,m, sampling=False, varel="", save_g=False):
	ofname = '../datasets/{}.dimacs'.format(G.name, n,m,varel)
	if path.exists(ofname):
		return 
	if sampling:
		edges = G.edges()
		edges = [(int(e[0]), int(e[1])) for e in edges]
		df = pd.DataFrame(edges)
		df.sort_values(by=[0], inplace=True)
		dimacs_graph =[]
		if save_g:
			with open(ofname, 'w') as f:
				f.write('c {}\n'.format(G.name))
				f.write('p edge\t{}\t{}\n'.format(n+1,m))

				output_edges = lambda x: f.write("e\t{}\t{}\n".format(x[0]+1, x[1]+1))
				df.apply(output_edges, axis=1)

#			if path.exists(ofname): print 'Wrote: ./{}'.format(ofname)
		else:
			output_edges = lambda x: "e\t{}\t{}\n".format(x[0]+1, x[1]+1)
			dimacs_graph = df.apply(output_edges, axis=1)
	else:
		edges = G.edges()
		edges = [(int(e[0]), int(e[1])) for e in edges]
		df = pd.DataFrame(edges)
		df.sort_values(by=[0], inplace=True)
		if save_g:
			with open(ofname, 'w') as f:
				f.write('c {}\n'.format(G.name))
				f.write('p edge\t{}\t{}\n'.format(n+1,m))

				output_edges = lambda x: f.write("e\t{}\t{}\n".format(x[0]+1, x[1]+1))
				df.apply(output_edges, axis=1)

#			if path.exists(ofname): print 'Wrote: ./{}'.format(ofname)
		else:
			output_edges = lambda x: "e\t{}\t{}\n".format(x[0]+1, x[1]+1)
			dimacs_graph =df.apply(output_edges, axis=1)
	if save_g:
		if path.exists(ofname): print '\tWrote: ./{}'.format(ofname)
		return [ofname]
	else:
		return dimacs_graph
