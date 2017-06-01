#!/usr/bin/env python
__author__ = 'saguinag' + '@' + 'nd.edu'
__version__ = "0.1.0"

##
## Description:
##

## TODO: some todo list

## VersionLog:

import argparse
import os
import re
import sys
import traceback
from collections import defaultdict
from itertools import combinations
from glob import glob
from json import dumps
import numpy as np
import networkx as nx
import pandas as pd

import tdec.PHRG as phrg
import tdec.tree_decomposition as td
from tdec.PHRG import graph_checks
from	 tdec.a1_hrg_cliq_tree import load_edgelist
from tdec.signature import hello

DBG=False


def get_parser ():
	parser = argparse.ArgumentParser(description='From dimacs tree to isomorphic reduced prod trees')
	parser.add_argument('--orig', required=True, help='Input tree edgelist file (dimacs)')
	parser.add_argument('--pathfrag', required=True, help='Input dimacs tree path fragment')
	parser.add_argument('--cnts', action='store_true', default=False, required=0,help='Just counts')
	parser.add_argument('--version', action='version', version=__version__)
	return parser

def Log_Info(str_to_print=""):
		if len(str_to_print)<8:
				print "~"*8

def listify_rhs(rhs_rule):
	if DBG: print type(rhs_rule), len(rhs_rule)
	rhs_clean= [f[1:-1] for f in re.findall("'.+?'", rhs_rule)]
	return rhs_clean

def rhs_tomultigraph(rhs_clean):
	'''
	Parse the RHS of each rule into a graph fragment
	:param x:
	:return:
	'''
	import re
	import networkx as nx

	# rhs_clean= [f[1:-1] for f in re.findall("'.+?'", x)]

	# rhs_clean = [f[1:-1] for f in re.findall("[^()]+", x)]
	G1 = nx.MultiGraph()
	for he in rhs_clean:
		epair,ewt = he.split(':')
		if ewt is "T":
			if len(epair.split(",")) == 1:	[G1.add_node(epair, label=ewt)]
			else: [G1.add_edge(epair.split(",")[0], epair.split(",")[1], label=ewt)]
		elif ewt is "N":
			if len(epair.split(",")) == 1:	[G1.add_node(epair, label=ewt)]
			else: [G1.add_edges_from(list(combinations(epair.split(","), 2)),label=ewt )]

	return G1

def rhs2multigraph(x):
	'''
	Parse the RHS of each rule into a graph fragment
	:param x:
	:return:
	'''
	import re
	from itertools import combinations
	import networkx as nx

	rhs_clean=[f[1:-1] for f in re.findall("'.+?'", x)]
	# rhs_clean = [f[1:-1] for f in re.findall("[^()]+", x)]
	G1 = nx.MultiGraph()
	for he in rhs_clean:
		epair,ewt = he.split(':')
		if ewt is "T":
			if len(epair.split(",")) == 1:	[G1.add_node(epair, label=ewt)]
			else: [G1.add_edge(epair.split(",")[0], epair.split(",")[1], label=ewt)]
		elif ewt is "N":
			if len(epair.split(",")) == 1:	[G1.add_node(epair, label=ewt)]
			else: [G1.add_edges_from(list(combinations(epair.split(","), 2)),label=ewt )]

	return G1



def isomorphic_test_from_dimacs_tree(orig, tdfname, gname="", iargs=""):
	# if whole tree path
	# else, assume a path fragment
	print '... path fragment:', tdfname
	print '... input graph	:', orig


	G = load_edgelist(orig) # load edgelist into a graph obj
	N = G.number_of_nodes()
	M = G.number_of_edges()
	# +++ Graph Checks
	if G is None: sys.exit(1)
	G.remove_edges_from(G.selfloop_edges())
	giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
	G = nx.subgraph(G, giant_nodes)
	graph_checks(G)
	# --- graph checks

	G.name = gname

	files = glob(tdfname+"*.dimacs.tree")
	prod_rules = {}
	stacked_df = pd.DataFrame()

	mat_dict = {}
	for i,x in enumerate(sorted(files)):
		mat_dict[os.path.basename(x).split(".")[0].split("_")[-1]]=i
		if DBG: print os.path.basename(x).split(".")[0].split("_")[-1]

	for tfname in files:
		tname = os.path.basename(tfname).split(".")
		tname = "_".join(tname[:2])

		with open(tfname, 'r') as f:	# read tree decomp from inddgo
			lines = f.readlines()
			lines = [x.rstrip('\r\n') for x in lines]

		cbags = {}
		bags = [x.split() for x in lines if x.startswith('B')]

		for b in bags:
			cbags[int(b[1])] = [int(x) for x in b[3:]]	# what to do with bag size?

		edges = [x.split()[1:] for x in lines if x.startswith('e')]
		edges = [[int(k) for k in x] for x in edges]

		tree = defaultdict(set)
		for s, t in edges:
			tree[frozenset(cbags[s])].add(frozenset(cbags[t]))
			if DBG: print '.. # of keys in `tree`:', len(tree.keys())

		root = list(tree)[0]
		root = frozenset(cbags[1])
		T = td.make_rooted(tree, root)
		# nfld.unfold_2wide_tuple(T) # lets me display the tree's frozen sets

		T = phrg.binarize(T)
		# root = list(T)[0]
		# root, children = T
		# td.new_visit(T, G, prod_rules, TD)
		# print ">>",len(T)

		td.new_visit(T, G, prod_rules)


		for k in prod_rules.iterkeys():
			if DBG: print k
			s = 0
			for d in prod_rules[k]:
				s += prod_rules[k][d]
			for d in prod_rules[k]:
				prod_rules[k][d] = float(prod_rules[k][d]) / float(s)	# normailization step to create probs not counts.
				if DBG: print '\t -> ', d, prod_rules[k][d]

		if DBG: print "--------------------"
		if DBG: print '- Prod. Rules'
		if DBG: print "--------------------"
		rules = []
		id = 0
		for k, v in prod_rules.iteritems():
			sid = 0
			for x in prod_rules[k]:
				rhs = re.findall("[^()]+", x)
				rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
				if DBG: print "r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]
				sid += 1
			id += 1

		df = pd.DataFrame(rules)
		print df.shape
		df['cate'] = tname
		stacked_df = pd.concat([df, stacked_df])

	if iargs['cnts']:
		return stacked_df,mat_dict
	else:
		np_sqr_mtrx = jaccard_coeff_isomorphic_rules_check(stacked_df, mat_dict)
		print gname
		df = pd.DataFrame(np_sqr_mtrx, columns=[x for x in sorted(mat_dict.keys())])
		df.index = sorted(mat_dict.keys())
		df.to_csv("Results/{}_isom_jaccardsim.tsv".format(gname), sep=",")
		
		return stacked_df,mat_dict #ToDo: not sure if I want to return this


def label_match(x, y):
	return x[0]['label'] == y[0]['label']

def jacc_dist_for_pair_dfrms(df1, df2):
	slen = len(df1)
	tlen = len(df2)
	# +++
	conc_df = pd.concat([df1, df2])
	# ---
	seen_rules = defaultdict(list)
	ruleprob2sum = defaultdict(list)
	cnrules = []
	cntr = 0
	for r in conc_df.iterrows():
		if DBG: print r[1]['rnbr'],
		if r[1]['lhs'] not in seen_rules.keys():
			seen_rules[r[1]['lhs']].append(r[1]['rnbr'])
			cnrules.append(r[1]['rnbr'])
			if DBG: print "+"
			cntr += 1
		else:	# lhs already seen
			# print df1[df1['rnbr']==seen_rules[r[1]['lhs']][0]]['rhs'].values
			# check the current rhs if the lhs matches to something already seen and check for an isomorphic match
			# rhs1 = listify_rhs(r[1]['rhs'])
			rhs1 = r[1]['rhs']
			rhs2 = conc_df[conc_df['rnbr'] == seen_rules[r[1]['lhs']][0]]['rhs'].values[0]
			G1 = rhs_tomultigraph(rhs1)
			G2 = rhs_tomultigraph(rhs2)
			if nx.faster_could_be_isomorphic(G1, G2):
				# print ' ',r[1]['rnbr'], r[1]['rhs'], '::', df1[df1['rnbr'] == seen_rules[r[1]['lhs']][0]]['rhs'].values
				if DBG: print ' <-curr', seen_rules[r[1]['lhs']][0], ':', conc_df[conc_df['rnbr'] == seen_rules[r[1]['lhs']][0]]['rnbr'].values, conc_df[conc_df['rnbr'] == seen_rules[r[1]['lhs']][0]]['cate'].values
				ruleprob2sum[seen_rules[r[1]['lhs']][0]].append(r[1]['rnbr'])
			else:
				seen_rules[r[1]['lhs']].append(r[1]['rnbr'])
				cnrules.append(r[1]['rnbr'])
				if DBG: print "+"
				cntr += 1
	
	if DBG: print "len(ruleprob2sum)", len(ruleprob2sum)
	if DBG: print	dumps(ruleprob2sum, indent=4, sort_keys=True)

	if not DBG: print "Overlapping rules	", len(ruleprob2sum.keys()), sum([len(x) for x in ruleprob2sum.values()])
	if DBG: print "Jaccard Sim:\t", (len(ruleprob2sum.keys())+sum([len(x) for x in ruleprob2sum.values()]))/ float(len(df1) + len(df2))
	return (len(ruleprob2sum.keys())+sum([len(x) for x in ruleprob2sum.values()]))/ float(len(df1) + len(df2))

def jaccard_coeff_isomorphic_rules_check_forfilepair(pr_grpby, mdf):
	if DBG: print pr_grpby[0], pr_grpby[1],
	return jacc_dist_for_pair_dfrms(mdf[mdf['cate']==pr_grpby[0]], \
													 mdf[mdf['cate']==pr_grpby[1]] )


def jaccard_coeff_isomorphic_rules_check(dfrm, headers_d):
	if dfrm.empty: return

	dfrm.columns = ['rnbr', 'lhs', 'rhs', 'pr', 'cate']
	gb = dfrm.groupby(['cate']).groups
	if DBG: print gb.keys()
	sqr_mtrx = np.zeros(shape=(len(headers_d),len(headers_d)))

	for p in combinations(sorted(gb.keys()), 2):
		if DBG: print [x.split("_")[1] for x in p],
		if DBG: print [headers_d[x.split("_")[1]] for x in p] #[0].split("_")[-1]
		j = headers_d[p[0].split("_dimacs")[0].split("_")[-1]]
		i = headers_d[p[1].split("_dimacs")[0].split("_")[-1]]
		sqr_mtrx[i,j] = jaccard_coeff_isomorphic_rules_check_forfilepair(p, dfrm)
		# break
		# sqr_mtrx[[headers_d[x.split("_")[1]] for x in p]] = jaccard_coeff_isomorphic_rules_check_forfilepair(p, dfrm)
	Log_Info()
	print sqr_mtrx
	return sqr_mtrx # numpy.savetxt("foo.csv", a, delimiter=",")

	exit()

	seen_rules = defaultdict(list)
	ruleprob2sum = defaultdict(list)
	cnrules = []
	cntr = 0

	for r in dfrm.iterrows():
		if DBG: print r[1]['rnbr'],
		if r[1]['lhs'] not in seen_rules.keys():
			seen_rules[r[1]['lhs']].append(r[1]['rnbr'])
			cnrules.append(r[1]['rnbr'])
			if DBG: print "+"
			cntr += 1
		else:	# lhs already seen
			# print df1[df1['rnbr']==seen_rules[r[1]['lhs']][0]]['rhs'].values
			# check the current rhs if the lhs matches to something already seen and check for an isomorphic match
			# rhs1 = listify_rhs(r[1]['rhs'])
			rhs1 = r[1]['rhs']
			rhs2 = dfrm[dfrm['rnbr'] == seen_rules[r[1]['lhs']][0]]['rhs'].values[0]
			G1 = rhs_tomultigraph(rhs1)
			G2 = rhs_tomultigraph(rhs2)
			if nx.faster_could_be_isomorphic(G1, G2):
				# print ' ',r[1]['rnbr'], r[1]['rhs'], '::', df1[df1['rnbr'] == seen_rules[r[1]['lhs']][0]]['rhs'].values
				if DBG: print ' <-curr', seen_rules[r[1]['lhs']][0], ':', dfrm[dfrm['rnbr'] == seen_rules[r[1]['lhs']][0]]['rnbr'].values, dfrm[dfrm['rnbr'] == seen_rules[r[1]['lhs']][0]]['cate'].values
				ruleprob2sum[seen_rules[r[1]['lhs']][0]].append(r[1]['rnbr'])
			else:
				seen_rules[r[1]['lhs']].append(r[1]['rnbr'])
				cnrules.append(r[1]['rnbr'])
				if DBG: print "+"
				cntr += 1

#	for k in ruleprob2sum.keys():
#		if DBG: print k
#		if DBG: print "	", ruleprob2sum[k]
#		if DBG: print "	", dfrm[dfrm['rnbr'] == k]['pr'].values+ sum(dfrm[dfrm['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
#		# dfrm[dfrm['rnbr'] == k]['pr'] += sum(dfrm[dfrm['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
#		c_val = dfrm[dfrm['rnbr'] == k]['pr'].values	+ sum(dfrm[dfrm['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
#		dfrm.set_value(dfrm[dfrm['rnbr'] == k].index, 'pr', c_val)
#		for r in ruleprob2sum[k]:
#			dfrm = dfrm[dfrm.rnbr != r]
#	print dfrm.shape

	# cnrules contains the rules we need to reduce df1 by
	# and ruleprob2sum will give us the new key for which pr will change.
	#	df1.to_csv("./ProdRules/"+name+"_prules.bz2",sep="\t", header="False", index=False, compression="bz2")
	return True

def isomorphic_check(prules, name):
	print '-' * 20
	print 'Isomorphic rules check (within file)'
	# for f in files:
	#	 df1 = pd.read_csv(f, index_col=0, compression='bz2', dtype=dtyps)
	df1 = pd.DataFrame(prules)
	df1.columns = ['rnbr', 'lhs', 'rhs', 'pr']
	print '... rules', df1.shape, 'reduced to',
	seen_rules = defaultdict(list)
	ruleprob2sum = defaultdict(list)
	cnrules = []
	cntr = 0
	for r in df1.iterrows():
		if DBG: print r[1]['rnbr'],
		if r[1]['lhs'] not in seen_rules.keys():
			seen_rules[r[1]['lhs']].append(r[1]['rnbr'])
			cnrules.append(r[1]['rnbr'])
			if DBG: print "+"
			cntr += 1
		else:	# lhs already seen
			# print df1[df1['rnbr']==seen_rules[r[1]['lhs']][0]]['rhs'].values
			# check the current rhs if the lhs matches to something already seen and check for an isomorphic match
			# rhs1 = listify_rhs(r[1]['rhs'])
			rhs1 = r[1]['rhs']
			rhs2 = df1[df1['rnbr'] == seen_rules[r[1]['lhs']][0]]['rhs'].values[0]
			G1 = rhs_tomultigraph(rhs1)
			G2 = rhs_tomultigraph(rhs2)
			# if nx.is_isomorphic(G1, G2, edge_match=label_match):
			if nx.faster_could_be_isomorphic(G1, G2):
				# print ' ',r[1]['rnbr'], r[1]['rhs'], '::', df1[df1['rnbr'] == seen_rules[r[1]['lhs']][0]]['rhs'].values
				if DBG: print ' <-curr', seen_rules[r[1]['lhs']][0], ':', df1[df1['rnbr'] == seen_rules[r[1]['lhs']][0]][
					'rnbr'].values
				ruleprob2sum[seen_rules[r[1]['lhs']][0]].append(r[1]['rnbr'])
			else:
				seen_rules[r[1]['lhs']].append(r[1]['rnbr'])
				cnrules.append(r[1]['rnbr'])
				if DBG: print "+"
				cntr += 1
	for k in ruleprob2sum.keys():
		if DBG: print k
		if DBG: print "	", ruleprob2sum[k]
		if DBG: print "	", df1[df1['rnbr'] == k]['pr'].values+ sum(df1[df1['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
		# df1[df1['rnbr'] == k]['pr'] += sum(df1[df1['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
		c_val = df1[df1['rnbr'] == k]['pr'].values	+ sum(df1[df1['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
		df1.set_value(df1[df1['rnbr'] == k].index, 'pr', c_val)
		for r in ruleprob2sum[k]:
			df1 = df1[df1.rnbr != r]
	print df1.shape

	# cnrules contains the rules we need to reduce df1 by
	# and ruleprob2sum will give us the new key for which pr will change.
	df1.to_csv("./ProdRules/"+name+"_prules.bz2",sep="\t", header=False, index=False, compression="bz2")

def main ():
	parser = get_parser()
	args = vars(parser.parse_args())
	hello()


	name = sorted(os.path.basename(args['orig']).split('.'), reverse=True, key=len)[0]
	stacked_pr_df,mat_d = isomorphic_test_from_dimacs_tree(args['orig'], args['pathfrag'], name, args)
	if args['cnts']:
		print #stacked_pr_df.to_string()
#		df = stacked_pr_df
#		gd = df.groupby([1]).count()
#		df.columns = ['rnbr', 'lhs', 'rhs', 'pr', 'cate']
#		gb = df.groupby(['cate']).groups
#		if DBG: print gb.keys()
#		sqr_mtrx = np.zeros(shape=(len(mat_d),len(mat_d)))
#		for p in combinations(sorted(gb.keys()), 2):
#			print p
#			r = jaccard_coeff_isomorphic_rules_check_forfilepair(p, df)
#			print r,type(r)
#			break

if __name__ == '__main__':
	try:
		main()
	except Exception, e:
		print str(e)
		traceback.print_exc()
		sys.exit(1)
	sys.exit(0)
