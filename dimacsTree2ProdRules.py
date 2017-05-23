#!/usr/bin/env python
__author__ = 'saguinag' + '@' + 'nd.edu'
__version__ = "0.1.0"

##
## fname "b2CliqueTreeRules.py"
##

## TODO: some todo list

## VersionLog:

#import net_metrics as metrics
import pandas as pd
import argparse, traceback
import os, sys
import networkx as nx
import re
from collections import deque, defaultdict, Counter
import tdec.tree_decomposition as td
import tdec.PHRG as phrg
#import tdec.probabilistic_cfg as pcfg
#import exact_phrg as xphrg
#import a1_hrg_cliq_tree as nfld
from tdec.a1_hrg_cliq_tree import load_edgelist

DEBUG = False


def get_parser ():
	parser = argparse.ArgumentParser(description='b2CliqueTreeRules.py: given a tree derive grammar rules')
	parser.add_argument('-t', '--treedecomp',required=True, help='input tree decomposition (dimacs file format)')
	parser.add_argument('--version', action='version', version=__version__)
	return parser

def dimacs_td_ct (tdfname, synthg=False):
	""" tree decomp to clique-tree """
	if isinstance(tdfname, list): [dimacs_td_ct(f) for f in tdfname]
	#	print '... input file:', tdfname

	fname = tdfname
	graph_name = os.path.basename(fname)
	gname = graph_name.split('.')[0]
	if synthg:
		gfname = 'datasets/'+gname+".dimacs"
	else:
		gfname = "datasets/out." + gname
	print os.path.basename(fname).split('.')[-2]
	tdh = os.path.basename(fname).split('.')[-2] # tree decomp heuristic
	tfname = gname+"."+tdh

	if synthg:
		G = load_edgelist(tdfname.split('.')[0]+".dimacs")
	else:
		G = load_edgelist(gfname)

	if DEBUG: print nx.info(G)

	with open(fname, 'r') as f:	# read tree decomp from inddgo
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
		if DEBUG: print '.. # of keys in `tree`:', len(tree.keys())
	if DEBUG: print tree.keys()
	root = list(tree)[0]
	if DEBUG: print '.. Root:', root
	root = frozenset(cbags[1])
	if DEBUG: print '.. Root:', root
	T = td.make_rooted(tree, root)
	if DEBUG: print '.. T rooted:', len(T)
	# nfld.unfold_2wide_tuple(T) # lets me display the tree's frozen sets

	T = phrg.binarize(T)

	prod_rules = {}
	td.new_visit(T, G, prod_rules)

	if DEBUG: print "--------------------"
	if DEBUG: print "- Production Rules -"
	if DEBUG: print "--------------------"

	for k in prod_rules.iterkeys():
		if DEBUG: print k
		s = 0
		for d in prod_rules[k]:
			s += prod_rules[k][d]
		for d in prod_rules[k]:
			prod_rules[k][d] = float(prod_rules[k][d]) / float(s)	# normailization step to create probs not counts.
			if DEBUG: print '\t -> ', d, prod_rules[k][d]

	rules = []
	id = 0
	for k, v in prod_rules.iteritems():
		sid = 0
		for x in prod_rules[k]:
			rhs = re.findall("[^()]+", x)
			rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
			if DEBUG: print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])

			sid += 1
		id += 1

	df = pd.DataFrame(rules)

	outdf_fname = "ProdRules/"+tfname+"_prules.bz2"

	if not os.path.isfile(outdf_fname):
		#		print '...',outdf_fname, "written"
		df.to_csv(outdf_fname, compression="bz2")
	else:
		print '...', outdf_fname, "file exists"

	return outdf_fname


def main ():
	parser = get_parser()
	args = vars(parser.parse_args())

	dimacs_td_ct(args['treedecomp'])	# gen synth graph


if __name__ == '__main__':
	try:
		main()
	except Exception, e:
		print str(e)
		traceback.print_exc()
		sys.exit(1)
	sys.exit(0)
