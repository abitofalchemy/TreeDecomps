import multiprocessing as mp
import os
import explodingTree as xt
import networkx as nx
import numpy as np
import pprint  as pp
import core.tree_decomposition as td
from glob import glob
from collections import deque, defaultdict, Counter
import core.PHRG as phrg
import re

DEBUG = False

def collect_results(result):
	#results.extend(result)
	results.append(result)

def collect_results_trees(result):
	#results.extend(result)
	results_trees.append(result)

def collect_prodrules(result):
	results_prs.append(result)

def write_prod_rules_to_tsv(prules, out_name):
	from pandas import DataFrame
	df = DataFrame(prules)
	# print "out_tdfname:", out_name
	df.to_csv("../ProdRules/" + out_name, sep="\t", header=False, index=False)


def dimacs_td_ct_fast(oriG, tdfname):
	#	print("dimacs_td_ct_fast",oriG, tdfname)
	""" tree decomp to clique-tree
	parameters:
	orig:			filepath to orig (input) graph in edgelist
	tdfname:	filepath to tree decomposition from INDDGO
	synthg:		when the input graph is a syth (orig) graph
	Todo:
	currently not handling sythg in this version of dimacs_td_ct
	"""
	G = oriG
	if G is None: return (1)
	# graph_checks(G)  # --- graph checks
	prod_rules = {}

	t_basename = os.path.basename(tdfname)
	out_tdfname = "../ProdRules/"+t_basename + ".prs"
#	print ("\t%s" % out_tdfname)
	#	print (" %s" % tdfname)

	if os.path.exists(out_tdfname):
		print "../ProdRules/" + out_tdfname, tdfname
		return out_tdfname
	
	with open(tdfname, 'r') as f:  # read tree decomp from inddgo
		lines = f.readlines()
		lines = [x.rstrip('\r\n') for x in lines]


	cbags = {}
	bags = [x.split() for x in lines if x.startswith('B')]

	for b in bags:
		cbags[int(b[1])] = [int(x) for x in b[3:]]  # what to do with bag size?

	edges = [x.split()[1:] for x in lines if x.startswith('e')]
	edges = [[int(k) for k in x] for x in edges]

#	print ("\t%s" % out_tdfname)
#	print ([x for x in edges])

	tree = defaultdict(set)
	for s, t in edges:
		tree[frozenset(cbags[s])].add(frozenset(cbags[t]))
		if DEBUG: print ('.. # of keys in `tree`:', len(tree.keys()))

	root = list(tree)[0]
	root = frozenset(cbags[1])
	T = td.make_rooted(tree, root)
	# nfld.unfold_2wide_tuple(T) # lets me display the tree's frozen sets

	T = phrg.binarize(T)
	root = list(T)[0]
	root, children = T
	# td.new_visit(T, G, prod_rules, TD)


	td.new_visit(T, G, prod_rules)

	if 0: print "--------------------"
	if 0: print "- Production Rules -"
	if 0: print "--------------------"

	for k in prod_rules.iterkeys():
		if DEBUG: print k
		s = 0
		for d in prod_rules[k]:
		  s += prod_rules[k][d]
		for d in prod_rules[k]:
		  prod_rules[k][d] = float(prod_rules[k][d]) / float(
			s)  # normailization step to create probs not counts.
		  if DEBUG: print '\t -> ', d, prod_rules[k][d]

#	print ">> [1]"
	rules = []
	id = 0
	for k, v in prod_rules.iteritems():
		sid = 0
		
		for x in prod_rules[k]:
			rhs = re.findall("[^()]+", x)
			rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
			# print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])
			sid += 1
		id += 1

#	print ">> [1]"
#	print rules
	if 0: print "--------------------"
	if 1: print '- P. Rules', len(rules)
	if 0: print "--------------------"

	# ToDo.
	# Let's save these rules to file or print proper
	print(out_tdfname)
	write_prod_rules_to_tsv(rules, out_tdfname)

	# g = pcfg.Grammar('S')
	# for (id, lhs, rhs, prob) in rules:
	#	g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

	# Synthetic Graphs
	#	hStars = grow_exact_size_hrg_graphs_from_prod_rules(rules, graph_name, G.number_of_nodes(), 20)
	#	# metricx = ['degree', 'hops', 'clust', 'assort', 'kcore', 'gcd'] # 'eigen'
	#	metricx = ['gcd','avgdeg']
	#	metrics.network_properties([G], metricx, hStars, name=graph_name, out_tsv=True)

	return out_tdfname


def transform_edgelist_to_dimacs(files):
	print ("Transform to dimacs")
	print ("-"*40)

	p = mp.Pool(processes=2)
	for f in files:
		gn = xt.graph_name(f)
		if os.path.exists('../datasets/{}.dimacs'): continue
		g = nx.read_gpickle(f)
		g.name = gn
		p.apply_async(xt.convert_nx_gObjs_to_dimacs_gObjs, args=([g], ), callback=collect_results)
		# xt.convert_nx_gObjs_to_dimacs_gObjs([g])
	p.close()
	p.join()
	#print (results)

def explode_to_trees(files, results_trees):
	print ("Explode to trees")
	print ("-"*40)

	var_els=['mcs','mind','minf','mmd','lexm','mcsm']
	for j,f in enumerate(files):
		gn = xt.graph_name(f)
		dimacs_file = "../datasets/{}.dimacs".format(gn)
		print (" ", gn,)
		p = mp.Pool(processes=2)
		for vael in var_els:
			p.apply_async(xt.dimacs_nddgo_tree_simple, args=(dimacs_file,vael, ), callback=collect_results_trees)
		# xt.dimacs_nddgo_tree_simple(f, vael)
		p.close()
		p.join()
		print(results_trees)
		
		if j == 0:
			asp_arr = np.array(results_trees)
			continue

		prs_np = np.array(results_trees)
		asp_arr = np.append(asp_arr, prs_np)


def star_dot_trees_to_prod_rules(files,results_prs):
	print ("Star dot trees to Production Rules")
	print ("-"*40)

	for j,f in enumerate(files):
		results_prs=[]
		gn = xt.graph_name(f)
		trees = glob("../datasets/{}*.tree".format(gn))

		pp = mp.Pool(processes=2)
		for t in trees:
			prs_fname = "../ProdRules/{}.prs".format(os.path.basename(t))
			if os.path.exists(prs_fname):
				print ("  {} file exits".format(prs_fname))
				continue
			oriG = xt.load_edgelist(f)
			pp.apply_async(dimacs_td_ct_fast, args=(oriG, t, ), callback=collect_prodrules)
		pp.close()
		pp.join()
		print (results_prs)
#		if j == 0:
#			rules_np = np.array(results_prs)
#			print (" ", rules_np.shape,"\t<= cumm. nbr of rules")
#			continue
#		prs_np = np.array(results_prs)
#		rules_np = np.append(rules_np, prs_np)
#		print (rules_np.shape)



#os.chdir(os.path.dirname(__file__))
print(os.getcwd())
dir= "../datasets"
p_files = [x[0]+"/"+f for x in os.walk(dir) for f in x[2] if f.endswith(".p")]
print

results = []
transform_edgelist_to_dimacs(p_files)
files = [x.rstrip(".p") for x in p_files]

print
results_trees=[]
explode_to_trees(files,results_trees)
pp.pprint( [x[0]+"/"+f for x in os.walk(dir) for f in x[2] if f.endswith(".tree")])

results_prs =[]
print
star_dot_trees_to_prod_rules(files,results_prs)

print

