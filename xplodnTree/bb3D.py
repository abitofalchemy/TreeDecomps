#!/usr/bin/env python

import multiprocessing as mp
# import explodingTree as xt
import os, sys
import re
from collections import deque, defaultdict, Counter
# from core.file_utils import edgelist_basic_info
from glob import glob
from core.stacked_prod_rules import stack_prod_rules_bygroup_into_list
from core.will_prod_rules_fire import will_prod_rules_fire
import core.tree_decomposition as td
import core.PHRG as phrg
import numpy as np
import pprint  as pp
from core.utils import Info

results = []
results_prs =[]
DEBUG=False

def write_prod_rules_to_tsv(prules, out_name):
	Info("write_prod_rules_to_tsv")
	from pandas import DataFrame
	df = DataFrame(prules)
	# print "out_tdfname:", out_name
	if not os.path.exists("../ProdRules"):
	  os.mkdir("../ProdRules")
	try:
		df.to_csv("../ProdRules/" + out_name, sep="\t", header=False, index=False)
	finally:
		print "\tWrote", "../ProdRules/" + out_name
	if os.path.exists("../ProdRules/" + out_name):
		print "Wrote", "../ProdRules/" + out_name

def dimacs_td_ct_fast(oriG, tdfname):
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
  prod_rules = {}

  t_basename = os.path.basename(tdfname)
  out_tdfname = os.path.basename(t_basename) + ".prs"
  if os.path.exists("../ProdRules/" + out_tdfname):
    # print "==> exists:", out_tdfname
    return out_tdfname
  if 0: print "../ProdRules/" + out_tdfname, tdfname

  with open(tdfname, 'r') as f:  # read tree decomp from inddgo
    lines = f.readlines()
    lines = [x.rstrip('\r\n') for x in lines]

  cbags = {}
  bags = [x.split() for x in lines if x.startswith('B')]

  for b in bags:
    cbags[int(b[1])] = [int(x) for x in b[3:]]  # what to do with bag size?

  edges = [x.split()[1:] for x in lines if x.startswith('e')]
  edges = [[int(k) for k in x] for x in edges]

  tree = defaultdict(set)
  for s, t in edges:
    tree[frozenset(cbags[s])].add(frozenset(cbags[t]))
    if DEBUG: print '.. # of keys in `tree`:', len(tree.keys())

  root = list(tree)[0]
  root = frozenset(cbags[1])
  T = td.make_rooted(tree, root)
  # nfld.unfold_2wide_tuple(T) # lets me display the tree's frozen sets

  T = phrg.binarize(T)
  root = list(T)[0]
  root, children = T
  # td.new_visit(T, G, prod_rules, TD)
  # print ">>",len(T)

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

  rules = []
  id = 0
  for k, v in prod_rules.iteritems():
    sid = 0
    for x in prod_rules[k]:
      rhs = re.findall("[^()]+", x)
      rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
      if 0: print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])
      sid += 1
    id += 1

  # print rules
  if 0: print "--------------------"
  if 0: print '- P. Rules', len(rules)
  if 0: print "--------------------"

  # ToDo.
  # Let's save these rules to file or print proper
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

def collect_results(result):
	#results.extend(result)
	results.append(result)

def collect_results_trees(result):
	#results.extend(result)
	results_trees.append(result)

def collect_prodrules(result):
	#results.extend(result)
	results_prs.append(result)

def run_external(args):
	import time

	running_procs = [
		Popen(['/usr/bin/my_cmd', '-i %s' % path], stdout=PIPE, stderr=PIPE)
		for path in '/tmp/file0 /tmp/file1 /tmp/file2'.split()]

	while running_procs:
		for proc in running_procs:
			retcode = proc.poll()
			if retcode is not None:  # Process finished.
				running_procs.remove(proc)
				break
			else:  # No process is done, wait a bit and check again.
				time.sleep(.1)
				continue

		# Here, `proc` has finished with return code `retcode`
		if retcode != 0:
			"""Error handling."""
		handle_results(proc.stdout)

# ^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_
# ^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_~^^~_

# files = [f.rstrip('\n\r') for f in open("datasets/datlst.txt","r").readlines()]
#
# # edgelist base info dict
# el_base_info_d  = {}
# el_base_info_d  = edgelist_basic_info(files)
#
# # print
# # print "Transform to dimacs"
# # print "-"*40
# p = mp.Pool(processes=2)
# for f in files:
# 	gn = xt.graph_name(f)
# 	if os.path.exists('datasets/{}.dimacs'): continue
# 	g = xt.load_edgelist(f)
# 	p.apply_async(xt.convert_nx_gObjs_to_dimacs_gObjs, args=([g], ), callback=collect_results)
# p.close()
# p.join()
# # print (results)
#
#
# print
# print "Explode to trees"
# print "-"*40
#
# var_els=['mcs','mind','minf','mmd','lexm','mcsm']
# results_trees=[]
# for j,f in enumerate(files):
#   gn = xt.graph_name(f)
#   dimacs_file = "datasets/{}.dimacs".format(gn)
#   print " ", gn,
#   p = mp.Pool(processes=2)
#   for vael in var_els:
#     p.apply_async(xt.dimacs_nddgo_tree_simple, args=(dimacs_file,vael, ), callback=collect_results_trees)
#     # xt.dimacs_nddgo_tree_simple(f, vael)
#   p.close()
#   p.join()
#
#   # print results_trees
#   # exit()
#   if j == 0:
#     asp_arr = np.array(results_trees)
#     # print " ", asp_arr.shape,"\t<= cumm. nbr of rules"
#     continue
#   prs_np = np.array(results_prs)
#   asp_arr = np.append(asp_arr, prs_np)
#   print asp_arr.shape
#
# print
# print "Star dot trees to Production Rules"
# print "-"*40
#
#
# for j,f in enumerate(files):
#   results_prs=[]
#   gn = xt.graph_name(f)
#   trees = glob("datasets/{}*.tree".format(gn))
#   print " ", gn,
#   pp = mp.Pool(processes=2)
#   for t in trees:
#     prs_fname = "ProdRules/{}.prs".format(os.path.basename(t))
#     if os.path.exists(prs_fname):
#       # print " ", prs_fname, "file exits"
#       continue
#     oriG = xt.load_edgelist(f)
#     pp.apply_async(dimacs_td_ct_fast, args=(oriG, t, ), callback=collect_prodrules)
#   pp.close()
#   pp.join()
#   if j == 0:
#     rules_np = np.array(results_prs)
#     print " ", rules_np.shape,"\t<= cumm. nbr of rules"
#     continue
#   prs_np = np.array(results_prs)
#   rules_np = np.append(rules_np, prs_np)
#   print rules_np.shape

#print
#print "Test production rules"
#print "-"*40
#
#
#for j,f in enumerate(files):
	#print " ", f
	#gn = xt.graph_name(f)
	#prs_files_l =glob("ProdRules/{}.dimacs.*.prs".format(gn))
	#n = el_base_info_d[gn]
	#will_prod_rules_fire(prs_files_l, n) # probe each group

# print
# print "Test intersectin (isomorphic) production rules subset"
# print "-"*40
# from core.baseball import recompute_probabilities
# from core.will_prod_rules_fire import probe_stacked_prs_likelihood_tofire
# from explodingTree import graph_name
#
# for f in files:
# 	gn = xt.graph_name(f)
# 	prs_files = glob("ProdRules/{}*prs".format(gn))
# 	staked_prs_df = stack_prod_rules_bygroup_into_list(prs_files) # from core.stacked_prod_rules
# 	print "*************" # recompute the probabilities for the group of prs
# 	df = recompute_probabilities(staked_prs_df) # from core.baseball
# 	# test if stacked prs can fire
# 	stck_fired = probe_stacked_prs_likelihood_tofire(df, graph_name(f), el_base_info_d[graph_name(f)])
# 	print (stck_fired)
# 	#
# 	break

if __name__ == '__main__':
	import sys
	from core.utils import graph_name

	if len(sys.argv) < 2:
		Info("add an out.* dataset with its full path")
		exit()
	f = sys.argv[1]
	f = "../datasets/" + graph_name(f) + "*.tree"
	ftrees = glob(f)


	orig = sys.argv[1] #"/Users/sal.aguinaga/KynKon/datasets/out.karate_club_graph"
	from core.utils import graph_name
	import networkx as nx

	gn = graph_name(orig)
	f = "../datasets/" + gn + "*.p"
	results = []
	for p in glob(f):
		pp.pprint(p)
		g = nx.read_gpickle(p)
		for tf in ftrees:
			print ("\t"), tf
			results.append(dimacs_td_ct_fast(g, tf))
	pp.pprint(results)

	sys.exit(0)