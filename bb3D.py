import multiprocessing as mp
from glob import glob
import explodingTree as xt 
import os, sys
# import networkx as nx
import re
from collections import deque, defaultdict, Counter
import tdec.tree_decomposition as td
import tdec.PHRG as phrg
import numpy as np


results = []
results_prs =[]
DEBUG=False

def write_prod_rules_to_tsv(prules, out_name):
  from pandas import DataFrame
  df = DataFrame(prules)
  # print "out_tdfname:", out_name
  df.to_csv("ProdRules/" + out_name, sep="\t", header=False, index=False)


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
  # graph_checks(G)  # --- graph checks
  prod_rules = {}

  t_basename = os.path.basename(tdfname)
  out_tdfname = os.path.basename(t_basename) + ".prs"
  if os.path.exists("ProdRules/" + out_tdfname):
    # print "==> exists:", out_tdfname
    return out_tdfname
  if 0: print "ProdRules/" + out_tdfname, tdfname

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


files = [f.rstrip('\n\r') for f in open("datasets/datlst.txt","r").readlines()]

print
print "Transform to dimacs"
print
p = mp.Pool(processes=2)
for f in files:
	gn = xt.graph_name(f)
	if os.path.exists('datasets/{}.dimacs'): continue
	g = xt.load_edgelist(f)
	p.apply_async(xt.convert_nx_gObjs_to_dimacs_gObjs, args=([g], ), callback=collect_results)
p.close()
p.join()
print (results)


print
print "Explode to trees"
print

var_els=['mcs','mind','minf','mmd','lexm','mcsm']
results_trees=[]
for f in files:
	gn = xt.graph_name(f)
	dimacs_file = "datasets/{}.dimacs".format(gn)
	if not os.path.exists(dimacs_file):
		print "  dimacs file does not exists"
		continue
	print dimacs_file,"----"
	p = mp.Pool(processes=2)
	for vael in var_els:
		if os.path.exists("datasets/{}.dimacs.{}.tree".format(gn,vael)):
			print "  file","datasets/{}.dimacs.{}.tree".format(gn,vael),"exists.."
			continue
		else:
			# print "filename",dimacs_file
			p.apply_async(xt.dimacs_nddgo_tree_simple, args=(f,vael, ), callback=collect_results_trees)
	p.close()
	p.join()

	print results_trees

print
print "Star dot trees to Production Rules"
print "-"*40


for j,f in enumerate(files):
  results_prs=[]
  gn = xt.graph_name(f)
  trees = glob("datasets/{}*.tree".format(gn))
  oriG = xt.load_edgelist(f)
  pp = mp.Pool(processes=2)
  for t in trees:
    # print " ",t
    pp.apply_async(dimacs_td_ct_fast, args=(oriG, t, ), callback=collect_prodrules)
    #dimacs_td_ct_fast(oriG, t)
  pp.close()
  pp.join()
  #results_prs = [dimacs_td_ct_fast(oriG, t) for t in trees]
  # print results_prs
  if j == 0:
    rules_np = np.array(results_prs)
    continue
  prs_np = np.array(results_prs)
  rules_np = np.append(rules_np, prs_np)

  print " ", np.shape(rules_np)