__author__ = 'saguinag' + '@' + 'nd.edu'
__version__ = "0.1.0"

##
##		tree_to_clique_tree
##

## TODO: some todo list

## VersionLog:

import tdec.net_metrics as metrics
import argparse, traceback
import os, sys
import networkx as nx
import re
from collections import deque, defaultdict, Counter
import tdec.tree_decomposition as td
import tdec.PHRG as phrg
import tdec.probabilistic_cfg as pcfg
import tdec.a1_hrg_cliq_tree as nfld
from tdec.a1_hrg_cliq_tree import load_edgelist
import glob
from tdec.PHRG import graph_checks
from pandas import DataFrame
from utils import graph_name

DEBUG = False


def grow_exact_size_hrg_graphs_from_prod_rules (prod_rules, gname, n, runs=1):
  """
	Args:
		rules: production rules (model)
		gname: graph name
		n:		 target graph order (number of nodes)
		runs:	how many graphs to generate

	Returns: list of synthetic graphs

	"""
  if n <= 0: sys.exit(1)

  g = pcfg.Grammar('S')
  for (id, lhs, rhs, prob) in prod_rules:
    g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

  num_nodes = n
  if DEBUG: print "Starting max size"
  g.set_max_size(num_nodes)
  if DEBUG: print "Done with max size"

  hstars_lst = []
  for i in range(0, runs):
    rule_list = g.sample(num_nodes)
    hstar = phrg.grow(rule_list, g)[0]
    hstars_lst.append(hstar)

  return hstars_lst


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
  graph_checks(G)  # --- graph checks
  prod_rules = {}

  t_basename = os.path.basename(tdfname)
  out_tdfname = os.path.basename(t_basename) + ".prs"
  if os.path.exists("ProdRules/" + out_tdfname):
    print "==> exists:", out_tdfname
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
  if 0:print '- P. Rules', len(rules)
  if 0: print "--------------------"

  '''
  # ToDo.
  # Let's save these rules to file or print proper
  df = DataFrame(rules)
  print "out_tdfname:", out_tdfname
  df.to_csv("ProdRules/" + out_tdfname, sep="\t", header=False, index=False)
  '''

  # g = pcfg.Grammar('S')
  # for (id, lhs, rhs, prob) in rules:
  #	g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

  # Synthetic Graphs
  #	hStars = grow_exact_size_hrg_graphs_from_prod_rules(rules, graph_name, G.number_of_nodes(), 20)
  #	# metricx = ['degree', 'hops', 'clust', 'assort', 'kcore', 'gcd'] # 'eigen'
  #	metricx = ['gcd','avgdeg']
  #	metrics.network_properties([G], metricx, hStars, name=graph_name, out_tsv=True)

  return ""


# def main ():
#   parser = get_parser()
#   args = vars(parser.parse_args())
#
#
# ## dimacs_td_ct(args['orig'], args['tree'] )	# gen synth graph
#

if __name__ == '__main__':
	f = "../datasets/karate*tree"
	orig = "/Users/sal.aguinaga/KynKon/datasets/out.karate_club_graph"
	pp.pprint(f)
	print
	pp.pprint(orig)
	sys.exit(0)
