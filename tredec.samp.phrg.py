#!/usr/bin/env python
__author__ = 'saguinag' + '@' + 'nd.edu'
__version__ = "0.1.0"

##
## fname "bdimacs_tree_hrggraphs.py"
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
#import exact_phrg as xphrg
import tdec.a1_hrg_cliq_tree as nfld
from tdec.a1_hrg_cliq_tree import load_edgelist
import glob

DEBUG = False


def get_parser ():
  parser = argparse.ArgumentParser(description='dimacs tree *list* to hrg graph')
  parser.add_argument('--orig', required=True, help='input edge list for the original (reference) graph')
  parser.add_argument('--tree', required=True, help='input tree decomposition (dimacs file format)')
  parser.add_argument('--version', action='version', version=__version__)
  return parser

def grow_exact_size_hrg_graphs_from_prod_rules(prod_rules, gname, n, runs=1):
  """
  Args:
    rules: production rules (model)
    gname: graph name
    n:     target graph order (number of nodes)
    runs:  how many graphs to generate

  Returns: list of synthetic graphs

  """
  if n <=0: sys.exit(1)


  g = pcfg.Grammar('S')
  for (id, lhs, rhs, prob) in prod_rules:
    g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

  print "n", n
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

def dimacs_td_ct (orig, tdfname):
  """ tree decomp to clique-tree """
  print '... input path:', tdfname
  fname = orig # original graph path
  graph_name = os.path.basename(fname)
  gname = os.path.basename(fname).split(".")
  gname = sorted(gname,reverse=True, key=len)[0]

  G = load_edgelist(fname) # load edgelist into a graph obj
  G.name = gname

  print nx.info(G)

  files = glob.glob(tdfname+"*.dimacs.tree")
  prod_rules = {}

  for tfname in files:
    with open(tfname, 'r') as f:  # read tree decomp from inddgo
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
    # root = list(T)[0]
    # root, children = T
    # td.new_visit(T, G, prod_rules, TD)
    # print ">>",len(T)

    td.new_visit(T, G, prod_rules)

    if 1: print "--------------------"
    if 1: print "-", len(prod_rules)
    if 1: print "--------------------"
  if 1: print "--------------------"
  if 1: print "- Production Rules -"
  if 1: print "--------------------"

  for k in prod_rules.iterkeys():
    if DEBUG: print k
    s = 0
    for d in prod_rules[k]:
      s += prod_rules[k][d]
    for d in prod_rules[k]:
      prod_rules[k][d] = float(prod_rules[k][d]) / float(s)  # normailization step to create probs not counts.
      if DEBUG: print '\t -> ', d, prod_rules[k][d]

  rules = []
  id = 0
  for k, v in prod_rules.iteritems():
    sid = 0
    for x in prod_rules[k]:
      rhs = re.findall("[^()]+", x)
      rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
      if 1: print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])
      sid += 1
    id += 1
  # print rules
  if 1: print "--------------------"
  print '- P. Rules',    len(rules)
  if 1: print "--------------------"
  
  g = pcfg.Grammar('S')
  for (id, lhs, rhs, prob) in rules:
    g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

  # Synthetic Graphs
  # print rules
  hStars = grow_exact_size_hrg_graphs_from_prod_rules(rules, graph_name, G.number_of_nodes(), 20)
  metricx = ['degree', 'hops', 'clust', 'assort', 'kcore', 'gcd'] # 'eigen'
  metrics.network_properties([G], metricx, hStars, name=graph_name, out_tsv=True)

  return


def main ():
  parser = get_parser()
  args = vars(parser.parse_args())
  dimacs_td_ct(args['orig'], args['tree'] )  # gen synth graph


if __name__ == '__main__':
  try:
    main()
  except Exception, e:
    print str(e)
    traceback.print_exc()
    sys.exit(1)
  sys.exit(0)
