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
from   collections import defaultdict
from glob import glob

import networkx as nx
import pandas as pd

import tdec.PHRG as phrg
import tdec.tree_decomposition as td
from tdec.PHRG import graph_checks
from   tdec.a1_hrg_cliq_tree import load_edgelist

DBG=False


def get_parser ():
  parser = argparse.ArgumentParser(description='From dimacs tree to isomorphic reduced prod trees')
  parser.add_argument('--orig', required=True, help='Input tree edgelist file (dimacs)')
  parser.add_argument('--ditree', required=True, help='Input tree decomposition file (dimacs)')
  parser.add_argument('--version', action='version', version=__version__)
  return parser

def listify_rhs(rhs_rule):
  print type(rhs_rule), len(rhs_rule)
  rhs_clean= [f[1:-1] for f in re.findall("'.+?'", rhs_rule)]
  return rhs_clean

def rhs_tomultigraph(rhs_clean):
  '''
  Parse the RHS of each rule into a graph fragment
  :param x:
  :return:
  '''
  import re
  from itertools import combinations
  import networkx as nx

  # rhs_clean= [f[1:-1] for f in re.findall("'.+?'", x)]

  # rhs_clean = [f[1:-1] for f in re.findall("[^()]+", x)]
  G1 = nx.MultiGraph()
  for he in rhs_clean:
    epair,ewt = he.split(':')
    if ewt is "T":
      if len(epair.split(",")) == 1:  [G1.add_node(epair, label=ewt)]
      else: [G1.add_edge(epair.split(",")[0], epair.split(",")[1], label=ewt)]
    elif ewt is "N":
      if len(epair.split(",")) == 1:  [G1.add_node(epair, label=ewt)]
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
      if len(epair.split(",")) == 1:  [G1.add_node(epair, label=ewt)]
      else: [G1.add_edge(epair.split(",")[0], epair.split(",")[1], label=ewt)]
    elif ewt is "N":
      if len(epair.split(",")) == 1:  [G1.add_node(epair, label=ewt)]
      else: [G1.add_edges_from(list(combinations(epair.split(","), 2)),label=ewt )]

  return G1



def isomorphic_test_from_dimacs_tree(orig, dimacs_tree_fname, gname):
  # if whole tree path
  # else, assume a path fragment
  tdfname = dimacs_tree_fname
  print '... input path:', tdfname

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

  files = glob("./datasets/"+gname+"*.dimacs.tree")

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

    if 1: print "--------------------"
    if 1: print "-", len(prod_rules)
    if 1: print "--------------------"
  if 1: print "--------------------"
  if 1: print "- Production Rules -"
  if 1: print "--------------------"

  for k in prod_rules.iterkeys():
    if DBG: print k
    s = 0
    for d in prod_rules[k]:
      s += prod_rules[k][d]
    for d in prod_rules[k]:
      prod_rules[k][d] = float(prod_rules[k][d]) / float(s)  # normailization step to create probs not counts.
      if DBG: print '\t -> ', d, prod_rules[k][d]

  if 1: print "--------------------"
  print '- Prod. Rules'
  if 1: print "--------------------"
  rules = []
  id = 0
  for k, v in prod_rules.iteritems():
    sid = 0
    for x in prod_rules[k]:
      rhs = re.findall("[^()]+", x)
      rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
      if 1: print "r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]
      sid += 1
    id += 1

  print "rules",  len(rules)

  isomorphic_check(rules, gname)


def label_match(x, y):
  return x[0]['label'] == y[0]['label']

def isomorphic_check(prules, name):
  print '-' * 20
  print 'Isomorphic rules check (within file)'
  # for f in files:
  #   df1 = pd.read_csv(f, index_col=0, compression='bz2', dtype=dtyps)
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
    else:  # lhs already seen
      # print df1[df1['rnbr']==seen_rules[r[1]['lhs']][0]]['rhs'].values
      # check the current rhs if the lhs matches to something already seen and check for an isomorphic match
      # rhs1 = listify_rhs(r[1]['rhs'])
      rhs1 = r[1]['rhs']
      rhs2 = df1[df1['rnbr'] == seen_rules[r[1]['lhs']][0]]['rhs'].values[0]
      G1 = rhs_tomultigraph(rhs1)
      G2 = rhs_tomultigraph(rhs2)
      if nx.is_isomorphic(G1, G2, edge_match=label_match):
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
    if DBG: print "  ", ruleprob2sum[k]
    if DBG: print "  ", df1[df1['rnbr'] == k]['pr'].values+ sum(df1[df1['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
    # df1[df1['rnbr'] == k]['pr'] += sum(df1[df1['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
    c_val = df1[df1['rnbr'] == k]['pr'].values  + sum(df1[df1['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
    df1.set_value(df1[df1['rnbr'] == k].index, 'pr', c_val)
    for r in ruleprob2sum[k]:
      df1 = df1[df1.rnbr != r]
  print df1.shape

  # cnrules contains the rules we need to reduce df1 by
  # and ruleprob2sum will give us the new key for which pr will change.
  df1.to_csv("./ProdRules/"+name+"_prules.bz2",sep="\t", header="False", index=False, compression="bz2")

def main ():
  parser = get_parser()
  args = vars(parser.parse_args())
  name = sorted(os.path.basename(args['ditree']).split('.'), reverse=True, key=len)[0]
  isomorphic_test_from_dimacs_tree(args['orig'], args['ditree'], name)

if __name__ == '__main__':
  try:
    main()
  except Exception, e:
    print str(e)
    traceback.print_exc()
    sys.exit(1)
  sys.exit(0)