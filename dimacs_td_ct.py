#!/usr/bin/env python
__author__ = 'saguinag' + '@' + 'nd.edu'
__version__ = "0.1.0"

##
## fname "dimacs_td_ct.py"
##

## TODO: some todo list

## VersionLog:

import argparse, traceback, optparse
import os, sys, time
import networkx as nx
from collections import deque, defaultdict, Counter
from enumhrgtree import tuplz_tree_graph
import pprint as pp
import tree_decomposition as td
import PHRG as phrg


def get_parser():
  parser = argparse.ArgumentParser(description='dimacs_td_ct: Convert tree decomposition to clique tree')
  parser.add_argument('-t', '--treedecomp', required=True, help='input tree decomposition (dimacs file format)')
  parser.add_argument('--version', action='version', version=__version__)
  return parser

def dimacs_td_ct(tdfname):
  ''' tree decomp to clique-tree'''
  fname = tdfname

  with open(fname, 'r') as f: # read tree decomp from inddgo
    lines = f.readlines()
    lines = [x.rstrip('\r\n') for x in lines]

  cbags = {}
  bags = [x.split() for x in lines if x.startswith('B')]
  bags = [x[1:] for x in bags]

  for x in bags:
    cbags[int(x[0])] = [int(k) for k in x[2:]]

  edges = [x.split()[1:] for x in lines if x.startswith('e')]
  edges =  [[int(k) for k in x] for x in edges]
  print edges
  print cbags

  tree = defaultdict(set)
  for s,t in edges:
    print s,t
    print cbags[s]
    print cbags[t]
    tree[frozenset(cbags[s])].add(frozenset(cbags[t]))

  pp.pprint (tree)
  for k,v in tree.items():
    print k,'\t',v

  root = list(tree)[0]
  T = td.make_rooted(tree, root)
  T = phrg.binarize(T)
  print
  print T
  exit()

  CT = nx.Graph()
  CT.add_edges_from(edges)

  for v in CT.nodes_iter():

    if v == '1':
      CT.node[v]['root']   = True
      CT.node[v]['parent'] = True

    CT.node[v]['cnode'] = cbags[int(v)]

  # print nx.info(CT)
  # print CT.edges()
  # print CT.nodes()
  print "CT.nodes(data=True)"
  print "==================="

  print CT.nodes(data=True)
  for k,v in CT.nodes_iter(data=True):
    print k,':\t', v
  #TODO: create the defaultdict from these node bags
  # print [v for v in  CT.nodes_iter()]
  # print CT.node['1']['root']
  print '... root:', nx.get_node_attributes(CT,'root').keys()
  root = nx.get_node_attributes(CT,'root').keys()[0]

  tree = defaultdict(set)

  #bag = frozenset(clique | {v})
  #tree[bag].add(tv)

  # print [c for c in CT.neighbors_iter(root)]
  # pprint.pprint  (T)
  # print len(T), type(T)
  # for x in T:
  #     print "{}".format(x)
  # root = list(T)[0]
  # T = td.make_rooted(T, root)
  # T = binarize(T)
  # print tuplz_tree_graph(root, CT) # need to convert to the tree ds in hrg



def main ():
  parser = get_parser()
  args = vars(parser.parse_args())

  dimacs_td_ct(args['treedecomp'])  # gen synth graph


if __name__ == '__main__':
  try:
    main()
  except Exception, e:
    print str(e)
    traceback.print_exc()
    sys.exit(1)
  sys.exit(0)
