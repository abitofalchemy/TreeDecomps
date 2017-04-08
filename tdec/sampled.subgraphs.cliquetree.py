__author__ = 'saguinag' + '@' + 'nd.edu'
__version__ = "0.1.0"

##
## fname
##

## TODO: some todo list

## VersionLog:

import argparse, traceback
import os, sys
from glob import glob
from a1_hrg_cliq_tree import load_edgelist
from collections import deque, defaultdict, Counter
import PHRG as phrg
import tree_decomposition as td
import re
from load_edgelist_from_dataframe import Pandas_DataFrame_From_Edgelist

import networkx as nx
from datetime import datetime
import pandas as pd

DEBUG = False

def get_parser():
  parser = argparse.ArgumentParser(description='multiple clique trees to one set of prod rules')
  parser.add_argument('--name', required=True, help='reference graph name')
  parser.add_argument('--tpath', required=True, help='input tree path (dir)')
  parser.add_argument('--version', action='version', version=__version__)
  return parser

def sampled_subgraphs_cliquetree(orig,tree_path):
    files =glob(tree_path+"*.dimacs.tree")
    prod_rules = {}
    graph_name = orig

    for fname in files:
        print '... input file:', fname

        df = Pandas_DataFrame_From_Edgelist([orig])[0]
        if df.shape[1]==3:
          G = nx.from_pandas_dataframe(df,'src', 'trg',['ts'])
        else:
          G = nx.from_pandas_dataframe(df, 'src', 'trg')
        print nx.info(G)

        with open(fname, 'r') as f:  # read tree decomp from inddgo
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
        if DEBUG: print tree.keys()
        # root = list(tree)[0]
        root = frozenset(cbags[1])
        if DEBUG: print '.. Root:', root
        T = td.make_rooted(tree, root)
        if DEBUG: print '.. T rooted:', len(T)
        # nfld.unfold_2wide_tuple(T) # lets me display the tree's frozen sets

        T = phrg.binarize(T)
        td.new_visit(T, G, prod_rules) # ToDo: here is where something funny is goin on.

        if DEBUG: print "--------------------"
        if DEBUG: print "- Production Rules -"
        if DEBUG: print "--------------------"

        for k in prod_rules.iterkeys():
            if DEBUG: print k
            s = 0
            for d in prod_rules[k]:
                s += prod_rules[k][d]
            for d in prod_rules[k]:
                prod_rules[k][d] = float(prod_rules[k][d]) / float(s)  # normailization step to create probs not counts.
                if DEBUG: print '\t -> ', d, prod_rules[k][d]
        print '... prod_rules size', len(prod_rules.keys())

    #  - production rules number -
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

    print graph_name
    graph_name = os.path.basename(graph_name)
    print graph_name
    outdf_fname = "./ProdRules/" + graph_name + ".prules"
    if not os.path.isfile(outdf_fname + ".bz2"):
        print '...', outdf_fname, "written"
        df.to_csv(outdf_fname + ".bz2", compression="bz2")
    else:
        print '...', outdf_fname, "file exists"

    return

if __name__ == '__main__':
  parser = get_parser()
  args = vars(parser.parse_args())

  treepath = args['tpath']
  gname = args["name"]
  print "... ", gname

  try:
    sampled_subgraphs_cliquetree(gname, treepath)
  except Exception, e:
    print 'ERROR, EXCEPTION: sampled_subgraphs_cliquetree'
    print str(e)
    traceback.print_exc()
    sys.exit(1)
  sys.exit(0)