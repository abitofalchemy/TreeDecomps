# #!/usr/bin/env python
# __author__ = 'saguinag' + '@' + 'nd.edu'
# __version__ = "0.1.0"
#
# ##
# ## fname "trde.dimacs_tree_2cliquetree.py"
# ##      Process a full graph
#
# ## TODO: some todo list
#
# ## VersionLog:
#
# import argparse
import os
import re
import sys
import traceback
from   collections import defaultdict
import networkx as nx
import pandas as pd
import numpy as np
import tdec.PHRG as phrg
import tdec.net_metrics as metrics
import tdec.probabilistic_cfg as pcfg
# import tdec.tree_decomposition as td
# from   tdec.a1_hrg_cliq_tree import load_edgelist,unfold_2wide_tuple
#
# DEBUG = False
#
# def get_parser ():
#     parser = argparse.ArgumentParser(description='Take a clique tree and genenerate HRG graphs')
#     parser.add_argument('--prules', required=True, help='Input tree decomposition file (dimacs)')
#     parser.add_argument('--orig', required=True, help='Reference graph file (edgelist)')
#     parser.add_argument('--version', action='version', version=__version__)
#     return parser
#
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

  #print "n", n
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
#
#
# def dimacs_td_ct (tdfname, orig_fname):
#   """ tree decomp to clique-tree """
#   if DEBUG: print '... input path:', tdfname
#   fname = tdfname
#   graph_name = os.path.basename(fname)
#   gname = graph_name.split('.')[0]
#   gfname = "datasets/out." + gname
#   if DEBUG: print '...', gfname
#
#   G = load_edgelist(orig_fname); print '... graph loaded'
#
#   with open(fname, 'r') as f:  # read tree decomp from inddgo
#     lines = f.readlines()
#     lines = [x.rstrip('\r\n') for x in lines]
#
#   cbags = {}
#   bags = [x.split() for x in lines if x.startswith('B')]
#
#   for b in bags:
#     cbags[int(b[1])] = [int(x) for x in b[3:]]  # what to do with bag size?
#
#   edges = [x.split()[1:] for x in lines if x.startswith('e')]
#   edges = [[int(k) for k in x] for x in edges]
#
#   tree = defaultdict(set)
#   for s, t in edges:
#     tree[frozenset(cbags[s])].add(frozenset(cbags[t]))
#
#   root = list(tree)[0]
#   root = frozenset(cbags[1])
#
#   T = td.make_rooted(tree, root)
#
#   T = phrg.binarize(T)
#   unfold_2wide_tuple(T) # lets me display the tree's frozen sets
#
#   prod_rules = {}
#   if DEBUG: print '... td.new_visit'
#   td.new_visit(T, G, prod_rules)
#
#   if DEBUG: print "--------------------"
#   if DEBUG: print "- Production Rules -"
#   if DEBUG: print "--------------------"
#
#   for k in prod_rules.iterkeys():
#     if DEBUG: print k
#     s = 0
#     for d in prod_rules[k]:
#       s += prod_rules[k][d]
#     for d in prod_rules[k]:
#       prod_rules[k][d] = float(prod_rules[k][d]) / float(s)  # normailization step to create probs not counts.
#       if DEBUG: print '\t -> ', d, prod_rules[k][d]
#
#   rules = []
#   id = 0
#   for k, v in prod_rules.iteritems():
#     sid = 0
#     for x in prod_rules[k]:
#       rhs = re.findall("[^()]+", x)
#       rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
#       if DEBUG: print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])
#       sid += 1
#     id += 1
#   if DEBUG: print "--------------------"
#   if DEBUG: print '- rules.len', len(rules)
#   if DEBUG: print "--------------------"
#
#
#
#   g = pcfg.Grammar('S')
#   for (id, lhs, rhs, prob) in rules:
#     g.add_rule(pcfg.Rule(id, lhs, rhs, prob))
#   if DEBUG: print 'Grammar g loaded.'
#
#   # Synthetic Graphs
#   g.set_max_size(G.number_of_nodes())
#   for i in range(10):
#     print g.sample(G.number_of_nodes())
#
#   hStars = grow_exact_size_hrg_graphs_from_prod_rules(rules, graph_name, G.number_of_nodes(), 20)
#   metricx = ['degree', 'hops', 'clust', 'assort', 'kcore', 'eigen', 'gcd']
#   metrics.network_properties([G], metricx, hStars, name=graph_name, out_tsv=True)
#
#   return
#
#
# def main(args):
#


# if __name__ == '__main__':
print len(sys.argv)
if len(sys.argv)<3:
    print "provide input production rules file: ./Results/*.bz2"
    print "example: python interxn.py ./Results/dsname_file.bz2 dsname"
    sys.exit(1)

df = pd.read_csv(sys.argv[1], index_col=0, compression='bz2')
# df = pd.read_csv(sys.argv[1], index_col=0, sep="\t")
# df= df[df['cate']=="ucidata-gama_minf_dimacs"]
f_ds = "/data/saguinag/datasets/out.{}".format(sys.argv[2])
graph_name= sys.argv[2]
gdf = pd.read_csv(f_ds, header=None, comment="%", sep="\t")
G = nx.from_pandas_dataframe(gdf, 0, 1)
num_nodes = G.number_of_nodes()


rules = []
for ix,row in df.iterrows():
    id = row[1]
    lhs = row[2]
    rhs = [x.strip("\'") for x in row[3].strip("\]\[").split(", ")]
    prob= row[4]
    print (id, lhs, rhs, prob)
    rules.append((id, lhs, rhs, prob))

# g = pcfg.Grammar('S')
# for ix,row in df.iterrows():
#     id = row[0]
#     lhs = row[1]
#     rhs = row[2]
#     prob= row[3]
#     # print (id, lhs, rhs, prob)
#     g.add_rule(pcfg.Rule(id, lhs, rhs, prob))
#     print id, lhs, rhs, prob


# rules = [
# ['r0.0', 'A', ['A:T'], 1.0],
# ['r1.0', 'A,B,C', ['C,A:N', 'B:T'], 0.5],
# ['r1.1', 'A,B,C', ['0,B:T', '0,C:T', '0,A:T'] ,0.5],
# ['r2.0', 'A,B', ['A:N', 'B:T'] ,1.0],
# ['r3.0', 'S', ['0,1:T', '0,2:T', '0,3:T', '1,3:T', '2,3:T', '2,3,0:N', '1,2,3,0:N'] ,1.0],
# ['r4.0', 'A,B,C,D,E,F,G', ['0,A:T', '0,B:T', '0,F:T', '0,G:T', '0,A,B,C,E,F:N', '0,A,B,C,D,E,F,G:N'], 1.0],
# ['r5.0', 'A,B,C,D,E,F,G,H',  ['0,C:T', '0,D:T', '0,E:T', '0,F:T', '0,G:T', '0,H:T', 'A,B,D,0,E:N'], 0.333333333333],
# ['r5.1', 'A,B,C,D,E,F,G,H',  ['0,A:T', '0,B:T', '0,C:T', '0,E:T', '0,G:T', '0,H:T', '0,A,B,C,D,E,F,H:N'], 0.333333333333],
# ['r5.2', 'A,B,C,D,E,F,G,H',  ['0,D:T', '0,E:T', '0,F:T', 'C,0,F,G,H:N', 'A,B,C,D,0,E,F,G:N'] ,0.333333333333],
# ['r6.0', 'A,B,C,D,E',  ['0,A:T', '0,B:T', '0,C:T', '0,D:T', '0,E:T', 'B,0,A:N'] ,0.25],
# ['r6.1', 'A,B,C,D,E',  ['0,A:T', '0,B:T', '0,C:T', '0,D:T', '0,E:T'] ,0.25],
# ['r6.2', 'A,B,C,D,E',  ['0,B:T', '0,C:T', '0,D:T', '0,E:T', '0,A:T'] ,0.25],
# ['r6.3', 'A,B,C,D,E',  ['0,A:T', '0,B:T', '0,C:T', '0,D:T', '0,B,C,D,E,A:N'] ,0.25],
# ['r7.0', 'A,B,C,D',  ['0,B:T', '0,C:T', '0,D:T', '0,A:T', '0,B,C,D,A:N'], 1.0],
# ['r8.0', 'A,B,C,D,E,F',  ['0,E:T', '0,F:T', '0,A,B,C,D,E,F:N'] ,0.5],
# ['r8.1', 'A,B,C,D,E,F',  ['0,B:T', '0,C:T', '0,D:T', '0,E:T', '0,F:T', '0,A:T', 'A,0,D,E,F:N'] ,0.5]
# ]


g = pcfg.Grammar('S')
for (id, lhs, rhs, prob) in rules:
    g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

print 'Grammar g loaded.'
# Synthetic Graphs
#num_nodes = int(sys.argv[-1])
g.set_max_size(num_nodes)

hStars = []
for i in range(20):
    rule_list = g.sample(num_nodes)
    hstar = phrg.grow(rule_list, g)[0]
    hStars.append(hstar)
    print i, hstar.number_of_nodes(), hstar.number_of_edges()

metricx = ['degree', 'hops', 'clust', 'gcd']
metrics.network_properties([G], metricx, hStars, name=graph_name, out_tsv=True)

# parser = get_parser()
# args = vars(parser.parse_args())
# try:
#     main(args)
# except Exception, e:
#     print str(e)
#     traceback.print_exc()
#     sys.exit(1)
# sys.exit(0)
