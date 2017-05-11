#!/usr/bin/env python
__author__ = 'saguinag' + '@' + 'nd.edu'
__version__ = "0.1.0"

##
## Description: isomorphic overlap, rules highlighted
##

## TODO: some todo list

## VersionLog:

import argparse
import os
import re
import sys
import traceback
from collections import defaultdict
from itertools import combinations
from glob import glob
from json import dumps
import numpy as np
import networkx as nx
import pandas as pd

import tdec.PHRG as phrg
import tdec.tree_decomposition as td
from tdec.PHRG import graph_checks
from   tdec.a1_hrg_cliq_tree import load_edgelist
from tdec.isomorph_interxn import isomorph_infile_reduce
from tdec.isomorph_interxn import isomorph_check_production_rules_pair, isomorph_intersection_2dfstacked

DBG = False
global args

def get_parser ():
  parser = argparse.ArgumentParser(description='From dimacs tree to isomorphic reduced prod trees')
  parser.add_argument('--orig', required=True, help='Input tree edgelist file (dimacs)')
  parser.add_argument('--pathfrag', required=True, help='Input dimacs tree path fragment')
  parser.add_argument('-verb', action='store_true', default=False, required=False,
                      help='Verbose (dev log info)')

  parser.add_argument('--version', action='version', version=__version__)
  return parser


def listify_rhs (rhs_rule):
  if DBG: print type(rhs_rule), len(rhs_rule)
  rhs_clean = [f[1:-1] for f in re.findall("'.+?'", rhs_rule)]
  return rhs_clean


def rhs_tomultigraph (rhs_clean):
  '''
  Parse the RHS of each rule into a graph fragment
  :param x:
  :return:
  '''
  import re
  import networkx as nx

  # rhs_clean= [f[1:-1] for f in re.findall("'.+?'", x)]

  # rhs_clean = [f[1:-1] for f in re.findall("[^()]+", x)]
  G1 = nx.MultiGraph()
  for he in rhs_clean:
    epair, ewt = he.split(':')
    if ewt is "T":
      if len(epair.split(",")) == 1:
        [G1.add_node(epair, label=ewt)]
      else:
        [G1.add_edge(epair.split(",")[0], epair.split(",")[1], label=ewt)]
    elif ewt is "N":
      if len(epair.split(",")) == 1:
        [G1.add_node(epair, label=ewt)]
      else:
        [G1.add_edges_from(list(combinations(epair.split(","), 2)), label=ewt)]

  return G1


def rhs2multigraph (x):
  '''
  Parse the RHS of each rule into a graph fragment
  :param x:
  :return:
  '''
  import re
  from itertools import combinations
  import networkx as nx

  rhs_clean = [f[1:-1] for f in re.findall("'.+?'", x)]
  # rhs_clean = [f[1:-1] for f in re.findall("[^()]+", x)]
  G1 = nx.MultiGraph()
  for he in rhs_clean:
    epair, ewt = he.split(':')
    if ewt is "T":
      if len(epair.split(",")) == 1:
        [G1.add_node(epair, label=ewt)]
      else:
        [G1.add_edge(epair.split(",")[0], epair.split(",")[1], label=ewt)]
    elif ewt is "N":
      if len(epair.split(",")) == 1:
        [G1.add_node(epair, label=ewt)]
      else:
        [G1.add_edges_from(list(combinations(epair.split(","), 2)), label=ewt)]

  return G1


def isomorphic_test_from_dimacs_tree (orig, tdfname, gname=""):
  # if whole tree path
  # else, assume a path fragment
  print '... input graph  :', os.path.basename(orig)
  print '... td path frag :', tdfname

  G = load_edgelist(orig)  # load edgelist into a graph obj
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

  files = glob(tdfname + "*.dimacs.tree")
  prod_rules = {}
  stacked_df = pd.DataFrame()

  mat_dict = {}
  for i, x in enumerate(sorted(files)):
    mat_dict[os.path.basename(x).split(".")[0].split("_")[-1]] = i
    if DBG: print os.path.basename(x).split(".")[0].split("_")[-1]

  for tfname in sorted(files):
    tname = os.path.basename(tfname).split(".")
    tname = "_".join(tname[:2])

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
    from json import dumps
    #print dumps(prod_rules, indent=4, sort_keys=True)

    for k in prod_rules.iterkeys():
      if DBG: print k
      s = 0
      for d in prod_rules[k]:
        s += prod_rules[k][d]
      for d in prod_rules[k]:
        prod_rules[k][d] = float(prod_rules[k][d]) / float(s)  # normailization step to create probs not counts.
        if DBG: print '\t -> ', d, prod_rules[k][d]

    if DBG: print "--------------------"
    if DBG: print '- Prod. Rules'
    if DBG: print "--------------------"
    rules = []
    #print dumps(prod_rules, indent=4, sort_keys=True)

    id = 0
    for k, v in prod_rules.iteritems():
      sid = 0
      for x in prod_rules[k]:
        rhs = re.findall("[^()]+", x)
        rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
        if DBG: print "r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]
        sid += 1
      id += 1

    df = pd.DataFrame(rules)
    df['cate'] = tname
    stacked_df = pd.concat([df, stacked_df])
    # print df.shape
  print "\nStacked prod rules\n", "~" * 20
  print "  ", stacked_df.shape
  if args['verb']: print stacked_df.to_string()
  stacked_df.to_csv("Results/{}_stacked_df.tsv".format(gname), sep="\t")
  if os.path.exists("Results/{}_stacked_df.tsv".format(gname)): print 'Wrote:', "Results/{}_stacked_df.tsv".format(gname)

  print "\nisomorphic union of the rules (_mod probs)\n", "~" * 20
  stacked_df.columns = ['rnbr', 'lhs', 'rhs', 'pr', df['cate'].name]
  iso_union,iso_interx = isomorph_intersection_2dfstacked(stacked_df)
  print "  ", iso_union.shape
  if args['verb']: print iso_union.to_string()

  print "\nIsomorphic intersection of the prod rules\n", "~" * 20
  print "  ", iso_interx.shape
  iso_interx.to_csv('Results/{}_isom_interxn.bz2'.format(gname), compression="bz2")
  if os.path.exists('Results/{}_isom_interxn.bz2'.format(gname)): print 'Wrote:', 'Results/{}_isom_interxn.bz2'.format(gname)
  #   # print k,v
  #   sid = 0
  #   for ix, r in iso_intxn[iso_intxn['lhs']==k].iterrows():
  #     print sid, k, r['rhs']
  #     sid += 1
  #   break
    #   rhs = re.findall("[^()]+", x)
    #   rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
    #   if DBG: print "r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]
    #   sid += 1
    # id += 1

  # ....:     print group
  # np_sqr_mtrx, max_ix_tup = jaccard_coeff_isomorphic_rules_check(stacked_df, mat_dict)
  # if DBG: print "Files with Max Jaccard Sim: ",
  # pair_var_elims = [[k for k,v in mat_dict.items() if m_ix == v][0] for m_ix in max_ix_tup]
  # if DBG: print pair_var_elims

  # union_prod_rules_from_pair(pair_var_elims, stacked_df)


  # print gname
  # df = pd.DataFrame(np_sqr_mtrx, columns=[x for x in sorted(mat_dict.keys())])
  # df.index = sorted(mat_dict.keys())
  # df.to_csv("Results/{}_isom.tsv".format(gname), sep=",")


def union_prod_rules_from_pair (var_elim_pair, stckd_df):
  if stckd_df.empty: return
  print '... union_prod_rules_from_pair'
  stckd_df.columns = ['rnbr', 'lhs', 'rhs', 'pr', 'cate']
  gb = stckd_df.groupby(['cate']).groups
  grp_prod_rules = []
  for x in gb.keys():
    [grp_prod_rules.append(x) for ve in var_elim_pair if ve in x]

  print var_elim_pair, grp_prod_rules
  df1 = stckd_df[stckd_df['cate'] == grp_prod_rules[0]]
  df2 = stckd_df[stckd_df['cate'] == grp_prod_rules[1]]
  print "DF shape:", df1.shape, df2.shape
  # print df1
  print
  # print df2
  print "Prior to i/f Redux", df1.shape, df2.shape
  rdf1 = isomorph_infile_reduce(df1)
  rdf2 = isomorph_infile_reduce(df2)
  print "After infile Redux", rdf1.shape, rdf2.shape
  #
  isomorph_check_production_rules_pair(rdf1, rdf2)


def label_match (x, y):
  return x[0]['label'] == y[0]['label']


def jacc_dist_for_pair_dfrms (df1, df2):
  slen = len(df1)
  tlen = len(df2)
  # +++
  conc_df = pd.concat([df1, df2])
  # ---
  seen_rules = defaultdict(list)
  ruleprob2sum = defaultdict(list)
  cnrules = []
  cntr = 0
  for r in conc_df.iterrows():
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
      rhs2 = conc_df[conc_df['rnbr'] == seen_rules[r[1]['lhs']][0]]['rhs'].values[0]
      G1 = rhs_tomultigraph(rhs1)
      G2 = rhs_tomultigraph(rhs2)
      if nx.is_isomorphic(G1, G2, edge_match=label_match):
        # print ' ',r[1]['rnbr'], r[1]['rhs'], '::', df1[df1['rnbr'] == seen_rules[r[1]['lhs']][0]]['rhs'].values
        if DBG: print ' <-curr', seen_rules[r[1]['lhs']][0], ':', \
        conc_df[conc_df['rnbr'] == seen_rules[r[1]['lhs']][0]]['rnbr'].values, \
        conc_df[conc_df['rnbr'] == seen_rules[r[1]['lhs']][0]]['cate'].values
        ruleprob2sum[seen_rules[r[1]['lhs']][0]].append(r[1]['rnbr'])
      else:
        seen_rules[r[1]['lhs']].append(r[1]['rnbr'])
        cnrules.append(r[1]['rnbr'])
        if DBG: print "+"
        cntr += 1

  if DBG: print "len(ruleprob2sum)", len(ruleprob2sum)
  if DBG: print  dumps(ruleprob2sum, indent=4, sort_keys=True)
  if DBG: print "len(df1) + len(df2)", len(df1), len(df2)
  if DBG: print "Overlapping rules  ", len(ruleprob2sum.keys()), sum([len(x) for x in ruleprob2sum.values()])
  if DBG: print "Jaccard Sim:\t", (len(ruleprob2sum.keys()) + sum([len(x) for x in ruleprob2sum.values()])) / float(
    len(df1) + len(df2))
  return (len(ruleprob2sum.keys()) + sum([len(x) for x in ruleprob2sum.values()])) / float(len(df1) + len(df2))


def jaccard_coeff_isomorphic_rules_check_forfilepair (pr_grpby, mdf):
  if DBG: print pr_grpby[0], pr_grpby[1],
  return jacc_dist_for_pair_dfrms(mdf[mdf['cate'] == pr_grpby[0]], \
                                  mdf[mdf['cate'] == pr_grpby[1]])


def jaccard_coeff_isomorphic_rules_check (dfrm, headers_d):
  ''' check dataframe 
  links:
  - http://stackoverflow.com/questions/24841271/finding-maximum-value-and-their-indices-in-a-sparse-lil-matrix-scipy-python
  '''
  if dfrm.empty: return

  dfrm.columns = ['rnbr', 'lhs', 'rhs', 'pr', 'cate']
  gb = dfrm.groupby(['cate']).groups
  if DBG: print gb.keys()
  sqr_mtrx = np.zeros(shape=(len(headers_d), len(headers_d)))

  for p in combinations(sorted(gb.keys()), 2):
    if DBG: print [x.split("_")[1] for x in p],
    if DBG: print [headers_d[x.split("_")[1]] for x in p]  # [0].split("_")[-1]
    j = headers_d[p[0].split("_")[1]]
    i = headers_d[p[1].split("_")[1]]

    sqr_mtrx[i, j] = jaccard_coeff_isomorphic_rules_check_forfilepair(p, dfrm)

  sqr_mtrx = np.asmatrix(sqr_mtrx)

  from numpy import unravel_index
  print "unravel_index"
  unravel_index(sqr_mtrx.argmax(), sqr_mtrx.shape)

  return sqr_mtrx, unravel_index(sqr_mtrx.argmax(), sqr_mtrx.shape)  # numpy.savetxt("foo.csv", a, delimiter=",")

  exit()

  seen_rules = defaultdict(list)
  ruleprob2sum = defaultdict(list)
  cnrules = []
  cntr = 0

  for r in dfrm.iterrows():
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
      rhs2 = dfrm[dfrm['rnbr'] == seen_rules[r[1]['lhs']][0]]['rhs'].values[0]
      G1 = rhs_tomultigraph(rhs1)
      G2 = rhs_tomultigraph(rhs2)
      if nx.is_isomorphic(G1, G2, edge_match=label_match):
        # print ' ',r[1]['rnbr'], r[1]['rhs'], '::', df1[df1['rnbr'] == seen_rules[r[1]['lhs']][0]]['rhs'].values
        if DBG: print ' <-curr', seen_rules[r[1]['lhs']][0], ':', dfrm[dfrm['rnbr'] == seen_rules[r[1]['lhs']][0]][
          'rnbr'].values, dfrm[dfrm['rnbr'] == seen_rules[r[1]['lhs']][0]]['cate'].values
        ruleprob2sum[seen_rules[r[1]['lhs']][0]].append(r[1]['rnbr'])
      else:
        seen_rules[r[1]['lhs']].append(r[1]['rnbr'])
        cnrules.append(r[1]['rnbr'])
        if DBG: print "+"
        cntr += 1

      #  for k in ruleprob2sum.keys():
      #    if DBG: print k
      #    if DBG: print "  ", ruleprob2sum[k]
      #    if DBG: print "  ", dfrm[dfrm['rnbr'] == k]['pr'].values+ sum(dfrm[dfrm['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
      #    # dfrm[dfrm['rnbr'] == k]['pr'] += sum(dfrm[dfrm['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
      #    c_val = dfrm[dfrm['rnbr'] == k]['pr'].values  + sum(dfrm[dfrm['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
      #    dfrm.set_value(dfrm[dfrm['rnbr'] == k].index, 'pr', c_val)
      #    for r in ruleprob2sum[k]:
      #      dfrm = dfrm[dfrm.rnbr != r]
      #  print dfrm.shape

  # cnrules contains the rules we need to reduce df1 by
  # and ruleprob2sum will give us the new key for which pr will change.
  #  df1.to_csv("./ProdRules/"+name+"_prules.bz2",sep="\t", header="False", index=False, compression="bz2")
  return True


def isomorphic_check (prules, name):
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
    if DBG: print "  ", df1[df1['rnbr'] == k]['pr'].values + sum(
      df1[df1['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
    # df1[df1['rnbr'] == k]['pr'] += sum(df1[df1['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
    c_val = df1[df1['rnbr'] == k]['pr'].values + sum(df1[df1['rnbr'] == r]['pr'].values for r in ruleprob2sum[k])
    df1.set_value(df1[df1['rnbr'] == k].index, 'pr', c_val)
    for r in ruleprob2sum[k]:
      df1 = df1[df1.rnbr != r]
  print df1.shape

  # cnrules contains the rules we need to reduce df1 by
  # and ruleprob2sum will give us the new key for which pr will change.
  df1.to_csv("./ProdRules/" + name + "_prules.bz2", sep="\t", header="False", index=False, compression="bz2")


def main(args):
  # parser = get_parser()
  # args = vars(parser.parse_args())
  gname = sorted(os.path.basename(args['orig']).split('.'), reverse=True, key=len)[0]
  isomorphic_test_from_dimacs_tree(args['orig'], args['pathfrag'], gname)


if __name__ == '__main__':
  parser = get_parser()
  args = vars(parser.parse_args())
  try:
    main(args)
  except Exception, e:
    print str(e)
    traceback.print_exc()
    sys.exit(1)
  sys.exit(0)
