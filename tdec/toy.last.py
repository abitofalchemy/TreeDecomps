import pandas as pd
import glob
import os
from itertools import combinations
from collections import defaultdict
import networkx as nx
import re

import pprint as pp
import numpy as np

DBG = False

# def load_bz2_file(ipath):
#   import bz2
#   bz_file = bz2.BZ2File(ipath)
#   lines_lst = bz_file.readlines()
#   for l in lines_lst[1:]:
#     # l = l.split()[2:3]
#     print l
#     break


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

def long_string_split(x):
  '''
  Parse the RHS of each rule into a graph fragment
  :param x:
  :return:
  '''
  import re
  import networkx as nx
  from itertools import combinations

  rhs_clean= [f[1:-1] for f in re.findall("'.+?'", x)]
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

def label_match(x, y):
  return x[0]['label'] == y[0]['label']

def string_to_list(x):
  '''
  Parse the RHS of each rule into a list
  :param x:
  :return:
  '''
  import re
  # rhs_clean= [f[1:-1] for f in re.findall("'.+?'", x)]
  rhs_clean = [f[1:-1] for f in re.findall("[^()]+", x)]
  return rhs_clean

def listify_rhs(rhs_rule):
  rhs_clean= [f[1:-1] for f in re.findall("'.+?'", rhs_rule)]
  return rhs_clean

files = glob.glob("/Users/saguinag/Theory/Grammars/ProdRules/ucidata-g*prules.bz2")
dtyps = {'rnbr': 'str', 'lhs': 'str', 'rhs': 'str', 'pr': 'float'}

print 'Rules Counts'
for p in [",".join(map(str, comb)) for comb in combinations(files, 2)]:
  p = p.split(',')
  df1 = pd.read_csv(p[0], index_col=0, compression='bz2', dtype=dtyps)
  df2 = pd.read_csv(p[1], index_col=0, compression='bz2', dtype=dtyps)
  print os.path.basename(p[0]).split('.')[0:2], os.path.basename(p[1]).split('.')[1], '\t',
  print df1.shape, df2.shape
  if not DBG: break

print '-'*20

print 'Rules Combos'
for p in [",".join(map(str, comb)) for comb in combinations(files, 2)]:
  p = p.split(',')
  df1 = pd.read_csv(p[0], index_col=0, compression='bz2',dtype=dtyps)
  df1.columns = ['rnbr', 'lhs', 'rhs', 'pr']
  df2 = pd.read_csv(p[1], index_col=0, compression='bz2',dtype=dtyps)
  df2.columns = ['rnbr', 'lhs', 'rhs', 'pr']

  print os.path.basename(p[0]).split('.')[0:2], os.path.basename(p[1]).split('.')[1], '\t'
  seen_rules = defaultdict(list)
  cntr   = 0
  for r in df1.iterrows():
    if not (r[1]['lhs'] in seen_rules.keys()):
      seen_rules[r[1]['lhs']].append(r[1]['rnbr'])
    else: # lhs already seen
      # print df1[df1['rnbr']==seen_rules[r[1]['lhs']][0]]['rhs'].values
      # check the current rhs if the lhs matches to something already seen and check for an isomorphic match
      rhs1 = listify_rhs(r[1]['rhs'])
      rhs2 = listify_rhs(df1[df1['rnbr']==seen_rules[r[1]['lhs']][0]]['rhs'].values[0])
      G1 = rhs_tomultigraph(rhs1)
      G2 = rhs_tomultigraph(rhs2)
      if nx.is_isomorphic(G1, G2, edge_match=label_match):
        # print ' ',r[1]['rnbr'], r[1]['rhs'], '::', df1[df1['rnbr'] == seen_rules[r[1]['lhs']][0]]['rhs'].values
        print ' ', r[1]['rnbr'], '::', df1[df1['rnbr'] == seen_rules[r[1]['lhs']][0]]['rnbr'].values
        seen_rules[r[1]['lhs']].append(r[1]['rnbr'])
  print seen_rules # seen rules is the set of minimal prod rules for this pair of files combo
  break

print 'Basic Comparison'
if 1:
  for p in [",".join(map(str, comb)) for comb in combinations(files, 2)]:
    p = p.split(',')
    df1 = pd.read_csv(p[0], index_col=0, compression='bz2',dtype=dtyps)
    df1.columns = ['rnbr', 'lhs', 'rhs', 'pr']
    df2 = pd.read_csv(p[1], index_col=0, compression='bz2',dtype=dtyps)
    df2.columns = ['rnbr', 'lhs', 'rhs', 'pr']

    print os.path.basename(p[0]).split('.')[0:2], os.path.basename(p[1]).split('.')[1], '\t',
    glist1 = df1.apply(lambda x: string_to_list(x['rhs'].strip('[]')), axis=1).values
    glist2 = df2.apply(lambda x: string_to_list(x['rhs'].strip('[]')), axis=1).values

    Differences = {tuple(i) for i in glist1} & {tuple(i) for i in glist2}
    print len(Differences), Differences
    # print [list(glist1),list(glist2)]#, glist2)

print 'Isomorphic Graph Frags Comparison'
if 0:
  for p in [",".join(map(str, comb)) for comb in combinations(files, 2)]:
    p = p.split(',')
    df1 = pd.read_csv(p[0], index_col=0, compression='bz2', dtype=dtyps)
    df1.columns = ['rnbr', 'lhs', 'rhs', 'pr']
    df2 = pd.read_csv(p[1], index_col=0, compression='bz2', dtype=dtyps)
    df2.columns = ['rnbr', 'lhs', 'rhs', 'pr']

    print os.path.basename(p[0]).split('.')[0:2], os.path.basename(p[1]).split('.')[1], '\t',
    glist1 = df1.apply(lambda x: rhs_tomultigraph(x['rhs'].strip('[]')), axis=1).values
    glist2 = df2.apply(lambda x: rhs_tomultigraph(x['rhs'].strip('[]')), axis=1).values
    #
    # print glist1.shape

    # Differences = {tuple(i) for i in glist1} & {tuple(i) for i in glist2}
    # print len(Differences)
    cntr = 0
    for t in zip(glist1,glist2):
      # print t[0].number_of_edges(), t[1].number_of_edges()
      if nx.is_isomorphic(t[0], t[1], edge_match=label_match):
      #   print [x.number_of_nodes() for x in t]
        cntr +=1

    print cntr
# for p in zip (files[1:], files):
#   df1 = pd.read_csv(p[0], index_col=0, compression='bz2',dtype=dtyps)
#   df1.columns = ['rnbr', 'lhs', 'rhs', 'pr']
#   df2 = pd.read_csv(p[1], index_col=0, compression='bz2',dtype=dtyps)
#   df2.columns = ['rnbr', 'lhs', 'rhs', 'pr']
#   # print df1.head()
#   # print df2.head()
#
#
#   print os.path.basename(p[0]).split('.')[0:2], os.path.basename(p[1]).split('.')[1], '\t'
#   glist1 = df1.apply(lambda x: string_to_list(x['rhs'].strip('[]')), axis=1).values
#   glist2 = df2.apply(lambda x: string_to_list(x['rhs'].strip('[]')), axis=1).values
#
#   Differences = {tuple(i) for i in glist1} & {tuple(i) for i in glist2}
#   print '  ', Differences
#
#
# if 0:
#   df1 = pd.read_csv(f1, index_col=0, compression='bz2',dtype=dtyps)
#   df1.columns=['rnbr','lhs','rhs','pr']
#   df2 = pd.read_csv(f2, index_col=0, compression='bz2', dtype=dtyps)
#   df2.columns=['rnbr','lhs','rhs','pr']
#
#
#   print df1.shape, df2.shape
#   df1.drop_duplicates()
#   df2.drop_duplicates()
#   print df1.shape, df2.shape
#   # df1 = df1.drop(['rnbr','pr'],axis=1)
#   # df2 = df2.drop(['rnbr','pr'],axis=1)
#   # print df1.head()
#   # print df2.head()
#   glist1 = df1.apply(lambda x: string_to_list(x['rhs'].strip('[]')), axis=1).values
#   glist2 = df2.apply(lambda x: string_to_list(x['rhs'].strip('[]')), axis=1).values
#
#   pairwise= zip (glist1, glist2)
#   print [idx for idx, pair in enumerate(pairwise) if pair[0] == pair[1]]