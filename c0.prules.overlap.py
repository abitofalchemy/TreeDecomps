#!/usr/bin/env python
__author__ = 'saguinag' + '@' + 'nd.edu'
__version__ = "0.1.0"

import pandas as pd
from glob import glob
import os
import argparse
import traceback
import sys
import networkx as nx
import pprint as pp

def peak_at_two_inpufiles (f1, f2):
  for f in [f1, f2]:
    print os.path.basename(f).split('.')[1],
  df1 = pd.read_csv(f1, index_col=0, compression='bz2')
  df1.columns = ['rnbr', 'lhs', 'rhs', 'pr']
  df1.drop(['rnbr', 'pr'], inplace=True, axis=1)
  # print df1.shape
  # print
  df2 = pd.read_csv(f2, index_col=0, compression='bz2')
  df2.columns = ['rnbr', 'lhs', 'rhs', 'pr']
  df2.drop(['rnbr', 'pr'], inplace=True, axis=1)
  # print df2.shape
  # print '- '*20
  # print pd.merge(df1, df2, left_on='rhs', right_on='rhs', how='inner',sort=False).head()
  # print pd.merge(df1, df2, left_on='rhs', right_on='rhs', how='inner',sort=False).shape
  # mdf = pd.concat ([df1,df2])
  # print mdf.shape
  # print mdf.drop_duplicates().shape
  print pd.merge(df1, df2, how='inner', on=['rhs']).shape


def rules_per_file (files):
  print "rules per file\n", "-" * 20
  for f in files:
    print os.path.basename(f).split('.')[1], "\t",
    df = pd.read_csv(f, index_col=0, compression='bz2')
    print df.shape
  return


def rules_overlap_between_files (files):
  for f1 in files:
    for f2 in files:
      if f1 is f2: continue
      df1 = pd.read_csv(f1, index_col=0, compression='bz2')
      df1.columns = ['rnbr', 'lhs', 'rhs', 'pr']
      df2 = pd.read_csv(f2, index_col=0, compression='bz2')
      df2.columns = ['rnbr', 'lhs', 'rhs', 'pr']
      #
      df1.drop(['rnbr', 'pr'], inplace=True, axis=1)
      df2.drop(['rnbr', 'pr'], inplace=True, axis=1)
      print os.path.basename(f1).split('.')[:2], 'vs', os.path.basename(f2).split('.')[1],'\t',
      # print df1.head()
      # print df2.head()
      print pd.merge(df1, df2, on=['rhs'],  how='inner').shape

  if 0:
    print "rules overlap between\n", "-" * 20
    mdf = pd.read_csv(files[0], index_col=0, compression='bz2')
    mdf.columns = ['rnbr', 'lhs', 'rhs', 'pr']
    mdf.drop(['rnbr', 'pr'], inplace=True, axis=1)
    print os.path.basename(files[0]).split('.')[1], ":"

    cdf = pd.DataFrame()  # collect overlap
    for f in files:
      print os.path.basename(f).split('.')[1], 'Overlap'
      df = pd.read_csv(f, index_col=0, compression='bz2')
      df.columns = ['rnbr', 'lhs', 'rhs', 'pr']
      df.drop(['rnbr', 'pr'], inplace=True, axis=1)
      cdf = pd.concat([cdf, pd.merge(df, mdf, left_on='lhs', right_on='lhs', how='inner', sort=False)])
      print pd.merge(df, mdf, left_on='lhs', right_on='lhs', how='inner', sort=False).shape

    print cdf.shape
  return

def long_string_split(x):
  '''
  Parse the RHS of each rule into a graph fragment
  :param x:
  :return:
  '''
  import re
  from itertools import combinations
  rhs_clean= [f[1:-1] for f in re.findall("'.+?'", x)]
  # print pd.merge(df, mdf, left_on='lhs', right_on='lhs', how='inner', sort=False).shape
  G1 = nx.Graph()
  for he in rhs_clean:
    epair,ewt = he.split(':')
    if ewt is "T":
      if len(epair.split(",")) == 1:  [G1.add_node(epair, label=ewt)]
      else: [G1.add_edge(epair.split(",")[0], epair.split(",")[1], weight=ewt)]
    elif ewt is "N":
      if len(epair.split(",")) == 1:  [G1.add_node(epair, label=ewt)]
      else: [G1.add_edges_from(list(combinations(epair.split(","), 2)),weight=ewt )]

  return G1


def isomorphic_overlap(files):
  '''

  :param files:
  :return:
  '''

  for f1 in files:
    for f2 in files:
      if f1 is f2: continue
      ##
      df1 = pd.read_csv(f1, index_col=0, compression='bz2')
      df1.columns = ['rnbr', 'lhs', 'rhs', 'pr']
      df2 = pd.read_csv(f2, index_col=0, compression='bz2')
      df2.columns = ['rnbr', 'lhs', 'rhs', 'pr']
      # print '... rhsG: graph fragments from RHSides'
      print os.path.basename(f1).split('.')[0:2], os.path.basename(f2).split('.')[1], '\t',
      df1['rhsG'] = df1.apply(lambda x: long_string_split(x['rhs'].strip('[]')), axis=1)
      df2['rhsG'] = df2.apply(lambda x: long_string_split(x['rhs'].strip('[]')), axis=1)

      isomorphic_rule_nbrs = 0
      for t1 in df1.iterrows():
        for t2 in df2.iterrows():
          # print t1[1].rnbr, t2[1].rnbr
          if nx.is_isomorphic(t2[1].rhsG, t1[1].rhsG):
            # isomorphic_rule_nbrs.append([os.path.basename(f1).split('.')[0],os.path.basename(f1).split('.')[1],
            #                             os.path.basename(f2).split('.')[1],t1[1].rnbr, t2[1].rnbr])
            # print t1[1][['rnbr','rhs']]
            # print t2[1][['rnbr','rhs']]
            # exit()
            isomorphic_rule_nbrs +=1

        # print df1.shape, df2.shape, len(isomorphic_rule_nbrs)
      print isomorphic_rule_nbrs


      # if 0:
      #   print os.path.basename(f1).rstrip('.prules.bz2'), 'vs', os.path.basename(f2).rstrip('.prules.bz2')
      #   df1.drop(['rnbr', 'pr'], inplace=True, axis=1)
      #   df2.drop(['rnbr', 'pr'], inplace=True, axis=1)
      #   print df1.merge(df2, on=('rhs'), suffixes=('_l','_r')).shape, len(df1), len(df2)
      #   # print df1.apply(long_string_split, axis=1)
      #   print df1.head(1)
      #   print df2.head(1)

def get_parser ():
  parser = argparse.ArgumentParser(description='b2CliqueTreeRules.py: given a tree derive grammar rules')
  parser.add_argument('-p', action="store_true", default=False)
  parser.add_argument('-g', '--graph', required=True, help='input graph (edgelist)')
  parser.add_argument('--version', action='version', version=__version__)
  return parser


def main ():
  parser = get_parser()
  args = vars(parser.parse_args())
  in_path = args['graph']
  files = glob(in_path + '*.bz2')

  rules_per_file(files) # number of rules in file
  print
  print 'Standard Overlap'
  rules_overlap_between_files(files)
  print
  print 'Isomorphic Graph Fragments'
  isomorphic_overlap(files)
  sys.exit(0)

  # peak_at_two_inpufiles(files[0],files[2])
  for f1 in files:
    for f2 in files:
      if f1 is not f2:
        peak_at_two_inpufiles(f1, f2)
  print


if __name__ == '__main__':
  try:
    main()
  except Exception, e:
    print str(e)
    traceback.print_exc()
    sys.exit(1)
  sys.exit(0)
