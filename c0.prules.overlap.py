#!/usr/bin/env python
__author__ = 'saguinag' + '@' + 'nd.edu'
__version__ = "0.1.0"

import pandas as pd
from glob import glob
import os
import argparse
import traceback
import sys


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

def display_rules(files):
  for f in files:
    print os.path.basename(f).split('.')[1], "\t",
    df = pd.read_csv(f, index_col=0, compression='bz2')
    print df.to_string()

def rules_per_file (files):
  print "rules per file\n", "-" * 20
  for f in files:
    print os.path.basename(f).split('.')[1], "\t",
    df = pd.read_csv(f, index_col=0, compression='bz2')
    print df.shape
  return


def rules_overlap_between_files (files):
  print "rules overlap between\n", "-" * 20
  mdf = pd.read_csv(files[0], index_col=0, compression='bz2')
  mdf.columns = ['rnbr', 'lhs', 'rhs', 'pr']
  mdf.drop(['rnbr', 'pr'], inplace=True, axis=1)
  print os.path.basename(files[0]).split('.')[1], ":"

  cdf = pd.DataFrame()  # collect overlap
  for f in files:
    print os.path.basename(f).split('.')[1], 'Overlap',
    df = pd.read_csv(f, index_col=0, compression='bz2')
    df.columns = ['rnbr', 'lhs', 'rhs', 'pr']
    df.drop(['rnbr', 'pr'], inplace=True, axis=1)
    cdf = pd.concat([cdf, pd.merge(df, mdf, left_on='lhs', right_on='lhs', how='inner', sort=False)])
    print pd.merge(df, mdf, left_on='lhs', right_on='lhs', how='inner', sort=False).shape

  print cdf.shape
  return


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
  print in_path
  files = glob(in_path + '*.bz2')
  
  rules_per_file(files) # number of rules in file
  display_rules(files)  # print rules to stdout 
  print
  rules_overlap_between_files(files)
  print 
  # peak_at_two_inpufiles(files[0],files[2])
  for f1 in files:
    for f2 in files:
      if f1 is not f2:
        peak_at_two_inpufiles(f1, f2)
  print

def isomorphic():
  G1 = nx.DiGraph()
  G2 = nx.DiGraph()

if __name__ == '__main__':
  try:
    main()
    isomorphic()
  except Exception, e:
    print str(e)
    traceback.print_exc()
    sys.exit(1)
  sys.exit(0)
