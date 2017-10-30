#!/usr/bin/env python
from glob import glob
from os import path 

import pandas as pd


files = glob("Results/*stacked_df.tsv")
for f in files:
  df = pd.read_csv (f, sep="\t", index_col=0)
  bn =[x for x in path.basename(f).split('.') if len(x)>3][0]
  print "({}, {})".format(bn, df.shape[0])
print 

#~# 
#~# intersection stats
#~# 
files = glob('Results/*isom_interxn.bz2')
dtyps = {'rnbr': 'str', 'lhs': 'str', 'rhs': 'str', 'pr': 'float'}
for f in files:
  df = pd.read_csv(f, index_col=0, compression='bz2', dtype=dtyps)
  bn = [x for x in path.basename(f).split('.') if len(x)>3][0]
  print "({}, {})".format(bn, df.shape[0])
