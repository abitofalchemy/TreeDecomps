#!/usr/bin/env python

import os
import pprint as pp
import pandas as pd
from core.isomorph_interxn import listify_rhs
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'small',
		  'figure.figsize': (1.6 * 7, 1.0 * 7),
		  'axes.labelsize': 'small',
		  'axes.titlesize': 'small',
		  'xtick.labelsize': 'small',
		  'ytick.labelsize': 'small'}
pylab.rcParams.update(params)
fig, ax = plt.subplots()

PRS_dir = "../ProdRules/"
dataset = "ucidata-gama"

files = [x[0]+"/"+f for x in os.walk(PRS_dir) for f in x[2] if f.startswith(dataset)]

#pp.pprint (files)
print(os.getcwd())
print
rhs_nonterm_nbrs = lambda RHS: [x for x in RHS if "N" in RHS]
mdf = pd.DataFrame()
for f in files:
	df = pd.read_csv(f, header=None, sep="\t")
	df['varel'] = (os.path.basename(f).split(".")[2])
	df['rhs']   = df[2].apply(listify_rhs)
	df['lhs_n'] = df[1].apply(lambda x: len(x.split(",")))
	df['rhs_n'] = df['rhs'].apply(lambda rhs: len([x for x in df['rhs'].values[0] if 'N' in x]) )
	# df['rhs_t'] = df['rhs'].apply(lambda rhs: len([x for x in df['rhs'].values[0] if 'N' in x]) )
	# # df['rhs_t']
	# print df.apply(lambda x: (len(x[1].split(",")), len(x['rhs'])), axis=1)
	# '"lhs:", len([x.split(',') for x in df[1]]), "rhs:", len(df['rhs'].values)
	mdf = pd.concat([df, mdf])
	# print(mdf.head())
	# print len(df.loc[0]['rhs'])
	print df.shape[0]
	# break
# mdf.plot()
# plt.savefig('tmpfig', bbox_inches='tight')
# print (mdf.head())
# print
# print (mdf.tail())