import pandas as pd

def recompute_probabilities(pd_data_frame):
	df = pd_data_frame
	df['rnogrp'] = df[0].apply(lambda x: x.split(".")[0])
	gb = df.groupby(['rnogrp']).groups
	
	ndf = df
	for k,v in gb.items():
		kcntr = 0
		for r in v:
			# print "{}.{}".format(k, kcntr), df[[1]].loc[r].values, df[[2]].loc[r].values[0] ,df[[3]].loc[r].values[0]/float(len(v))
			ndf.loc[r] = pd.Series(["{}.{}".format(k, kcntr), list(df[[1]].loc[r].values)[0], df[[2]].loc[r].values[0] ,df[[3]].loc[r].values[0]/float(len(v))])
			kcntr += 1
	
	ndf = ndf.drop('rnogrp', axis=1)
	print ndf.head()
	return ndf

from sys import argv,exit
import os

if len(argv) <=1: 
	print '--> needs argument, provide a *.prs filename'
	exit(1)

fname = argv[1]
if not os.path.exists(fname): 
	print 'file does not exists, try a new file'
	exit(1)

df = pd.read_csv(fname, header=None, sep="\t")
df = recompute_probabilities(df)
df.to_csv(fname.split('.')[0]+"_rc.tsv", header=False, index=False, sep="\t") # rcprs = recomputed prod rules
if os.path.exists(fname.split('.')[0]+"_rc.tsv"): print 'Saved file:', fname.split('.')[0]+"_rc.tsv"



