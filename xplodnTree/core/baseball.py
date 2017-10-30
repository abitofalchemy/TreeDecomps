"""
Recomputes probabilities
"""
import pandas as pd
from pprint import pprint as pp
from td_isom_jaccard_sim import listify_rhs

def dbg(in_str):
	print ("{}".format(in_str))

def recompute_probabilities(pd_data_frame):
	df = pd_data_frame
	df = df.reset_index(drop=True)
	print df.to_string()

	if df.columns[0]=='rnbr':
		df['rnogrp'] = df["rnbr"].apply(lambda x: x.split(".")[0])
	else:
		df['rnogrp'] = df[0].apply(lambda x: x.split(".")[0])


	gb = df.groupby(['rnogrp']).groups
	for k,v in gb.items():
		kcntr = 0
		# print k, v
		# print
		for r in v:
			prob_f = df["prob"].loc[r]/sum([df["prob"].loc[x] for x in v])
			# df.loc[r] = pd.Series(["{}.{}".format(k, kcntr), list(df["lhs"].loc[r]), \
			# 	df["rhs"].loc[r], prob_f])
			df.set_value(v, 'prob', prob_f)
			kcntr += 1
	df.drop('rnogrp', axis=1, inplace=True)
	print df.tail()
	return df

#TODO WORKING on getting this to hand the mdf being passed
#		as argument
# def listify_rhs(rhs_obj):
# 	print type (rhs_obj[0]), len(rhs_obj[0])



def recompute_probabilities_two(pd_data_frame):
	df = pd_data_frame
	df['rnogrp'] = df[0].apply(lambda x: x.split(".")[0])
	gb = df.groupby(['rnogrp']).groups
	for k,v in gb.items():
		print k
		print "  ", len(gb[k])
	ndf = df
	for k,v in gb.items():
		kcntr = 0
		for r in v:
			ndf.set_value(r, [0], "{}.{}".format(k, kcntr))
			ndf.set_value(r, [3], df[[3]].loc[r].values[0]/float(len(v)))
			# rhs = df[[2]].loc[r].values[0]
			# ndf.loc[r]= pd.Series(["{}.{}".format(k, kcntr),
			# 						df[[1]].loc[r].values, #list(df[[1]].loc[r].values)[0],
			# 						listify_rhs(rhs[0]),
			# 						df[[3]].loc[r].values[0]/float(len(v))])
			# # ndf.loc[r] = pd.Series(	["{}.{}".format(k, kcntr),
			# 						df[[1]].loc[r].values, #list(df[[1]].loc[r].values)[0],
			# 						listify_rhs(rhs[0]),
			# 						df[[3]].loc[r].values[0]/float(len(v))])
			kcntr += 1

	ndf = ndf.drop('rnogrp', axis=1)
	print ndf.head()
	return ndf

if __name__ == '__main__':
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
