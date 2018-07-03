#!/usr/bin/env python
'''
Working on getting rules we read to grow Hstars
'''
__version__="0.1.0"
import argparse
import sys
import traceback
import pandas as pd
import os
import matplotlib
matplotlib.use('pdf')
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from utils import graph_name, listify_rhs
from glob import glob
import PHRG as PHRG
import probabilistic_cfg as pcfg
import pprint as pp

params = {'legend.fontsize': 'smaller',
		  'figure.figsize': (1.6 * 7, 1.0 * 4),
		  'axes.labelsize': 'smaller',
		  'axes.titlesize': 'smaller',
		  'xtick.labelsize': 'smaller',
		  'ytick.labelsize': 'smaller'}
pylab.rcParams.update(params)




def proc_prod_rules_single(fname):
	gn = graph_name(fname)
	PRS_dir = "../ProdRules/"
	files = glob(PRS_dir + "*{}*prs*".format(gn))

	pp.pprint (files)
	# print(os.getcwd())
	# print
	rhs_nonterm_nbrs = lambda RHS: [x for x in RHS if "N" in RHS]
	mdf = pd.DataFrame()
	for f in files:
		df = pd.read_csv(f, header=None, sep="\t")
		df['varel'] = (os.path.basename(f).split(".")[2])
		df['rhs']   = df[2].apply(listify_rhs)
		df['lhs_n'] = df[1].apply(lambda x: len(x.split(",")))
		df['rhs_n'] = df['rhs'].apply(lambda rhs: len([x for x in df['rhs'].values[0] if 'N' in x]) )
		df['rhs_n'] = df['rhs'].apply(lambda x: len([k for k in x if 'N' in k]))
		# df['rhs_t'] = df['rhs'].apply(lambda rhs: len([x for x in df['rhs'].values[0] if 'N' in x]) )
		# # df['rhs_t']
		# print df.apply(lambda x: (len(x[1].split(",")), len(x['rhs'])), axis=1)
		# '"lhs:", len([x.split(',') for x in df[1]]), "rhs:", len(df['rhs'].values)
		mdf = pd.concat([df, mdf])
		# print(mdf.head())
		# print len(df.loc[0]['rhs'])
		# print (df['lhs'].shape)
	print df.head()
	gb = mdf.groupby('varel').groups
	# print (gb['lexm'])
	# print (df.describe())
	# df.boxplot(ax=xa[1])
	# print
	# print (mdf.tail())
	fig, xa = plt.subplots(1, len(gb.keys()))
	# mdf.groupby('varel').hist(ax=xa[0])

	# Visualize pairplot of df
	# sns.pairplot(mdf, hue='varel');
	for j,ve in enumerate(gb.keys()):
		print (ve)
		# mdf[mdf['varel']==ve][['lhs_n','rhs_n']].hist(ax=xa[0])
		# xa[0].histogram(mdf[mdf['varel']==ve].lhs_n)
		numBins=4
		xa[j].hist(mdf[mdf['varel']==ve][['lhs_n','rhs_n']], numBins, alpha=0.8)
		if j == 0: xa[j].legend(('lhs_n','rhs_n'))
		xa[j].set_title(ve)
		# mdf[mdf['varel']==ve].hist(ax=xa[0],x=ve,y=)
		# mdf.loc(gb[ve].values).head() #.hist(ax=xa[j],label=ve)
		# print (mdf.loc(gb[ve]))

	plt.savefig('tmpfig',bbox_inches='tight')

def proc_prod_rules_orig(fname):
	gn = graph_name(fname)
	df = pd.read_csv(fname, header=None, sep="\t")


	df['rhs']   = df[2].apply(listify_rhs)
	print df['rhs'].apply(lambda x: [k for k in x if 'N' in k]).head()
	df['rhs_n'] = df['rhs'].apply(lambda x: len([k for k in x if 'N' in k]))
	df['lhs_n'] = df[1].apply(lambda x: len(x.split(",")))
	print df.head()

	# df['lhs_n'] = df[1].apply(lambda x: len(x.split(",")))
	# df['rhs_n'] = df['rhs'].apply(lambda rhs: len([x for x in df['rhs'].values[0] if 'N' in x]) )
	# 	# df['rhs_t'] = df['rhs'].apply(lambda rhs: len([x for x in df['rhs'].values[0] if 'N' in x]) )
	# 	# # df['rhs_t']
	# 	# print df.apply(lambda x: (len(x[1].split(",")), len(x['rhs'])), axis=1)
	# 	# '"lhs:", len([x.split(',') for x in df[1]]), "rhs:", len(df['rhs'].values)
	# 	mdf = pd.concat([df, mdf])
	# 	# print(mdf.head())
	# 	# print len(df.loc[0]['rhs'])
	# 	# print (df['lhs'].shape)
	#
	# print (mdf.describe())
	# print (mdf.tail())

def get_prod_rules(fname):
	df = pd.read_csv(fname, sep="\t", header= None)
	nf = df[[0,1,3]]
	# nf['lhs'] = df[1].apply(lambda x: x.split(','))
	nf['rhs'] = df[2].apply(listify_rhs)
	# nf = df[1].apply(lambda x: x.split(','), axis=1)
	# print nf.head()
	rules = nf[[0,1,'rhs',3]].values.tolist()
	return rules

def hstar_fixed_graph_gen(args):
	import networkx as nx

	orig_fname = args['grow'][0]
	gn = graph_name(orig_fname)
	if os.path.exists("../datasets/{}.p".format(gn)):
		origG = nx.read_gpickle("../datasets/{}.p".format(gn))
	else:
		print ("we load edgelist into an nx.obj")

	prs_files = glob("../ProdRules/{}*prs".format(gn))
	for f in prs_files:
		prod_rules = get_prod_rules(f)
		g = pcfg.Grammar('S')
		for (id, lhs, rhs, prob) in prod_rules:
			# print (id, lhs, rhs, prob)
			g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

		# exit() # Takes this out
		# ToDo: We nee to get these rules in the right format

		num_nodes = origG.number_of_nodes()

		print "Starting max size"
		g.set_max_size(num_nodes)

		print "Done with max size"

		Hstars = []

		num_samples = 20
		print '*' * 40
		for i in range(0, num_samples):
			rule_list = g.sample(num_nodes)
			hstar = PHRG.grow(rule_list, g)[0]
			Hstars.append(hstar)
	import pickle
	pickle.dump({'origG': origG, 'hstars': Hstars}, open('../Results/{}_hstars.p'.format(gn), "wb"))
	if os.path.exists('../Results/{}_hstars.p'.format(gn)): print ("Pickle written")


def main(args):
	if not(args['fldr'] is None):
		print ("-"*40)
		print ("Process Prod rules in Folder")
		print ("="*40)
		proc_prod_rules_folder(args['orig'])
	elif not(args['orig'] is None):
		print ("-" * 40)
		print ("Process Prod rules in Folder")
		print ("=" * 40)
		proc_prod_rules_single(args['orig'][0])
	elif not(args['prs'] is None): # ToDo: need to fix something here
		print ("-" * 40)
		print ("Process Single Prod rules")
		print ("=" * 40)
		proc_prod_rules_orig(args['prs'][0])
	elif not(args['grow'] is None):
		# if args['orig'] is None:
		# 	print ("Please include the path to original dataset: python prs.py --orig path/fname")
		# 	exit(1)
		hstar_fixed_graph_gen(args)

def get_parser ():
	parser = argparse.ArgumentParser(description='xplodnTree tree decomposition')
	parser.add_argument('--fldr', nargs=1, required=False, help="PRS Folder")
	parser.add_argument('--orig', nargs=1, required=False, help="edgelist source file")
	parser.add_argument('--prs', nargs=1, required=False, help="Single PRS file")
	parser.add_argument('--grow',nargs=1, required=0, help="grow with given prs")
	parser.add_argument('--version',   action='version', version=__version__)
	return parser

# if __name__ == '__main__':
# 	'''ToDo: clean the edglists, write them back to disk and then run inddgo on 1 component graphs
# 	'''
#
# 	parser = get_parser()
# 	args = vars(parser.parse_args())
# 	try:
# 		main(args)
# 	except Exception, e:
# 		print (str(e))
# 		traceback.print_exc()
# 		sys.exit(1)
# 	sys.exit(0)