#!/usr/bin/env python
__version__="0.1.0"

# make the other metrics work
# generate the txt files, then work on the pdf otuput
import pandas as pd
import sys
import os
import re
import shelve
import numpy as np
import pprint as pp
import argparse, traceback
import tdec.graph_sampler as gs
import tdec.probabilistic_cfg as pcfg
import networkx as nx
import tdec.PHRG as phrg

from glob import glob
from td_isom_jaccard_sim import listify_rhs
from tdec.load_edgelist_from_dataframe import Pandas_DataFrame_From_Edgelist
from collections import Sequence

DBG = False

def genflat(l, ltypes=Sequence):
	# by https://stackoverflow.com/users/95810/alex-martelli
	l = list(l)
	while l:
		while l and isinstance(l[0], ltypes):
			l[0:1] = l[0]
		if l: yield l.pop(0)

def summarize_listify_rule(rhs_rule):
	if DBG: print type(rhs_rule), len(rhs_rule)
	rhs_clean= [f[1:-1] for f in re.findall("'.+?'", rhs_rule)]
	return [len(x.split(",")) for x in rhs_clean if "N" in x]


def willFire_check(dat_frm):
	""" Checks if the subset of prod rules will fire
	:param dat_frm:
	:return bool
	"""
	ret_val = False
	
	if not len(dat_frm):
		return ret_val

	#print [nt for nt in dat_frm[1] if "S" in nt]#.apply(lambda x: [nt for nt in x if "S" in nt])
	nt_symbs_s = [nt for nt in dat_frm[1] if "S" in nt]
	if not len(nt_symbs_s):
		print nt_symbs_s
		print "_S:" # list(dat_frm[1].values)
		return ret_val
	else:
		# print dat_frm.iloc[1][1], dat_frm.iloc[1][2]
		rhs_els = dat_frm[2].apply(summarize_listify_rule)
		lhs_els = dat_frm[1].apply(lambda x: len(x.split(",")))
		df =  pd.concat([lhs_els, rhs_els], axis=1)
		# Test if for each rhs NT we have an equal sized LHS
		# d = defaultdict(list)
		# print df.head()
		# print df.groupby([1]).groups.keys()
		lhs_keys = df.groupby([1]).groups.keys()
		key_seen = {}
		for k in lhs_keys:
			if k == 1: continue
			if k in list(genflat(df[2].values)):
				key_seen[k] = True
			else:
				key_seen[k] = False
		# print key_seen
		# print not any(key_seen.values())
		ret_val = not any(x is False for x in key_seen.values())

		return ret_val


def tst_prod_rules_isom_intrxn(fname,origfname):
	"""
	Test the isomorphic subset of rules
	
	:param fname:	isom intersection rules file
	:param origfname: reference input network (dataset) edgelist file
	:return: 
	"""
	# Get the original file
	fdf = Pandas_DataFrame_From_Edgelist([origfname])
	origG = nx.from_pandas_dataframe(fdf[0], 'src', 'trg')
	
	# Read the subset of prod rules
	df = pd.read_csv(fname, header=None,	sep="\t", dtype={0: str, 1: list, 2: list, 3: float})
	g = pcfg.Grammar('S')
	
	if not willFire_check(df):
		print "-"*10, fname, "contains production rules that WillNotFire"
		return None
	else:
		print "+"*40
	# Process dataframe
	from td_isom_jaccard_sim import listify_rhs
	for (id, lhs, rhs, prob) in df.values:
		rhs = listify_rhs(rhs)
		g.add_rule(pcfg.Rule(id, lhs, rhs, float(prob)))
	
	print "\n","."*40 #print 'Added the rules to the datastructure'

	num_nodes = origG.number_of_nodes()

	# print "Starting max size", 'n=', num_nodes
	g.set_max_size(num_nodes)
	# print "Done with max size"

	Hstars = []

	ofname   = "FakeGraphs/"+ origG.name+ "isom_ntrxn.shl"
	database = shelve.open(ofname)
	
	num_samples = 20 #
	print '~' * 40
	for i in range(0, num_samples):
		rule_list = g.sample(num_nodes)
		hstar		 = phrg.grow(rule_list, g)[0]
		Hstars.append(hstar)
		print hstar.number_of_nodes(), hstar.number_of_edges()

	print '-' * 40
	database['hstars'] = Hstars
	database.close()

def tst_prod_rules_level1_individual(in_path):
	# files = glob("ProdRules/moreno_lesmis_lesmis.*_iprules.tsv")
	mdf = pd.DataFrame()
	for f in sorted(files, reverse=True):
		df = pd.read_csv(f, header=None, sep="\t")
		mdf = pd.concat([mdf, df])
		# print f, mdf.shape
	# print mdf.head()

		g = pcfg.Grammar('S')
		from td_isom_jaccard_sim import listify_rhs
		for (id, lhs, rhs, prob) in df.values:
			rhs = listify_rhs(rhs)
			# print (id), (lhs), (rhs), (prob)
			g.add_rule(pcfg.Rule(id, lhs, rhs, float(prob)))


		num_nodes = 16 # G.number_of_nodes()
		print "Starting max size", 'n=', num_nodes
		g.set_max_size(num_nodes)
		print "Done with max size"
		Hstars = []
		print '-' * 40
		try:
			rule_list = g.sample(num_nodes)
		except Exception, e:
			print str(e)
			continue
		hstar		 = phrg.grow(rule_list, g)[0]
		Hstars.append(hstar)
		print '+' * 40
		# break

# def tst_prod_rules_canfire(infname):
	# print infname

	# print df[2].apply(lambda x: listify_rhs(x.split()[0]))
# tst_prod_rules_isom_intrxn("Results/moreno_vdb_vdb_isom_itrxn.tsv")
# tst_prod_rules_level1_individual("ProdRules/moreno_lesmis_lesmis.*_iprules.tsv")
#		tst_prod_rules_isom_intrxn(fname)


#_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~#
def get_parser ():
	parser = argparse.ArgumentParser(description="Test the reduced set of prod rules")
	parser.add_argument('--prs',nargs=1, required=True, help="Filename to prod rules file")
	parser.add_argument('--orig',nargs=1, required=True, help="Filename to original dataset file")
	parser.add_argument('--version', action='version', version=__version__)
	return parser

def main (argsd):
	if argsd['prs']:
		tst_prod_rules_isom_intrxn(argsd['prs'][0], argsd['orig'][0])
	else:
		print "I am not sure what you are trying to do"
		sys.exit(1)



#test_isom_subset_of_prod_rules("Results/moreno_lesmis_lesmis_isom_itrxn.tsv")
#print
#test_isom_subset_of_prod_rules("Results/moreno_vdb_vdb_isom_itrxn.tsv")
#print
#test_isom_subset_of_prod_rules("Results/ucidata-gama_isom_itrxn.tsv")
#print
#test_isom_subset_of_prod_rules("Results/ucidata-zachary_isom_itrxn.tsv")
#
###
### tst_prod_rules_canfire("Results/ucidata-gama_stcked_prs_isom_itrxn.tsv")
###
#infname ="Results/ucidata-gama_isom_itrxn.tsv"
#df = pd.read_csv(infname, header=None,	sep="\t", dtype={0: str, 1: str, 2: list, 3: float})
#
#df['lhs_n']=[len(x.split(',')) for x in df[1].values]
#df['els_n']=[len(listify_rhs(x)) for x in df[2].values]
#df['nt_els']=[len([k for k in listify_rhs(x) if ":N" in k]) for x in df[2].values]
#print '#'*10
#rhs_nt_els_nbr = {}
#for y,x in df[[1,2]].values:
#	# print x
#	rhs_nts =[]
#	for k in listify_rhs(x):
#		if ":N" in k:
#			# print ' ', len(k.split(','))
#			rhs_nts.append(len(k.split(',')))
#	rhs_nt_els_nbr[len(y.split(','))] = rhs_nts
#for k,v in rhs_nt_els_nbr.items():
#	print k,'\t',v
#
#print '^'*20
#print 'rhs',rhs_nt_els_nbr
#print 'lhs',[len(x.split(',')) for x in df[1].values]
## print	df[[1,2,'lhs_n','els_n','nt_els']].head()
#print set(df['lhs_n']) & set(rhs_nt_els_nbr)
#print 'Test .... if each rhs is not in lhs ... we cannot fire (?)'

if __name__ == '__main__':
	'''ToDo: 
		[] clean the edglists, write them back to disk and then run inddgo on 1 component graphs
		[] Add WillFire
	'''
	parser = get_parser()
	args = vars(parser.parse_args())
	try:
		main(args)
	except Exception, e:
		print str(e)
		traceback.print_exc()
		sys.exit(1)
	sys.exit(0)
