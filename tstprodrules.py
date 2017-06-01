# make the other metrics work
# generate the txt files, then work on the pdf otuput
__version__ = "0.1.0"
import pandas as pd
import sys
import os
import re
import numpy as np
from glob import glob
import pprint as pp
import argparse, traceback
import tdec.graph_sampler as gs
import tdec.probabilistic_cfg as pcfg
from td_isom_jaccard_sim import listify_rhs

DBG = False

def tst_prod_rules_isom_intrxn(fname):
	df = pd.read_csv(fname, header=None,  sep="\t", dtype={0: str, 1: str, 2: list, 3: float})
	g = pcfg.Grammar('S')
	from td_isom_jaccard_sim import listify_rhs
	for (id, lhs, rhs, prob) in df.values:
		rhs = listify_rhs(rhs)
		# print (id), (lhs), (rhs), (prob)
		g.add_rule(pcfg.Rule(id, lhs, rhs, float(prob)))

	num_nodes = 16 # G.number_of_nodes()

	# print "Starting max size", 'n=', num_nodes
	g.set_max_size(num_nodes)

	# print "Done with max size"
	Hstars = []

	num_samples = 20
	print
	print '-' * 40
	for i in range(0, num_samples):
		rule_list = g.sample(num_nodes)
	print '+' * 40

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
		hstar     = phrg.grow(rule_list, g)[0]
		Hstars.append(hstar)
		print '+' * 40
		# break

# def tst_prod_rules_canfire(infname):
	# print infname

	# print df[2].apply(lambda x: listify_rhs(x.split()[0]))
# tst_prod_rules_isom_intrxn("Results/moreno_vdb_vdb_isom_itrxn.tsv")
# tst_prod_rules_level1_individual("ProdRules/moreno_lesmis_lesmis.*_iprules.tsv")
def test_isom_subset_of_prod_rules(fname):
	print fname
	try:
		tst_prod_rules_isom_intrxn(fname)
	except Exception, e:
		print str(e)
		print fname, "----- does not work"

test_isom_subset_of_prod_rules("Results/moreno_lesmis_lesmis_isom_itrxn.tsv")
print
test_isom_subset_of_prod_rules("Results/moreno_vdb_vdb_isom_itrxn.tsv")
print
test_isom_subset_of_prod_rules("Results/ucidata-gama_isom_itrxn.tsv")
print
test_isom_subset_of_prod_rules("Results/ucidata-zachary_isom_itrxn.tsv")

##
## tst_prod_rules_canfire("Results/ucidata-gama_stcked_prs_isom_itrxn.tsv")
##
infname ="Results/ucidata-gama_isom_itrxn.tsv"
df = pd.read_csv(infname, header=None,  sep="\t", dtype={0: str, 1: str, 2: list, 3: float})

df['lhs_n']=[len(x.split(',')) for x in df[1].values]
df['els_n']=[len(listify_rhs(x)) for x in df[2].values]
df['nt_els']=[len([k for k in listify_rhs(x) if ":N" in k]) for x in df[2].values]
print '#'*10
rhs_nt_els_nbr = {}
for y,x in df[[1,2]].values:
	# print x
	rhs_nts =[]
	for k in listify_rhs(x):
		if ":N" in k:
			# print ' ', len(k.split(','))
			rhs_nts.append(len(k.split(',')))
	rhs_nt_els_nbr[len(y.split(','))] = rhs_nts
for k,v in rhs_nt_els_nbr.items():
	print k,'\t',v

print '^'*20
print 'rhs',rhs_nt_els_nbr
print 'lhs',[len(x.split(',')) for x in df[1].values]
# print  df[[1,2,'lhs_n','els_n','nt_els']].head()
print set(df['lhs_n']) & set(rhs_nt_els_nbr)
print 'Test .... if each rhs is not in lhs ... we cannot fire (?)'
