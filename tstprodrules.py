# make the other metrics work
# generate the txt files, then work on the pdf otuput
__version__ = "0.1.0"
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import sys
import os
import re
from glob import glob
import networkx as nx
import tdec.PHRG as phrg
import tdec.tree_decomposition as td
import tdec.probabilistic_cfg as pcfg
import tdec.net_metrics as metrics
import tdec.load_edgelist_from_dataframe as tdf
import pprint as pp
import argparse, traceback
import tdec.graph_sampler as gs

DBG = False

# fname = "Results/moreno_lesmis_lesmis_isom_itrxn.tsv" # (ctrl) exact phrg prod rules; should converge
# df = pd.read_csv(fname, header=None,  sep="\t", dtype={0: str, 1: str, 2: list, 3: float})
# g = pcfg.Grammar('S')
# from td_isom_jaccard_sim import listify_rhs
# for (id, lhs, rhs, prob) in df.values:
# 	rhs = listify_rhs(rhs)
# 	# print (id), (lhs), (rhs), (prob)
# 	g.add_rule(pcfg.Rule(id, lhs, rhs, float(prob)))
#
# num_nodes = 16 # G.number_of_nodes()
#
# print "Starting max size", 'n=', num_nodes
# g.set_max_size(num_nodes)
#
# print "Done with max size"
# Hstars = []
#
# num_samples = 20
# print '-' * 40
# for i in range(0, num_samples):
# 	rule_list = g.sample(num_nodes)
# print '+' * 40
#fname = "Results/ucidata-gama_stcked_prs.tsv"

# #fname = "ProdRules/ucidata-gama_stcked_prs.tsv"
# #fname = "Results/ucidata-gama_stcked_prs_isom_itrxn.tsv"
# #fname = "ProdRules/tst.tsv"


files = glob("ProdRules/moreno_lesmis_lesmis.*_iprules.tsv")
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
