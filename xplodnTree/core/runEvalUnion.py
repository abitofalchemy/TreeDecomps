"""
Evaluate FakeGraphs
"""
import traceback
import argparse
import os
import sys
import pandas as pd
import multiprocessing
import pprint as pp
import tdec.probabilistic_cfg as pcfg
import pprint as pp

from glob import glob
from explodingTree import graph_name
from baseball import recompute_probabilities_two

__version__ = "0.1.0"

#_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~#
# def var_elims_tree_prs_union(fname):

def runEvalUnion(graph_name):
	# get the var elim .prs files
	# TODO finish this in a loop
	var_elim_prs_files = glob("ProdRules/{}*lcc*.tree.prs".format(graph_name))
	mdf = pd.DataFrame()
	for f in var_elim_prs_files:
		df = pd.read_csv(f, sep="\t", header=None, )
		mdf = pd.concat([mdf, df])
		# print mdf.shape
	## --
	toRules = lambda x: (x[0], x[1], x[2], x[3])
	prod_rules = mdf.apply(toRules, axis=1).values
	# pp.pprint (prod_rules[:4])
	## --
	df = recompute_probabilities_two(mdf)

	df.to_csv(graph_name.split('.')[0]+"_rc.tsv", header=False, index=False, sep="\t") # rcprs = recomputed prod rules
	if os.path.exists(graph_name.split('.')[0]+"_rc.tsv"): print 'Saved file:', graph_name.split('.')[0]+"_rc.tsv"

	# g = pcfg.Grammar('S')
	# for (id, lhs, rhs, prob) in prod_rules:
	# 	g.add_rule(pcfg.Rule(id, lhs, rhs, prob))
	#
	# num_nodes = 1899#G.number_of_nodes()
	#
	# print "Starting max size", 'n=', num_nodes
	# g.set_max_size(num_nodes)
	#
	# print "Done with max size"
	#
	# Hstars = []
	#
	# num_samples = 20
	# print '*' * 40
	# for i in range(0, num_samples):
	#   rule_list = g.sample(num_nodes)
	#   hstar = phrg.grow(rule_list, g)[0]
	#   Hstars.append(hstar)
	# print hstar.number_of_nodes(), hstar.number_of_edges()



def main():
	parser = get_parser()
	inargs = vars(parser.parse_args())
	print inargs

	ifname = inargs['orig'][0]
	gname = graph_name(ifname)
	#fgFiles = glob('FakeGraphs/*'+gname +"*")
	#print (len(fgFiles), "number of files")
	print ("%% EvalUnion %%")
	runEvalUnion(gname)
	exit()

	print ("%%")
	prsfiles = glob('ProdRules/{}_lcc_{}.prs'.format(gname, [x for x in [0,1]]))
	mdf = pd.DataFrame() 	# masterDF
	for f in prsfiles:		# concat prod rules files
		df  = pd.read_csv(f, sep="\t", header=None)
		mdf = pd.concat([df, mdf])
	mdf.to_csv('ProdRules/{}_concat.prs'.format(gname), sep="\t", header=None, index=None)
	return



def get_parser ():
	parser = argparse.ArgumentParser(description='Clique trees for HRG graph model.')
	parser.add_argument('--orig',nargs=1, required=False, help="edgelist input file")
	parser.add_argument('--version', action='version', version=__version__)
	return parser

if __name__ == '__main__':
	'''ToDo: clean the edglists, write them back to disk and then run inddgo on 1 component graphs
	'''
	try:
		main()
	except Exception, e:
		print str(e)
		traceback.print_exc()
		sys.exit(1)
	sys.exit(0)
