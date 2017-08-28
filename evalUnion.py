"""
Evaluate FakeGraphs
"""
import traceback
import argparse
import os
import sys 
import pandas as pd

from glob import glob
from explodingTree import graph_name
__version__ = "0.1.0"

#_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~_~#
def main():
	parser = get_parser()
	inargs = vars(parser.parse_args())
	print inargs

	ifname = inargs['orig'][0]
	gname = graph_name(ifname)
	#fgFiles = glob('FakeGraphs/*'+gname +"*")
	#print (len(fgFiles), "number of files")
	print ("%% EvalUnion %%")
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

