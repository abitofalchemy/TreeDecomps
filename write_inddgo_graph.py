__author__ = 'saguinag'+'@'+'nd.edu'
__version__ = "0.1.0"

##
## fname
##

## TODO: some todo list

## VersionLog:

import argparse,traceback
import os, sys, time
import networkx as nx
from datetime import datetime
import pandas as pd


def nx_edges_to_nddgo_graph(G):
	if G.name is None:
		t_str = datetime.now().strftime("%Y-%m-%d_%H_%M")
	edges = G.edges()
	edges = [(int(e[0]), int(e[1])) for e in edges]
	df = pd.DataFrame(edges)
	df.sort_values(by=[0],inplace=True)
	ofname = '{}.dimacs'.format(G.name)

	with open (ofname, 'w') as f:
		f.write('c {}\n'.format(G.name))
		f.write('p edge\t{}\t{}\n'.format(G.number_of_nodes(), G.number_of_edges()))
		# for e in df.iterrows():
		output_edges = lambda x: f.write("e\t{}\t{}\n".format(x[0], x[1]))
		df.apply(output_edges,axis=1)
			#f.write("e\t{}\t{}\n".format(e[0]+1,e[1]+1))
	if os.path.exists(ofname): print 'Wrote: ./{}'.format(ofname)

def get_parser():
	parser = argparse.ArgumentParser(description='gen_hrg: Generate synthetic graph using HRG model')
	parser.add_argument('-g', '--graph', required=True, help='input graph (edgelist)')
	parser.add_argument('--version', action='version', version=__version__)
	return parser

if __name__ == '__main__':
	parser = get_parser()
	args = vars(parser.parse_args())
	fname = args['graph']

	print "... ", fname
	gname = os.path.basename(fname).split('.')[1]
	print "... ", gname
	G = nx.read_edgelist(fname, comments="%", data=False)
	G.name = gname
	print "... info",nx.info(G)
	try:
		nx_edges_to_nddgo_graph(G)
		#save_plot_figure_2disk(plotname=plt_filename)
		#print 'Saved plot to: '+plt_filename
	except Exception, e:
		print 'ERROR, UNEXPECTED SAVE PLOT EXCEPTION'
		print str(e)
		traceback.print_exc()
		os._exit(1)
	sys.exit(0)
