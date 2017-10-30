__author__ = 'saguinag'+'@'+'nd.edu'
__version__ = "0.1.0"

##
## fname
##

## TODO: some todo list

## VersionLog:

import argparse,traceback,optparse
import os, sys, time
import networkx as nx
import numpy as np
import pandas as pd
import comp_metrics as cm
import probabilistic_cfg as pcfg

def gen_nx_graph_obj (fname):
    G = nx.read_edgelist(fname, comments="%")
    hstar = cm.synthetic_graph_generator(G,'hrg')
    fname = os.path.basename(fname).split('.')[1]
    files = [ f for f in os.listdir("./Results") if os.path.isfile(os.path.join("./Results",f)) ]
    files=  [f for f in files if fname in f]
    try:
			nx.write_edgelist(hstar, "Results/{}{}_hstar.edgelist.bz2".format(fname,len(files)))
    except Exception, e:
			print 'ERROR, UNEXPECTED Write output file EXCEPTION'
			print str(e)
			traceback.print_exc()
			os._exit(1)
    return "Results/{}{}_hstar.edgelist.bz2".format(fname,len(files))

def get_parser():
	parser = argparse.ArgumentParser(description='gen_hrg: Generate synthetic graph using HRG model')
	parser.add_argument('-g', '--graph', required=True, help='input graph (edgelist)')
	parser.add_argument('--version', action='version', version=__version__)
	return parser

def main():
    parser = get_parser()
    args = vars(parser.parse_args())

    print gen_nx_graph_obj(args['graph']) # gen synth graph

if __name__ == '__main__':
	try:
		main()
		#save_plot_figure_2disk(plotname=plt_filename)
		#print 'Saved plot to: '+plt_filename
	except Exception, e:
		print 'ERROR, UNEXPECTED SAVE PLOT EXCEPTION'
		print str(e)
		traceback.print_exc()
		os._exit(1)
	sys.exit(0)
