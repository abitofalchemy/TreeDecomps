__author__ = 'saguinag'+'@'+'nd.edu'
__version__ = "0.1.0"

##
## fname
##

## TODO: some todo list

## VersionLog:

import argparse,traceback,optparse
import os, sys, time
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import comp_metrics as  nm


def get_parser():
	parser = argparse.ArgumentParser(description='netprops: computers network properties')
	parser.add_argument('-g',  '--graph',	required=True, help='input graph (edgelist)')
	parser.add_argument('-d',  '--degree',	action='store_true', help='compute degree distribution')
	parser.add_argument('-p',  '--hopplot',	action='store_true', help='compute hopplot')
	parser.add_argument(       '--gcd',	action='store_true', help='graphlet correlation distance')
	parser.add_argument('-m',  '--modularity',	action='store_true', help='compute modularity')
	# parser.add_argument('-s',  '--save',action='store_true', help='Save to disk with unique names')
	parser.add_argument('-a', '--all', action='store_true', help='Compute all net props')
	parser.add_argument('--version', action='version', version=__version__)
	return parser

def main():
	parser = get_parser()
	args = vars(parser.parse_args())

	# checks on the inpout file (make sure it's an edgelist and and readable)
	print '... input:', args['graph']
	cnprops = []
	if args['degree'] is True:
		cnprops.append('degree')

	if args['hopplot']  is True:
		cnprops.append('hopplot')

	if args['modularity'] is True:
		# ('modularity','hrg'))
		print '... net property:', 'modularity'
		cnprops.append('modularity')
	if args['gcd'] is True:
		cnprops.append('gcd')
	if args['all'] is True:
		cnprops = list('all')

	print nm.compute_net_properties( args['graph'], cnprops )

	print '... Done.'

if __name__ == '__main__':
	# g = command_line_runner()

	# ## View/Plot the graph to a file
	# fig = plt.figure(figsize=(1.6*6,1*6))
	# ax0 = fig.add_subplot(111)

	# nx.draw_networkx(g[1],ax=ax0)
	# plt_filename="/tmp/outfig"

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
