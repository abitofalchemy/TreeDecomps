#!/usr/bin/env python
import argparse
import traceback
import os,sys
import networkx as nx

from graph_datasets import read_dataset

"""
start arbitrarily at any node
from that node do BFS and mark nodes
explore triangles
keep track of visited nodes
alpha1 and alpha 2 involutions
	alpha1 wedges
	alpha2 triangles
"""

def gfrag_gmaps(nx_graph):
	G = nx_graph
	he_id = G.number_of_nodes()
	Darts = dict()
	dart = {}
	for v in G.nodes():
		neigs = list(G.neighbors(v))

		for vn in neigs:
			dart[v] = [he_id, he_id+1]
			he_id += 1


		print ('Dart:',dart)

		break




def submaps(graph_fname):
	graph = read_dataset(graph_fname)
	cmaps = gfrag_gmaps(graph)
	print(cmaps)


def get_parser():
    parser = argparse.ArgumentParser(description='break up given graph into its submaps.')
    parser.add_argument('-g', '--graph', help='input graph (path)', default=1, type=str)
	#
    return parser

def command_line_runner():
	parser = get_parser()
	args = vars(parser.parse_args())
	if len(sys.argv)<2:
		parser.print_help()
		exit()
	return args

if __name__ == '__main__':
	args = command_line_runner()
	print(args)
	try:
		submaps(args['graph'])
	except Exception as e:
		print (str(e))
		traceback.print_exc()
		os._exit(1)
	sys.exit(0)
