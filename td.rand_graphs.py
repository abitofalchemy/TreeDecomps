#!/usr/bin/env python
__version__="0.1.0"
__name__="rand_graphs"

# ToDo:
# [] process mult dimacs.trees to hrg

import sys
import math
import numpy as np
import traceback
import argparse
import os
import glob
import networkx as nx
import pandas as pd
from tdec.PHRG import graph_checks
import subprocess
import math
import tdec.graph_sampler as gs
import platform


def get_parser ():
	parser = argparse.ArgumentParser(description='Process random graphs')
	parser.add_argument('--version', action='version', version=__version__)
	return parser

def dimacs_nddgo_tree(dimacsfnm_lst, heuristic):
	# print heuristic,dimacsfnm_lst

	for dimacsfname in dimacsfnm_lst:
		nddgoout = ""
		if platform.system() == "Linux":
		  args = ["bin/linux/serial_wis -f {} -nice -{} -w {}.tree".format(dimacsfname, heuristic, dimacsfname)]
		else:
		  args = ["bin/mac/serial_wis -f {} -nice -{} -w {}.tree".format(dimacsfname, heuristic, dimacsfname)]
		while not nddgoout:
			popen = subprocess.Popen(args, stdout=subprocess.PIPE, shell=True)
			popen.wait()
			# output = popen.stdout.read()
			out, err = popen.communicate()
			nddgoout = out.split('\n')
		print nddgoout
	return dimacsfname+".tree"

def load_edgelist(gfname):
  import pandas as pd
  try:
	edglst = pd.read_csv(gfname, comment='%', delimiter='\t')
	# print edglst.shape
	if edglst.shape[1]==1: edglst = pd.read_csv(gfname, comment='%', delimiter="\s+")

  except Exception, e:
	print "EXCEPTION:",str(e)
	traceback.print_exc()
	sys.exit(1)

  if edglst.shape[1] == 3:
	edglst.columns = ['src', 'trg', 'wt']
  elif edglst.shape[1] == 4:
	edglst.columns = ['src', 'trg', 'wt','ts']
  else:
	edglst.columns = ['src', 'trg']
  g = nx.from_pandas_dataframe(edglst,source='src',target='trg')
  g.name = os.path.basename(gfname)
  return g



def nx_edges_to_nddgo_graph_sampling(graph, n, m, peo_h):
	G = graph
	if n is None and m is None: return
	# n = G.number_of_nodes()
	# m = G.number_of_edges()
	nbr_nodes = 256
	basefname = 'datasets/{}_{}'.format(G.name, peo_h)

	K = int(math.ceil(.25*G.number_of_nodes()/nbr_nodes))
	print "--", nbr_nodes, K, '--';

	for j,Gprime in enumerate(gs.rwr_sample(G, K, nbr_nodes)):
		# if gname is "":
		#	 # nx.write_edgelist(Gprime, '/tmp/sampled_subgraph_200_{}.tsv'.format(j), delimiter="\t", data=False)
		#	 gprime_lst.append(Gprime)
		# else:
		#	 # nx.write_edgelist(Gprime, '/tmp/{}{}.tsv'.format(gname, j), delimiter="\t", data=False)
		#	 gprime_lst.append(Gprime)
		# # print "...  files written: /tmp/{}{}.tsv".format(gname, j)


		edges = Gprime.edges()
		edges = [(int(e[0]), int(e[1])) for e in edges]
		df = pd.DataFrame(edges)
		df.sort_values(by=[0], inplace=True)

		ofname = basefname+"_{}.dimacs".format(j)

		with open(ofname, 'w') as f:
		  f.write('c {}\n'.format(G.name))
		  f.write('p edge\t{}\t{}\n'.format(n,m))
		  # for e in df.iterrows():
		  output_edges = lambda x: f.write("e\t{}\t{}\n".format(x[0], x[1]))
		  df.apply(output_edges, axis=1)
		# f.write("e\t{}\t{}\n".format(e[0]+1,e[1]+1))
		if os.path.exists(ofname): print 'Wrote: {}'.format(ofname)

	return basefname

def edgelist_dimacs_graph(orig_graph, peo_h, prn_tw = False):
	fname = orig_graph
	gname = os.path.basename(fname).split(".")
	gname = sorted(gname,reverse=True, key=len)[0]

	if ".tar.bz2" in fname:
		from tdec.read_tarbz2 import read_tarbz2_file
		edglst = read_tarbz2_file(fname)
		df = pd.DataFrame(edglst,dtype=int)
		G = nx.from_pandas_dataframe(df,source=0, target=1)
	else:
		G = nx.read_edgelist(fname, comments="%", data=False, nodetype=int)
	# print "...",  G.number_of_nodes(), G.number_of_edges()
	# from numpy import max
	# print "...",  max(G.nodes()) ## to handle larger 300K+ nodes with much larger labels

	N = max(G.nodes())
	M = G.number_of_edges()
	# +++ Graph Checks
	if G is None: sys.exit(1)
	G.remove_edges_from(G.selfloop_edges())
	giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
	G = nx.subgraph(G, giant_nodes)
	graph_checks(G)
	# --- graph checks

	G.name = gname

	# print "...",  G.number_of_nodes(), G.number_of_edges()
	if G.number_of_nodes() > 500 and not prn_tw:
		return (nx_edges_to_nddgo_graph_sampling(G, n=N, m=M, peo_h=peo_h), gname)
	else:
		return (nx_edges_to_nddgo_graph(G, n=N, m=M, varel=peo_h), gname)

def print_treewidth (in_dimacs, var_elim):
	nddgoout = ""
	if platform.system() == "Linux":
	  args = ["bin/linux/serial_wis -f {} -nice -{} -width".format(in_dimacs, var_elim)]
	else:
	  args = ["bin/mac/serial_wis -f {} -nice -{} -width".format(in_dimacs, var_elim)]
	while not nddgoout:
		popen = subprocess.Popen(args, stdout=subprocess.PIPE, shell=True)
		popen.wait()
		# output = popen.stdout.read()
		out, err = popen.communicate()
		nddgoout = out.split('\n')
	print nddgoout
	return nddgoout

def convert_nx_gObjs_to_dimacs_gObjs(nx_gObjs):
  '''
  Take list of graphs and convert to dimacs
  '''
  dimacs_glst=[]
  for G in nx_gObjs:
    N = max(G.nodes())
    M = G.number_of_edges()
    # +++ Graph Checks
    if G is None: sys.exit(1)

    G.remove_edges_from(G.selfloop_edges())
    giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
    G = nx.subgraph(G, giant_nodes)
    graph_checks(G)
    # --- graph checks
    G.name = "synthG_{}_{}".format(N,M)

    from tdec.arbolera import nx_edges_to_nddgo_graph
    dimacs_glst.append(nx_edges_to_nddgo_graph(G, n=N, m=M, save_g=True))

  return dimacs_glst

def tree_decomposition_with_varelims(dimacs_fnames, vElim):
  '''
  Tree decomp given a var elim
  Args:
    dimacs_fnames A list of filenames of corresponding dimacs graph format graphs
    vElim         Variable elimination method to use with
  '''
  for f in dimacs_fnames:
    print f
    break

def main ():
  # parser = get_parser()
  # args = vars(parser.parse_args())
  print "Hello" 
  n_nodes_set = [math.pow(2,x) for x in range(4,12,1)]
  n_edges_set = {}
  for n in n_nodes_set:
    n_edges_set[n] = nx.fast_gnp_random_graph(int(n), 0.5).number_of_edges()

  print n_nodes_set
  print n_edges_set
  ba_gObjs = [nx.barabasi_albert_graph(n, np.random.choice(range(1,int(n)))) for n in n_nodes_set]
  #~# 
  #~# convert to dimacs graph 
  dimacs_gObjs = convert_nx_gObjs_to_dimacs_gObjs(ba_gObjs,)
  var_el_m = ['lexm','mcs','mcsm','mind','minf','mmd']
  print dimacs_gObjs

  #~#
  #~# decompose the given graphs
  tree_objs = tree_decomposition_with_varelims(dimacs_gObjs, var_el_m)
  
  #~#
  #~# dimacs tree to HRG clique tree 
  clq_trees = convert_dimacs_tree_objs_to_hrg_clique_trees(tree_objs)

  #~# 
  #~# get stacked HRG prod rules
  stck_prod_rules = get_hrg_prod_rules(clq_trees)
  #~# 
  #~# get the isomophic overlap
  intxn_prod_rules = get_isom_overlap_in_stacked_prod_rules(stck_prod_rules)


try:
  main()
except Exception, e:
  print str(e)
  traceback.print_exc()
  sys.exit(1)
sys.exit(0)
