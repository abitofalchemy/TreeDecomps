#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import networkx as nx
import traceback
import sys
import os
import subprocess

from explodingTree import graph_name,load_edgelist,edgelist_to_dimacs,dimacs_nddgo_tree
from tredec_samp_phrg import dimacs_td_ct
from tdec.load_edgelist_from_dataframe import Pandas_DataFrame_From_Edgelist
from glob import glob

import tdec.graph_sampler as gs
import pprint as pp
import tdec.PHRG as phrg

"""
Process a given (orig) reference graph and ensure 
- sampling
- edglist to dimacs conversion
- dimacs to tree (decomposition) 
- [x] .tree to clique tree -> to production rules
- [todo] process prod rules to get Union 
	==> get graphs from the union of Varl Elim Prod Rules
- [todo] process prod rules to find/get isomorphic 
"""

def dbginfo(x,comment=False):
	print "\t",
	if comment:
		print "==>", x
		return 

	print "==>\t", type(x)
	if isinstance(x, list): print "\tlen:",len(x)


def sample_rand_subgraphs_in(g):
	dbginfo("sample_rand_subgraphs_in(g):",1)
	if g.number_of_nodes() <500 : 
		return [g]
	glst = gs.rwr_sample(g, 2, 300)
	return glst

def write_subgraph_toeltsv(sgNbr,x,gname):
#	dbginfo("write_subgraph_toeltsv",1)
	out_el_fname = ".tmp_edgelists/{}_{}_edgelist.tsv".format(gname,sgNbr)
	if not os.path.exists(out_el_fname):
		nx.write_edgelist(x, out_el_fname, delimiter="\t", data=False)
	return out_el_fname


def exec_call(arg_a):
	"""
	call_out = ""
	args = ["python exact_phrg.py --orig {} --prs".format(arg_a)]
	print args
	while not call_out:
		popen = subprocess.Popen(args, stdout=subprocess.PIPE, shell=True)
		popen.wait()
		out, err = popen.communicate()
		call_out = out.split('\n')
	print call_out, out, err
	"""
	G = load_edgelist(arg_a)
	prod_rules = phrg.probabilistic_hrg_deriving_prod_rules(G)
	pp.pprint(prod_rules)

import multiprocessing 
def inddgo_xproc(f_instance, heuristic):
	nddgoout = ""
	outfname = dimacsfname+"."+heuristic+".tree"
	if platform.system() == "Linux":
		args = ["bin/linux/serial_wis -f {} -nice -{} -w {} -decompose_only".format(dimacsfname, heuristic, outfname)]
	else:
		args = ["bin/mac/serial_wis -f {} -nice -{} -w {} -decompose_only".format(dimacsfname, heuristic, outfname)]
	while not nddgoout:
		popen = subprocess.Popen(args, stdout=subprocess.PIPE, shell=True)
		popen.wait()
		# output = popen.stdout.read()
		out, err = popen.communicate()
		nddgoout = out.split('\n')

	results.append(outfname)

def dimacs_inddgo_tree_decomps(var_elim, gname):
	'''
	Scan folder for .dimacs files and if not converted to .tree,
	convert it. 
	'''
	#dbginfo("dimacs_inddgo_tree_decomps",1)
#	dimacs_files = glob("datasets/{}*.dimacs".format(gname))
#	dimacs_files = "datasets/{}.dimacs.{}.tree".format(gname, var_elim)
#	print "\t===>", gname
	multiprocessing.Process(target=dimacs_nddgo_tree, args=([gname],var_elim,)).start()

def convert_dimacs_trees_to_cliquetrees(gname):
#	dbginfo("convert_dimacs_trees_to_cliquetrees",1)
	ct_files = glob("datasets/{}*.dimacs*.tree".format(gname))
	for f in ct_files:
		multiprocessing.Process(target=dimacs_td_ct, args=("datasets/out.{}".format(gname),f,)).start()

def concat_phrg_prod_rules(glst, graph_name):
	elfiles = [write_subgraph_toeltsv(j,x,graph_name) for j,x in enumerate(glst)]
	for f in elfiles:
		multiprocessing.Process(target=edgelist_to_dimacs, args=(f,)).start()
	dbginfo("edgelists to dimacs, processig ... ",1 )
	return elfiles

def main():
	gname = graph_name(sys.argv[1])
	print gname
	concat_prs = "ProdRules/{}_concat.prs".format(gname)

	if not os.path.exists(concat_prs):
		G	=		load_edgelist(sys.argv[1])
		print "[<>]","red the graph"
		lcc =		max(nx.connected_component_subgraphs(G), key=len)	# find largest conn component
		Glst =	sample_rand_subgraphs_in(lcc) #
		print "[<>]","got the Glst LCCs"
		
		concat_phrg_prod_rules([x for x in Glst], G.name) # subgraphs base prod rules
		
		dimacs_files = glob("datasets/{}*.dimacs".format(gname))
		var_el_lst = ['mcs','mind','minf','mmd','lexm','mcsm']
		for gfname in dimacs_files:
			for ve in var_el_lst:
				multiprocessing.Process(target= dimacs_inddgo_tree_decomps, args=(ve,gfname,)).start()
		print "[<>]","checks on the edgelist vs the orig graph"
		
		## --
		convert_dimacs_trees_to_cliquetrees(gname)
		print "[<>]","convert_dimacs_trees_to_cliquetrees"
		
		## --
		elfiles = glob(".tmp_edgelists/{}*tsv".format(gname))
		subgraphs = [load_edgelist(f) for f in elfiles]
		prod_rules = []
		prod_rules = [phrg.probabilistic_hrg_deriving_prod_rules(G) for G in subgraphs]
		import itertools
		prod_rules = list(itertools.chain.from_iterable(prod_rules))
		pd.DataFrame(prod_rules).to_csv(concat_prs, sep="\t", header=False, index=False)
		
		## --
		dimacs_files = glob("datasets/{}*.dimacs".format(gname))
		var_el_lst = ['mcs','mind','minf','mmd','lexm','mcsm']
		for gfname in dimacs_files:
			for ve in var_el_lst:
				multiprocessing.Process(target= dimacs_inddgo_tree_decomps, args=(ve,gfname,)).start()

		print "[<>]","checks on the edgelist vs the orig graph"
		
		
	print "[<>]","concat hrg prod_rules:", concat_prs
	


if __name__ == '__main__':
	'''ToDo: cli
	'''
	try:
		main()
	except Exception, e:
		print str(e)
		traceback.print_exc()
		sys.exit(1)
	sys.exit(0)
		
