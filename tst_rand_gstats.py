from glob import glob
import os
import networkx as nx
import numpy as np
import pandas as pd
from exact_phrg import grow_exact_size_hrg_graphs_from_prod_rules
import tdec.probabilistic_cfg as pcfg
import traceback

def graph_stats_and_visuals(gobjs=None):
	"""
	graph stats & visuals
	:gobjs: input nx graph objects
	:return: 
	"""
	import matplotlib
	matplotlib.use('pdf')
	import matplotlib.pyplot as plt
	import matplotlib.pylab as pylab
	params = {'legend.fontsize': 'small',
						'figure.figsize': (1.6 * 7, 1.0 * 7),
						'axes.labelsize': 'small',
						'axes.titlesize': 'small',
						'xtick.labelsize': 'small',
						'ytick.labelsize': 'small'}
	pylab.rcParams.update(params)
	import matplotlib.gridspec as gridspec

	print "BA G(V,E)"
	if gobjs is None:
		gobjs = glob("datasets/synthG*.dimacs")
	dimacs_g = {}
	for fl in gobjs:
		with open(fl, 'r') as f:
			l=f.readline()
			l=f.readline().rstrip('\r\n')
			bn = os.path.basename(fl)
			dimacs_g[bn] = [int(x) for x in l.split()[-2:]]
		print "%d\t%s" %(dimacs_g[bn][0], dimacs_g[bn][1])

	print "BA Prod rules Stacked"
	for k in dimacs_g.keys():
		fname = "ProdRules/"+k.split('.')[0]+".prs"
		if os.path.exists(fname):
			f_sz = np.loadtxt(fname, delimiter="\t", dtype=str)
			print k.split("_")[1], len(f_sz)

	print "BA Prod rules Isom Subset"
	for k in dimacs_g.keys():
			fname = "Results/"+k.split('.')[0]+"_isom_interxn.tsv"
			if os.path.exists(fname):
				f_sz = np.loadtxt(fname, delimiter="\t", dtype=str)
				print k.split("_")[1], len(f_sz)

#	print "Synth HRG graphs from Isom Subset"
#	from exact_phrg import grow_exact_size_hrg_graphs_from_prod_rules
#	for k in dimacs_g.keys():
#		fname = "Results/"+k.split('.')[0]+"_isom_interxn.tsv"
#		if os.path.exists(fname):
#			print k.split('.')[0], k.split("_")[1]
#
#			rules = np.loadtxt(fname, delimiter="\t", dtype=str)#[str,list, list, float]) # subset
#			g = pcfg.Grammar('S')
#			for (id, lhs, rhs, prob) in rules:
#				g.add_rule(pcfg.Rule(id, lhs, rhs, float(prob)))
#			#
#			num_nodes = int(k.split("_")[1])
#			print "Starting max size", num_nodes
#			try:
#				g.set_max_size(num_nodes)
#				print "Done with max size"
#			except  Exception, e:
#				print "!!!", str(e)
#				traceback.print_exc()
#				continue
#			
#			hstars_lst = []
#			print "  ",
#			for i in range(0, 10):
#				print '>',
#				try:
#					rule_list = g.sample(num_nodes)
#				except Exception, e:
#					print "!!!", str(e)
#					traceback.print_exc()
#					rule_list = g.sample(num_nodes)
#
#				hstar = phrg.grow(rule_list, g)[0]
#				hstars_lst.append(hstar)


#			hStars = grow_exact_size_hrg_graphs_from_prod_rules(rules, k.split('.')[0],
#																													int(k.split("_")[1]),10)
#			print '...', [g.number_of_nodes() for g in hStars]

#				hStars = grow_exact_size_hrg_graphs_from_prod_rules(rules, graph_name, G.number_of_nodes(),10)
def graph_gen_isom_interxn(fbname):
	import pandas as pd
	fname = 'ProdRules/'+fbname
	df = pd.read_csv(fname,index_col=0,compression="bz2")
	g = pcfg.Grammar('S')
	for (id, lhs, rhs, prob) in df.values:
#		print id, lhs, rhs, float(prob)
		g.add_rule(pcfg.Rule(id, lhs, rhs, float(prob)))

	num_nodes = int(fbname.split("_")[1])+1
	print "Starting max size", num_nodes
	try:
		g.set_max_size(num_nodes)
		print "Done with max size"
	except  Exception, e:
		print "!!!", str(e)
		traceback.print_exc()
		g.set_max_size(num_nodes)

	hstars_lst = []
	for i in range(0, 10):
		print " "*100,i
		try:
			rule_list = g.sample(num_nodes)
		except Exception, e:
			print "!!!", str(e)
			traceback.print_exc()
			continue


def ba_edges(v_lst):
	ba_graphs_d = {}
	eol = []
	for n_v in v_lst:
		e_o.append(np.random.choice(range(0,n_v)))
	return e_o

#		ba_graphs_d[n_v] = nx.barabasi_albert_graph(n_v, e_o)
#	for k,v in ba_graphs_d.iteritems():
#		print k, v.number_of_nodes(), v.number_of_edges()

def ba_control_hrg(v_lst):
	
	data = []
	for n_v in v_lst:
		g_obj = nx.barabasi_albert_graph(n_v, np.random.choice(range(1,n_v)))
		print "ba", g_obj.number_of_nodes(), g_obj.number_of_edges()
		data.append(g_obj)
		hrg = hrg_graph_gen(


#~# Main
#graph_stats_and_visuals()
#graph_gen_isom_interxn("synthG_127_3072.mmd_prules.bz2")
ba_control_hrg([10,20,50,100])
