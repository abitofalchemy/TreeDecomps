from glob import glob
import tdec.net_metrics as metrics
import os
import csv
import math
import networkx as nx
import numpy as np
import pandas as pd
from exact_phrg import grow_exact_size_hrg_graphs_from_prod_rules
import tdec.probabilistic_cfg as pcfg
import tdec.PHRG as phrg
import traceback
from td_rndGraphs import convert_nx_gObjs_to_dimacs_gObjs



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
#			except	Exception, e:
#				print "!!!", str(e)
#				traceback.print_exc()
#				continue
#
#			hstars_lst = []
#			print "	",
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
	except	Exception, e:
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
def save_nxgobjs_to_disk(gObjs, pickleOutF):
	import pickle
	pickle.dump(gObjs, open(pickleOutF, "wb"))
	if os.path.exists(pickleOutF): print '  ','Wrote gObjs list to a pickle file:', pickleOutF
	return

def ba_control_hrg(v_lst):
	grow_graphs = False
	v_lst = [int(n) for n in v_lst]
	data = []
	prules_lst = []
	for n_v in v_lst:
		nxgobj = nx.barabasi_albert_graph(n_v, np.random.choice(range(1,n_v)))
		nxgobj.name = "ba_%d_%d" %(nxgobj.number_of_nodes(), nxgobj.number_of_edges())

		print "ba", nxgobj.number_of_nodes(), nxgobj.number_of_edges()
		data.append(nxgobj)
		prod_rules = phrg.probabilistic_hrg_deriving_prod_rules(nxgobj)
		prules_lst.append(nxgobj.name)
		prules_lst.append(prod_rules)
		g = pcfg.Grammar('S')
		for (id, lhs, rhs, prob) in prod_rules:
			g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

		num_nodes = nxgobj.number_of_nodes()

		print "Starting max size", 'n=', num_nodes
		g.set_max_size(num_nodes)

		print "Done with max size"

		Hstars = []

		num_samples = 10
		print '*' * 40
		for i in range(0, num_samples):
			try:
				rule_list = g.sample(num_nodes)
			except Exception, e:
				print str(e)
				traceback.print_exc()
				continue #sys.exit(1)

			hstar = phrg.grow(rule_list, g)[0]
			Hstars.append(hstar)
		if 0:
			metricx = ['degree','clust', 'hop', 'gcd']
			metrics.network_properties([nxgobj], metricx, Hstars, name=nxgobj.name, out_tsv=False)

	#	convert_nx_gObjs_to_dimacs_gObjs(data)
	save_nxgobjs_to_disk(data, "datasets/ba_cntrl_%d_%d.p"%(v_lst[0],v_lst[-1]))
	with open("Results/ba_cntrl_%d_%d.tsv"%(v_lst[0],v_lst[-1]),'wb') as fout:
	    writer = csv.writer(fout,delimiter='\t')
	    writer.writerows(prules_lst)
#~# Main
#graph_stats_and_visuals()
#graph_gen_isom_interxn("synthG_127_3072.mmd_prules.bz2")
ba_control_hrg([math.pow(2,x) for x in range(4,8,1)])
