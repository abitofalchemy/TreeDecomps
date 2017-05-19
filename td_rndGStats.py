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
	if os.path.exists(pickleOutF): print '	','Wrote gObjs list to a pickle file:', pickleOutF
	return

def ba_control_hrg(v_lst):
	grow_graphs = False
	v_lst = [int(n) for n in v_lst] # set of nodes to generate BA graphs
	data = []
	prules_lst = []
	for n_v in v_lst:
		nxgobj = nx.barabasi_albert_graph(n_v, np.random.choice(range(1,n_v)))
		nxgobj.name = "ba_%d_%d" %(nxgobj.number_of_nodes(), nxgobj.number_of_edges())

		print "ba", nxgobj.number_of_nodes(), nxgobj.number_of_edges()
		data.append(nxgobj)
		prod_rules = phrg.probabilistic_hrg_deriving_prod_rules(nxgobj)
		df = pd.DataFrame(list(prod_rules))
		ofname = "Results/ba_cntrl_%d.tsv"%(n_v)
		df.to_csv(ofname, sep="\t", header=False, index=False)
		

		prules_lst.append(prod_rules)
		g = pcfg.Grammar('S')
		for (id, lhs, rhs, prob) in df.values:
			g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

		num_nodes = nxgobj.number_of_nodes()

		print "  ","Starting max size", 'n=', num_nodes
		g.set_max_size(num_nodes)
		print "  ","Done with max size"

		Hstars = []
		num_samples = 10
		for i in range(0, num_samples):
			try:
				rule_list = g.sample(num_nodes)
			except Exception, e:
				print str(e)
				traceback.print_exc()
				continue #sys.exit(1)

			hstar = phrg.grow(rule_list, g)[0]
			Hstars.append(hstar)
		print "  ", 'Save BA production rules'
		# with open("Results/ba_cntrl_%d.tsv"%(n_v),'wb') as fout:
		# 	writer = csv.writer(fout,delimiter='\t')
		# 	writer.writerows(prules_lst)
		
		
		if os.path.exists(ofname):
				print '\tSaved to disk:',ofname
		if 0:
			metricx = ['degree','clust', 'hop', 'gcd']
			metrics.network_properties([nxgobj], metricx, Hstars, name=nxgobj.name, out_tsv=False)
		break
	
	#	convert_nx_gObjs_to_dimacs_gObjs(data)
	print '#~#'*4
	print 'Save nxobjs to disk'
	save_nxgobjs_to_disk(data, "datasets/ba_cntrl_%d_%d.p"%(v_lst[0],v_lst[-1]))
	


def graph_gen_isom_interxn(fbsname):
	import pandas as pd
	if 0: df = pd.read_csv(fbsname,index_col=0,compression="bz2")
	df = pd.read_csv(fbsname, sep="\t", header=None)
	for i,r in df.iterrows():
		print list(r[2])
		break
	
	
	with open (fbsname, 'r') as f:
		lines = f.readlines()
		lines = [x.rstrip('\r\n') for x in lines]
		lines = [x.split('\t') for x in lines]
	for x in lines:
		print x
	exit()

	#	g = pcfg.Grammar('S')
	#	for (id, lhs, rhs, prob) in df.values:
	#		print id, lhs, rhs, float(prob)
	#		g.add_rule(pcfg.Rule(id, lhs, rhs, float(prob)))
	#	fbname = os.path.basename(fbsname)
	#	print fbname.split("_")[-1].rstrip(".tsv")
	#	num_nodes = int(fbname.split("_")[-1].rstrip(".tsv"))
	#	print "Starting max size", num_nodes
	#	g.set_max_size(num_nodes)
	#	print "Done with max size"
	#
	#	hstars_lst = []
	#	for i in range(0, 10):
	#		try:
	#			rule_list = g.sample(num_nodes)
	#			print ">"*10,i
	#		except Exception, e:
	#			print "!!!", str(e)
	#			traceback.print_exc()
	#			continue
	lines = [
("r0.0",	"A,B",	['A,B:N', 'A:N'],	0.857142857143),
("r0.1",	"A,B",	['A:N', 'B:N'],	0.142857142857),
("r1.0",	"S",	['0,1:T', '0,1:N', '0:N'],	1.0),
("r2.0",	"A",	['0,A:T', 'A:N', 'A:N'],	0.0666666666667),
("r2.1",	"A",	['A:T'],	0.0666666666667),
("r2.2",	"A",	['0,A:T', '0:N', '0:N'],	0.0666666666667),
("r2.3",	"A",	['0,A:T', '0:N'],	0.133333333333),
("r2.4",	"A",	['0,A:T'],	0.666666666667)]
	g = pcfg.Grammar('S')
	for (id, lhs, rhs, prob) in lines:
		print (id, lhs, rhs, prob)
		g.add_rule(pcfg.Rule(id, lhs, rhs, float(prob)))
	
	num_nodes = int(fbsname.split("_")[-1].rstrip(".tsv"))
	print "  ","Starting max size", 'n=', num_nodes
	g.set_max_size(num_nodes)
	print "  ","Done with max size"
	
	Hstars = []
	num_samples = 10
	for i in range(0, num_samples):
		try:
			rule_list = g.sample(num_nodes)
		except Exception, e:
			print str(e)
			traceback.print_exc()
			continue #sys.exit(1)
		
		hstar = phrg.grow(rule_list, g)[0]
		Hstars.append(hstar)
	print "  ", 'Save BA production rules'

#~# Main
if __name__ == '__main__':
	#graph_stats_and_visuals()
	#graph_gen_isom_interxn("synthG_127_3072.mmd_prules.bz2")
	if 0: ba_control_hrg([math.pow(2,x) for x in range(4,8,1)])
	if 0: graph_gen_isom_interxn("synthG_127_3072.mmd_prules.bz2")
	if 1: graph_gen_isom_interxn("Results/ba_cntrl_16.tsv")


