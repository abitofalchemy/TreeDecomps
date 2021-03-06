from glob import glob
import tdec.net_metrics as metrics
import os
import re
import sys
import csv
import math
import pickle
import networkx as nx
import numpy as np
import pandas as pd
from exact_phrg import grow_exact_size_hrg_graphs_from_prod_rules
import tdec.probabilistic_cfg as pcfg
import tdec.PHRG as phrg
import traceback
from td_rndGraphs import convert_nx_gObjs_to_dimacs_gObjs



def graph_stats_and_visuals(gobjs=None):
	"""Graph stats & visuals
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
#	import pickle
#	pickle.dump(gObjs, open(pickleOutF, "wb"))
#	if os.path.exists(pickleOutF): print '	','Wrote gObjs list to a pickle file:', pickleOutF
	out_edgelist_fnames = []
	
	for j,g in enumerate(gObjs):
		ofname = "datasets/"+g.name+"_%d.tsv"%j
		out_edgelist_fnames.append(ofname)
		nx.write_edgelist(g, ofname,data=False)
	
	return out_edgelist_fnames

def ba_control_hrg(v_lst):
	grow_graphs = False
	v_lst = [int(n) for n in v_lst] # set of nodes to generate BA graphs
	data = []
	prules_lst = []
	for n_v in v_lst:
		# nxgobj = nx.barabasi_albert_graph(n_v, np.random.choice(range(1,n_v)))
		nxgobj = nx.barabasi_albert_graph(n_v,3)
		nxgobj.name = "ba_%d_%d" %(nxgobj.number_of_nodes(), nxgobj.number_of_edges())

		print "ba", nxgobj.number_of_nodes(), nxgobj.number_of_edges()
		data.append(nxgobj)
		prod_rules = phrg.probabilistic_hrg_deriving_prod_rules(nxgobj)
		df = pd.DataFrame(list(prod_rules))
		out_base_fname = "ba_cntrl_%d"%(n_v)
		ofname = "Results/" + out_base_fname + ".tsv" #_________________
		df.to_csv(ofname, sep="\t", header=False, index=False)


		prules_lst.append(prod_rules)
		g = pcfg.Grammar('S')
		for (id, lhs, rhs, prob) in df.values:
			g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

		num_nodes = nxgobj.number_of_nodes()

		print "	","Starting max size", 'n=', num_nodes
		g.set_max_size(num_nodes)
		print "	","Done with max size"

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
		print "	", 'Save BA production rules'



		if os.path.exists(ofname):
				print '\tSaved to disk:',ofname
		if 0:
			metricx = ['degree','clust', 'hop', 'gcd']
			metrics.network_properties([nxgobj], metricx, Hstars, name=nxgobj.name, out_tsv=False)

	
	#	convert_nx_gObjs_to_dimacs_gObjs(data)
	print '#~#'*4
	print 'Save nxobjs to disk'
	edglst_fnames_lst = save_nxgobjs_to_disk(data, "datasets/" + out_base_fname + "_%d.p"%v_lst[-1])

	return out_base_fname, v_lst[-1], edglst_fnames_lst

def graph_gen_isom_interxn(in_fname="", orig_el=None):
	"""Generate graph using isomorphic intersection of the production rules.
	Keyword arguments:
		in_fname -- file basename bz2 for the production rules?
		orig_el	-- original graph's edgelist (optional)


	Example: Pass a file derived from an input file to test that this works by defualt.
	"""
	in_fname = in_fname
	df = pd.read_csv(in_fname, delimiter='\t', header=None)
	rhs_clean = lambda rhs_rule: [f[1:-1] for f in re.findall("'.+?'", rhs_rule)]
	try:
		df['rhslst'] = df[2].apply(rhs_clean)
	except Exception, e:
		print str(e)
		df['rhslst'] = df['rhs'].apply(rhs_clean)

	df = df[[0, 1, 'rhslst',3]]
	#~#	// ** //
	g = pcfg.Grammar('S')
	for (id, lhs, rhs, prob) in df.values:
		# print (id, lhs, rhs, prob)
		g.add_rule(pcfg.Rule(id, lhs, rhs, float(prob)))
	#
	fbname = os.path.basename(in_fname)
	print fbname.split("_")[1]
	
	#	try:
	#		num_nodes = int(fbname.split("_")[1].strip(".tsv"))
	#	except Exception, e:
	#		print str(e)
	from tdec.load_edgelist_from_dataframe import Pandas_DataFrame_From_Edgelist
	eldf = Pandas_DataFrame_From_Edgelist([orig_el])[0]
	G = nx.from_pandas_dataframe(eldf, source='src', target='trg')
	G.name = [x for x in os.path.basename(orig_el).split(".") if len(x)>3][0]

	num_nodes = G.number_of_nodes()

	print "Starting max size", num_nodes
	g.set_max_size(num_nodes)
	print "Done with max size"

	hStars = []
	for i in range(0, 10):
		try:
			rule_list = g.sample(num_nodes)
		except Exception, e:
			print "!!!", str(e)
		#	traceback.print_exc()
			continue
		hstar = phrg.grow(rule_list, g)[0]
		print "~^~"*10,hstar.number_of_nodes(), hstar.number_of_edges()
		hStars.append(hstar)

	if orig_el is not None:
		import tdec.net_metrics as metrics
		metricx = ['degree','clust', 'hop', 'gcd']
#		G = nx.read_edgelist(orig_el)
#		G.name = os.path.basename(orig_el).split(".")[0]
		metrics.network_properties([G], metricx, hStars, name=G.name, out_tsv=True)



#	lines = [
#		("r0.0",	"A,B",	['A,B:N', 'A:N'],	0.857142857143),
#		("r0.1",	"A,B",	['A:N', 'B:N'],	0.142857142857),
#		("r1.0",	"S",	['0,1:T', '0,1:N', '0:N'],	1.0),
#		("r2.0",	"A",	['0,A:T', 'A:N', 'A:N'],	0.0666666666667),
#		("r2.1",	"A",	['A:T'],	0.0666666666667),
#		("r2.2",	"A",	['0,A:T', '0:N', '0:N'],	0.0666666666667),
#		("r2.3",	"A",	['0,A:T', '0:N'],	0.133333333333),
#		("r2.4",	"A",	['0,A:T'],	0.666666666667)]
#	g = pcfg.Grammar('S')
#	for (id, lhs, rhs, prob) in lines:
#		print (id, lhs, rhs, prob)
#		g.add_rule(pcfg.Rule(id, lhs, rhs, float(prob)))
#
#	num_nodes = int(in_fname.split("_")[-1].rstrip(".tsv"))
#	print "	","Starting max size", 'n=', num_nodes
#	g.set_max_size(num_nodes)
#	print "	","Done with max size"
#
#	Hstars = []
#	num_samples = 10
#	for i in range(0, num_samples):
#		try:
#			rule_list = g.sample(num_nodes)
#		except Exception, e:
#			print str(e)
#			traceback.print_exc()
#			continue #sys.exit(1)
#		
#		hstar = phrg.grow(rule_list, g)[0]
#		Hstars.append(hstar)
#	print "	", 'Save BA production rules'

def control_test():
	'''ba_cntrl_128.tsv holds baseline production rules for
		the BA random grpah of 128 nodes
	'''
	graph_gen_isom_interxn(in_fname="Results/ba_cntrl_128.tsv", orig_el="datasets/ba_128_375_2.tsv")


#~# Main
if __name__ == '__main__':
	if 0:
		bsnm_fname,last_inrange,elst_fnames = ba_control_hrg([math.pow(2,x) for x in range(5,8,1)])
		print " Basename:",bsnm_fname
		print "			|V|:",last_inrange
		print "Edgelists:", elst_fnames

		for el in elst_fnames:
			synth_prs ="_".join(bsnm_fname.split("_")[:2])
			print synth_prs
			synth_prs ="Results/"+bsnm_fname+".tsv"
			graph_gen_isom_interxn(in_fname=synth_prs)
			print synth_prs, el

	if 0: control_test()
	'''import sys
	if len(sys.argv) < 2:
		print "provide an isom_interxn.tsv file"
		sys.exit(1)
	
	is_in_fname = sys.argv[1]
	orig = sys.argv[2]
	print is_in_fname
	print orig
	'''
	is_in_files = glob("Results/synthG*isom_interxn.tsv")
	num_n_lst   = [os.path.basename(f).split("_")[1] for f in is_in_files]
	v_lst       = [int(v) for v in num_n_lst]	
	
	for f in is_in_files:
		num_n   = os.path.basename(f).split("_")[1]
		v = int(num_n)
		oirg = nx.barabasi_albert_graph(v, 3)
		orig = "datasets/bar_alb_" + str(v+1) + "_exp3.tsv"

		print f,
		print orig
		if os.path.exists(orig):
			graph_gen_isom_interxn(in_fname=f, orig_el= orig) 
		else:
			print orig + " file does not exists"
	
	if 0:
		graph_gen_isom_interxn(
					in_fname=	"Results/synthG_31_87_isom_interxn.tsv",
					orig_el=	"datasets/bar_alb_32_exp3.tsv")

# graph_gen_isom_interxn(in_fname="Results/ba_cntrl_128.tsv")

#	if 0: graph_gen_isom_interxn("synthG_127_3072.mmd_prules.bz2")
#	for f in glob('Results/ba_cntrl*.tsv'):
#		print f
##		graph_gen_isom_interxn(f)
#		break

