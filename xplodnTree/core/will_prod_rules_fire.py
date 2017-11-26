import os
import pickle
import pandas as pd
import probabilistic_cfg as pcfg
from utils import Info
from PHRG import grow

def probe_stacked_prs_likelihood_tofire(df, fname="", nbr_nodes=0):
	Info("probe stacked prs likelihood tofire")
	g = pcfg.Grammar('S')
	df = df[['rnbr', 'lhs', 'rhs', 'prob']] # ToDo: need to drop the gname column
	for (id, lhs, rhs, prob) in df.values.tolist(): # 21Nov17
		g.add_rule(pcfg.Rule(id, lhs, rhs, float(prob)))
	num_nodes = int(nbr_nodes)
	g.set_max_size(num_nodes)
	try:
		g.set_max_size(num_nodes)
	except Exception, e: # print "Done with max size"
		print "\t:", e
		# return False
		os._exit(1)
	finally:
		bsn = os.path.basename(fname)
		fire_bool = True
	'''Added this new pice on 21Nov17'''
	Hstars = []
	num_runs = 20
	for i in range(0, num_runs):
		rule_list = g.sample(num_nodes)
		hstar = grow(rule_list, g)[0] # fixed-size graph generation
		Hstars.append(hstar)

	return (fire_bool, Hstars)

def will_prod_rules_fire(prs_files_lst, nbr_nodes):
	if not len(prs_files_lst): return
	ret_val = []

	for fname in prs_files_lst:
		# Read the subset of prod rules
		df = pd.read_csv(fname, header=None, sep="\t", dtype={0: str, 1: list, 2: list, 3: float})
		g = pcfg.Grammar('S')
		from td_isom_jaccard_sim import listify_rhs
		for (id, lhs, rhs, prob) in df.values:
			rhs = listify_rhs(rhs)
			g.add_rule(pcfg.Rule(id, lhs, rhs, float(prob)))

		num_nodes = nbr_nodes[0]
		# 		print "Starting max size", 'n=', num_nodes[0], type(num_nodes)


	return ret_val

# Hstars = []
#
# ofname   = "FakeGraphs/"+ origG.name+ "_isom_ntrxn.shl"
# database = shelve.open(ofname)
#
# num_samples = 20 #
# print '~' * 40
# for i in range(0, num_samples):
# 	rule_list = g.sample(num_nodes)
# 	hstar		 = phrg.grow(rule_list, g)[0]
# 	Hstars.append(hstar)
# 	print hstar.number_of_nodes(), hstar.number_of_edges()
#
# print '-' * 40
# database['hstars'] = Hstars
# database.close()
#
# exit()
