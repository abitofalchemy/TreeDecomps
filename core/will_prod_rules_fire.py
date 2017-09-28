import core.tstprodrules as tprs
import pandas as pd
import probabilistic_cfg as pcfg
import os
from collections import defaultdict
from utils import Info


def probe_stacked_prs_likelihood_tofire(df, fname="", nbr_nodes=0):
	Info("{}, {}".format(df.shape, nbr_nodes))
	print df.head()
	g = pcfg.Grammar('S')
	for (id, lhs, rhs, prob) in df.values:
		g.add_rule(pcfg.Rule(id, lhs, rhs, float(prob)))

	num_nodes = nbr_nodes[0]
	try:
		g.set_max_size(num_nodes)
	except Exception, e: # print "Done with max size"
		print "\t", e
		return False
	finally:
		bsn = os.path.basename(fname)
		return True

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
