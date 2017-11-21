import multiprocessing as mp
import os
import re
from collections import defaultdict

import networkx as nx

import load_edgelist_from_dataframe as dfe

# Log Info
# [ ] todo - finish edgelist_basic_info so it writes a file in json form (key, value) of 
#			a given graph {gn: (V,E)}

results= defaultdict(tuple)

def graph_name(fname):
	gnames= [x for x in os.path.basename(fname).split('.') if len(x) >3][0]
	if len(gnames):
		return gnames
	else:
		return gnames[0]

def collect_results(result):
	gn,v,e = result
	results[gn] = (v,e)
		
def edgelist_basic_info(fn_lst):
	# if the file exists ... read it and return as dict
	if os.path.exists('.graph_base_info.json'):
		print " ", "base info file exists ... "
		g_base_info_dict = load_graph_base_info()
	else:
		resuts=defaultdict(tuple)
		p = mp.Pool(processes=2)
		for f in fn_lst:
			res_d = net_info(f,)
			collect_results(res_d)
		# p.close()
		# p.join()
	
		write_graph_base_info(results)
		g_base_info_dict = results
	return g_base_info_dict 
	
	
def net_info(edgelist_fname):
	dfs = dfe.Pandas_DataFrame_From_Edgelist([edgelist_fname])
	df = dfs[0]

	try:
		g = nx.from_pandas_dataframe(df, 'src', 'trg', edge_attr=['ts'])
	except  Exception, e:
		g = nx.from_pandas_dataframe(df, 'src', 'trg')

	if df.empty:
		g = nx.read_edgelist(edgelist_fname,comments="%")
	gn = graph_name(edgelist_fname)
	
	return (gn, g.number_of_nodes(), g.number_of_edges())	


def write_graph_base_info(ddict):
	import json
	try:
		with open('.graph_base_info.json', 'w') as fp:
			json.dump(ddict, fp)
		return True
	except IOError:
		print "unable to write to disk"
		return False

def load_graph_base_info():
	import json
	try:
		with open('.graph_base_info.json', 'r') as fp:
			data = json.load(fp)
	except IOError:
		print "Failed to read file"
		exit()
	return data

def Info(_str):
	print "  >> {}".format(_str)

def listify_rhs(rhs_rule):
	rhs_clean= [f[1:-1] for f in re.findall("'.+?'", rhs_rule)]
	return rhs_clean
