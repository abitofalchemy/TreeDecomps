import pandas as pd
import glob
import os
from itertools import combinations
from collections import defaultdict,Counter
import networkx as nx
import re

import pprint as pp
import numpy as np

DBG = False

def rhs_tomultigraph(rhs_clean):
	'''
	Parse the RHS of each rule into a graph fragment
	:param x:
	:return:
	'''
	import re
	from itertools import combinations
	import networkx as nx
	if isinstance(rhs_clean, str): 
		rhs_clean= [f[1:-1] for f in re.findall("'.+?'", rhs_clean)]

	# rhs_clean = [f[1:-1] for f in re.findall("[^()]+", x)]
	G1 = nx.MultiGraph()
	for he in rhs_clean:
		epair,ewt = he.split(':')
		if ewt is "T":
			if len(epair.split(",")) == 1:	[G1.add_node(epair, label=ewt)]
			else: [G1.add_edge(epair.split(",")[0], epair.split(",")[1], label=ewt)]
		elif ewt is "N":
			if len(epair.split(",")) == 1:	[G1.add_node(epair, label=ewt)]
			else: [G1.add_edges_from(list(combinations(epair.split(","), 2)),label=ewt )]

	return G1

def rhs2multigraph(x):
	'''
	Parse the RHS of each rule into a graph fragment
	:param x:
	:return:
	'''
	import re
	from itertools import combinations
	import networkx as nx

	rhs_clean=[f[1:-1] for f in re.findall("'.+?'", x)]
	# rhs_clean = [f[1:-1] for f in re.findall("[^()]+", x)]
	G1 = nx.MultiGraph()
	for he in rhs_clean:
		epair,ewt = he.split(':')
		if ewt is "T":
			if len(epair.split(",")) == 1:	[G1.add_node(epair, label=ewt)]
			else: [G1.add_edge(epair.split(",")[0], epair.split(",")[1], label=ewt)]
		elif ewt is "N":
			if len(epair.split(",")) == 1:	[G1.add_node(epair, label=ewt)]
			else: [G1.add_edges_from(list(combinations(epair.split(","), 2)),label=ewt )]

	return G1

def long_string_split(x):
	'''
	Parse the RHS of each rule into a graph fragment
	:param x:
	:return:
	'''
	import re
	import networkx as nx
	from itertools import combinations

	rhs_clean= [f[1:-1] for f in re.findall("'.+?'", x)]
	G1 = nx.MultiGraph()
	for he in rhs_clean:
		epair,ewt = he.split(':')
		if ewt is "T":
			if len(epair.split(",")) == 1:	[G1.add_node(epair, label=ewt)]
			else: [G1.add_edge(epair.split(",")[0], epair.split(",")[1], label=ewt)]
		elif ewt is "N":
			if len(epair.split(",")) == 1:	[G1.add_node(epair, label=ewt)]
			else: [G1.add_edges_from(list(combinations(epair.split(","), 2)),label=ewt )]

	return G1

def label_match(x, y):
	return x[0]['label'] == y[0]['label']

def string_to_list(x):
	'''
	Parse the RHS of each rule into a list
	:param x:
	:return:
	'''
	import re
	# rhs_clean= [f[1:-1] for f in re.findall("'.+?'", x)]
	rhs_clean = [f[1:-1] for f in re.findall("[^()]+", x)]
	return rhs_clean

def listify_rhs(rhs_rule):
	rhs_clean= [f[1:-1] for f in re.findall("'.+?'", rhs_rule)]
	return rhs_clean

def isomorphic_frags_from_prod_rules(lst_files):
	'''
	isomorphic_frags_from_prod_rules
	:param file_paths: input list of files (with full paths)
	:return:
	'''
	for p in [",".join(map(str, comb)) for comb in combinations(lst_files, 2)]:
		p = p.split(',')
		dtyps = {'rnbr': 'str', 'lhs': 'str', 'rhs': 'str', 'pr': 'float'}
		df1 = pd.read_csv(p[0], index_col=0, compression='bz2', dtype=dtyps)
		df1.columns = ['rnbr', 'lhs', 'rhs', 'pr']
		df2 = pd.read_csv(p[1], index_col=0, compression='bz2', dtype=dtyps)
		df2.columns = ['rnbr', 'lhs', 'rhs', 'pr']

		print os.path.basename(p[0]).split('.')[0:2], os.path.basename(p[1]).split('.')[1], '\t'
		# glist1 = df1.apply(lambda x: rhs2multigraph(x['rhs'].strip('[]')), axis=1).values
		# glist2 = df2.apply(lambda x: rhs2multigraph(x['rhs'].strip('[]')), axis=1).values
		#
		glist1 = df1['rnbr'].values
		glist2 = df2['rnbr'].values

		cntr = 0
		for i in xrange(len(glist1)):
			for j in range (i,len(glist2),1):
				if (df1[df1['rnbr'] == glist1[i]].lhs.values == df2[df2['rnbr'] == glist2[j]].lhs.values):
					if DBG: print df1[df1['rnbr']== glist1[i]].lhs.values, df2[df2['rnbr'] == glist2[j]].lhs.values
					if DBG: print i,j
					G1 = rhs2multigraph(df1[df1['rnbr'] == glist1[i]].rhs.values[0].strip('[]'))
					G2 = rhs2multigraph(df2[df2['rnbr'] == glist2[j]].rhs.values[0].strip('[]'))
					if nx.is_isomorphic(G1, G2, edge_match=label_match):
						cntr +=1

						if not DBG: print '	>>>> isomorphic <<<<<',
						if not DBG: print '	{},{}:'.format(i,j), df1[df1['rnbr']== glist1[i]].rnbr.values, df2[df2['rnbr'] == glist2[j]].rnbr.values

		#		 print glist1[i], glist2[j]
		#		 if nx.is_isomorphic(t[0], t[1], edge_match=label_match):
		#			 cntr += 1
		#
		#	 print
		#

		# # Differences = {tuple(i) for i in glist1} & {tuple(i) for i in glist2}
		# # print len(Differences)
		print cntr

def isomorph_infile_reduce(dfx):
	seen_rules = defaultdict(list)
	cntr = 0
	for index, r in dfx.iterrows():
		print r['rnbr'],
		if r['lhs'] not in seen_rules.keys():
			seen_rules[r['lhs']].append(r['rnbr'])
			if DBG: print "+"
			cntr += 1
		else:	# lhs already seen
			rhs1 = r['rhs']
			print seen_rules[r['lhs']][0]
			rhs2 = dfx[dfx['rnbr'] == seen_rules[r['lhs']][0]]['rhs'].values[0]
			G1 = rhs_tomultigraph(rhs1)
			G2 = rhs_tomultigraph(rhs2)

			if nx.is_isomorphic(G1, G2, edge_match=label_match):
				# print ' ',r['rnbr'], r['rhs'], '::', df1[df1['rnbr'] == seen_rules[r['lhs']][0]]['rhs'].values
				if DBG: print ' <-curr', seen_rules[r['lhs']][0], ':', dfx[dfx['rnbr'] == seen_rules[r['lhs']][0]]['pr'].values

				sum_pr = dfx[dfx['rnbr'] == seen_rules[r['lhs']][0]]['pr'].values[0] + r['pr']
				dfx.set_value(dfx[dfx['rnbr'] == seen_rules[r['lhs']][0]].index.values[0], 'pr', sum_pr)
				dfx = dfx.drop(dfx[dfx.rnbr == r['rnbr']].index)
			else:
				seen_rules[r['lhs']].append(r['rnbr'])
				if DBG: print "+"
				cntr += 1
				# exit()
				# print seen_rules.values()
	return dfx # reduced dataframe

# def isomorph_intersection_2dfstacked(dfx):
#	 seen_rules = defaultdict(list)
#	 overlap_pairs = defaultdict(list)
#	 dfx['iso'] = False
#	 dfx = dfx.reset_index()
#	 # print dfx.to_string()
#	 df_interx = pd.DataFrame()#(columns = dfx.columns)
#	 for x,y in combinations(dfx.index.values, 2): # for every pair
#		 lhs1 = dfx.loc[x]['lhs']
#		 lhs2 = dfx.loc[y]['lhs']
#		 if lhs1 == lhs2: # where the lhs are equal
#			 G1 = rhs_tomultigraph(dfx.loc[x]['rhs'])
#			 G2 = rhs_tomultigraph(dfx.loc[y]['rhs'])
#			 # test if their RHS are isomorphic
#			 if nx.is_isomorphic(G1, G2, edge_match=label_match):
#				 # print x, y, lhs1, lhs2	# dfx.loc[x][['rnbr','lhs']].values
#				 # print dfx.loc[x], type(dfx.loc[x])#,dfx.loc[y][['index','rnbr']].values
#				 row = dfx.loc[x]
#				 # print dfx.loc[x]['pr'] + dfx.loc[y]['pr']
#				 row.set_value(label='pr', value=dfx.loc[x]['pr']+dfx.loc[y]['pr'])
#				 if DBG: print "{} + {} = {}".format(dfx.loc[x]['pr'], dfx.loc[y]['pr'], dfx.loc[x]['pr']+dfx.loc[y]['pr'])
#				 df_interx = df_interx.append(row)
#				 dfx.set_value(x, 'iso', True)
#				 dfx.set_value(y, 'iso', True)
#				 # overlap_pairs[x].append(row)

def isomorph_intersection_2dfstacked(dfx):
	print "\t>> input DF shape", dfx.shape
	seen_rules = defaultdict(list)
	overlap_pairs = defaultdict(list)
	dfx['iso'] = False
	dfx = dfx.reset_index()
	# print dfx.to_string()
	df_interx = pd.DataFrame()#(columns = dfx.columns)
	for x,y in combinations(dfx.index.values, 2): # for every pair
		lhs1 = dfx.loc[x]['lhs']
		lhs2 = dfx.loc[y]['lhs']
		if lhs1 == lhs2: # where the lhs are equal
			G1 = rhs_tomultigraph(dfx.loc[x]['rhs'])
			G2 = rhs_tomultigraph(dfx.loc[y]['rhs'])
			# test if their RHS are isomorphic
#			if nx.is_isomorphic(G1, G2, edge_match=label_match):
			if nx.faster_could_be_isomorphic(G1, G2):
				if (dfx.loc[y]['iso']==False):
					seen_rules[(x,dfx.loc[x]['lhs'])].append(y)
					dfx.set_value(x, 'iso', True)
					dfx.set_value(y, 'iso', True)
					# df_interx = df_interx.append(dfx.loc[x])
					# overlap_pairs[x].append(row)


	# print "	", df_interx.shape
	df_union = dfx[ dfx['iso'] == True ]
	# print "	", df_union.shape
	df_interx = []

	lhs_cnt = Counter()
	for x,y in seen_rules.keys():
		lhs_cnt[y] += 1
	# print lhs_cnt
	lhs_counts_dict = {}
	for k in lhs_cnt.keys(): # compute totals for each lhs
		# print k
		rhs_els = 0
		for k2, v2 in seen_rules.iteritems():
			if k2[1] in k: # weird that I have to use "in"
				rhs_els += 1 + len(v2)
		# print "	", rhs_els
		lhs_counts_dict[k]=rhs_els

	for k,v in seen_rules.iteritems():
		# print "#"
		# rprob = dfx.loc[k[0]]['pr']
		cntr = 1 + len(v)
		df_union.set_value(k[0], 'pr', cntr/float(lhs_counts_dict[k[1]]))
		# df_union.set_value(k[0], 'rnbr', "r%d.%d" % (id,sid))
		df_interx.append(df_union.loc[k[0]].values)

	df_interx = pd.DataFrame(df_interx)
	# df_interx = df_interx.sort_values(by=[1])

	# print df_interx.to_string()
	gb = df_interx.groupby([2]).groups
	id = 0
	for k,v in gb.iteritems():
		sid=0
		for w in v:
			# print "r%d.%d" % (id,sid)
			df_interx.set_value(w, [1], "r%d.%d" % (id,sid))
			sid += 1
		id += 1
	# print df_interx.to_string()
	print "\tdf_interx", df_interx.shape, "new shape"
	# print dfx[ dfx['iso'] == True ] # this is the union of the rules
	# print df_interx# this is the intersetion with modified probs
	# df_interx = df_interx.reset_index(drop=True)
	# # print df_interx.head()
	# print "isomorph_infile_reduce", df_interx.shape
	# # df = isomorph_infile_reduce(df_interx)
	# df_interx[['rnbr', 'pr', 'iso', 'lhs', 'rhs']].to_csv("Results/intersection_rules.tsv", sep="\t", index=False)
	# gb = dfx.groupby(['iso']).groups
	# import pprint as pp
	# pp.pprint(gb)

	return df_union, df_interx


	# cntr = 0
	# for index, r in dfx.iterrows():
	#	 print r['rnbr'],
	#	 if r['lhs'] not in seen_rules.keys():
	#		 seen_rules[r['lhs']].append(r['rnbr'])
	#		 print "+"
	#		 cntr += 1
	#	 else:	# lhs already seen
	#		 # check current rhs if lhs matches to something already seen & check for an isomorphic match
	#		 rhs1 = r['rhs']
	#		 rhs2 = dfx[dfx['rnbr'] == seen_rules[r['lhs']][0]]['rhs'].values[0]
	#		 G1 = rhs_tomultigraph(rhs1)
	#		 G2 = rhs_tomultigraph(rhs2)
	#		 if nx.is_isomorphic(G1, G2, edge_match=label_match):
	#			 # print ' ',r['rnbr'], r['rhs'], '::', df1[df1['rnbr'] == seen_rules[r['lhs']][0]]['rhs'].values
	#			 print ' <-curr', seen_rules[r['lhs']][0], ':', dfx[dfx['rnbr'] == seen_rules[r['lhs']][0]]['pr'].values
	#
	#			 sum_pr = dfx[dfx['rnbr'] == seen_rules[r['lhs']][0]]['pr'].values[0] + r['pr']
	#			 dfx.set_value(dfx[dfx['rnbr'] == seen_rules[r['lhs']][0]].index.values[0], 'pr', sum_pr)
	#			 # dfx = dfx.drop(dfx[dfx.rnbr == r['rnbr']].index)
	#		 else:
	#			 seen_rules[r['lhs']].append(r['rnbr'])
	#			 print "+"
	#			 cntr += 1
	#
	# import json
	# print json.dumps(seen_rules, indent=4, sort_keys=True)
	# return dfx # reduced dataframe

def isomorph_intersection(dfa, dfb):
	print '-'*20
	print 'Rules Intra'
	seen_rules = defaultdict(list)
	cntr	 = 0
	for index, r in dfa.iterrows():
		print r['rnbr'],
		if	r['lhs'] not in seen_rules.keys():
			seen_rules[r['lhs']].append(r['rnbr'])
			print "+"
			cntr += 1
		else: # lhs already seen
			# print df1[df1['rnbr']==seen_rules[r['lhs']][0]]['rhs'].values
			# check the current rhs if the lhs matches to something already seen and check for an isomorphic match
			rhs1 = r['rhs']
			rhs2 = dfa[dfa['rnbr']==seen_rules[r['lhs']][0]]['rhs'].values[0]
			G1 = rhs_tomultigraph(rhs1)
			G2 = rhs_tomultigraph(rhs2)
			if nx.is_isomorphic(G1, G2, edge_match=label_match):
				# print ' ',r['rnbr'], r['rhs'], '::', df1[df1['rnbr'] == seen_rules[r['lhs']][0]]['rhs'].values
				print ' <-curr', seen_rules[r['lhs']][0],':', dfa[dfa['rnbr'] == seen_rules[r['lhs']][0]]['pr'].values

				sum_pr = dfa[dfa['rnbr'] == seen_rules[r['lhs']][0]]['pr'].values[0] +r['pr']
				dfa.set_value(dfa[dfa['rnbr'] == seen_rules[r['lhs']][0]].index.values[0], 'pr',	sum_pr)
				dfa = dfa.drop(dfa[dfa.rnbr ==r['rnbr']].index)
			else:
				seen_rules[r['lhs']].append(r['rnbr'])
				print "+"
				cntr += 1
				# exit()
		# print seen_rules.values()

	import json
	print json.dumps(seen_rules,indent=4, sort_keys=True)
	# print
	dfa[['rnbr','pr']].to_csv("Results/tmp.tsv", sep="\t", index=False) # "{}, {}".format(y[1]['rnbr'],y[1]['pr']), axis=1)
	# dfa = shrink_dataframe_by(dfa, seen_rules)

def isomorph_check_production_rules_pair(df1, df2):
	mdf = pd.concat([df1,df2])
	# print mdf.shape
	# print mdf.head()
	# print mdf.tail()
	mdf = isomorph_intersection_2dfstacked(mdf)
	# print mdf.shape
