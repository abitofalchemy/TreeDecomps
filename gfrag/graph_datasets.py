#!/usr/bin/env python
import os
from pandas import DataFrame, read_csv
import networkx as nx

def read_dataset(filename):
	fname, fext = os.path.splitext(filename)
	print(fname, fext)
	# exit()
	if fext == ".tsv":
		data_lst = read_csv(filename, delimiter="\s+", header=None)
	elif fext == ".csv":
		data_lst = read_csv(filename, delimiter=",", header=None)
	elif fext == ".bz2":
		from load_tarbz2_datasets import load_tarbz2_dataset
		data_bz2 = load_tarbz2_dataset(filename)
		return data_bz2
	else:
		raw_data = list(filter(lambda l: len(l)>0, open(filename).readlines()))
		x_dat = []
		for x in raw_data:
			x = x.rstrip("\r\n")
			x_dat.append(x.split())

		data_lst = DataFrame(x_dat)
	if (data_lst.shape[1]==3):
		data_lst.columns  = ('src','trg','e_props')
		graph = nx.from_pandas_edgelist(data_lst, 'src', 'trg', edge_attr=['e_props'])
	else:
		data_lst.columns  = ('src','trg')
		graph = nx.from_pandas_edgelist(data_lst, 'src', 'trg')

	return graph

# test
# graph = read_dataset("../datasets/as20000102_1.tsv")
# print(nx.info(graph))
