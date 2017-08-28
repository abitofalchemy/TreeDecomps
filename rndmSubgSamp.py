import pandas as pd
import networkx as nx
import traceback
import sys

from explodingTree import graph_name
from tdec.load_edgelist_from_dataframe import Pandas_DataFrame_From_Edgelist

def readEdglstGraph(fname):
	df = Pandas_DataFrame_From_Edgelist([fname])[0]
	G	= nx.from_pandas_dataframe(df, source='src',target='trg')
	Gc = max(nx.connected_component_subgraphs(G), key=len)
	gname = graph_name(fname)


def main():
	print sys.argv

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
		
