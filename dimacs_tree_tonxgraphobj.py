import networkx as nx
import sys
import os

def dimacs_tree_to_nx_graph(fname):
	with open (fname, 'r') as f:
		lines = f.readlines()

	nodes = [line.rstrip('\r\n') for line in lines if line.startswith("B")]
	edges = [line.rstrip('\r\n') for line in lines if line.startswith("e")]
	
	nodes = [n for n in nodes]
	vid   = [int(l.split()[1]) for l in nodes]
	clqsz = [l.split()[2] for l in nodes]
	edges = [x.split()[1:] for x in edges]
	edges = [(int(x),int(y)) for x,y in edges]
	
	G = nx.Graph()
	for i in range(0,len(vid)):
		G.add_node(vid[i], clqsize=clqsz[i])
	G.add_edges_from(edges)
	
	print G.nodes()
	print G.edges()
	print nx.info(G)
	return G
	
'''
if __name__ == '__main__':
	if len(sys.argv) < 2:
		sys.exit('Usage: %s dimacs_input_tree' % sys.argv[0])

	if not os.path.exists(sys.argv[1]):
		sys.exit('ERROR: File %s was not found!' % sys.argv[1])
	try:
		dimacs_tree_to_nx_graph(sys.argv[1])
	except Exception, e:
		os._exit(1)
	os._exit(0)
'''
