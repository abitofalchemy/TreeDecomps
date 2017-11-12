import os
import networkx as nx
import sys

# --< BEGIN >--
ds_dir =      "./datasets/"   # Directory path containing your network datasets of interest
results_dir = "./Results/" # Replace this with the directory you want to save the results
tmp_dir =     "/tmp"          # Replace this with the directory you want to save the SFrame to

if len(sys.argv) != 3:
	print ("\nInfidemo.py: Missing arguments...\n")
	print ("\tUsage: python infidemo.py fName(Path) recurrence#(int)")
	print ("\tTry again.")
	exit()
nFname = str(sys.argv[1])
rnbr = int(sys.argv[2])

# NB: here I use hypertext dataset b/c it is small
# nFname = ds_dir+"out.sociopatterns-hypertext" # get file from url above
# nFname = "/Users/sal.aguinaga/Boltzmann/TreeDecomps/datasets/ucidata-gama/out.ucidata-gama"
# nFname = "/Users/sal.aguinaga/Boltzmann/InfinityMirrorGCT/datasets/as20000102/out.as20000102"
gname  = [x for x in os.path.basename(nFname).split('.') if len(x)>3][0]

# tweak the call to read_edgelist depending on your dataset
# graph  = nx.read_edgelist(nFname, comments='%',  nodetype=int, data=(('weight',int),('ts',int)))
graph  = nx.read_edgelist(nFname, comments='%',  nodetype=int, data=False)

graph.name=gname
# graph = nx.karate_club_graph()
# graph.name="kchop"
print nx.info(graph)


import core.PHRG as phrg
import core.probabilistic_cfg as pcfg

G = graph
Hstars = [] # synthetic (stochastically generate) graphs using the graph grammars
ProdRulesKth =[]
# Note that therem might not be convergence, b/c the graph may degenerate early
for j in range(0,10): # nbr of times to do Inf. Mirr. tst
	for k in range(1,rnbr+1): # nbr of times to feedback the resulting graph
		prdrls = {}
		prod_rules = phrg.probabilistic_hrg_deriving_prod_rules(G)
		# print len(prod_rules)
		# initialize the Grammar g
		g = pcfg.Grammar('S')

		for (id, lhs, rhs, prob) in prod_rules:
			g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

		num_nodes = G.number_of_nodes()
		g.set_max_size(num_nodes)

		print "Done initializing the grammar data-structure"
		# Generate a synthetic graph using HRGs
		try:
			rule_list = g.sample(num_nodes)
		except Exception, e:
			print str(e)
			rule_list = g.sample(num_nodes)
			break
		hstar = phrg.grow(rule_list, g)[0]
		G = hstar # feed back the newly created graph
	# store the last synth graph & restart
	Hstars.append(hstar) #

# Warning
# If the rules are not able to generate a graph rerun this step or add a try/catch to retry or continue
print ("~..~"*10)
nx.write_gpickle(Hstars[0],  "Results/inf_mir_{}_{}_{}.p".format(graph.name,k,j))
if os.path.exists("Results/inf_mir_{}_{}_{}.p".format(graph.name,k,j)):
	print ("Results/inf_mir_{}_{}_{}.p written".format(graph.name,k,j))
