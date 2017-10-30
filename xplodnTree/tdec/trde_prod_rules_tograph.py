#!/usr/bin/env python
__author__ = 'saguinag' + '@' + 'nd.edu'
__version__ = "0.1.0"

##
## fname:...... trde_prod_rules_tograph.py
## description: from a reduced set of production rules, generate a graph
##

## TODO: some todo list

## VersionLog:

import csv

import networkx as nx

import trde_exact_phrg as xphrg
import net_metrics as metrics

in_file_path = "sandbox/out.mcsm.mod.csv"
rules = list(csv.reader(open(in_file_path, 'rb'), delimiter='\t'))

# g = pcfg.Grammar('S')
# for (id, lhs, rhs, prob) in rules:
#   prob = float(prob)
#   g.add_rule(pcfg.Rule(id, lhs, rhs, prob))
#   print type(prob)

  # print "{}, {}, {}, {}".format(id,lhs,rhs,prob)


G=nx.karate_club_graph()
print G.number_of_nodes()
# # Synthetic Graphs
hStars = xphrg.grow_exact_size_hrg_graphs_from_prod_rules(rules, \
                                                          "zachary.mcsm.mod",\
                                                          G.number_of_nodes(), 50)
metricx = ['degree', 'hops', 'clust', 'assort', 'kcore', 'eigen', 'gcd']
metrics.network_properties([G], metricx, hStars, name="zachary.mcsm.mod", out_tsv=True)

