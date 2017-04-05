#!/usr/bin/env python
__author__ = 'saguinag' + '@' + 'nd.edu'
__version__ = "0.1.0"

import os
import pprint as pp
import re
import networkx as nx
import argparse
import graph_sampler as gs
import math
import tree_decomposition as td

# prod_rules = {}
DEBUG =  False

def graph_checks (G):
  ## Target number of nodes
  global num_nodes
  num_nodes = G.number_of_nodes()

  if not nx.is_connected(G):
    if DEBUG: print "Graph must be connected";
    os._exit(1)

  if G.number_of_selfloops() > 0:
    if DEBUG: print "Graph must be not contain self-loops";
    os._exit(1)


def matcher (lhs, N):
  if lhs == "S":
    return [("S", "S")]
  m = []
  for x in N:
    if len(x) == lhs.count(",") + 1:
      i = 0
      for y in lhs.split(","):
        m.append((x[i], y))
        i += 1
      return m


def binarize (tree):
  (node, children) = tree
  children = [binarize(child) for child in children]
  if len(children) <= 2:
    return (node, children)
  else:
    # Just make copies of node.
    # This is the simplest way to do it, but it might be better to trim unnecessary members from each copy.
    # The order that the children is visited is arbitrary.
    binarized = (node, children[:2])
    for child in children[2:]:
      binarized = (node, [binarized, child])
    return binarized


def grow (rule_list, grammar, diam=0):
  D = list()
  newD = diam
  H = list()
  N = list()
  N.append(["S"])  # starting node
  ttt = 0
  # pick non terminal
  num = 0
  for r in rule_list:
    rule = grammar.by_id[r][0]
    lhs_match = matcher(rule.lhs, N)
    e = []  # edge list
    match = []
    for tup in lhs_match:
      match.append(tup[0])
      e.append(tup[1])
    lhs_str = "(" + ",".join(str(x) for x in sorted(e)) + ")"

    new_idx = {}
    n_rhs = rule.rhs
    if 0: print lhs_str, "->", n_rhs
    for x in n_rhs:
      new_he = []
      he = x.split(":")[0]
      term_symb = x.split(":")[1]
      for y in he.split(","):
        if y.isdigit():  # y is internal node
          if y not in new_idx:
            new_idx[y] = num
            num += 1
            if diam > 0 and num >= newD and len(H) > 0:
              newD = newD + diam
              newG = nx.Graph()
              for e in H:
                if (len(e) == 1):
                  newG.add_node(e[0])
                else:
                  newG.add_edge(e[0], e[1])
                # D.append(metrics.bfs_eff_diam(newG, 20, 0.9))
          new_he.append(new_idx[y])
        else:  # y is external node
          for tup in lhs_match:  # which external node?
            if tup[1] == y:
              new_he.append(tup[0])
              break
      # prod = "(" + ",".join(str(x) for x in new_he) + ")"
      if term_symb == "N":
        N.append(sorted(new_he))
      elif term_symb == "T":
        H.append(new_he)  # new (h)yper(e)dge
      # print n_rhs, new_he
    match = sorted(match)
    N.remove(match)

  newG = nx.Graph()
  for e in H:
    if (len(e) == 1):
      newG.add_node(e[0])
    else:
      newG.add_edge(e[0], e[1])

  return newG, D


def probabilistic_hrg (G, num_samples=1, n=None):
  '''
  Args:
  ------------
	G: input graph (nx obj)
	num_samples:   (int) in the 'grow' process, this is number of
	               synthetic graphs to generate
	n: (int) num_nodes; number of nodes in the resulting graphs
	Returns: List of synthetic graphs (H^stars)
  '''
  graphletG = []

  if DEBUG: print G.number_of_nodes()
  if DEBUG: print G.number_of_edges()

  G.remove_edges_from(G.selfloop_edges())
  giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
  G = nx.subgraph(G, giant_nodes)

  if n is None:
    num_nodes = G.number_of_nodes()
  else:
    num_nodes = n

  if DEBUG: print G.number_of_nodes()
  if DEBUG: print G.number_of_edges()

  graph_checks(G)

  if DEBUG: print
  if DEBUG: print "--------------------"
  if DEBUG: print "-Tree Decomposition-"
  if DEBUG: print "--------------------"

  prod_rules = {}
  if num_nodes >= 500:
    for Gprime in gs.rwr_sample(G, 2, 300):
      T = td.quickbb(Gprime)
      root = list(T)[0]
      T = td.make_rooted(T, root)
      T = binarize(T)
      root = list(T)[0]
      root, children = T
      td.new_visit(T, G, prod_rules, TD)
  else:
    T = td.quickbb(G)
    root = list(T)[0]
    T = td.make_rooted(T, root)
    T = binarize(T)
    root = list(T)[0]
    root, children = T

    td.new_visit(T, G, prod_rules, TD)

  if DEBUG: print
  if DEBUG: print "--------------------"
  if DEBUG: print "- Production Rules -"
  if DEBUG: print "--------------------"

  for k in prod_rules.iterkeys():
    if DEBUG: print k
    s = 0
    for d in prod_rules[k]:
      s += prod_rules[k][d]
    for d in prod_rules[k]:
      prod_rules[k][d] = float(prod_rules[k][d]) / float(s)  # normailization step to create probs not counts.
      if DEBUG: print '\t -> ', d, prod_rules[k][d]

  rules = []
  id = 0
  for k, v in prod_rules.iteritems():
    sid = 0
    for x in prod_rules[k]:
      rhs = re.findall("[^()]+", x)
      rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
      if DEBUG: print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])
      sid += 1
    id += 1
  # print rules
  exit()

  g = pcfg.Grammar('S')
  for (id, lhs, rhs, prob) in rules:
    # print type(id), type(lhs), type(rhs), type(prob)
    g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

  if DEBUG: print "Starting max size"
  num_nodes = num_nodes
  num_samples = num_samples

  g.set_max_size(num_nodes)

  if DEBUG: print "Done with max size"

  Hstars = []

  for i in range(0, num_samples):
    rule_list = g.sample(num_nodes)
    # print rule_list
    hstar = grow(rule_list, g)[0]
    # print "H* nodes: " + str(hstar.number_of_nodes())
    # print "H* edges: " + str(hstar.number_of_edges())
    Hstars.append(hstar)

  return Hstars


# def probabilistic_hrg_deriving_prod_rules(G, num_samples=1, n=None):
# hrg_baseline.py
def probabilistic_hrg_deriving_prod_rules (G, K=1, n=None, gname=""):
  '''
	Rule extraction procedure

		'''
  if G is None: return

  G.remove_edges_from(G.selfloop_edges())
  giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
  G = nx.subgraph(G, giant_nodes)

  if n is None:
    num_nodes = G.number_of_nodes()
  else:
    num_nodes = n

  graph_checks(G)

  if DEBUG: print
  if DEBUG: print "--------------------"
  if DEBUG: print "-Tree Decomposition-"
  if DEBUG: print "--------------------"

  if num_nodes >= 500:
    for j,Gprime in enumerate(gs.rwr_sample(G, K, 200)):
      if gname is "":
        nx.write_edgelist(Gprime, '/tmp/sampled_subgraph_200_{}.tsv'.format(j), delimiter="\t", data=False)
      else:
        nx.write_edgelist(Gprime, '/tmp/{}{}.tsv'.format(gname, j), delimiter="\t", data=False)
        print "...  files written: /tmp/{}{}.tsv".format(gname, j)

  return

def get_parser():
  parser = argparse.ArgumentParser(description="sample a large graph and write the subgraph to file")
  parser.add_argument('-g', '--graph', required=True, help='input graph (edgelist)')
  parser.add_argument('--version', action='version', version=__version__)
  return parser

if __name__ == "__main__":
  parser = get_parser()
  args = vars(parser.parse_args())
  fname = args['graph']

  print "... ", fname
  gname = os.path.basename(fname)
  print "... ", gname
  G = nx.read_edgelist(fname, comments="%", data=False)
  G.name = gname

  num_nodes = G.number_of_nodes()
  print num_nodes
  k_subgraphs_nbr = int(math.ceil(.1*num_nodes/200))
  print k_subgraphs_nbr

  prod_rules = {}
  p_rules = probabilistic_hrg_deriving_prod_rules(G, K=k_subgraphs_nbr, gname=gname)

  exit(0)

