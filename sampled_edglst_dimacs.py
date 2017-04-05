__author__ = 'saguinag' + '@' + 'nd.edu'
__version__ = "0.1.0"

##
## fname
##

## TODO: some todo list
#   I do a mapping of nodes in one direction, but do I need to do it on
#   on the ohter way?

## VersionLog:

import argparse, traceback
import os, sys, time
import networkx as nx
from datetime import datetime
import numpy as np
import pandas as pd

def nx_edges_to_nddgo_graph (G, sampling=False):
  if sampling:

    edges = G.edges()
    edges = [(int(e[0]), int(e[1])) for e in edges]
    df = pd.DataFrame(edges)
    df.sort_values(by=[0], inplace=True)
    ofname = 'datasets/{}.dimacs'.format(G.name)

    with open(ofname, 'w') as f:
      f.write('c {}\n'.format(G.name))
      f.write('p edge\t{}\t{}\n'.format(G.number_of_nodes(), G.number_of_edges()))
      # for e in df.iterrows():
      output_edges = lambda x: f.write("e\t{}\t{}\n".format(x[0], x[1]))
      df.apply(output_edges, axis=1)
    # f.write("e\t{}\t{}\n".format(e[0]+1,e[1]+1))
    if os.path.exists(ofname): print 'Wrote: ./{}'.format(ofname)
  else:
    if G.name is None:
      t_str = datetime.now().strftime("%Y-%m-%d_%H_%M")
    edges = G.edges()
    edges = [(int(e[0]), int(e[1])) for e in edges]
    df = pd.DataFrame(edges)
    df.sort_values(by=[0], inplace=True)
    ofname = 'datasets/{}.dimacs'.format(G.name)

    with open(ofname, 'w') as f:
      f.write('c {}\n'.format(G.name))
      f.write('p edge\t{}\t{}\n'.format(G.number_of_nodes(), G.number_of_edges()))
      # for e in df.iterrows():
      output_edges = lambda x: f.write("e\t{}\t{}\n".format(x[0], x[1]))
      df.apply(output_edges, axis=1)
    # f.write("e\t{}\t{}\n".format(e[0]+1,e[1]+1))
    if os.path.exists(ofname): print 'Wrote: ./{}'.format(ofname)

def map_original_node_ids(f):
  print ">", f
  df = pd.read_csv(f,header=None, delimiter="\t")
  nf = pd.concat([df[0], df[1]])
  nf.drop_duplicates()
  c = 1
  mp = dict()
  for k in nf.values:
    if k not in mp.keys():
      mp[k] = c
      c += 1

  return mp
  # return dict(zip(nf.values, range(1,len(nf))))

def edgelist_to_nddgo_graph_dimacs(edge_list_fname):
  fname = edge_list_fname

  gname = os.path.basename(fname).split('.')[1]
  if len(gname) <= 3:
    gname = os.path.basename(fname).split('.')[0]

  print "... ", gname

  indata = np.loadtxt(fname, comments=("%"), delimiter="\t", dtype=None)
  max_vnode = np.max(indata)
  mapping_d = map_original_node_ids(fname)
  G = nx.read_edgelist(fname, comments="%", data=False, nodetype=int)
  if 0: G = nx.relabel_nodes(G1, mapping_d) # let's hold off on the mapping
  G.name = gname

  edges = G.edges()
  edges = [(int(e[0]), int(e[1])) for e in edges]
  df = pd.DataFrame(edges)
  df.sort_values(by=[0], inplace=True)
  ofname = 'datasets/{}.dimacs'.format(G.name)

  with open(ofname, 'w') as f:
    f.write('c {}\n'.format(G.name))
    f.write('p edge\t{}\t{}\n'.format(int(max_vnode), G.number_of_edges()))
    # for e in df.iterrows():
    output_edges = lambda x: f.write("e\t{}\t{}\n".format(x[0], x[1]))
    df.apply(output_edges, axis=1)
  # f.write("e\t{}\t{}\n".format(e[0]+1,e[1]+1))
  if os.path.exists(ofname): print 'Wrote: ./{}'.format(ofname)


def get_parser():
  parser = argparse.ArgumentParser(description='Convert edgelist to dimacs graph format')
  parser.add_argument('--edglst', required=True, help='input graph (edgelist) filename')
  parser.add_argument('--version', action='version', version=__version__)
  return parser

if __name__ == '__main__':
  parser = get_parser()
  args = vars(parser.parse_args())
  fname = args['edglst']

  try:
    edgelist_to_nddgo_graph_dimacs(fname)
  except Exception, e:
    print 'ERROR, UNEXPECTED EXCEPTION'
    print str(e)
    traceback.print_exc()
    sys.exit(1)
  sys.exit(0)
