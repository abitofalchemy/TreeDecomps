__version__="0.1.0"

import sys
import traceback
import argparse
import os
import networkx as nx
import pandas as pd
from tdec.PHRG import graph_checks
import subprocess

def get_parser ():
    parser = argparse.ArgumentParser(description='given a tree derive grammar rules')
    parser.add_argument('--orig', required=True, help='input tree decomposition (dimacs file format)')
    parser.add_argument('--peoh', required=True, help='mcs, lexm, mind, minf, etc')
    parser.add_argument('--version', action='version', version=__version__)
    return parser

def dimacs_nddgo_tree(dimacsfname, heuristic):
    args = ["bin/mac/serial_wis -f {} -nice -{} -w {}.tree".format(dimacsfname, heuristic, dimacsfname)]
    nddgoout = ""
    while not nddgoout:
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, shell=True)
        popen.wait()
        # output = popen.stdout.read()
        out, err = popen.communicate()
        nddgoout = out.split('\n')
    print nddgoout
    return dimacsfname+".tree"

def load_edgelist(gfname):
  import pandas as pd
  try:
    edglst = pd.read_csv(gfname, comment='%', delimiter='\t')
    # print edglst.shape
    if edglst.shape[1]==1: edglst = pd.read_csv(gfname, comment='%', delimiter="\s+")

  except Exception, e:
    print "EXCEPTION:",str(e)
    traceback.print_exc()
    sys.exit(1)

  if edglst.shape[1] == 3:
    edglst.columns = ['src', 'trg', 'wt']
  elif edglst.shape[1] == 4:
    edglst.columns = ['src', 'trg', 'wt','ts']
  else:
    edglst.columns = ['src', 'trg']
  g = nx.from_pandas_dataframe(edglst,source='src',target='trg')
  g.name = os.path.basename(gfname)
  return g

def nx_edges_to_nddgo_graph (G,n,m, sampling=False):
    ofname = 'datasets/{}.dimacs'.format(G.name)
    print '...', ofname

    if sampling:

        edges = G.edges()
        edges = [(int(e[0]), int(e[1])) for e in edges]
        df = pd.DataFrame(edges)
        df.sort_values(by=[0], inplace=True)

        with open(ofname, 'w') as f:
          f.write('c {}\n'.format(G.name))
          f.write('p edge\t{}\t{}\n'.format(n,m))
          # for e in df.iterrows():
          output_edges = lambda x: f.write("e\t{}\t{}\n".format(x[0], x[1]))
          df.apply(output_edges, axis=1)
        # f.write("e\t{}\t{}\n".format(e[0]+1,e[1]+1))
        if os.path.exists(ofname): print 'Wrote: ./{}'.format(ofname)
    else:
        edges = G.edges()
        edges = [(int(e[0]), int(e[1])) for e in edges]
        df = pd.DataFrame(edges)
        df.sort_values(by=[0], inplace=True)

        with open(ofname, 'w') as f:
          f.write('c {}\n'.format(G.name))
          f.write('p edge\t{}\t{}\n'.format(n,m))
          # for e in df.iterrows():
          output_edges = lambda x: f.write("e\t{}\t{}\n".format(x[0], x[1]))
          df.apply(output_edges, axis=1)
        # f.write("e\t{}\t{}\n".format(e[0]+1,e[1]+1))
        if os.path.exists(ofname): print 'Wrote: ./{}'.format(ofname)
    return ofname

def edgelist_dimacs_graph(orig_graph):
    fname = orig_graph
    gname = os.path.basename(fname).split(".")
    gname = sorted(gname,reverse=True)[0]
    # if args['sampling']:
    #   mapping_d = map_original_node_ids(fname)
    #   G1 = nx.read_edgelist(fname, comments="%", data=False, nodetype=int)
    #   G = nx.relabel_nodes(G1, mapping_d)
    # else:
    G = nx.read_edgelist(fname, comments="%", data=False, nodetype=int)

    # +++ Graph Checks
    if G is None: sys.exit(1)
    G.remove_edges_from(G.selfloop_edges())
    giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
    G = nx.subgraph(G, giant_nodes)
    graph_checks(G)
    # --- graph checks

    G.name = gname

    print "...",  G.number_of_nodes(), G.number_of_edges()
    return (nx_edges_to_nddgo_graph(G, n=G.number_of_nodes(), m=G.number_of_edges()), gname)


def main ():
    parser = get_parser()
    args = vars(parser.parse_args())
    # edglst_dimacs_tree_phrg(args['orig'], args['peoh'])  # gen synth graph
    dimacs_g, gname = edgelist_dimacs_graph(args['orig'])
    dimacs_t = dimacs_nddgo_tree(dimacs_g, args['peoh'])
    print dimacs_t, args['orig']
    # args = ["echo", "--clqtree {}".format(dimacs_t),\
    #         "--orig {}".format(args['orig'])]
    # out = ""
    # # while not out:
    # popen = subprocess.Popen(args, stdout=subprocess.PIPE, shell=True)
    # popen.wait()
    # # output = popen.stdout.read()
    # out, err = popen.communicate()
    # nddgoout = out.split('\n')
    # print nddgoout

if __name__ == '__main__':
    try:
        main()
    except Exception, e:
        print str(e)
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)
