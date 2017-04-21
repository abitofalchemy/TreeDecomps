__version__="0.1.0"

# ToDo:
# [] process mult dimacs.trees to hrg

import sys
import traceback
import argparse
import os
import glob
import networkx as nx
import pandas as pd
from tdec.PHRG import graph_checks
import subprocess
import math
import tdec.graph_sampler as gs
import platform

global args

def get_parser ():
    parser = argparse.ArgumentParser(description='Given an edgelist and PEO heuristic perform tree decomposition. Example: '
                                                 '"python tredec.dimacs.tree.py --orig out.filename --peoh mcs"')
    parser.add_argument('--orig', required=True, help='Input/reference graph in edgelist format (or tar.bz2)')
    parser.add_argument('--peoh', required=True, help='Var. elim method such as mcs, lexm, mind, minf, etc')
    parser.add_argument('-tw', action='store_true',  default=False, required=False, help='print treewidth given var elim method')
    parser.add_argument('--version', action='version', version=__version__)
    return parser

def dimacs_nddgo_tree(dimacsfnm_lst, heuristic):
    # print heuristic,dimacsfnm_lst

    for dimacsfname in dimacsfnm_lst:
        nddgoout = ""
        if platform.system() == "Linux":
          args = ["bin/linux/serial_wis -f {} -nice -{} -w {}.tree".format(dimacsfname, heuristic, dimacsfname)]
        else:
          args = ["bin/mac/serial_wis -f {} -nice -{} -w {}.tree".format(dimacsfname, heuristic, dimacsfname)]
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

def nx_edges_to_nddgo_graph (G,n,m, sampling=False, peoh=""):
    # print args['peoh']
    ofname = 'datasets/{}_{}.dimacs'.format(G.name, peoh)
    # print '...', ofname

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

    return [ofname]

def nx_edges_to_nddgo_graph_sampling(graph, n, m, peo_h):
    G = graph
    if n is None and m is None: return
    # n = G.number_of_nodes()
    # m = G.number_of_edges()
    nbr_nodes = 256
    basefname = 'datasets/{}_{}'.format(G.name, peo_h)

    K = int(math.ceil(.25*G.number_of_nodes()/nbr_nodes))
    print "--", nbr_nodes, K, '--';

    for j,Gprime in enumerate(gs.rwr_sample(G, K, nbr_nodes)):
        # if gname is "":
        #     # nx.write_edgelist(Gprime, '/tmp/sampled_subgraph_200_{}.tsv'.format(j), delimiter="\t", data=False)
        #     gprime_lst.append(Gprime)
        # else:
        #     # nx.write_edgelist(Gprime, '/tmp/{}{}.tsv'.format(gname, j), delimiter="\t", data=False)
        #     gprime_lst.append(Gprime)
        # # print "...  files written: /tmp/{}{}.tsv".format(gname, j)


        edges = Gprime.edges()
        edges = [(int(e[0]), int(e[1])) for e in edges]
        df = pd.DataFrame(edges)
        df.sort_values(by=[0], inplace=True)

        ofname = basefname+"_{}.dimacs".format(j)

        with open(ofname, 'w') as f:
          f.write('c {}\n'.format(G.name))
          f.write('p edge\t{}\t{}\n'.format(n,m))
          # for e in df.iterrows():
          output_edges = lambda x: f.write("e\t{}\t{}\n".format(x[0], x[1]))
          df.apply(output_edges, axis=1)
        # f.write("e\t{}\t{}\n".format(e[0]+1,e[1]+1))
        if os.path.exists(ofname): print 'Wrote: {}'.format(ofname)

    return basefname

def edgelist_dimacs_graph(orig_graph, peo_h, prn_tw = False):
    fname = orig_graph
    gname = os.path.basename(fname).split(".")
    gname = sorted(gname,reverse=True, key=len)[0]

    if ".tar.bz2" in fname:
        from tdec.read_tarbz2 import read_tarbz2_file
        edglst = read_tarbz2_file(fname)
        df = pd.DataFrame(edglst,dtype=int)
        G = nx.from_pandas_dataframe(df,source=0, target=1)
    else:
        G = nx.read_edgelist(fname, comments="%", data=False, nodetype=int)
    # print "...",  G.number_of_nodes(), G.number_of_edges()
    # from numpy import max
    # print "...",  max(G.nodes()) ## to handle larger 300K+ nodes with much larger labels

    N = max(G.nodes())
    M = G.number_of_edges()
    # +++ Graph Checks
    if G is None: sys.exit(1)
    G.remove_edges_from(G.selfloop_edges())
    giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
    G = nx.subgraph(G, giant_nodes)
    graph_checks(G)
    # --- graph checks

    G.name = gname

    # print "...",  G.number_of_nodes(), G.number_of_edges()
    if G.number_of_nodes() > 500 and not prn_tw:
        return (nx_edges_to_nddgo_graph_sampling(G, n=N, m=M, peo_h=peo_h), gname)
    else:
        return (nx_edges_to_nddgo_graph(G, n=N, m=M, peoh=peo_h), gname)

def print_treewidth (in_dimacs, var_elim):
    nddgoout = ""
    if platform.system() == "Linux":
      args = ["bin/linux/serial_wis -f {} -nice -{} -width".format(in_dimacs, var_elim)]
    else:
      args = ["bin/mac/serial_wis -f {} -nice -{} -width".format(in_dimacs, var_elim)]
    while not nddgoout:
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, shell=True)
        popen.wait()
        # output = popen.stdout.read()
        out, err = popen.communicate()
        nddgoout = out.split('\n')
    print nddgoout
    return nddgoout

def main ():
    parser = get_parser()
    args = vars(parser.parse_args())
    # edgelist to dimacs, if tw then do not sample for a large graph and try to derive
    # a clique tree / tree decomposition using the given var. elim heuristic
    dimacs_g, gname = edgelist_dimacs_graph(args['orig'], args['peoh'], args['tw'])

    if args['tw']:
        print_treewidth(dimacs_g[0], args['peoh'])
        if len(dimacs_g) == 1:
            dimacs_t = dimacs_nddgo_tree(dimacs_g, args['peoh'])
    else:
        if len(dimacs_g) == 1:
            dimacs_t = dimacs_nddgo_tree(dimacs_g, args['peoh'])
        else:
            import time
            time.sleep(2)
            subgraphs_lst= glob.glob(dimacs_g+"*dimacs")
            # process multi subgraphs
            dimacs_t = dimacs_nddgo_tree(subgraphs_lst, args['peoh']) # pass list of subgraphs
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
