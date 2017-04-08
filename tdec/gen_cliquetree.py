__author__ = 'saguinag' + '@' + 'nd.edu'
__version__ = "0.1.0"

##
## gen_cliquetree
##

## TODO: some todo list

## VersionLog:

import argparse, traceback
import os, sys, pprint
import networkx as nx
import graph_sampler as gs
import tree_decomposition as td
import PHRG as phrg
import walk_ct as wct
import numpy as np

def treewidth(parent, children,twlst ):
  twlst.append(parent)
  for x in children:
    if isinstance(x, (tuple,list)):
      treewidth(x[0],x[1],twlst)
    else:
      print type(x), len(x)



def hrg_clique_tree (G):
  if G is None: return

  #  ------------------ ##
  #  tree decomposition
  #  ------------------ ##
  num_nodes = G.number_of_nodes()

  prod_rules = {}
  if num_nodes >= 500:
    for Gprime in gs.rwr_sample(G, 2, 300):
      T = td.quickbb(Gprime)
      root = list(T)[0]
      T = td.make_rooted(T, root)
      T = phrg.binarize(T)
      root = list(T)[0]
      root, children = T
      td.new_visit(T, G, prod_rules)
  else:
    T = td.quickbb(G)
    root = list(T)[0]
    T = td.make_rooted(T, root)
    T = phrg.binarize(T)
    root = list(T)[0]
    root, children = T
    td.new_visit(T, G, prod_rules)

  # pprint.pprint (children)
  return root, children

def gen_clique_tree (fname,tw):
    try:
        G = nx.read_edgelist(fname, comments="%", data=False)
    except Exception:
        G = nx.read_edgelist(fname,data=False)
    fname = os.path.basename(fname).split('.')[1]
    files = [f for f in os.listdir("./Results") if os.path.isfile(os.path.join("./Results", f))]
    files = [f for f in files if fname in f]
    G.name = fname

    root, children =  hrg_clique_tree(G)

    if tw:
        tw=[]
        treewidth(root, children,tw)
    print '___ {}'.format(fname)
    print '    Treewidth:', np.max([len(x)-1 for x in tw])
    print '    Tree:'
    if not tw:
        wct.walk_ct(root, children, '\t') # print the tree to stdout


def get_parser ():
  parser = argparse.ArgumentParser(description='gen_hrg: Generate synthetic graph using HRG model')
  parser.add_argument('-g', '--graph', required=True, help='input graph (edgelist)')
  parser.add_argument('--tw', action='store_true', help='print treewidth')
  parser.add_argument('--version', action='version', version=__version__)
  return parser


def main ():
  parser = get_parser()
  args = vars(parser.parse_args())

  gen_clique_tree(args['graph'],args['tw'])  #


if __name__ == '__main__':
  try:
    main()
    # save_plot_figure_2disk(plotname=plt_filename)
    # print 'Saved plot to: '+plt_filename
  except Exception, e:
    print 'ERROR, UNEXPECTED SAVE PLOT EXCEPTION'
    print str(e)
    traceback.print_exc()
    os._exit(1)
  sys.exit(0)
