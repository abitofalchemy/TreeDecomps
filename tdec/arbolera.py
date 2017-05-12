__version__="0.1.0"

from pandas import DataFrame
from os import path

def nx_edges_to_nddgo_graph(G,n,m, sampling=False, varel="", save_g=False):
  ofname = 'datasets/{}_{}_{}{}.dimacs'.format(G.name, n,m,varel)
  if sampling:
    edges = G.edges()
    edges = [(int(e[0]), int(e[1])) for e in edges]
    df = DataFrame(edges)
    df.sort_values(by=[0], inplace=True)
    dimacs_graph =[]
    if save_g:
      with open(ofname, 'w') as f:
        f.write('c {}\n'.format(G.name))
        f.write('p edge\t{}\t{}\n'.format(n+1,m))

        output_edges = lambda x: f.write("e\t{}\t{}\n".format(x[0]+1, x[1]+1))
        df.apply(output_edges, axis=1)
      
#      if path.exists(ofname): print 'Wrote: ./{}'.format(ofname)
    else:
      output_edges = lambda x: "e\t{}\t{}\n".format(x[0]+1, x[1]+1)
      dimacs_graph = df.apply(output_edges, axis=1)
  else:
    edges = G.edges()
    edges = [(int(e[0]), int(e[1])) for e in edges]
    df = DataFrame(edges)
    df.sort_values(by=[0], inplace=True)
    if save_g:
      with open(ofname, 'w') as f:
        f.write('c {}\n'.format(G.name))
        f.write('p edge\t{}\t{}\n'.format(n+1,m))

        output_edges = lambda x: f.write("e\t{}\t{}\n".format(x[0]+1, x[1]+1))
        df.apply(output_edges, axis=1)

#      if path.exists(ofname): print 'Wrote: ./{}'.format(ofname)
    else:
      output_edges = lambda x: "e\t{}\t{}\n".format(x[0]+1, x[1]+1)
      dimacs_graph =df.apply(output_edges, axis=1)
  if save_g:
    if path.exists(ofname): print '\tWrote: ./{}'.format(ofname)
    return [ofname]
  else:
    return dimacs_graph
