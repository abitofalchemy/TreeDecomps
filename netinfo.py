#!/usr/bin/env python
import networkx as nx
import os, sys
import dataframe_from_temporal_edgelist as dfe


if len(sys.argv)>1:
  ifile = sys.argv[1]
else:
  exit(1)
print ifile, 

dfs = dfe.Pandas_DataFrame_From_Edgelist([ifile])
df = dfs[0]

try:
    g = nx.from_pandas_dataframe(df, 'src', 'trg',edge_attr=['ts'])
except  Exception, e:
    #print str(e)
    g = nx.from_pandas_dataframe(df, 'src', 'trg')

if df.empty:
  g = nx.read_edgelist(ifile,comments="%")

print os.path.basename(ifile), g.number_of_nodes(), g.number_of_edges()

