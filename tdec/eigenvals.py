import networkx as nx
import numpy as np
import pandas as pd

f = np.loadtxt("/data/saguinag/datasets/time_stamped_datasets/out.email-Eu-core-Dept1", dtype=int, comments="%")
df = pd.DataFrame(f)
if df.shape[1]==4:
  df.columns=['src','trg', 'wt', 'ts']
elif df.shape[1]==3:
  df.columns=['src','trg', 'ts']
else:
  df.columns=['src','trg']
g = nx.from_pandas_dataframe(df, 'src', 'trg')
print nx.info(g)

df = pd.DataFrame.from_dict(g.degree().items())
eigv = nx.adjacency_spectrum(g)
df['eig']= abs(eigv)
print df.groupby([1]).count().head()

gb = df.groupby([1]).groups
Xeigv = [(k,df.loc[v]['eig'].sum()) for k,v in gb.iteritems()]
print len(Xeigv)


for i,x in Xeigv:
  print "%d\t%.3f" % (i,x)

  

