import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], delimiter=",", header=0, index_col=None)
print df.columns 
df = df[ ["Reporter Code", u'Partner Code', u'Trade Value (US$)']]
# df[2] = df[2]/float(df[2].max())
df['Trade Value (US$)'] = df['Trade Value (US$)']/df['Trade Value (US$)'].max()
df['Trade Value (US$)'] = df['Trade Value (US$)'].apply(lambda x: "%.6f" % x)

df.to_csv(sys.argv[1].split(".")[0] + ".edglst",
            sep="\t", header=None, index=False)



