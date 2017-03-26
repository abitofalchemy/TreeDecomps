import pandas as pd
from glob import glob
import os
import re

def peak_at_two_inpufiles(f1, f2):
    for f in [f1, f2]:
        print os.path.basename(f).split('.')[1],
    df1 = pd.read_csv(f1, index_col=0, compression='bz2')
    df1.columns =['rnbr','lhs','rhs','pr']
    df1.drop(['rnbr','pr'],inplace=True,axis=1)
    # print df1.shape
    # print
    df2 = pd.read_csv(f2, index_col=0, compression='bz2')
    df2.columns =['rnbr','lhs','rhs','pr']
    df2.drop(['rnbr','pr'],inplace=True,axis=1)
    # print df2.shape
    # print '- '*20
    # print pd.merge(df1, df2, left_on='rhs', right_on='rhs', how='inner',sort=False).head()
    # print pd.merge(df1, df2, left_on='rhs', right_on='rhs', how='inner',sort=False).shape
    # mdf = pd.concat ([df1,df2])
    # print mdf.shape
    # print mdf.drop_duplicates().shape
    print pd.merge(df1, df2, how='inner', on=['rhs']).shape

def rules_per_file(files):
    print "rules per file\n","-"*20
    for f in files:
        print os.path.basename(f).split('.')[1],"\t",
        df = pd.read_csv(f, index_col=0, compression='bz2')
        print df.shape
    return

def rules_overlap_between_files(files):
    print "rules overlap between\n","-"*20
    mdf = pd.read_csv(files[0], index_col=0, compression='bz2')
    mdf.columns =['rnbr','lhs','rhs','pr']
    mdf.drop(['rnbr','pr'],inplace=True,axis=1)
    print os.path.basename(files[0]).split('.')[1],":"

    cdf = pd.DataFrame() # collect overlap
    for f in files:
        print os.path.basename(f).split('.')[1],'Overlap',
        df = pd.read_csv(f, index_col=0, compression='bz2')
        df.columns =['rnbr','lhs','rhs','pr']
        df.drop(['rnbr','pr'],inplace=True,axis=1)
        cdf = pd.concat([cdf, pd.merge(df, mdf, left_on='lhs', right_on='lhs', how='inner',sort=False)])
        print pd.merge(df, mdf, left_on='lhs', right_on='lhs', how='inner',sort=False).shape

    print cdf.shape
    return

data_files_dir = "/Users/saguinag/Theory/Grammars/ProdRules"
files = glob(data_files_dir+'/*.bz2')

rules_per_file(files)
print
rules_overlap_between_files(files)
# peak_at_two_inpufiles(files[0],files[2])
for f1 in files:
    for f2 in files:
        if f1 is not f2:
            peak_at_two_inpufiles(f1, f2)
    print

exit(0)

# mdf = pd.read_csv(files.pop(), index_col=0, compression='bz2')
# mdf.columns =['rnbr','lhs','rhs','pr']
# mdf.drop(['rnbr','pr'],inplace=True,axis=1)
#
# # print mdf.shape
#
# for f in files:
#     print mdf.shape
#     df = pd.read_csv(f, index_col=0, compression='bz2')
#     df.columns =['rnbr','lhs','rhs','pr']
#     df.drop(['rnbr','pr'],inplace=True,axis=1)
#     mdf = pd.concat([df, mdf])
#
# # We now have a master df
# print mdf.shape
# print mdf.drop_duplicates().shape
#
# print 'LHS stats'
# elscnt = lambda r: len(r['lhs'].strip("[]").split(','))
# print mdf.apply(elscnt, axis=1).describe()
# print
# print 'RHS stats'
# elscnt = lambda r: len(re.findall(r"'(.*?)'", r['rhs'].strip("[]"), re.DOTALL))
# print mdf.apply(elscnt, axis=1).describe()
#
# ## do a join
# files = glob(data_files_dir+'/*.bz2')
# print files[0]
# mdf = pd.read_csv(files[0], index_col=0, compression='bz2')
# mdf.columns =['rnbr','lhs','rhs','pr']
# mdf.drop(['rnbr','pr'],inplace=True,axis=1)
# print mdf.shape
# for f in files:
#     print os.path.basename(f).split('.')[1],
#     df = pd.read_csv(f, index_col=0, compression='bz2')
#     df.columns =['rnbr','lhs','rhs','pr']
#     df.drop(['rnbr','pr'],inplace=True,axis=1)
#     # print 'df',df.shape
#     # print df.head()
#     # mdf = pd.merge(df, mdf, on='rhs', how='inner')
#     print pd.merge(df, mdf, left_on='lhs', right_on='lhs', how='inner',sort=False).shape
#
#     # jdf= df.join(mdf, left_on='lhs', right_on='lhs', how='inner', lsuffix="_x")
#     # print 'jdf', jdf.shape


# print mdf.shape
# print mdf
