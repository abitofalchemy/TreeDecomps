#title           :pyscript.py
#description     :This will create a header for a python script.
#author          :@saguinag
#date            :
#version         :0.4
#usage           :python pyscript.py
#notes           :
#python_version  :2.6.6
#======================
# http://ericasadun.com/2016/12/04/running-python-in-xcode-step-by-step/

def load_tarbz2_dataset(fpath):
    from pandas import read_csv
    df = read_csv(fpath, sep="\t",comment="%", header=None)
    return df

import networkx as nx
import os
os.chdir(os.path.dirname(__file__))
print(os.getcwd())

#from glob import glob
#d_files = glob("../datasets/out.*")
#import subprocess
#p = subprocess.Popen("find ../datasets -name 'out.*' -type f", stdout=subprocess.PIPE, shell=True)
#(output, err) = p.communicate()
#decode('utf-8').strip()
#d_files = output.decode('utf-8').split("\b\n")
#print (len(d_files))


dir="../datasets"
d_files = [x[0]+"/"+f for x in os.walk(dir) for f in x[2] if f.startswith("out.")]
for f in d_files:
    df =load_tarbz2_dataset(f)
    g = nx.from_pandas_dataframe(df, source=0, target=1) # edgelist
    nx.write_gpickle(g, f+".p")
print ("Done converting to gpickle")


