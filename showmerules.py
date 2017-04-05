__version__="0.1.0"
from glob import glob
import pandas as pd
import os
import argparse

def get_parser ():
  parser = argparse.ArgumentParser(description='print rules for a given set')
  parser.add_argument('--gpath', required=True, help='input graph (edgelist)')
  parser.add_argument('--version', action='version', version=__version__)
  return parser


if __name__ == '__main__':
  parser = get_parser()
  args = vars(parser.parse_args())
  in_path = args['gpath']

  files = glob(in_path + '*.bz2')
  for f in files:
      with open("Results/out.txt", "a") as outf:
        outf.write("%s\n"%f)
      df1 = pd.read_csv(f, index_col=0, compression='bz2')
      #df1[[1,2]].to_csv("Results/out.txt", mode="a", header=False, sep="\t",index=False)
      print df1.to_string()





