from glob import glob
import sys
from os import path
import numpy as np
import pandas as pd


# if __name__ == '__main__':
print len(sys.argv)
if len(sys.argv)<2:
	print "provide input production rules file: out.dataset_fname"
	sys.exit(1)

print sys.argv[1]
#~#
#~# Stacked (union) of production rules
#~#
stacked_df = pd.read_csv(sys.argv[1], index_col=0, sep="\t")
print df.head()

#~#
#~# Basic Overlapping of  rules
#~#

#
##~#
##~# Isomorphic Intersection of Rules
##~#
#print stats_isomorphic_rule_overlap(stacked_rules)
#
