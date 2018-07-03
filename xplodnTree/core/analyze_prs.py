#!/usr/bin/env python

import multiprocessing as mp
from utils import Info, graph_name
import sys, os
import pprint as pp

from glob import glob
from isomorph_overlap_hl import stack_prod_rules_bygroup_into_list
from prs import proc_prod_rules_orig


results = []

def prs_count_per(prs_lst):
	for f in prs_lst:
		pp.pprint ([os.path.basename(f), len( open(f).readlines())])


if __name__ == '__main__':
	if len(sys.argv) < 2:
		Info("add an out.* dataset with its full path")
		exit()

	f = sys.argv[1]
	gn = graph_name(f)

	f = "../ProdRules/" + gn + "*.prs"
	files = glob(f)

	prs_cnt_per = prs_count_per(files)
	# prs_stack = stack_prod_rules_bygroup_into_list(files)

	sys.exit(0)