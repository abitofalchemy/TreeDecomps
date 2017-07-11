# Start with this line to compute all
python explodingTree.py --orig datasets/out.brunsonsouthernwomensouthernwomen

# This next line re-computes the probabilities for the entire *set* of graphs
python baseball.py ProdRules/brunsonsouthernwomensouthernwomen_lcc.prs
python tstprodrules.py --orig datasets/out.brunsonsouthernwomensouthernwomen --prs ProdRules/brunsonsouthernwomensouthernwomen_lcc_rc.tsv

# This tst the isomorphic (intersection) subset of the rules
python tstprodrules.py --orig datasets/out.brunsonsouthernwomensouthernwomen --prs Results/brunsonsouthernwomensouthernwomen_isom_interxn.tsv



