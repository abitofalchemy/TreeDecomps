find ~/KynKon/datasets -name 'out.*' | parallel /Users/sal.aguinaga/Boltzmann/TreeDecomps/xplodnTree/xplotree_subgraphs_prs.py {}
find ~/KynKon/datasets -name 'out.as' | parallel /Users/sal.aguinaga/Boltzmann/TreeDecomps/xplodnTree/explodingTree.py --base {}

# python xplodnTree.py --orig /Users/sal.aguinaga/KynKon/datasets/ucidata-gama/out.ucidata-gama
# python prs.py --orig /Users/sal.aguinaga/KynKon/datasets/ucidata-gama/out.ucidata-gama
# python fast_fixed_phrg.py --orig   /Users/sal.aguinaga/KynKon/datasets/maayan-Stelzl/out.maayan-Stelzl -prs
#
#
# python isomorph_overlap_hl.py --orig /Users/sal.aguinaga/KynKon/datasets/ucidata-gama/out.ucidata-gama --pathfrag ../ProdRules/ucidata-gama.dimacs
#
# python isomorph_overlap_hl.py --orig ../datasets/out.ucidata-gama --pathfrag ../ProdRules/ucidata-gama.dimacs
# python isomorph_overlap_hl.py --orig /Users/sal.aguinaga/KynKon/datasets/ucidata-gama/out.ucidata-gama --fire ../Results/ucidata-gama_isom_interxn.bz2
