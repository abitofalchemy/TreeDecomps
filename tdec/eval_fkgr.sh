#!/bin/bash
# Start with this line to compute all
#python explodingTree.py --orig datasets/out.brunsonsouthernwomensouthernwomen

# This next line re-computes the probabilities for the entire *set* of graphs
#python baseball.py ProdRules/brunsonsouthernwomensouthernwomen_lcc.prs
#python tstprodrules.py --orig datasets/out.brunsonsouthernwomensouthernwomen --prs ProdRules/brunsonsouthernwomensouthernwomen_lcc_rc.tsv

# This tst the isomorphic (intersection) subset of the rules
#python tstprodrules.py --orig datasets/out.brunsonsouthernwomensouthernwomen --prs Results/brunsonsouthernwomensouthernwomen_isom_interxn.tsv

TS=`date +"%d%b%y_%H%M"`
bname=`basename "$1" | cut -d'.' -f1`
logname="Results/"$bname"_"$TS".log"
bname=`basename "$1"`
bname="${bname#*.}"
logname="Results/ba_ctrl_"$TS".log"

## Print metrics and save to log files
#   Arguments:
#   original (reference) dataset in eddgelist format
#   
# find datasets -name "$bname"_*dimacs.tree | parallel python "td.phrg.py --orig '$1' --clqtree" {} ">" {}.log
# python td.isom_jaccard_sim.py --orig /data/saguinag/datasets/out.ucidata-gama  --pathfrag datasets/ucidata-gama_
# cat tdec/maindatasets | parallel python "td.isom_jaccard_sim.py --orig {} --pathfrag datasets/'$bname'_"
#echo $1 | parallel python "td.isom_jaccard_sim.py --orig {}  --pathfrag datasets/'$bname'_"
#echo 'python explodingTree.py --orig datasets/'$1' >/dev/null 2>&1 &&'
# python tstprodrules.py --orig datasets/$1 --prs "Results/"$bname"_isom_interxn.tsv" >>Results/$bname".log"2>&1 &
# echo 'python tstprodrules.py --orig datasets/'$1' --prs Results/'$bname'_isom_interxn.tsv >>Results/'$bname'.log 2>&1 &'

#echo 'python explodingTree.py --orig datasets/'$1' >>/home/saguinag/Logs/'$bname'.log 2>&1 &' 
#python explodingTree.py --orig datasets/$1 >>/home/saguinag/Logs/$bname.log 2>&1 & 
python explodingTree.py --orig $1 --synthchks >> Results/syncks_$bname.log 2>&1 &
