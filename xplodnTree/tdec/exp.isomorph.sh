#!/bin/bash
TS=`date +"%d%b%y_%H%M"`
bname=`basename "$1" | cut -d'.' -f1`
logname="Results/"$bname"_"$TS".log"
bname=`basename "$1"`
bname="${bname#*.}"
## Print metrics and save to log files
#   Arguments:
#   original (reference) dataset in eddgelist format
#   
# find datasets -name "$bname"_*dimacs.tree | parallel python "td.phrg.py --orig '$1' --clqtree" {} ">" {}.log
# python td.isom_jaccard_sim.py --orig /data/saguinag/datasets/out.ucidata-gama  --pathfrag datasets/ucidata-gama_
# cat tdec/maindatasets | parallel python "td.isom_jaccard_sim.py --orig {} --pathfrag datasets/'$bname'_"
#echo $1 | parallel python "td.isom_jaccard_sim.py --orig {}  --pathfrag datasets/'$bname'_"
python td.interxn.py $1 >$logname
