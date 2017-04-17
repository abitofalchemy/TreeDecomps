#!/bin/bash
TS=`date +"%d%b%y_%H%M"`
bname=`basename "$1" | cut -d'.' -f1`
#logname="~/Logs/"$bname"_"$TS".log"
bname=`basename "$1"`
bname="${bname#*.}"
echo $bname
## Print metrics and save to log files
#python tredec.phrg.py --orig datasets/out.ucidata-zachary --clqtree datasets/ucidata-zachary_mcs.dimacs.tree
#find ./datasets -name "$bname"*.tree -type f | parallel tredec.phrg.py --orig --clqtree {} 
# echo python tredec.phrg.py --orig datasets/out.ucidata-zachary --clqtree datasets/ucidata-zachary_mcs.dimacs.tree
find datasets -name "$bname"_*.tree | parallel python "tredec.phrg.py --orig '$1' --clqtree" {} ">" {}.log
