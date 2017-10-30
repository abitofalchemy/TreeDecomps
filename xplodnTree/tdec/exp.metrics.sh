#!/bin/bash
TS=`date +"%d%b%y_%H%M"`
bname=`basename "$1" | cut -d'.' -f1`
#logname="~/Logs/"$bname"_"$TS".log"
bname=`basename "$1"`
bname="${bname#*.}"
echo $bname
## Print metrics and save to log files
#   Arguments:
#   original (reference) dataset in eddgelist format
#   
find datasets -name "$bname"_*dimacs.tree | parallel python "td.phrg.py --orig '$1' --clqtree" {} ">" {}.log
