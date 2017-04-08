#!/bin/bash
TS=`date +"%d%b%y_%H%M"`
bname=`basename "$1" | cut -d'.' -f1`
#logname="~/Logs/"$bname"_"$TS".log"
bname=`basename "$1"`
bname="${bname#*.}"
echo $bname
## To process the entire graph
## edgeslist to dimacs
python tredec.dimacs.tree.py --orig $1 --peoh $2
python tredec.phrg.py --clqtree datasets/"$bname" --orig $1



