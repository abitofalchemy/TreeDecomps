#!/bin/bash
TS=`date +"%d%b%y_%H%M"`
bname=`basename "$1" | cut -d'.' -f1`
#logname="~/Logs/"$bname"_"$TS".log"
bname=`basename "$1"`
bname="${bname#*.}"
echo $bname

## edgeslist to dimacs
python a1_write_inddgo_graph.py -g $1 &&

## dimacs to tree

./bin/serial_wis -f ./datasets/"$bname".dimacs -nice -$2 -w Results/"$bname"."$2".dimacs.tree &&

## Process tree to HRGs

python b1_dimacs_tree_to_cliquetree.py -t Results/"$bname"."$2".dimacs.tree
