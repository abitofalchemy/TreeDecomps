#!/bin/bash
TS=`date +"%d%b%y_%H%M"`
bname=`basename "$1" | cut -d'.' -f1`
#logname="~/Logs/"$bname"_"$TS".log"
bname=`basename "$1"`
bname="${bname#*.}"
# echo $bname

## edgeslist to dimacs
## dimacs to tree
cat tdec/heuristics | parallel python "tredec.dimacs.tree.py --orig '$1' --varel {}"

## Process tree to HRGs
## python b1_dimacs_tree_to_cliquetree.py -t Results/"$bname"."$2".dimacs.tree > ~/Logs/"$bname"_"$2".log

