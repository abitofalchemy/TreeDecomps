#!/bin/bash
TS=`date +"%d%b%y_%H%M"`
bname=`basename "$1" | cut -d'.' -f1`
#logname="~/Logs/"$bname"_"$TS".log"
bname=`basename "$1"`
bname="${bname#*.}"
echo $bname

# Clique Tree Rules
python b2CliqueTreeRules.py -t Results/ucidata-zachary.dimacs.tree

# c0.prules.overlap.py
python c0.prules.overlap.py -p -g Results/moreno_lesmis_lesmis


