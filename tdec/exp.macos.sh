#!/bin/bash
TS=`date +"%d%b%y_%H%M"`
bname=`basename "$1" | cut -d'.' -f1`
#logname="~/Logs/"$bname"_"$TS".log"
bname=`basename "$1"`
bname="${bname#*.}"
echo $bname
## To process the entire graph
## edgeslist to dimacs
# python trde.edgelist_inddgo_graph.py -g $1 &&
#
# ## dimacs to tree
# ./bin/mac/serial_wis -f ./datasets/"$bname".dimacs -nice -$2 -w Results/"$bname"."$2".dimacs.tree &&
#
# ## Process tree to HRGs
# ./trde.dimacs_tree_2cliquetree.py --clqtree Results/"$bname"."$2".dimacs.tree --orig $1

python tredec.dimacs.tree.py --orig  $1 --peoh $2
