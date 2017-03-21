TS=`date +"%d%b%y_%H%M"`
bname=`basename "$1" | cut -d'.' -f2`
logname="~/Logs/"$bname"_"$TS".log"

## edgeslist to dimacs
python a1_write_inddgo_graph.py -g /Users/saguinag/Theory/DataSets/contact/out.contact

## dimacs to tree

./INDDGO/bin/serial_wis -f ./datasets/contact.dimacs -nice -mcs -w Results/contact.dimacs.tree

## Process tree to HRGs

./b1_dimacs_tree_to_cliquetree.py -t Results/contact.dimacs.tree
