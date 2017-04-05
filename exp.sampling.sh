#!/bin/bash
TS=`date +"%d%b%y_%H%M"`
bname=`basename "$1" | cut -d'.' -f1`
#logname="~/Logs/"$bname"_"$TS".log"
bname=`basename "$1"`
bname="${bname#*.}"
echo $bname


## subgraphs
python sample_edgelist_tosubgraphs.py -g $1 &&

## edgeslist to dimacs
find /tmp/ -name 'sampled_subgraph*' | parallel python a1_write_inddgo_graph.py -g {}&&

## dimacs to tree
find datasets/ -name '*.dimacs' | parallel ./bin/serial_wis -f {} -nice -$2 -w {}.tree && 
find ./datasets -name 'sampled_subgraph_200*dimacs' | parallel ./bin/mac/serial_wis -f {} -nice -mcs  -w {}.tree


## Process multiple trees from subgraphs and join
python sampled.subgraphs.cliquetree.py -tpath /tmp/ -gname "$bname"

  532  python sample_edgelist_tosubgraphs.py -g datasets/out.higgs-activity_time
  540  python a1_write_inddgo_graph.py -g /tmp/sampled_subgraph_200_0.tsv
  542  ./bin/serial_wis -f ./datasets/sampled_subgraph_200_0.dimacs -nice -lexm -w ./datasets/sampled_subgraph_200_0.dimacs.tree

## -- ##
python sample_edgelist_tosubgraphs.py -g datasets/out.higgs-activity_time

python sampled_edglst_dimacs.py --edglst /tmp/sampled_subgraph_200_0.tsv

./bin/mac/serial_wis -f datasets/sampled_subgraph_200_0.dimacs -nice -mcs -w ./Results/sampled_subgraph_200_0.dimacs.tree



