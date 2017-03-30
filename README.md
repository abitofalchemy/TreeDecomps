# Tree Decompositions

We investigate the effect of different tree decompositions on 
graph grammars. 

## StarLog
- 30Mar17: Looking at overlaps using isomorphism
- 28Mar17: Working on getting sampled graphs to generate trees
- 27Mar17: The sampled graphs need to handle the node ids being much Larger than |V|
- 23Mar17: 
- 19Mar17: Need to make sure the conversion to binarized tree works okay
- 08Mar16: plot/show the progression from the BoardExample file to 
  the binarized version and test other graphs


# Experiments

## How to Run Experiments

- Setup the graphs (edgelist network datasets) 
  `./setup_working_graphs.sh`

- Run from a script to generate clique trees given a dataset and POE heuristic
  `./experiments.sh datasets/out.ucidata-zachary mcs`

## POE Heuristic Results
-mind : generates an elim. ordering using min degree heuristic
-mmd : generates an elim. ordering using multiple min degree heuristic
-minf : generates an elim. ordering using min fill heuristic
-bmf : generates an elim. ordering using a batched min fill heuristic
-beta : generates an elim. ordering using the beta heuristic
-metmmd : generates an elim. ordering using METIS mmd heuristic
-metnnd : generates an elim. ordering using METS node ND heuristic
-mcsm : generates an elim. ordering using mcsm euristic
-mcs  : generates an elim. ordering using mcs
-lexm : generates an elim. ordering using lex-m bfs heuristic

Dataset | mcs | mind | 
--------|-----|------|
jazz    |
lesmis  |
zachary |

## Workflow (Rstudio)

[Main Workflow (Rstudio)](ctrlRtdecomp.Rmd)
`python write_inddgo_graph.py -g ~/Theory/DataSets/out.brunson_southern-women_southern-women`


# References
- http://www.cs.princeton.edu/courses/archive/spr04/cos226/lectures/maxflow.4up.pdf
- http://sahandsaba.com/thirty-python-language-features-and-tricks-you-may-not-know.html
- [book mcs tid bit](https://books.google.com/books?id=NFm7BQAAQBAJ&pg=PA186&lpg=PA186&dq=python+algorithm+maximum+cardinality+search+sample+code&source=bl&ots=YAod8M0QFx&sig=7xD9NF5EBK0cNwQgkD-nHkrcZVk&hl=en&sa=X&ved=0ahUKEwj7hqfJ99_SAhWBbSYKHecDCCwQ6AEIQjAG#v=onepage&q=python%20algorithm%20maximum%20cardinality%20search%20sample%20code&f=false)
- http://code.activestate.com/recipes/221251-maximum-cardinality-matching-in-general-graphs/

# One Liners
- ` python -c "import networkx as nx; g=nx.read_edgelist('datasets/out.ucidata-zachary', comments='%'); print nx.info(g);"`
