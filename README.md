# Tree Decompositions

Investigating the effect of different tree decompositions on 
graph grammars. 

## How to Run Experiments
- Generate a tree decomposition using "minf" variable elimination algorithm
  `./tredec.dimacs.tree.py --orig datasets/out.contact --peoh minf`

- Print treewidth & compute a tree decomposition given the variable elimination using flag: `varel`
  `./tredec.dimacs.tree.py --orig datasets/out.contact -peoh minf -tw`

- Generate HRG graphs from the tree decomposition (`.dimacs.tree` file) generated in one of the above steps.
  `./tredec.phrg.py --orig datasets/out.contact --clqtree datasets/contact_minf.dimacs.tree`

- Batch processing on DSG1:
Tree decompositions: `tdec/exp.linux.sh datasets/out.contact`
HRG graph geneation: `tdec/exp.metrics.sh datasets/out.contact`
Jaccard Similarity:  `./tredec.isomorph_dimacs_tree.py --orig datasets/out.contact --pathfrag datasets/contact_` Jaccard Similarity b/w pairs of Production Rules from each of the Tree decomposition heuristics.

### Jaccard Dist
- http://stackoverflow.com/questions/11911252/python-jaccard-distance-using-word-intersection-but-not-character-intersection
- [Jaccard Similarity](http://infolab.stanford.edu/~ullman/mmds/ch3.pdf)

### sampling
`python tredec.dimacs.tree.py --orig ~/Theory/DataSets/out.subelj_euroroad_euroroad --peoh mcs`
`python tredec.samp.phrg.py --orig ~/Theory/DataSets/out.subelj_euroroad_euroroad --tree datasets/subelj_euroroad_euroroad`

## Battery of Datasets
- Download the datasets; run script `download_datasets.sh` for details


## 
<!--[Main Workflow (Rstudio)](tree_decomps.Rmd)-->

## StarLog

Date   | Notes
-------|------------------------------------------------------------------
07Apr17| Working on refactoring the files to make it more usable; fixing issues with sampling; ToDo: process mult trees from sampling

06Apr17| tree decomposition: 'fails here' issue of strange d[x] 
03Apr17| Got the isomorph working nees tuning; ToDo: Sampling: sampled.subgraphs.cliquetree.py
03Apr17| isomorphic prod rules check; (1) within a set and (2) between sets; TODO: zoom in on zachary lexm and mcsm iso overlap; **within file, using iso check, use the resluting set to build graphs**
30Mar17| Looking at overlaps using isomorphism
29Mar17| Work on the overlap data; the sampling is not working out so easily
28Mar17| Try sampling smaller graphs and reference the original `sample_edgelist_tosubgraphs.py`
27Mar17| The sampled graphs need to handle the node ids being much Larger than V.size
19Mar17| Need to make sure the conversion to binarized tree works okay
08Mar16| plot/show the progression from the BoardExample file to the binarized version and test other graphs
17Mar17| The workflow: 
29Jan17| `dimacs_td_ct` ToDo: TD to Prod Rules from NDDGO
17Jan17| INDDGO - Baseline TD plotting; Compute Network Statistics `gen_cliquetree` has issues for "set changed size during iteration" The problem is with `RuntimeError: Set changed size during iteration`  
12Jan17| ToDo: TD to Prod Rules from NDDGO 
07Jan17| Done: python call to generate hrg (hstar) graph objects
10Jan17| ToDo: python call to generate kron graphs (??)
11Jan17| Figure out how to take a TD from inddgo and derive a set of production rules
11Jan17| Expand related work & experiments


<!--- Run from a script to generate clique trees given a dataset and variable elimination heuristic (poeh)-->
<!--`tdec/exp.linux.sh datasets/out.ucidata-zachary` This script converts and edglist to `.dimacs` and uses that to run INDDGO to generate a tree decomposition. The script passes as one of the arguments each of the variable elimination heuristics we selected for this project. -->
<!--* to run a single example do:-->
<!--`python  tredec.dimacs.tree.py --orig datasets/out.ucidata-zachary --peoh mcs` this will sample if the graph exceeds 500 nodes. To avoid sampling, do `python  tredec.dimacs.tree.py --orig datasets/out.ucidata-zachary --peoh mcs -tw` this will print the treewidth and  -->

