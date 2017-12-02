find ~/Kynkon/datasets -name 'out.*' -type f 

python xplodnTree.py --orig $1
python prs.py --orig $1 
python xplodnTree.py --orig $1
python prs.py --grow $1 
01Dec17 | File "explodingTree.py", line 61, in explode_to_trees
01Dec17 | fix the explode to tree
