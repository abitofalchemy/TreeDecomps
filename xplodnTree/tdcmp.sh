find ~/Kynkon/datasets -name 'out.*' -type f 

python xplodnTree.py --orig $1
python prs.py --orig $1 
python xplodnTree.py --orig $1
python prs.py --grow $1 
