from subprocess import *
import os
##
## check datasets
## 
print ("// check datasets \\\\")

c = "find ./datasets -name 'out.*' -type f"
handle = Popen(c, stdin=PIPE, stderr=PIPE, stdout=PIPE, shell=True)
stdout_value = handle.communicate()[0]
print stdout_value
dataset_files = stdout_value
print "-"*10

##
## check orig graph gpickles
## 
c = "find ./datasets -name '*.pickle' -type f"#.format(fname)
handle = Popen(c, stdin=PIPE, stderr=PIPE, stdout=PIPE, shell=True)
stdout_value = handle.communicate()[0]
print stdout_value
print "-"*10

##
## check dimacs
##
for f in dataset_files.split("\n"):
	fname = os.path.basename(f)
	fname = [x for x in fname.split('.') if len(x)>3]
	if len(fname) ==1: fname=fname[0]
	c = "find ./datasets -name '{}*.dimacs' -type f".format(fname)
	handle = Popen(c, stdin=PIPE, stderr=PIPE, stdout=PIPE, shell=True)
	stdout_value = handle.communicate()[0]
	print stdout_value
print "-"*10

##
## check tree decomposition 
##
for f in dataset_files.split("\n"):
	fname = os.path.basename(f)
	fname = [x for x in fname.split('.') if len(x)>3]
	if len(fname) ==1: fname=fname[0]
	c = "find ./datasets -name '{}*.dimacs.tree' -type f".format(fname)
	handle = Popen(c, stdin=PIPE, stderr=PIPE, stdout=PIPE, shell=True)
	stdout_value = handle.communicate()[0]
	print stdout_value
print "-"*10

##
## check production rules 
##
for f in dataset_files.split("\n"):
	fname = os.path.basename(f)
	fname = [x for x in fname.split('.') if len(x)>3]
	if len(fname) ==1: fname=fname[0]
	c = "find ./ProdRules -name '{}*.prs' -type f".format(fname)
	handle = Popen(c, stdin=PIPE, stderr=PIPE, stdout=PIPE, shell=True)
	stdout_value = handle.communicate()[0]
	print stdout_value
print "-"*10
