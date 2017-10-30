import multiprocessing as mp
import os
import explodingTree as xt

os.chdir(os.path.dirname(__file__))
print(os.getcwd())

print ("Transform to dimacs")
print ("-"*40)

dir= "../datasets"
p_files = [x[0]+"/"+f for x in os.walk(dir) for f in x[2] if f.endswith(".p")]

p = mp.Pool(processes=4)
for f in p_files:
     gn = xt.graph_name(f)
     if os.path.exists('datasets/{}.dimacs'): continue
     g = xt.load_edgelist(f)
     p.apply_async(xt.convert_nx_gObjs_to_dimacs_gObjs, args=([g], ), callback=collect_results)
p.close()
p.join()
print (results)

