import numpy as np 
X=np.loadtxt("datasets/contact_eigenv.tsv")
y = np.linspace(0,len((X), 70)
print y
exit()
C = np.random.choice(X[:,0], 70, replace=False)
D = [X[int(x),1] for x in sorted(C)]
Y = np.array([[int(x) for x in sorted(C)], D])
Y = np.transpose(Y)
np.savetxt("datasets/contact_eigenv_trim.tsv", Y, delimiter="\t")

