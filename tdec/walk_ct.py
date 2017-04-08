# ,': has',len(kids),'kids',type(kids)
# print type(kids),len(kids)
# for x in kids:
#   # print type(x)
#   if isinstance(x, frozenset):
#     print '\t{}+->{}'.format(parent, x)
#   elif len(x) > 1:
#
#     print [type(child) for child in x]
#   else:
#     print type(x)
#

def walk_ct(parent, children, indnt):
  print parent
  for x in children:
    if isinstance(x, (tuple,list)):
      print indnt,
      walk_ct(x[0],x[1], indnt+"\t")
    else:
      print type(x), len(x)


