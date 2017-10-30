import tarfile, os

def read_tarbz2_file(tarbz2_fpath=""):
  tf = tarfile.open(tarbz2_fpath, "r:bz2")
  outfn = [x for x in tf.getnames() if "out." in x]
  orig_dat = tf.extractfile(dict(zip( tf.getnames(), tf.getmembers()))[outfn[0]] ).readlines()
  print (type, orig_dat[0])
  orig_dat = [x.rstrip(b"\r\n") for x in orig_dat if b"%" not in x]

  return orig_dat
