import numpy as np

def get_mat(x,y):
  """
  Consider Ax=y, you have plenty of x and y couples and you are
  looking for A (x and y are the same size ie. A is square
  but it could be extended if neccessary)
  Just give a list of x and the list of associated y to this function :)
  """
  ns = len(x) # Number of samples (x and y)
  s = x[0].shape[0] # Size of x (and y)
  N = s*s # Size of the output matrix
  M = ns*s
  assert ns>s,"Not enough samples!"

  m = np.zeros((M,N)) # only 1/s of the matrix will be filled, rest will be 0
  p = np.empty(M)
  for k in range(ns):
    for i in range(s):
      m[k*s+i,s*i:s*i+s] = x[k]
      p[k*s+i] = y[k][i]
  return np.linalg.lstsq(m,p)[0].reshape(s,s)


if __name__ == "__main__":
  print("Performing test of get_mat...")
  x=y=8
  ns = 400

  samples = [np.random.random(x) for i in range(ns)]

  realA = np.random.random((x,y))

  fsamples = [np.dot(realA,s) for s in samples]
  r = get_mat(samples,fsamples)
  assert np.allclose(r,realA),"Uhm, something went wrong!"
  print("Success")
