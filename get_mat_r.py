import numpy as np

def get_mat(x,y):
  """
  Consider Ax=y, you have plenty of x and y couples and you are
  looking for A (x and y can have different sizes ie. A can be rectangular)
  Just give a list of x and the list of associated y to this function :)
  The number of samples must at least the dimension of x
  """
  if not isinstance(x[0],np.ndarray):
    x = [np.array(i) for i in x]
  if not isinstance(y[0],np.ndarray):
    y = [np.array(i) for i in y]
  ns = len(x) # Number of samples (x and y)
  s = x[0].shape[0] # Size of x
  assert ns>=s,"Not enough samples!"
  t = y[0].shape[0] # Size of y
  N = s*t # Size of the output matrix
  M = ns*t

  m = np.zeros((M,N)) # only 1/s of the matrix will be filled, rest will be 0
  p = np.empty(M)
  for k in range(ns):
    for i in range(t):
      m[k*t+i,s*i:s*i+s] = x[k]
      p[k*t+i] = y[k][i]
  return np.linalg.lstsq(m,p,rcond=None)[0].reshape(t,s)


if __name__ == "__main__":
  print("Performing test of get_mat...")
  x = 8
  y = 12
  ns = 400

  samples = [np.random.random(x) for i in range(ns)]

  realA = np.random.random((y,x))

  fsamples = [np.dot(realA,s) for s in samples]
  r = get_mat(samples,fsamples)
  assert np.allclose(r,realA),"Uhm, something went wrong!"
  print("Success")
