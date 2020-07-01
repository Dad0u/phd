import numpy as np
norm = np.linalg.norm

def get_mat(x,y,init=None):
  """
  Consider Ax=y, you have plenty of x and y couples and you are
  looking for A. See get_mat or get_mat_r for this "simple" case.
  Now let's spice things up: you don't fully know the x vectors !
  You want the A matrix AND the last u coordinates
  of all the partially known vectors (u = t-s)

  /   \   /x\   / \
  | A | x |-| = |y|
  \   /   \r/   \ /

  Dimensions:
  A: t*t (sought)
  x: s (known n times)
  r: u (sought n times)
  y: t (known n times)
  with t = s+u

  Therefore you have t²+n*u unknown and n*t equations (so you need n >= t²/s)

  n must be large but beware, this version will not scale as well as the
  other get_mat as it has an O(n²) complexity instead of O(n)
  """
  if not isinstance(x,np.ndarray):
    x = np.array(x)
  if not isinstance(y,np.ndarray):
    y = np.array(y)
  n = len(x) # Number of samples (x and y)
  s = x[0].shape[0] # Size of x
  t = y[0].shape[0] # Size of y
  u = t-s # Size of r
  print("n{} s{} t{}".format(n,s,t))
  print("eq{} x{}".format(n*t,n*u+t*t))
  assert t >= s,"Wut? I have no unknown in x!"
  assert n*s>=t*t,"Not enough samples!"

  def f(arr):
    a = arr[:t*t].reshape(t,t)
    r = arr[t*t:].reshape(n,u)

    return np.sum([norm(np.dot(a,np.hstack((i,k)))-j)**2 for i,j,k in zip(x,y,r)])

  if init is None:
    init = np.ones(t*t+n*u)
  #init = np.hstack((realA.flatten(),np.zeros(n*u)))
  #init = np.hstack((realA.flatten(),samples[:,-u:].flatten()))
  print("f(0)",f(init))

  from scipy.optimize import minimize
  sol = minimize(f,init)
  A = sol.x[:t*t].reshape(t,t)
  r = sol.x[t*t:].reshape(n,s-t)

  print("A=",A)
  print("f(sol)=",f(sol.x))
  return A,r

if __name__ == "__main__":
  print("Performing test of get_mat...")
  s = 4
  t = 6
  n = 60

  samples = np.random.random((n,t))

  realA = np.random.random((t,t))
  print("realA=",realA)

  #fsamples = [np.dot(realA,x) for x in samples]
  fsamples = np.dot(realA,samples.T).T
  A,r = get_mat([x[:s] for x in samples],fsamples)
