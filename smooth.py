import numpy as np


def smooth(a,n=100):
  return np.mean(a[:(len(a)//n)*n].reshape(len(a)//n,n),axis=1)


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  N = 18926

  t = np.arange(0,10,10/N)
  y = np.sin(t)+(np.random.rand(N)*.2-.1)

  plt.plot(t,y)
  plt.plot(smooth(t),smooth(y))
  plt.show()
