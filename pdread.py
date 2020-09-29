# Version 1.0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

LJFILE = 'lj.csv.gz'
CORRELFILE = 'gpucorrel.csv.gz'
CUTFILE = 'cut.txt'


# If a folder contains a 'cut.txt' file, it will read the timestamp
# and consider the next folder to be part of the same test
# They will be concatenated and returned as a single dataframe

# Each list in paths is a single test
# Each list in cuts is the end timestamp for each test
# 1 elt shorter than the corresponding list in paths
paths = sorted(glob("data/*/"))

# Making a list of test containing the split times
tests = []
new = True
for p in paths:
  if new:
    tests.append([p])
  else:
    tests[-1].append(p)
  new = not os.path.exists(p+CUTFILE)
# Make tests hashable (for cachedfunc)
tests = [tuple(l) for l in tests]


def read_single(path):
  """
  Reads a single folder (ex: 01a_...)

  Returns a Pandas DataFrame inculding correl and lj.
  At every timestamp, the unknown values are interpolated.
  The result is NOT resampled (timestamps are those from both the csv files)
  """
  #print("[read_single] Called with",path)
  lj = pd.read_csv(path+LJFILE, delimiter=',\\s+',engine='python')
  lj['t(s)'] = pd.to_timedelta(lj['t(s)'], unit='s')
  lj = lj.set_index('t(s)')

  correl = pd.read_csv(path+CORRELFILE,
      delimiter=',\\s+',engine='python')
  correl['t(s)'] = pd.to_timedelta(correl['t(s)'], unit='s')
  correl = correl.set_index('t(s)')

  return pd.concat([lj,correl],axis=1)


def read_test(paths):
  """
  Reads several protions of a test and builds a virtual "continuous" test
  If freq is specified (see Pandas resample doc for the format),
  it will resample the dataframe to space evenly the timestamps
  Else, it is simply interpolated to fill the empty spaces

  Takes the folders and the timestamps where each portion of the test ends
  (except for the last one)
  """
  #print("[read] Called with",paths)
  # Read all the tests
  frames = [read_single(path) for path in paths]
  # Cut them at the proper length
  cuts = []
  for p in paths:
    try:
      cuts.append(float(np.loadtxt(p+CUTFILE)))
    except OSError:
      cuts.append(None)
  frames = [frame[frame.index<pd.Timedelta(cut,'s')] if cut else frame
      for frame,cut in zip(frames,cuts)]
  # Offset the timestamps accordingly
  for frame,cut in zip(frames[1:],np.cumsum(cuts[:-1])):
    frame.index += pd.Timedelta(cut,'s')
  # Concatenate and resample
  return pd.concat(frames)


def read_all(tests=tests):
  """
  Reads all the tests in the folder
  """
  return [read_test(t) for t in tests]


if __name__ == '__main__':
  print(f"Found {len(tests)} different tests")
  for i,t in enumerate(tests):
    print(i+1,t)
    #frame = read_test(t)
    #frame = read_test(t).interpolate()
    frame = read_test(t).resample('10ms').mean().interpolate()
    frame.plot(y='exx(%)')
    frame.plot('exx(%)','F(N)')
  plt.show()
