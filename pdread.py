# Version 1.0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


# If a folder contains a 'cut.txt' file, it will read the timestamp
# and consider the next folder to be part of the same test
# They will be concatenated and returned as a single dataframe

# Each list in paths is a single test
# Each list in cuts is the end timestamp for each test
# 1 elt shorter than the corresponding list in paths
paths = sorted(glob("data/*/"))
cuts = []
for p in paths:
  try:
    cuts.append(np.loadtxt(p+'cut.txt'))
  except OSError:
    cuts.append(None)

# Making a list of test containing the split times
test = []
new = True
for p,c in zip(paths,cuts):
  if new:
    test.append([(p,c)])
  else:
    test[-1].append((p,c))
  new = c is None


print(f"Found {len(test)} different tests")


def read_single(path):
  """
  Reads a single folder (ex: 01a_...)

  Returns a Pandas DataFrame inculding correl and lj.
  At every timestamp, the unknown values are interpolated.
  The result is NOT resampled (timestamps are those from both the csv files)
  """
  lj = pd.read_csv(path+'lj.csv.gz', delimiter=',\\s+',engine='python')
  lj['t(s)'] = pd.to_timedelta(lj['t(s)'], unit='s')
  lj = lj.set_index('t(s)')

  correl = pd.read_csv(path+'gpucorrel.csv.gz',
      delimiter=',\\s+',engine='python')
  correl['t(s)'] = pd.to_timedelta(correl['t(s)'], unit='s')
  correl = correl.set_index('t(s)')

  data_high = pd.concat([lj,correl],axis=1).interpolate('time')
  #data = data_high.resample('10ms').median()
  return data_high


def read(freq='10ms',paths=paths[0],cuts=cuts):
  """
  Reads several protions of a test and builds a virtual "continuous" test
  Will resample the dataframe to space evenly the timestamps

  Takes the folders and the timestamps where each portion of the test ends
  (except for the last one)
  """
  # Read all the tests
  frames = [read_single(p) for p in paths]
  # Cut them at the proper length
  frames = [frame[frame.index<pd.Timedelta(cut,'s')]
      for frame,cut in zip(frames,cuts)] + [frames[-1]]
  # Offset the timestamps accordingly
  for frame,cut in zip(frames[1:],np.cumsum(cuts)):
    frame.index += pd.Timedelta(cut,'s')
  # Concatenate and resample
  return pd.concat(frames).resample(freq).mean()


if __name__ == '__main__':
  frame = read()
  frame.plot(y='exx(%)')
  frame.plot('exx(%)','F(N)')
  plt.show()
