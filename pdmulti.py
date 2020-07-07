import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from phd import pdread,pdaudio

THRESH_FILE = 'audio_thresh.txt'
try:
  thresh = float(np.loadtxt(THRESH_FILE))
except OSError:
  thresh = 50
  print(f"Warning, {THRESH_FILE} not found. Using default value {thresh}")

tests = pdread.read_all()

audios = [pdaudio.read_audio(test) for test in pdread.tests]

for t,a in zip(tests,audios):
  data = pd.concat([t,a]).interpolate('time').resample('10ms').mean()
  t = np.array(data.index,float)/1e9
  plt.plot(t,data['exx(%)'])
  audiodmg = data['audio_lvl'] > thresh
  plt.plot(t[audiodmg],data['exx(%)'][audiodmg],'o')
plt.show()
