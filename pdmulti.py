import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from phd import pdread,pdaudio,pdlocalstrain,pdlocalthermo

THRESH_FILE = 'thresh.txt'
default_thresh = {'audio':50,'localstrain':40,'localthermo':30}

try:
  with open(THRESH_FILE,'r') as f:
    thresh = eval(f.read())
  assert isinstance(thresh,dict)
except Exception:
  print(f"[pdmulti] WARN {THRESH_FILE} not found. Using default values")
  thresh = default_thresh
for k in default_thresh:
  if k not in thresh:
    print(f"[pdmulti] WARN {THRESH_FILE} does not contain {k}. Using default")
    thresh[k] = default_thresh[k]

tests = pdread.read_all()

audios = [pdaudio.read_audio(test) for test in pdread.tests]

lstrain = [pdlocalstrain.read_localstrain(test) for test in pdread.tests]

lthermo = [pdlocalthermo.read_localthermo(test) for test in pdread.tests]

for t,a,ls,lt in zip(tests,audios,lstrain,lthermo):
  data = pd.concat([t,a,ls,lt]).interpolate('time').resample('10ms').mean()
  t = np.array(data.index,float)/1e9
  plt.plot(t,data['exx(%)'],label='$\\epsilon_{xx}$(%)')
  audiodmg = data['audio_lvl'] > thresh['audio']
  plt.plot(t[audiodmg],data['exx(%)'][audiodmg],'o',label='audio')
  lstraindmg = data['localstrain'] > thresh['localstrain']
  plt.plot(t[lstraindmg],data['exx(%)'][lstraindmg]+.01,'o',label='correl')
  lthermodmg = data['localthermo'] > thresh['localthermo']
  plt.plot(t[lthermodmg],data['exx(%)'][lthermodmg]+.02,'o',label='thermo')
  plt.legend()
plt.show()
