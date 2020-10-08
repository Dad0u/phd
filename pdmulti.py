import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from phd import pdread,pdaudio,pdlocalstrain,pdlocalthermo

THRESH_FILE = 'thresh.txt'
default_thresh = {'audio':50,'localstrain':40,'localthermo':30,'fdmg':25}

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


def tplot(data,**kwargs):
  t = np.array(data.index,dtype=float)/1e9
  plt.plot(t,data,**kwargs)


for test,a,ls,lt in zip(tests,audios,lstrain,lthermo):
  plt.figure()
  fsmooth = test['F(N)'].resample('20ms').mean()
  fdmg = (fsmooth.rolling(10).mean().diff()-fsmooth.diff()).abs()
  data = pd.concat([test,a,ls,lt,fdmg.to_frame(name='fdmg')]).sort_index()
  tplot(data['exx(%)'].dropna(),label='$\\epsilon_{xx}$(%)')
  data['exx(%)'].interpolate('time',inplace=True)

  tplot(data[data['fdmg'] > thresh['fdmg']]['exx(%)']-.01,
      marker='o',linestyle='',label='Fdmg')
  tplot(data[data['audio_lvl'] > thresh['audio']]['exx(%)'],
      marker='o',linestyle='',label='Audio')
  tplot(data[data['localstrain'] > thresh['localstrain']]['exx(%)']+.01,
      marker='o',linestyle='',label='Correl')
  tplot(data[data['localthermo'] > thresh['localthermo']]['exx(%)']+.02,
      marker='o',linestyle='',label='Thermo')
  plt.legend()
plt.show()
