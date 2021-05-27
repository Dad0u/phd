import pandas as pd
import numpy as np
from scipy.io import wavfile
from mystuff.cachedfunc import cachedfunc

WAVFILE = 'audio_denoised.wav'
OFFSETFILE = 'offset_audio.txt'
CUTFILE = 'cut.txt'
SAMPLERATE = 192000  # kHz, samplerate of the wav file


@cachedfunc('audio.p')
def read_audio(paths, freq='10ms', samplerate=SAMPLERATE):
  """
  To compute a "noise level" from the mic during the test
  We are working on a complete test (ie could be several wav files)

  paths : List of the dirs containing the data
  freq : Frequency of the output (Hz)
  samplerate : samplerate of the wav file
  """
  total_offset = 0
  frames = []
  for p in paths:
    offset = np.loadtxt(p + OFFSETFILE)
    try:
      cut = np.loadtxt(p + CUTFILE)
    except OSError:
      cut = None
    f, w = wavfile.read(p + WAVFILE)
    wav = w[int(offset * samplerate):cut and int((cut + offset) * samplerate)]
    dwav = pd.DataFrame(np.abs(wav), columns=['audio_lvl'])
    t = np.arange(0, len(dwav)) / SAMPLERATE
    t += total_offset
    total_offset += cut if cut else 0
    dwav.index = pd.to_timedelta(t, unit='s')
    dwav.index.name = 't(s)'
    frames.append(dwav.resample(freq).mean())
  return pd.concat(frames)


def count_evt(frame, thresh=100):
  ser = frame['audio_lvl']
  d = (ser > thresh).astype(int)
  return (d.diff() == 1).cumsum()
