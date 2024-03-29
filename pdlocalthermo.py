import pandas as pd
import numpy as np
from glob import glob
import cv2
from mystuff.cachedfunc import cachedfunc

CUTFILE = 'cut.txt'
MASKFILE = 'irmask.tif'
IMGPATH = 'IR/*.tif'
XIPATH = 'imagesXiD-face/*.tiff'
IRTHRESH = 'irthresh.txt'  # Value of a pixel (diff of DL) to be considered hot
# Not to be confused with the number of hot pixel threshold (see pdmulti)


class TimeGetter:
  """
  To retreive the timestamp of an image given its index

  Uses the Ximea images (the cameras were in sync)
  """

  def __init__(self, path):
    xilist = sorted(glob(path + XIPATH))
    # Remove the reference image
    xilist = [i for i in xilist if 'ref' not in i]
    self.tlist = [float(s.split('_')[-1][:-5]) for s in xilist]

  def get(self, i):
    if isinstance(i, str):
      i = int(i.split('_')[-1][:-4])
    return self.tlist[i]


H = '[read_localthermo] '
W = 'Warning: '


@cachedfunc('localthermo.p')
def read_localthermo(paths):
  """
  To compute a damage evolution based on local temperature elevations
  """
  total_offset = 0
  frames = []
  for path in paths:
    print(H + "Processing", path)
    try:
      cut = np.loadtxt(path + CUTFILE)
    except OSError:
      cut = np.inf
    imglist = sorted(glob(path + IMGPATH),
                     key=lambda s: int(s.split('_')[-1][:-4]))
    last = cv2.imread(imglist[0], cv2.IMREAD_ANYDEPTH)
    h, w = last.shape
    try:
      mask = cv2.imread(path + MASKFILE, 0).astype(float) / 255
    except AttributeError:
      print(H + W + "Mask not found! Using default")
      margin = .2  # 20% margin on the default mask
      mask = np.zeros((h, w))
      mask[int(margin * h):int((1 - margin) * h),
           int(margin * w):int((1 - margin) * w)] = 1
    tg = TimeGetter(path)
    if len(tg.tlist) != len(imglist):
      print(H + W + "There are {} Ximea images and {} IR images 🤔".format(
        len(tg.tlist), len(imglist)))
      imglist = imglist[:min(len(tg.tlist), len(imglist))]
    try:
      irthresh = int(np.loadtxt(IRTHRESH))
    except OSError:
      print(H + W + f"{IRTHRESH} not found, using default value")
      irthresh = 30
    r = []
    for imgname in imglist[1:]:
      t = tg.get(imgname)
      if t >= cut:
        break
      img = cv2.imread(imgname, cv2.IMREAD_ANYDEPTH).astype(float)
      diff = img - last
      last = img
      # r.append((t,np.sum(mask*diff**2)))
      mdiff = mask * (diff - diff[np.where(mask)].mean())
      r.append((t + total_offset, np.count_nonzero(mdiff > irthresh)))

    total_offset += min(cut, t)
    data = pd.DataFrame(r, columns=['t(s)', 'localthermo'])
    data['t(s)'] = pd.to_timedelta(data['t(s)'], unit='s')
    frames.append(data.set_index('t(s)'))
  return pd.concat(frames)
