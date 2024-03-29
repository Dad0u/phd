import pandas as pd
import numpy as np
import tables
import os
import cv2
from mystuff.cachedfunc import cachedfunc

CORRELFILES = ['post/disflow-face/optflow_rel.hdf',
    'post/disflow-face/optflow_rel_10.hdf',
    'post/disflow-face/disflow_rel.hdf']

CUTFILE = 'cut.txt'
MASKFILE = 'mask.tif'

RMFIELDS = ['x','y','r','exx','eyy','exy']


def get_time(s):
  return float(b'.'.join(s.split(b'_')[-1].split(b'.')[:-1]))


@cachedfunc('localstrain.p')
def read_localstrain(paths):
  """
  To compute a damage evolution based on the occurence of local strain

  The file read is the displacement between two successive images
  The strain is computed, squared and summed within the mask
  """
  total_offset = 0
  frames = []
  for path in paths:
    try:
      cut = np.loadtxt(path+CUTFILE)
    except OSError:
      cut = np.inf
    print("Searching in",CORRELFILES)
    cfile = next(path+p for p in CORRELFILES if os.path.exists(path+p))
    assert cfile, "No correl file available!"
    hdf = tables.open_file(cfile,'r')
    names = hdf.root.names
    table = hdf.root.table
    n,h,w,_ = table.shape
    try:
      mask = cv2.imread(path+MASKFILE,0).astype(float)/255
    except AttributeError:
      print("[read_localstrain] Warning, mask not found! Using default")
      margin = .2 # 20% margin on the default mask
      mask = np.zeros((h,w))
      mask[int(margin*h):int((1-margin)*h),int(margin*w):int((1-margin)*w)] = 1
    r = []
    last_t = get_time(names[0][0])
    for field,name in zip(table,names):
      print("Processing",name[1])
      new_t = get_time(name[1])
      # The events occured somewhere between last_t and new_t
      # Let's take the average
      t = (new_t+last_t)/2
      last_t = new_t
      if t >= cut:
        break
      v = 0
      for i in range(2):
        for f in np.gradient(field[:,:,i]):
          v += np.sum(mask*(f-f[np.where(mask)].mean())**2)
      r.append((t+total_offset,v))
    total_offset += min(cut,t)
    data = pd.DataFrame(r,columns=['t(s)','localstrain'])
    data['t(s)'] = pd.to_timedelta(data['t(s)'],unit='s')
    frames.append(data.set_index('t(s)'))
    hdf.close()
  return pd.concat(frames)
