import pandas as pd
import numpy as np
import tables
import os
import cv2
from phd.fields import get_fields,OrthoProjector
from mystuff.cachedfunc import cachedfunc

CORRELFILES = ['post/disflow-face/optflow_rel.hdf',
    'post/disflow-face/optflow_rel_10.hdf']

CUTFILE = 'cut.txt'
MASKFILE = 'mask.tif'

RMFIELDS = ['x','y','r','exx','eyy','exy']


@cachedfunc('localdisp.p')
def read_localstrain(paths):
  """
  To compute a damage evolution based on the occurence of local strain

  The file read is the displacement between two successive images
  Rigid body motion and first order strain are removed
  the residual is squared, summed and returned
  """
  total_offset = 0
  frames = []
  for path in paths:
    try:
      cut = np.loadtxt(path+CUTFILE)
    except OSError:
      cut = np.inf
    cfile = next(path+p for p in CORRELFILES if os.path.exists(path+p))
    assert cfile, "No correl file available!"
    h = tables.open_file(cfile,'r')
    names = h.root.names
    table = h.root.table
    n,h,w,_ = table.shape
    try:
      mask = cv2.imread(path+MASKFILE,0).astype(float)/255
    except OSError:
      print("[read_localstrain] Warning, mask not found! Using default")
      margin = .2 # 20% margin on the default mask
      mask = np.zeros((h,w))
      mask[int(margin*h):int((1-margin)*h),int(margin*w):int((1-margin)*w)] = 1
    mask2 = np.stack([mask]*2,axis=2)
    base = get_fields(RMFIELDS,h,w)
    for i in range(len(RMFIELDS)):
      base[:,:,:,i] *= mask2
    proj = OrthoProjector(base)
    r = []
    for field,name in zip(table,names):
      print(name[1])
      t = float(b'.'.join(name[1].split(b'_')[-1].split(b'.')[:-1]))
      if t >= cut:
        break
      f = np.sum((field*mask2-proj.get_full(field))**2)
      r.append((t+total_offset,f))
      print("DEBUG",r[-1])
    total_offset += min(cut,t)
    data = pd.DataFrame(r,columns=['t(s)','correl_dmg'])
    data['t(s)'] = pd.to_timedelta(data['t(s)'],unit='s')
    frames.append(data.set_index('t(s)'))
  return pd.concat(frames)
