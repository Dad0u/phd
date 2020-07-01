import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def show_color(f,quiver=(15,20),maxi=None):
  h,w,_ = f.shape
  ampl = (f[:,:,0]**2+f[:,:,1]**2)**.5
  #ampl /= ampl.max()
  if maxi is None:
    ampl /= np.percentile(ampl,95)
  else:
    ampl /= maxi
  angl = -np.arctan2(f[:,:,1],-f[:,:,0])
  angl = (angl+np.pi)/(2*np.pi)
  #r = np.stack([angl,np.ones((w,h),dtype=np.uint8),ampl],axis=2)
  r = hsv_to_rgb(np.stack([angl,ampl,np.ones((h,w),dtype=np.uint8)],axis=2))
  if hasattr(f,"mask"):
    r = r*(1-np.stack((f.mask[:,:,0],)*3,axis=2))
  #plt.imshow(angl)
  #plt.imshow(r)
  #plt.imshow(hsv_to_rgb(r))
  plt.imshow(r)
  if quiver:
    stepy = h//quiver[0]
    stepx = w//quiver[1]
    plt.quiver(np.arange(0,w,stepx),np.arange(0,h,stepy),
        f[::stepy,::stepx,0],-f[::stepy,::stepx,1])
  plt.show()


if __name__ == "__main__":
  import tables
  f = "/media/vic/071abe08-d1fe-4e35-bfca-eadaab1e7217/Essais/19.01-uni45/"+\
      "epr2/post/dis/optflow.hdf"
  h = tables.open_file(f)
  f = h.root.table[500]

  show_color(f)

  h.close()
