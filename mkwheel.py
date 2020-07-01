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
  #plt.show()


H = 300
W = 300

mh  = H/2
mw = W/2

a = np.empty((H,W,2))
for i in range(H):
  for j in range(W):
    ni = (i-mh)/mh
    nj = (j-mw)/mw
    a[i,j,1] = ni
    a[i,j,0] = nj

mask = np.zeros((H,W,2))

for i in range(H):
  for j in range(W):
    ni = (i-mh)/mh
    nj = (j-mw)/mw
    if ni**2+nj**2>1:
      mask[i,j,0] = 1
      mask[i,j,1] = 1


#show_color(a,quiver=None)
show_color(np.ma.array(a,mask=mask),quiver=None)

plt.savefig("wheel.png",dpi=200)
