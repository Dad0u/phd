import cv2
import numpy as np


def remap(a,r,interp=cv2.INTER_CUBIC):
  """
  Remaps a using given r the displacement as a result from correlation
  """
  imy,imx = a.shape
  x,y = np.meshgrid(range(imx),range(imy))
  return cv2.remap(a.astype(np.float32),
      (x+r[:,:,0]).astype(np.float32),(y+r[:,:,1]).astype(np.float32),interp)


def get_res(a,b,r):
  #return b-remap(a,-r)
  return a-remap(b,r)


def resample2(im,order=1):
  """
  Stupid resampling to divide resolution by 2
  """
  assert isinstance(order,int) and order >= 1,"Order must be an int >=1"
  y,x = im.shape
  if y%2:
    im = im[:-1,:]
    print("Warning, dropped pixels on axis 0 to resample!")
  if x%2:
    im = im[:,:-1]
    print("Warning, dropped pixels on axis 1 to resample!")
  if order == 1:
    return im[::2,::2]+im[1::2,::2]+im[::2,1::2]+im[1::2,1::2]
  else:
    return resample2(resample2(im,order-1))

# float sigma=.6f
# int fpi=5
# float alpha = 1.f
# float delta=.5f
# float gamma=5.f
# float downscaleFactor = .95f
