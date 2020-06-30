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
  assert isinstance(order,int) and order >= 0,"Order must be an int >=1"
  if order == 0:
    return im
  if 1 in im.shape:
    raise RuntimeError("Cannot resample image, dim < 1")
  imf = im.astype(np.float)
  y,x = im.shape
  if y%2:
    imf = imf[:-1,:]
    print("Warning, dropped pixels on axis 0 to resample!")
  if x%2:
    imf = imf[:,:-1]
    print("Warning, dropped pixels on axis 1 to resample!")
  if order == 1:
    return ((imf[::2,::2]+imf[1::2,::2]+imf[::2,1::2]+imf[1::2,1::2])/4
            ).astype(im.dtype)
  else:
    return resample2(resample2(im,order-1))
