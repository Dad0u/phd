"""
@title: Coarse to fine variational refinement
@author: Victor Couty
@date 22/04/2020
@version 1.1
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


default = dict(
    alpha=30,
    delta=5,
    gamma=2,
    fpi=30,
    sor=10,
    omega=1.6,
    nstages=100,
    minsize=200,
    rfactor=.95,
    interp_d=cv2.INTER_AREA, # When reducing
    interp_u=cv2.INTER_CUBIC, # When enlarging
    progress=True
  )


class Vref:
  """
  Coarse-to-fine variational refinement using OpenCV
  """
  def __init__(self,**kwargs):
    self.vr = cv2.VariationalRefinement_create()
    for k,v in default.items():
      setattr(self,k,kwargs.pop(k,v))
    if kwargs:
      raise AttributeError(f"Unknown parameter(s):{kwargs}")
    self.h = self.w = -1

  @property
  def gamma(self):
    return self.vr.getGamma()

  @gamma.setter
  def gamma(self,v):
    self.vr.setGamma(v)

  @property
  def fpi(self):
    return self.vr.getFixedPointIterations()

  @fpi.setter
  def fpi(self,v):
    self.vr.setFixedPointIterations(v)

  @property
  def sor(self):
    return self.vr.getSorIterations()

  @sor.setter
  def sor(self,v):
    self.vr.setSorIterations(v)

  @property
  def omega(self):
    return self.vr.getOmega()

  @omega.setter
  def omega(self,v):
    self.vr.setOmega(v)

  def calc_pyramid(self):
    """
    Computes the shape of the image on all the levels
    """
    self.shapelist = []
    f = 1
    while len(self.shapelist) < self.nstages \
        and min(self.h,self.w)*f >= self.minsize:
      self.shapelist.append((int(round(self.h*f)),int(round(self.w*f))))
      f *= self.rfactor
    if self.progress:
      self.total = sum([i*j for i,j in self.shapelist])

  def resample_img(self,ima,imb):
    """
    Makes lists of the images at every resolution of the pyramid
    """
    self.imalist = [ima]
    self.imblist = [imb]
    for y,x in self.shapelist[1:]:
      self.imalist.append(cv2.resize(self.imalist[-1],(x,y),
        interpolation=self.interp_d))
      self.imblist.append(cv2.resize(self.imblist[-1],(x,y),
        interpolation=self.interp_d))

  def print_progress(self,erase=True):
    print(("\r" if erase else "") +
        "{:.2f} %".format(self.processed/self.total*100),end="",flush=True)

  def calc(self,ima,imb,f=None):
    """
    Compute the variational refinement with the coarse-to-fine approach
    """
    if ima.shape != (self.h,self.w):
      self.h,self.w = ima.shape
      self.calc_pyramid()
    # Prepare all the images
    self.resample_img(ima,imb)
    if f is None: # No field given, start from 0
      f = np.zeros(self.shapelist[-1]+(2,),dtype=np.float32)
    else: # Resample it to the resolution of the first level
      f = cv2.resize(f,self.shapelist[-1][::-1],
          interpolation=self.interp_d)*self.rfactor**len(self.shapelist)
    # Compute the first field
    self.vr.setAlpha(self.alpha*self.rfactor**(len(self.shapelist)-1))
    self.vr.setDelta(self.delta/self.rfactor**(len(self.shapelist)-1))
    #print("\ni=",0)
    #print("alpha=",self.vr.getAlpha())
    #print("delta=",self.vr.getDelta())
    #print("shape=",self.shapelist[-1])
    self.vr.calc(self.imalist[-1],self.imblist[-1],f)
    if self.progress:
      i,j = self.shapelist[-1]
      self.processed = i*j
      self.print_progress(False)

    # Working our way up the pyramid (skipping the lowest)
    for i,(ima,imb,shape) in enumerate(list(zip(
        self.imalist,self.imblist,self.shapelist))[-2::-1]):
      f = cv2.resize(f,shape[::-1],interpolation=self.interp_u)/self.rfactor
      self.vr.setAlpha(self.alpha*self.rfactor**(len(self.shapelist)-i-2))
      self.vr.setDelta(self.delta/self.rfactor**(len(self.shapelist)-i-2))
      #print("\n\ni=",i+1)
      #print("alpha=",self.vr.getAlpha())
      #print("delta=",self.vr.getDelta())
      #print("shape=",shape)
      self.vr.calc(ima,imb,f)
      if self.progress:
        self.processed += shape[0]*shape[1]
        self.print_progress()
    if self.progress:
      print("")
    return f


if __name__ == '__main__':
  # Quick example
  ima = cv2.imread('/home/vic/test.local/ca.tif',0)
  imb = cv2.imread('/home/vic/test.local/cb.tif',0)

  v = Vref(progress=True)
  f = v.calc(ima,imb,None)
  plt.imshow(f[:,:,0])
  plt.show()
