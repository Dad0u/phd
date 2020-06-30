#!/usr/bin/python3
#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from time import time,sleep


class Displayer(object):
  """
    imgs: list of indexes/names
    clim: Scale to use
      val (float between 0 and 50)
        Will cap to np.percentile(img,val) and 100-val
      [min,max]: use min and max
  """
  def __init__(self,keys,getter,scale=0,cmap='viridis',cache=True):
    self.keys = keys
    self.get = getter
    self.scale = scale
    self.cmap = cmap
    self.cache = cache
    self.last_v = -1
    self.cimg = {}
    self.cclim = {}
    self.root = tk.Tk()
    self.slider = tk.Scale(self.root,orient='horizontal',
        from_=1,to=len(self.keys),length=500)
    self.slider.pack()

    self.key_label = tk.Label(self.root,text='Hello')
    self.key_label.pack()
    self.scale_label = tk.Label(self.root,text='Hello')
    self.scale_label.pack()

  def get_scale(self,img):
    try:
      # If it is a tuple/list of two values, simply return them
      a,b = self.scale
      return a,b
    except TypeError:
      # Then it should be a float/int, let's compute the percentiles
      return np.percentile(img,self.scale),np.percentile(img,100-self.scale)

  def show_interactive(self,event=None):
    v = self.slider.get()-1
    if v != self.last_v:
      key = self.keys[v]
      self.show(key)
      self.last_v = v
    self.root.after(50,self.show_interactive)

  def show(self,key):
    if key not in self.cimg:
      img = self.get(key)
      if self.cache:
        self.cimg[key] = img
    else:
      img = self.cimg[key]
    if key in self.cclim:
      lo,hi = self.cclim[key]
    else:
      lo,hi = self.get_scale(img)
      self.cclim[key] = lo,hi

    self.key_label.configure(text=str(key))
    self.scale_label.configure(text="[%f, %f]"%(lo,hi))
    plt.clf()
    plt.imshow(img,clim=(lo,hi),cmap=self.cmap)
    plt.pause(.001)

  def interactive(self):
    self.root.after(50,self.show_interactive)
    self.root.mainloop()
    self.end()

  def end(self):
    plt.close('all')

  def animate(self,freq=10):
    t1 = time()
    for i,key in enumerate(self.keys):
      self.show(key)
      self.slider.set(i)
      t0 = t1
      t1 = time()
      delay = 1/freq - t1 + t0
      if delay > 0:
        sleep(delay)


if __name__ == '__main__':
  img = (plt.imread('/home/vic/test/lena.png')*255).astype(np.uint8)

  d = Displayer(range(255),lambda i: img+i,scale=(100,200),cache=False)

  d.interactive()
  #d.animate()
