#!/usr/bin/python3
#coding: utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from time import time, sleep

matplotlib.use('TKAgg')


def normalize(img, lo, hi):
  return (np.clip(img, lo, hi).astype(float) - lo) / (hi - lo)


class Displayer:
  """
    imgs: list of indexes/names
    clim: Scale to use
      val (float between 0 and 50)
        Will cap to np.percentile(img,val) and 100-val
      [min,max]: use min and max
  """

  def __init__(self, keys, getter, scale=0, cmap='viridis', cache=True):
    self.keys = keys
    self.get = getter
    self.scale = scale
    self.cmap = cmap
    self.cache = cache
    self.last_v = -1
    self.cimg = {}
    self.cclim = {}

  def init_window(self):
    self.root = tk.Tk()
    self.slider = tk.Scale(self.root, orient='horizontal',
                           from_=1, to=len(self.keys), length=500)
    self.slider.pack()

    self.key_label = tk.Label(self.root, text='Hello')
    self.key_label.pack()
    self.scale_label = tk.Label(self.root, text='Hello')
    self.scale_label.pack()

  def get_scale(self, img):
    try:
      # If it is a tuple/list of two values, simply return them
      a, b = self.scale
      return a, b
    except TypeError:
      # Then it should be a float/int, let's compute the percentiles
      return (np.percentile(img, self.scale),
              np.percentile(img, 100 - self.scale))

  def show_interactive(self, event=None):
    v = self.slider.get() - 1
    if v != self.last_v:
      key = self.keys[v]
      self.show(key)
      self.last_v = v
    self.root.after(250, self.show_interactive)

  def get_img(self, key):
    if key not in self.cimg:
      img = self.get(key)
      if self.cache:
        self.cimg[key] = img
    else:
      img = self.cimg[key]
    if key in self.cclim:
      lo, hi = self.cclim[key]
    else:
      lo, hi = self.get_scale(img)
      self.cclim[key] = lo, hi
    return (img, lo, hi)

  def show(self, key):
    img, lo, hi = self.get_img(key)
    self.key_label.configure(text=str(key))
    self.scale_label.configure(text="[%f, %f]" % (lo, hi))
    if not hasattr(self, 'im'):
      self.im = plt.imshow(img, clim=(lo, hi), cmap=self.cmap)
      plt.draw()
      plt.pause(.01)
    # plt.clf()
    # plt.imshow(img,clim=(lo,hi),cmap=self.cmap)
    self.im.set_data(img)
    self.im.set_clim(lo, hi)
    plt.draw()
    plt.pause(.1)

  def interactive(self):
    self.init_window()
    self.root.after(250, self.show_interactive)
    self.root.mainloop()
    self.end()

  def end(self):
    plt.close('all')

  def animate(self, freq=10):
    self.init_window()
    t1 = time()
    for i, key in enumerate(self.keys):
      self.show(key)
      self.slider.set(i)
      t0 = t1
      t1 = time()
      delay = 1 / freq - t1 + t0
      if delay > 0:
        sleep(delay)

  def make_video(self, vidname, fps=30.):
    import cv2
    cm = getattr(matplotlib.cm, self.cmap)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    img, *_ = self.get_img(self.keys[0])
    vid = cv2.VideoWriter(vidname, fourcc, fps, (img.shape[::-1]))
    try:
      for i, k in enumerate(self.keys):
        print(f"{i}/{len(self.keys)}")
        img, lo, hi = self.get_img(k)
        img_c = (cm(normalize(img, lo, hi)) * 255).astype(np.uint8)
        vid.write(img_c[:, :, -2::-1])
    finally:
      vid.release()


if __name__ == '__main__':
  img = (plt.imread('/home/vic/Images/speckle.png') * 255).astype(np.uint8)

  d = Displayer(range(255), lambda i: img + i, scale=(100, 200), cache=False)

  d.interactive()
  # d.animate()
  # d.make_video('test.mp4')
