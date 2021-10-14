#!/usr/bin/python3
#coding: utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from time import time, sleep
from multiprocessing import Process, Queue

matplotlib.use('TKAgg')


def normalize(img, lo, hi):
  return (np.clip(img, lo, hi).astype(float) - lo) / (hi - lo)


class Async_iter(Process):
  """
  Can be used to iterate over a list of objects and apply
  asynchronously a function (like an async map)

  For example to load images to be processed in an other process

  namelist: List of keys

  load: function to call on each key

  length: The number of element to load in advance

  sleep_delay: How long will the process sleep if there no elements are needed
  """
  def __init__(self, namelist, load, length=3, sleep_delay=.1):
    super().__init__()
    self.namelist = namelist
    self.load = load
    self.q = Queue()
    self.length = length
    self.sleep_delay = sleep_delay

  def __iter__(self):
    self.start()
    return self

  def run(self):
    for name in self.namelist:
      while self.q.qsize() >= self.length:
        sleep(self.sleep_delay)
      self.q.put(self.load(name))

  def __next__(self):
    while self.q.qsize() == 0 and self.is_alive():
      sleep(self.sleep_delay)
    if self.q.qsize() > 0:
      return self.q.get()
    raise StopIteration


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

  def mkimg_color(self, key):
    cm = getattr(matplotlib.cm, self.cmap)
    img, lo, hi = self.get_img(key)
    return (cm(normalize(img, lo, hi)) * 255).astype(np.uint8)[:, :, -2::-1]

  def make_video(self, vidname, fps=30.):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    img, *_ = self.get_img(self.keys[0])
    vid = cv2.VideoWriter(vidname, fourcc, fps, (img.shape[::-1]))
    ai = Async_iter(self.keys, self.mkimg_color)
    try:
      for i, img in enumerate(ai):
        print(f"{i}/{len(self.keys)}")
        vid.write(img)
    finally:
      vid.release()
      ai.terminate()


if __name__ == '__main__':
  img = (plt.imread('/home/vic/Images/speckle.png') * 255).astype(np.uint8)

  d = Displayer(range(0, 255, 3), lambda i: img + i, scale=(100, 200), cache=False)

  #d.interactive()
  # d.animate()
  d.make_video('test.mp4')
