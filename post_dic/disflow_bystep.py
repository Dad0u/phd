#!/usr/bin/python3
#coding: utf-8

from glob import glob
import cv2
import numpy as np
import tables
import os
import datetime
from time import time, sleep
from multiprocessing import Process, Queue

# 1.2 Switch residual and total type from Int16 to Float32
# 1.3 Images now opening asynchronously
version = "1.3"


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


try:
  dis_class = cv2.optflow.createOptFlow_DIS
except AttributeError:
  dis_class = cv2.DISOpticalFlow_create


def format_time(t):
  i = int(t)
  s = i % 60
  i //= 60
  m = i % 60
  i //= 60
  h = i % 24
  d = i // 24
  if d:
    return "{} days {:02d}:{:02d}:{:02d}".format(d, h, m, s)
  elif h:
    return "{:02d}:{:02d}:{:02d}".format(h, m, s)
  elif m:
    return "{:02d}:{:02d}".format(m, s)
  else:
    frac = "{:.3f}".format(t - int(t))[2:]
    return "{:02d}.{}s".format(s, frac)


def unique_name(full_name):
  """
  Make sure that the specified file name does NOT exist, may add
  an index at the end
  """
  if '.' in full_name:
    l = full_name.split('.')
    name = '.'.join(l[:-1])
    ext = l[-1]
  else:
    name = full_name
    ext = ''
  i = 1
  s = full_name
  while os.path.exists(s):
    s = name + "_{:05d}.".format(i) + ext
    i += 1
  if s != full_name:
    print("WARNING: {} exists, using {} instead".format(full_name, s))
  return s


def remap(a, r, interp=cv2.INTER_CUBIC):
  """
  Remaps a using r the displacement as returned by correlation
  returns B such as B(x) = A(x+r)
  """
  imy, imx = a.shape
  x, y = np.meshgrid(range(imx), range(imy))
  return cv2.remap(a.astype(np.float32),
                   (x + r[:, :, 0]).astype(np.float32),
                   (y + r[:, :, 1]).astype(np.float32),
                   interp)


def compose(f1, f2):
  """
  Gives A->C knowing A->B and B->C
  """
  rf2 = np.stack([remap(f2[:, :, 0], f1), remap(f2[:, :, 1], f1)], axis=2)
  return f1 + rf2


def get_res(a, b, r):
  # return b-remap(a,-r)
  return remap(b, r) - a


def scalar_res(res):
  """
  How to represent the residual with a scalar ?
  """
  return (np.sum(res**2) / res.size)**.5
  #cropped = res[300:2200,1000:7200]
  # return (np.sum(cropped**2)/cropped.size)**.5


def calc_flow(file_list,
              out_rel='optflow_rel.hdf',  # pair by pair flow (cannot be None)
              out_total='optflow_total.hdf',  # Cumulated flow
              out_res='optflow_res.hdf',  # Residual
              complevel=0,
              complevel_res=0,
              complevel_tot=0,
              use_last=True,
              open_func=lambda s: cv2.imread(s, 0),
              # Preset medium
              finest_scale=0,
              gd_iterations=25,
              patch_size=4,
              patch_stride=1,
              alpha=20,
              delta=5,
              gamma=1,
              iterations=20):

  infos = dict()
  infos['start_time'] = str(datetime.datetime.now())
  infos['dir'] = os.getcwd()
  infos['host'] = os.uname()[1]
  infos['algo'] = 'Disflow-rel'
  infos['algo_version'] = version
  infos['finest_scale'] = finest_scale
  infos['gd_iterations'] = gd_iterations
  infos['patch_size'] = patch_size
  infos['patch_stride'] = patch_stride
  infos['alpha'] = alpha
  infos['delta'] = delta
  infos['gamma'] = gamma
  infos['iterations'] = iterations
  infos['opencv'] = cv2.__version__
  infos_s = str(infos).encode('utf-8')
  o_img = open_func(file_list[0])
  height, width = o_img.shape
  size = 8 * height * width * len(file_list) / 2**20
  output_size = 2 * size if out_total else size
  if out_res:
    output_size += size / 2  # Same type as the fields, but only 1 component
  print("Estimated output size: {:.2f} MB".format(output_size))

  # Opening the main output file
  hrel = tables.open_file(unique_name(out_rel), 'w')

  # Creating the infos node
  hrel.create_array(hrel.root, 'infos', [infos_s])

  # If compression is asked, create the filter
  filt = tables.Filters(complevel=complevel) if complevel else None
  # Create the array at the node 'table'
  arr = hrel.create_earray(hrel.root, 'table', tables.Float32Atom(),
                           (0, height, width, 2), expectedrows=len(file_list),
                           filters=filt)
  # Create the array of the names of the images
  max_size = max([len(i.encode('utf-8')) for i in file_list])
  names = hrel.create_earray(hrel.root, 'names', tables.StringAtom(max_size),
                             (0, 2), expectedrows=len(file_list))
  res_arr = hrel.create_earray(hrel.root, 'res', tables.Float32Atom(), (0,),
                               expectedrows=len(file_list) - 1)

  # If asked, create the file and array for the residual
  if out_res:
    hres = tables.open_file(unique_name(out_res), 'w')
    filt_r = tables.Filters(complevel=complevel_res) if complevel_res\
        else None
    arr_r = hres.create_earray(hres.root, 'table', tables.Float32Atom(),
                               (0, height, width), expectedrows=len(file_list),
                               filters=filt_r)
    names_r = hres.create_earray(hres.root, 'names',
                                 tables.StringAtom(max_size), (0, 2),
                                 expectedrows=len(file_list) - 1)

  # If asked, create the total (cumulated field)
  if out_total:
    htot = tables.open_file(unique_name(out_total), 'w')
    filt_t = tables.Filters(complevel=complevel_tot) if complevel_res\
        else None
    arr_t = htot.create_earray(htot.root, 'table', tables.Float32Atom(),
                               (0, height, width, 2),
                               expectedrows=len(file_list),
                               filters=filt_t)
    names_t = htot.create_earray(htot.root, 'names',
                                 tables.StringAtom(max_size), (0, 2),
                                 expectedrows=len(file_list) - 1)

  # Creating the optflow class
  dis = dis_class()
  dis.setFinestScale(finest_scale)
  dis.setGradientDescentIterations(gd_iterations)
  dis.setPatchSize(patch_size)
  dis.setPatchStride(patch_stride)
  dis.setVariationalRefinementAlpha(alpha)
  dis.setVariationalRefinementDelta(delta)
  dis.setVariationalRefinementGamma(gamma)
  dis.setVariationalRefinementIterations(iterations)

  r = None
  t0 = t2 = time()
  total = np.zeros((height, width, 2), dtype=np.float32)
  # Main loop (can catch kb interrupt)
  fb = file_list[0]
  imb = open_func(fb)
  ai = Async_iter(file_list[1:], open_func)
  try:
    for i, (f, img) in enumerate(zip(file_list[1:], ai)):
      fa = fb
      ima = imb
      fb = f
      imb = img
      print("Image {}/{}: {}".format(i + 1, len(file_list), fb))
      # Adding the names of the two images
      names.append([[fa.encode('utf-8'), fb.encode('utf-8')]])
      # Opening the second image
      print("Computing optflow...")
      # Should we initialize the field ?
      r = dis.calc(ima, imb, r if use_last else None)
      # Adding the result to the table
      arr.append(r[None])
      # Computing the residual
      print("Done. Computing residual...")
      res = get_res(ima, imb, r)
      res_arr.append(np.array([scalar_res(res)]))
      if out_res:
        arr_r.append(res[None])
        names_r.append([[fa.encode('utf-8'), fb.encode('utf-8')]])
      if out_total:
        total = compose(total, r)
        arr_t.append(total[None])
        names_t.append([[fa.encode('utf-8'), fb.encode('utf-8')]])
      print("Done.")
      t1 = t2
      t2 = time()
      print("Last loop took {}".format(format_time(t2 - t1)))
      print("  ETA1 {}".format(format_time(
          (t2 - t1) * (len(file_list) - i - 1))))
      print("Elapsed time: {}".format(format_time(t2 - t0)))
      print("  ETA2 {}".format(
          format_time((t2 - t0) / (i + 1) * (len(file_list) - i - 1))))
  except KeyboardInterrupt:
    print("Interrupted !")  # Support de la reprise ?
    ai.terminate()

  print("Correlation finished !")
  hrel.create_array(hrel.root, 'elapsed', [time() - t0])
  print("Closing main hdf file..")
  hrel.close()
  print("Done.")
  if out_res:
    print("Closing residual hdf file..")
    hres.close()
    print("Done.")
  if out_total:
    print("Closing total flow hdf file..")
    htot.close()
    print("Done.")


if __name__ == '__main__':
  path = "/home/vic/Thèse/Essais/TU-carac/21-04-TU-cycles/01-00/data/Wed_Apr_21_15-50-51/imagesXiD-face/*.tiff"
  img_list = sorted(glob(path))

  calc_flow(img_list[::50], alpha=10, delta=1, gamma=0, iterations=2,
            finest_scale=2, out_res=None)
