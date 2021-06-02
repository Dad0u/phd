#!/usr/bin/python3
#coding: utf-8

from glob import glob
import cv2
import numpy as np
import tables
import os
import datetime
from time import time

# 1.1 Replaced Float64Atom to Float32Atom
version = "1.1"


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


def calc_flow(original_image,
              file_list,
              out_file='optflow.hdf',
              out_res='optflow_res.hdf',
              complevel=0,
              complevel_res=1,
              use_last=True,
              open_func=lambda s: cv2.imread(s, 0),
              # Preset medium
              finest_scale=0,
              gd_iterations=25,
              patch_size=8,
              patch_stride=3,
              alpha=20,
              delta=5,
              gamma=10,
              iterations=5):

  infos = dict()
  infos['start_time'] = str(datetime.datetime.now())
  infos['dir'] = os.getcwd()
  infos['host'] = os.uname()[1]
  infos['algo'] = 'Disflow'
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
  o_img = open_func(original_image)
  height, width = o_img.shape
  output_size = 8. * height * width * len(file_list) / 2**20
  if out_res:
    output_size *= 1.05  # Rough pessimistic estimate with zlib(1) compression
  print("Estimated output size: {:.2f} MB".format(output_size))

  # Opening the main output file
  h = tables.open_file(unique_name(out_file), 'w')

  # Creating the infos node
  h.create_array(h.root, 'infos', [infos_s])

  # If compression is asked, create the filter
  filt = tables.Filters(complevel=complevel) if complevel else None
  # Create the array at the node 'table'
  arr = h.create_earray(h.root, 'table', tables.Float32Atom(),
                        (0, height, width, 2),
                        expectedrows=len(file_list),
                        filters=filt)
  # Create the array of the names of the images
  max_size = max([len(i.encode('utf-8'))
                 for i in file_list + [original_image]])
  names = h.create_earray(h.root, 'names', tables.StringAtom(max_size),
                          (0, 2), expectedrows=len(file_list))
  res_arr = h.create_earray(h.root, 'res', tables.Float32Atom(), (0,),
                            expectedrows=len(file_list))

  # If asked, create the file and array for the residual
  if out_res:
    h_res = tables.open_file(unique_name(out_res), 'w')
    filt_r = tables.Filters(complevel=complevel_res) if complevel_res\
        else None
    arr_r = h.create_earray(h_res.root, 'table', tables.Int16Atom(),
                            (0, height, width), expectedrows=len(file_list),
                            filters=filt_r)
    names_r = h.create_earray(h_res.root, 'names', tables.StringAtom(max_size),
                              (0, 2), expectedrows=len(file_list))
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
  # Main loop (can catch kb interrupt)
  try:
    for i, f in enumerate(file_list):
      print("Image {}/{}: {}".format(i + 1, len(file_list), f))
      # Adding the names of the two images
      names.append([[original_image.encode('utf-8'), f.encode('utf-8')]])
      # Opening the second image
      print("Computing optflow...")
      img = open_func(f)
      # Should we initialize the field ?
      if use_last:
        r = dis.calc(o_img, img, r)
      else:
        r = dis.calc(o_img, img, None)
      # Adding the result to the table
      arr.append(r[None])
      # Computing the residual
      print("Done. Computing residual...")
      res = get_res(o_img, img, r)
      res_arr.append(np.array([scalar_res(res)]))
      if out_res:
        arr_r.append(res[None])
        names_r.append([[original_image.encode('utf-8'), f.encode('utf-8')]])
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

  print("Correlation finished !")
  h.create_array(h.root, 'elapsed', [time() - t0])
  print("Closing main hdf file..")
  h.close()
  print("Done.")
  if out_res:
    print("Closing residual hdf file..")
    h_res.close()
    print("Done.")


if __name__ == '__main__':
  path = "/home/vic/Thèse/Essais/19.01-uni45/epr2/img_local/*.tif"
  img_list = sorted(glob(path))

  calc_flow(img_list[0], img_list[::10], alpha=10,
            delta=1, gamma=0, iterations=20)
