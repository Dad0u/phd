import numpy as np
import cv2
import os
import datetime
from time import time

from gpucorrel import GPUCorrel

version = "2.0"


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


def calc_flow_gpu(original_image,
                  file_list,
                  fields=['x', 'y', 'r', 'exx', 'eyy', 'exy'],
                  csv_out="correl_gpu.csv",
                  hdf_out=None,
                  res_out=None,
                  open_func=lambda s: cv2.imread(s, 0)):

  # Getting some info on the current run
  infos = dict()
  infos['start_time'] = str(datetime.datetime.now())
  infos['algo'] = "iDIC-GPU"
  infos['algo_version'] = version
  infos['dir'] = os.getcwd()
  infos['host'] = os.uname()[1]
  infos_s = str(infos).encode('utf-8')

  # Opening the first image
  img0 = open_func(original_image)
  # Instanciating the correl object
  correl = GPUCorrel(img0.shape,
                     fields=fields)
  correl.set_ref(img0)

  r = np.empty((0, len(fields)))
  height, width = img0.shape

  # Writing the HDF file to store the full_fields
  if hdf_out:
    import tables
    h = tables.open_file(unique_name(hdf_out), 'w')
    # Creating the infos node
    h.create_array(h.root, 'infos', [infos_s])

    arr = h.create_earray(h.root, 'table', tables.Float32Atom(),
                          (0, height, width, 2), expectedrows=len(file_list))
    max_size = max([len(i.encode('utf-8'))
                   for i in file_list + [original_image]])
    names = h.create_earray(h.root, 'res', tables.Float64Atom(), (0,),
                            expectedrows=len(file_list))
    fields = np.stack(
        [f.get() for f in correl.get_fields(height, width)], axis=-1)

  # Writing the HDF file to store the residual
  if res_out:
    import tables
    hres = tables.open_file(unique_name(res_out), 'w')
    # Creating the infos node
    hres.create_array(hres.root, 'infos', [infos_s])
    arr_res = hres.create_earray(hres.root, 'table', tables.Float32Atom(),
                                 (0, height, width),
                                 expectedrows=len(file_list))
    names_res = hres.create_earray(hres.root, 'names',
                                   tables.StringAtom(max_size),
                                   (0, 2), expectedrows=len(file_list))
    arr_res_scalar = hres.create_earray(hres.root, 'scalar',
                                        tables.Float32Atom(),
                                        (0,), expectedrows=len(file_list))

  t0 = t2 = time()
  try:
    for i, img_name in enumerate(file_list):
      print("Image {}/{}: {}".format(i + 1, len(file_list), img_name))
      print("Opening and computing optflow...")
      img = open_func(img_name)
      disp = correl.compute(img)
      r = np.vstack((r, disp))
      if hdf_out:
        print("Done. Writing full field...")
        arr.append(sum([d * f for d, f in zip(disp, fields)])[None])
        names.append([[original_image.encode('utf-8'),
                       img_name.encode('utf-8')]])
      if res_out:
        print("Done. Writing residual...")
        # To make sure we updated the residual!
        c = correl.correl[0]
        c._makeDiff.prepared_call(c.grid, c.block,
                                  c.devOut.gpudata,
                                  c.devX.gpudata,
                                  c.devFieldsX.gpudata,
                                  c.devFieldsY.gpudata)
        res = c.devOut.get()
        # arr_res.append(get_res(img0,img,full_field)[None])
        arr_res.append(res[None])
        names_res.append([[original_image.encode('utf-8'),
                           img_name.encode('utf-8')]])
        arr_res_scalar.append([correl.correl[0].res])
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
    print("Interrupted!")

  np.savetxt(unique_name(csv_out), r)
  if hdf_out:
    h.close()
  if res_out:
    hres.close()
  correl.clean()


if __name__ == '__main__':
  from glob import glob
  path = "/home/vic/Thèse/Essais/20.07-2-Biaxe/01-plexi/img/img_0*.tiff"
  img_list = sorted(glob(path))[::10]
  calc_flow_gpu(img_list[0], img_list[1:],
                open_func=lambda s: cv2.imread(s, 0)[1500:-1500, 1500:-1500])
