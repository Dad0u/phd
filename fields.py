#coding: utf-8
# Tools to generate and manipulate 2D fields
# Dimension is (h,w,2),
# the third axis contain the displacement along respectively x and y

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def ones(h, w):
  return np.ones((h, w), dtype=np.float32)


def zeros(h, w):
  return np.zeros((h, w), dtype=np.float32)


Z = None


def z(h, w):
  global Z
  if Z is None or Z[0].shape != (h, w):
    sh = 1 / (w * w / h / h + 1)**.5
    sw = w * sh / h
    Z = np.meshgrid(np.linspace(-sw, sw, w, dtype=np.float32),
                    np.linspace(-sh, sh, h, dtype=np.float32))
  return Z


def get_field(s, h, w):
  """
  Generates a field corresponding to the string s with size (h,w)

  Possible strings are :
  x y r exx eyy exy eyx exy2 r z uxx uyy uxy vxx vyy vxy
  """
  if s == 'x':
    """
    Rigid body motion along x in pixels
    """
    return ones(h, w), zeros(h, w)
  elif s == 'y':
    """
    Rigid body motion along y in pixels
    """
    return zeros(h, w), ones(h, w)
  elif s == 'r':
    """
    Rotation in degrees
    """
    u, v = z(h, w)
    # Ratio (angle) of the rotation
    # Should be π/180 to be 1 for 1 deg
    # Z has and amplitude of 1 in the corners
    # 360 because h²+w² is twice the distance center-corner
    r = (h**2 + w**2)**.5 * np.pi / 360
    return v * r, -u * r
  elif s == 'exx':
    """
    Elongation along x in %
    """
    return (np.concatenate((np.linspace(-w / 200, w / 200, w,
            dtype=np.float32)[np.newaxis, :],) * h, axis=0),
            zeros(h, w))
  elif s == 'eyy':
    """
    Elongation along y in %
    """
    return (zeros(h, w),
            np.concatenate((np.linspace(-h / 200, h / 200, h,
                                        dtype=np.float32)[:, np.newaxis],) * w, axis=1))
  elif s == 'exy':
    """
    "Shear" derivative of y along x in %
    """
    return (np.concatenate((np.linspace(-h / 200, h / 200, h,
            dtype=np.float32)[:, np.newaxis],) * w, axis=1),
            zeros(h, w))
  elif s == 'eyx':
    """
    "Shear" derivative of x along y in %
    """
    return (zeros(h, w),
            np.concatenate((np.linspace(-w / 200, w / 200, w,
                                        dtype=np.float32)[np.newaxis, :],) * h, axis=0))
  elif s == 'exy2':
    """
    Sum of the two previous definitions of shear in %
    """
    return ((w / (w * h)**.5) * np.concatenate((np.linspace(-h / 200, h / 200, h,
            dtype=np.float32)[:, np.newaxis],) * w, axis=1),
            (h / (w * h)**.5) * np.concatenate((np.linspace(-w / 200, w / 200, w,
                                                            dtype=np.float32)[np.newaxis, :],) * h, axis=0))

  elif s == 'z':
    """
    Zoom in %
    """
    u, v = z(h, w)
    r = (h**2 + w**2)**.5 / 200
    return u * r, v * r
  elif s == 'uxx':
    """
    ux = x²
    """
    return (np.concatenate(((np.linspace(-1, 1, w, dtype=np.float32)**2)
            [np.newaxis, :],) * h, axis=0),
            zeros(h, w))
  elif s == 'uyy':
    """
    ux = y²
    """
    return (np.concatenate(((np.linspace(-1, 1, h, dtype=np.float32)**2)
            [:, np.newaxis],) * w, axis=1),
            zeros(h, w))
  elif s == 'uxy':
    """
    ux = x*y
    """
    return (np.array([[k * j for j in np.linspace(-1, 1, w)]
            for k in np.linspace(-1, 1, 2)], dtype=np.float32),
            zeros(h, w))
  elif s == 'vxx':
    """
    uy = x²
    """
    return (zeros(h, w),
            np.concatenate(((np.linspace(-1, 1, 2,
                                         dtype=np.float32)**2)[np.newaxis, :],) * h, axis=0))
  elif s == 'vyy':
    """
    uy = y²
    """
    return (zeros(h, w),
            np.concatenate(((np.linspace(-1, 1, 2,
                                         dtype=np.float32)**2)[:, np.newaxis],) * w, axis=1))
  elif s == 'vxy':
    """
    uy = x*y
    """
    return (zeros(h, w),
            np.array([[k * j for j in np.linspace(-1, 1, 2)]
                      for k in np.linspace(-1, 1, 2)], dtype=np.float32))
  else:
    print("WTF?", s)
    raise NameError("Unknown field string: " + s)


def get_fields(l, h, w):
  """
  Calls get_field for each string in l

  Returns a single numpy array of dim (h,w,2,N), N being the number of fields
  """
  r = np.empty((h, w, 2, len(l)), dtype=np.float32)
  for i, s in enumerate(l):
    if isinstance(s, np.ndarray):
      r[:, :, :, i] = s
    else:
      r[:, :, 0, i], r[:, :, 1, i] = get_field(s, h, w)
  return r


class Fielder:
  """
  Class to build the fields from the projected vector

  Useful to rebuild the full field when using GPUCorrel class
  """

  def __init__(self, flist, h, w):
    self.nfields = len(flist)
    self.h = h
    self.w = w
    fields = get_fields(flist, h, w)
    self.fields = [fields[:, :, :, i] for i in range(fields.shape[3])]

  def get(self, *x):
    return sum([i * f for i, f in zip(x, self.fields)])


class Projector:
  """
  Tools to project fields on a base of given fields
  """

  def __init__(self, base, check_orthogonality=True):
    if isinstance(base, list):
      self.base = base
    else:
      self.base = [base[:, :, :, i] for i in range(base.shape[3])]
    self.fielder = Fielder(self.base, *self.base[0].shape[:2])
    self.norms2 = [np.sum(b * b) for b in self.base]
    if check_orthogonality:
      from itertools import combinations
      s = []
      for a, b in combinations(self.base, 2):
        s.append(abs(np.sum(a * b)))
      maxs = max(s)
      if maxs / self.base[0].size > 1e-6:
        print("WARNING, base does not seem orthogonal!")
        print(s)
        if input("Continue ?").strip().lower() not in "yo":
            raise Exception("Base not orthogonal")

  def get_scal(self, flow):
    return [np.sum(vec * flow) / n2 for vec, n2 in zip(self.base, self.norms2)]

  def get_full(self, flow):
    return self.fielder.get(*self.get_scal(flow))


class OrthoProjector(Projector):
  """
  Like Projector, but the base is orthogonalized at first
  It allows the projection to happen on a non-orthogonal base
  Warning : Unicity of the result is not guaranteed
  """

  def __init__(self, base):
    vec = [base[:, :, :, i] for i in range(base.shape[3])]
    new_base = [vec[0]]
    for v in vec[1:]:
      p = Projector(new_base, check_orthogonality=False)
      new_base.append(v - p.get_full(v))
    Projector.__init__(self, new_base, check_orthogonality=False)


def avg_ampl(f):
  """
  Returns the average amplitude of a field
  """
  return (np.sum(f[:, :, 0]**2 + f[:, :, 1]**2) / f.size * 2)**.5


def remap(a, r, interp=cv2.INTER_CUBIC):
  """
  Remaps a using given r the displacement as a result from correlation
  """
  imy, imx = a.shape
  x, y = np.meshgrid(range(imx), range(imy))
  return cv2.remap(a.astype(np.float32),
                   (x + r[:, :, 0]).astype(np.float32), (y + r[:, :, 1]).astype(np.float32), interp)


def get_res(a, b, r):
  # return b-remap(a,-r)
  return a - remap(b, r)


def resample2(im, order=1):
  """
  Stupid resampling to divide resolution by 2
  """
  assert isinstance(order, int) and order >= 0, "Order must be an int >=1"
  if order == 0:
    return im
  if 1 in im.shape:
    raise RuntimeError("Cannot resample image, dim < 1")
  imf = im.astype(np.float)
  y, x = im.shape
  if y % 2:
    imf = imf[:-1, :]
    print("Warning, dropped pixels on axis 0 to resample!")
  if x % 2:
    imf = imf[:, :-1]
    print("Warning, dropped pixels on axis 1 to resample!")
  if order == 1:
    return ((imf[::2, ::2] + imf[1::2, ::2] + imf[::2, 1::2] + imf[1::2, 1::2]) / 4
            ).astype(im.dtype)
  else:
    return resample2(resample2(im, order - 1))


def show_color(f, quiver=(15, 20), maxi=None, show=True):
  h, w, _ = f.shape
  ampl = (f[:, :, 0]**2 + f[:, :, 1]**2)**.5
  #ampl /= ampl.max()
  if maxi is None:
    maxi = np.percentile(ampl, 95)
    print("max scale=", maxi)
  ampl /= maxi
  angl = -np.arctan2(f[:, :, 1], -f[:, :, 0])
  angl = (angl + np.pi) / (2 * np.pi)
  #r = np.stack([angl,np.ones((w,h),dtype=np.uint8),ampl],axis=2)
  r = hsv_to_rgb(
    np.stack([angl, ampl, np.ones((h, w), dtype=np.uint8)], axis=2))
  if hasattr(f, "mask"):
    r = r * (1 - np.stack((f.mask[:, :, 0],) * 3, axis=2))
  # plt.imshow(angl)
  # plt.imshow(r)
  # plt.imshow(hsv_to_rgb(r))
  plt.imshow(r)
  if quiver:
    stepy = h // quiver[0]
    stepx = w // quiver[1]
    plt.quiver(np.arange(0, w, stepx), np.arange(0, h, stepy),
               f[::stepy, ::stepx, 0], -f[::stepy, ::stepx, 1])
  if show:
    plt.show()
