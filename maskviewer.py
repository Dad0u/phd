import cv2
from iterview import Displayer


def remap(a,out_range):
  mini,maxi = out_range
  return (a-a.min())/(a.max()-a.min())*(maxi-mini)+mini


def viewmask(imlist,mask):
  mask2 = remap(mask.astype(float),(.5,1))
  d = Displayer(imlist,lambda s: mask2*cv2.imread(s,2))
  d.interactive()


if __name__ == '__main__':
  from glob import glob
  import sys
  if len(sys.argv) == 3:
    imgpath = sys.argv[1]
    maskpath = sys.argv[2]
  else:
    imgpath = '*.tiff'
    maskpath = '../mask.tif'
  l = sorted(glob(imgpath))
  assert l,"No images found"
  print(f'Found {len(l)} images')
  mask = cv2.imread(maskpath,0)
  if mask is None:
    raise FileNotFoundError("Mask not found")
  viewmask(l,mask)
