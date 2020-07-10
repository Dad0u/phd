import cv2
import numpy as np


def mkhole(r,coord):
  cy,cx = coord
  for y in range(cy-r,cy+r):
    for x in range(cx-r,cx+r):
      if (y-cy)**2+(x-cx)**2<r**2:
        a[y,x] = 0


h,w = 1680,7920
ymin,ymax = 320,1370
xmin,xmax = 900,7690
r = 90
cy,cx = 866,4310


a = np.zeros((h,w),dtype=np.uint)
a[ymin:ymax,xmin:xmax] = 255

mkhole(r,(cy,cx))
mkhole(r,(cy,cx-20))
mkhole(r,(cy,cx-40))

cv2.imwrite('data/01a_Tue_Mar__3_15-46-43/mask.tif',a)
