import cv2
import numpy as np

h,w = 180,640
ymin,ymax = 55,133
xmin,xmax = 77,572

# Hole
cy,cx = 98,313
r = 9

a = np.zeros((h,w),dtype=np.uint)
a[ymin:ymax,xmin:xmax] = 255

if r:
  for y in range(cy-r,cy+r):
    for x in range(cx-r,cx+r):
      if (y-cy)**2+(x-cx)**2<r**2:
        a[y,x] = 0
cv2.imwrite('mask.tif',a)
