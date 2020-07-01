import numpy as np
import smbus


def toint(lower,upper):
  return 256*upper+lower


class AMG8833():
  """
  Class to read AMG8833 infrared array sensors
  """
  def __init__(self,addr=0x69,bus=1):
    self.addr = addr
    self.bus = smbus.SMBus(bus)

  def read(self):
    r = np.empty((8,8),dtype=np.uint16)
    for line in range(8):
      data = self.bus.read_i2c_block_data(self.addr,0x80+16*line,16)
      #print("data=",data)
      r[7-line,:] = [toint(a,b) for a,b in zip(data[::2],data[1::2])]
    print("min,max=",r.min(),r.max())
    return r


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  import matplotlib.animation as ani
  a = AMG8833()
  calib = a.read()
  fig = plt.figure()
  im = plt.imshow(calib)

  def update(*args):
    im.set_array(a.read()-calib+256)
    im.autoscale()
    return im,

  ani = ani.FuncAnimation(fig,update,interval=100,blit=True)
  plt.show()
