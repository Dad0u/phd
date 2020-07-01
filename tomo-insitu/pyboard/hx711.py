import machine

class HX711():
  def __init__(self,dt,sck):
    self.sck = machine.Pin(sck,3)
    self.dt = machine.Pin(dt,1)
    self.lastmode = 1

  def read_once(self,nextmode=1):
    self.sck.value(0)
    while self.dt.value() == 1:
      pass
    v = 0
    self.sck.value(1)
    self.sck.value(0)
    s = self.dt.value()
    for i in range(23):
      self.sck.value(1)
      self.sck.value(0)
      v += 2**(22-i)*self.dt.value()
    for i in range(nextmode):
      self.sck.value(1)
      self.sck.value(0)
    if s:
      return -2**23+v
    else:
      return v

  def read(self,n=1,mode=1):
    global lastmode
    if mode != self.lastmode:
      #print("Changing mode!")
      self.read_once(mode)
      lastmode = mode
    if n > 1:
      return sum([self.read_once(mode) for i in range(n)])/float(n)
    return self.read_once(mode)
