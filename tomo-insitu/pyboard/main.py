import net
from hx711 import HX711
import socket
import utime
import json
import machine

header = b'crappy_h\x01\x02\x03'

label1 = "j1"
dt1 = 23
sck1 = 22
label2 = "j2"
dt2 = 21
sck2 = 19
label3 = "bat"
ai3 = 34

def encode_size(s):
  h = []
  while s:
    h.append(s%256)
    s //= 256
  return bytes([len(h)]+h)

def run():
  s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
  s.bind(('',1148))
  s.listen(1)
  cl,ip = s.accept()
  t0 = utime.ticks_ms()
  j1 = HX711(dt1,sck1)
  j2 = HX711(dt2,sck2)
  ai3 = machine.ADC(machine.Pin(34,1))

  while 1:
    data = json.dumps(
        {'t(s)':[(utime.ticks_ms()-t0)/1000],
          label1:[j1.read()],
          label2:[j2.read()],
          #label3:[ai3.read()]
        }).encode('ascii')

    cl.send(header+encode_size(len(data))+data)
