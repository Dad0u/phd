#coding:utf-8

from __future__ import print_function,division

from pymodbus.client.sync import ModbusSerialClient
import crappy
from time import time
tofloat = crappy.tool.convert_data.data_to_float32

class Torqueforce(crappy.inout.InOut):
  def __init__(self,port='/dev/ttyUSB0',baudrate=115200):
    self.port = port
    self.baudrate = baudrate
    self.m = ModbusSerialClient('rtu',port=self.port,baudrate=self.baudrate,
        parity='E')

  def open(self):
    self.m.connect()

  def get_data(self):
    l = self.m.read_holding_registers(32,4,unit=1).registers
    return [time(),tofloat(l[:2]),tofloat(l[2:])]

  def close(self):
    self.m.close()

io = crappy.blocks.IOBlock("TorqueForce",labels=['t(s)','F(kN)','C(Nm)'],
    verbose=True)

graph_f = crappy.blocks.Grapher(('t(s)','F(kN)'))
graph_c = crappy.blocks.Grapher(('t(s)','C(Nm)'))

crappy.link(io,graph_f)
crappy.link(io,graph_c)

crappy.start()
