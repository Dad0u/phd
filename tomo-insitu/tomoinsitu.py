#coding: utf-8

from pymodbus.client.sync import ModbusTcpClient
from fractions import Fraction

import crappy.tool.convert_data as cvt

K = 32 # Offset between registers of different motors

class Register():
  def __init__(self,offset,dtype='bool',rw=False):
    self.offset = offset
    self.dtype = dtype
    self.rw = rw

  @property
  def length(self):
    return 1 + ('32' in self.dtype)

available_types = ['int32','int16','uint32','uint16','float32','bool']

holding_reg = {
    'CMD_POS':(0,'float32'),
    'CMD_DIST':(2,'float32'),
    'CMD_VEL':(4,'float32'),
    'ACC':(6,'float32'),
    'DEC':(8,'float32'),
    'FDEC':(10,'float32'),
    'GEAR_RATIO_NUM':(12,'int32'),
    'GEAR_RATIO_DEN':(14,'uint32'),
    }

input_reg = {
    'VEL':(0,'float32'),
    'POS':(2,'float32'),
    'STATE':(4,'uint16'),
    }

coil = {
    'POWER':0,
    'MV_ABS':1,
    'MV_REL':2,
    'MV_VEL':3,
    'STOP':4,
    'ACK_ERR':5,
    'GEAR_IN':6,
    'GEAR_OUT':7,
    'MV_DIR':8,
    }

state = {
  0:'INIT',
  1:'WAIT',
  20:'POWERON',
  25:'POWEROFF',
  30:'HOME',
  40:'MV_VEL',
  50:'MV_ABS',
  60:'MV_REL',
  70:'CONNEXION',
  80:'DISCONNEXION',
  100:'ERROR',
    }
reg = {}
coils = {}
for mot in range(1,4):
  for name,(offset,dtype) in holding_reg.items():
    assert dtype in available_types,"Invalid type "+str(dtype)
    reg[name+str(mot)] = Register(mot*K+offset,dtype,True)
  for name,(offset,dtype) in input_reg.items():
    reg[name+str(mot)] = Register(mot*K+offset,dtype,False)
  for name,offset in coil.items():
    coils[name+str(mot)] = mot*K+offset
mot=4 # Codeur
for name,(offset,dtype) in input_reg.items():
  reg[name+str(mot)] = Register(mot*K+offset,dtype,False)


def data2val(data,dtype):
  assert dtype in available_types,"Invalid type: "+dtype
  assert len(data) == 1+('32' in dtype),"Invalid data length"
  return getattr(cvt,'data_to_'+dtype)(data)



  #if dtype == 'float32':
  #  return cvt.data_to_float32(data)
  #if dtype == 'int32':
  #  return cvt.data_to_int32(data)
  #if dtype == 'int16':
  #  return cvt.data_to_int16(data)
  #if dtype == 'uint32':
  #  return cvt.data_to_uint32(data)
  #if dtype == 'uint16':
  #  return data[0]
  #else:
  #  raise AttributeError("Unknown type: "+dtype)

def val2data(data,dtype):
  assert dtype in available_types,"Invalid type: "+dtype
  return getattr(cvt,dtype+'_to_data')(data)
  #if dtype == 'float32':
  #  return cvt.float32_to_data(data)
  #if dtype == 'int32':
  #  return cvt.int32_to_data(data)
  #if dtype == 'int16':
  #  return cvt.int16_to_data(data)
  #if dtype == 'uint32':
  #  return cvt.uint32_to_data(data)
  #if dtype == 'uint16':
  #  return cvt.uint16_to_data(data)
  #else:
  #  raise AttributeError("Unknown type: "+dtype)


class KollTomo():
  """
-----
|   |
| 1 |     |
|   |   -----
-----   |   |
  |     | 3 |
        |   |
        -----

  |
-----
|   |
| 2 |
|   |
-----
This class is to drive the rotating machine meant to
perform traction tests while in the microtomograph and rotating
the sample while applying the charge

The motors 1 and 2 are supposed to be synchronized to avoid torsion
on the sample but could also be used to deliberatly apply torsion
They can be synchronized with a chosen gear ratio with an encoder
The goal is to measure the rotation of the table to act as if the sample
was standing on  the table.

This code is meant to run with the appropriate software written on the
variators, to link the modbus registers to the corresponding actions/values

There are 3 types of methods:
    - Low level methods: connect, disconnect, read and write
    - Motor specific methods, they start with 'motor_'
        They are not supposed to be used as is by the user to drive
        the machine, use the machine specific methods to do so
    - Machine specific methods
  """

  # ===== Low level methods =====
  def __init__(self,ip="192.168.0.109",port=502):
    self.h = ModbusTcpClient(ip,port)
    self.geared = False
    self.powered = dict(zip(range(1,4),[False]*3))
    self.geared = dict(zip(range(1,4),[False]*3))
    l = list(coils.keys())+list(reg.keys())
    assert len(l) == len(set(l)),"Duplicate register name!"
    self.registers = reg
    self.coils = coils

  def connect(self):
    """
    Connect to the Modbus controller
    """
    assert self.h.connect(),"Could not connect!"

  def disconnect(self):
    """
    Disconnect from the Modbus controller
    """
    self.h.close()

  def read(self,regname):
    """
    Wrapper to read a Modbus register/coil given by its name
    According to the dtype of the register, it will convert the
    data and return it as the dtype (float32,int32,int16,uint...)
    Coils are read one by one and returned as bools
    """
    if regname in self.coils:
      return self.h.read_coils(self.coils[regname]).bits[0]
    reg = self.registers[regname] # Will raise an error if key is not defined
    if reg.rw:
      return data2val(self.h.read_holding_registers(
        reg.offset,reg.length).registers,reg.dtype)
    else:
      return data2val(
          self.h.read_input_registers(
            reg.offset,reg.length).registers,reg.dtype)

  def write(self,regname,value):
    """
    Wrapper to write into a Modbus register/coil given by its name
    According to the dtype of the register, it will convert the
    data and write by converting based on the dtype
    """
    if regname in self.coils:
      return self.h.write_coil(self.coils[regname],bool(value))
    reg = self.registers[regname] # Will raise an error if key is not defined
    assert reg.rw,"Cannot write to an input register!"
    return self.h.write_registers(reg.offset,val2data(value,reg.dtype))


  # ===== Motor specific methods =====
  def motor_on(self,motor):
    """
    Poweron a motor
    """
    assert motor in range(1,4),"Invalid motor number!"
    if not self.powered[motor]:
      self.write("POWER%d"%motor,True)
      self.powered[motor] = True

  def motor_off(self,motor):
    """
    Turn off a motor
    """
    assert motor in range(1,4),"Invalid motor number!"
    if not self.powered[motor]:
      return
    s = self.read('STATE%d'%motor)
    if 'MV_' in state[s]:
      self.write('STOP%d'%motor,1)
    if self.geared[motor]:
      self.motor_gear_out(motor)
    self.write("POWER%d"%motor,False)
    self.powered[motor] = False

  def motor_gear_in(self,motor,ratio=1):
    """
    Link this motor to the encoder
    """
    if not self.powered[motor]:
      return
    f = Fraction(ratio)
    if f > 1:
      f = 1/((1/f).limit_denominator(2**16-1))
    else:
      f = f.limit_denominator(2**16-1)
    print("RATIO approximated by %d/%d (=%f)"%(
      f.numerator,f.denominator,float(f)))
    if self.geared[motor]:
      self.motor_gear_out(motor)
    self.write('GEAR_RATIO_NUM%d'%motor,f.numerator)
    self.write('GEAR_RATIO_DEN%d'%motor,f.denominator)
    self.write('GEAR_IN%d'%motor,True)
    self.geared[motor] = True

  def motor_gear_out(self,motor):
    if not self.powered[motor]:
      return
    if self.geared[motor]:
      self.write('GEAR_OUT%d'%motor,True)
      self.geared[motor] = False

  def motor_set_speed(self,motor,speed):
    if speed == 0:
      self.motor_stop(motor)
      return
    if 'MV_' in state[self.read('STATE%d'%motor)]:
      self.write('STOP%d'%motor,True)
    sign,speed = speed >= 0,abs(speed)
    s = state[self.read('STATE%d'%motor)]
    while s != 'WAIT':
      if s == 'ERROR':
        raise IOError("Motor %d is in error state!"%motor)
      s = state[self.read('STATE%d'%motor)]
    self.write('CMD_VEL%d'%motor,speed)
    self.write('MV_DIR%d'%motor,sign)
    self.write('MV_VEL%d'%motor,True)

  def motor_set_pos(self,motor,pos,vel=360,acc=3600,dec=3600,rel=False):
    if 'MV_' in state[self.read('STATE%d'%motor)]:
      self.write('STOP%d'%motor,True)
    s = state[self.read('STATE%d'%motor)]
    while s != 'WAIT':
      if s == 'ERROR':
        raise IOError("Motor %d is in error state!"%motor)
      s = state[self.read('STATE%d'%motor)]
    self.write('CMD_VEL%d'%motor,vel)
    self.write('ACC%d'%motor,acc)
    self.write('DEC%d'%motor,dec)
    if rel:
      self.write('CMD_DIST%d'%motor,pos)
      self.write('MV_REL%d'%motor,True)
    else:
      self.write('MV_DIR%d'%motor,False)
      self.write('CMD_POS%d'%motor,pos)
      self.write('MV_ABS%d'%motor,True)

  def motor_stop(self,motor):
    if self.geared[motor]:
      self.motor_gear_out(motor)
    self.write('STOP%d'%motor,True)

  def motor_get_speed(self,motor):
    return self.read('VEL%d'%motor)

  def motor_get_pos(self,motor):
    return self.read('POS%d'%motor)

  def motor_get_state(self,motor):
    return self.read('STATE%d'%motor)

  def motor_ack_err(self,motor):
    self.write('ACK_ERR%d'%motor,True)


  # ===== Machine methods =====
  def stop(self):
    self.motor_stop(3)

  def stop_all(self):
    for m in (1,2,3):
      self.motor_stop(m)

  def on(self):
    for i in (1,2,3):
      self.motor_on(i)

  def off(self):
    for i in (1,2,3):
      self.motor_off(i)

  def gear_in(self,ratio):
    """
    Link the motors 1 and 2 position to the encoder (identified as
    a 4th motor)
    You cen specify a ratio between the angle of the encoder and
    the angle of the motors
    """
    self.motor_gear_in(1,ratio)
    self.motor_gear_in(2,-ratio)

  def gear_out(self):
    for i in (1,2):
      self.motor_gear_out(i)

  def set_speed(self,speed):
    self.motor_set_speed(3,speed)

  def set_pos(self,pos,**kwargs):
    self.motor_set_pos(3,pos,**kwargs)

  def get_speed(self):
    return self.motor_get_speed(3)

  def get_pos(self):
    return self.motor_get_pos(3)
