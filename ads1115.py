import smbus
from time import sleep

gain_d = {
    0:6.144,
    1:4.096,
    2:2.048,
    3:1.024,
    4:0.512,
    5:0.256
  }


rate_d = {
    8:0,
    16:1,
    32:2,
    64:3,
    128:4,
    250:5,
    475:6,
    860:7
  }


def toint(l):
    high,low = l
    val = (high << 8) + low
    if val & 0x8000:
        return val - 65536
    return val


class ADS1115():
    """
    Class for ADS 1115 ADC

    Can read the voltage on any channel
    """
    def __init__(self,addr,bus=1):
        self.addr = addr
        self.bus = smbus.SMBus(bus)

    def read(self,chan,gain=1,rate=128):
        config_h = 0x81 # OS mode+mode
        config_h |= (4+chan) << 4
        config_h |= gain << 1
        config_l = rate_d[rate] << 5 | 3 # Disable queue
        #print("DEBUG: config=",config_h,config_l)
        self.bus.write_i2c_block_data(self.addr,1,[config_h,config_l])
        sleep(1./128)
        return toint(
            self.bus.read_i2c_block_data(self.addr,0,2))/32768.*gain_d[gain]
