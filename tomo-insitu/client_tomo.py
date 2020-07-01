#coding: utf-8

import crappy

#def prcd(data):
#  print("GOT",data)
#  print("BIN:",bin(int(data['j1']))[2:].rjust(24,'0'))
#  return data

#s = crappy.blocks.Client('10.42.0.93',load_method='json')
#s = crappy.blocks.Client('10.42.0.156',load_method='json')
s = crappy.blocks.Client('10.42.0.230',load_method='json')
g1 = crappy.blocks.Grapher(('t(s)','j1'),interp=False)
g2 = crappy.blocks.Grapher(('t(s)','j2'),interp=False)

crappy.link(s,g1,condition=crappy.condition.Moving_med(3))
crappy.link(s,g2,condition=crappy.condition.Moving_med(3))

crappy.start()
