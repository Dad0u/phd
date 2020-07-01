#coding: utf-8

from __future__ import print_function,division
import mmap

def get_section(m,start,end):
  a = m.find(start)
  if a == -1:
    raise AttributeError("No such label: '%s'"%start)
  m.seek(a)
  b = m.find(end)
  if b == -1:
    raise AttributeError("No such label: '%s'"%end)
  m.seek(b) # To place the cursor at the end
  return m[a:b].split(b"\n")[1:-1]

def get_mesh(m):
  mesh = {}
  for l in get_section(m,b"COOR_3D",b"FINSF"):
    #c = filter(bool,l.strip().split(b" "))
    c = [i for i in l.strip().split(b" ") if i]
    mesh[c[0]] = (float(c[1]),float(c[2]),float(c[3]))
  return mesh

def get_disp(m):
  disp = {}
  for l in get_section(m,b"CHAMP AUX NOEUDS DE NOM SYMBOLIQUE  DEPL",b"-->")[2:]:
    #c = filter(bool,l.strip().split(b" "))
    c = [i for i in l.strip().split(b" ") if i]
    if not c: continue
    disp[c[0]] = (float(c[1]),float(c[2]),float(c[3]))
  return disp

def read(filename):
  with open(filename,'r') as f:
    m = mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
    # Get only the nodes on the top face (z=0)
    mesh = {k:v for k,v in get_mesh(m).items() if v[-1] == 0}
    d = get_disp(m)
    m.close()
    return mesh,d

def show_fields(mesh,field,scale=10):
  import matplotlib.pyplot as plt
  ax = plt.axes()
  for node in mesh:
    ax.arrow(mesh[node][0],mesh[node][1],
        scale*field[node][0], scale*field[node][1],
        width=.1,head_width=.5,head_length=.3)
  ax.relim()
  ax.set_aspect('equal')
  ax.set_xlim([-20,20])
  ax.set_ylim([-20,20])



