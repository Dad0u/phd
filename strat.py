#coding: utf-8
from __future__ import division,print_function

import numpy as np


def rotate(M,theta):
  s = np.sin(theta)
  c = np.cos(theta)
  Ts = np.array([[c*c,s*s,-2*s*c],[s*s,c*c,2*s*c],[s*c,-s*c,c*c-s*s]])
  Te = np.array([[c*c,s*s,s*c],[s*s,c*c,-s*c],[-2*s*c,2*s*c,c*c-s*s]])
  return np.dot(np.dot(Ts,M),Te)


def getQ(El,Et,Glt,nu_lt,theta=0):
  a = 1/(1-nu_lt*nu_lt*Et/El)
  Q11 = El*a
  Q22 = Et*a
  Q12 = nu_lt*Q22
  Q66 = Glt
  Q = np.array([[Q11,Q12,0],[Q12,Q22,0],[0,0,Q66]])
  if theta:
    return rotate(Q,theta)
  return Q


def mkMat(*layers):
  h = 0
  for Q,z in layers:
    assert Q.shape == (3,3),"Incorrect Q:"+str(Q)
    h += z
  print("Overall thickness:",h)
  A = np.zeros((3,3))
  B = np.zeros((3,3))
  D = np.zeros((3,3))
  z1 = -h*5e-4
  for Q,h in layers:
    h *= 1e-3
    z0 = z1
    z1 = z0+h
    A += h*Q
    B += .5*(z1*z1-z0*z0)*Q
    D += 1/3*(z1**3-z0**3)*Q
  r = np.empty((6,6))
  r[0:3,0:3] = A
  r[3:6,0:3] = B
  r[0:3,3:6] = B
  r[3:6,3:6] = D
  return r
