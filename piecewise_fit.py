# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# Si n points de rupture
# => 2*(n+1) paramètres

# NON =================
# k0 pente de la demi-droite de gauche

# x1 coord x du premier point
# y1 coord y du premier point
# x2 coord x du deuxième point
# y2 coord y du deuxième point
# ...
# xn coord x du nème point
# yn coord y du nème point

# k1 pente de la demi-droite de droite
# ======================

# k0 pente de la demi-droite de gauche
# y0 ordonnée à l'origine de la demi-droite de gauche
# x1 coord x du 1er point
# k1 pente du segment après le premier point
# x2 coord x du 2ème point
# k2 pente du segment suivant
# etc...
# xn coord x du dernier point
# kn pente du de la demi-droite de droite


def pw_lin(t,*args):
  assert len(args)%2 == 0
  #n = len(args) // 2
  k0,y0 = args[:2]
  x,k = zip(*sorted(zip(args[2::2],args[3::2])))
  k = (k0,) + k
  y = [y0]
  last = y0+k[0]*x[0]
  #print("x=",x)
  #print("k=",k)
  for u,v,m in zip(x[:-1],x[1:],k[1:-1]):
    #print(f"Processing segment x={u} to {v}")
    #print(f"k={m}")
    #print(f"y0={last}")
    last += (v-u)*m
    #print(f"y1={last}")
    y.append(last-m*v)
    #print("Appending",y[-1])
  y.append(last-k[-1]*x[-1])
  #print("y=",y)
  #return np.piecewise(t,[t<i for i in x],
  conds = [t<x[0]]+[np.logical_and(t>=a,t<b) for a,b in zip(x[:-1],x[1:])]
  funcs = [lambda t,m=m,p=p: m*t+p for m,p in zip(k,y)]
  #for f in funcs:
  #  print("f(0)=",f(0))
  #  plt.plot(t,f(t))
  #plt.show()

  #print(" ====== CONDS ========\n",conds)
  #print(" ====== FUNCS ========\n",funcs)
  return np.piecewise(t,conds,funcs)


x = np.arange(-1,5,.1)
#plt.plot(x,pw_lin(x,1,0,0,2,1,-1,3,1))
plt.plot(x,pw_lin(x,1,1,1,2,2,-1))
plt.show()
