#!/usr/bin/python3
#coding: utf-8

import matplotlib.pyplot as plt

def mkbar(fname,cmap='viridis',clim=(0,.1),dpi=400):
    plt.imshow([[0]],clim=clim,cmap=cmap)
    plt.colorbar()
    plt.savefig(fname,dpi=dpi,transparent=True)
    plt.close('all')

if __name__ == '__main__':
    mkbar('bar.png')
