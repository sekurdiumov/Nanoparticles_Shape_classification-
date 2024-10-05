# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:46:30 2019

@author: sk1u19
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy.lib import scimath as SC
import time
import matplotlib.colors
import csv
import random as rnd
import scipy.io as scio
from scipy.ndimage import rotate

from Shapes import Shapes
from Vector_Diffraction import Vector_Diffraction as VD 

#plt.style.use('dark_background') 
plt.style.use('default') 
matplotlib.rcParams.update({'font.size' : 30})
pi = np.pi

   
## ============================================================================
## Main:
## ============================================================================
# constants
Nmax = 6000
Nmay = 6000
L0 = 60
PWL = 100
spacing = 1/PWL   
print ('Points per wavelength: ', PWL)                                                
inputField = np.ones((Nmax, Nmay))*0

NX2 = int(Nmax/2)
NY2 = int(Nmay/2)
x = np.linspace(-NX2, NX2, Nmax)*spacing
y = np.linspace(-NY2, NY2, Nmay)*spacing
X,Y = np.meshgrid(x,y)

z = np.array([2])
N0 = 0

doplot = 0

fnames = ['ellipses_data.mat',
          'triangles_data.mat',
          'rectangles_data.mat',
          'hexagons_data.mat',
          'rings_data.mat',
          'stars_data.mat',
          'zigzags_data.mat',
          'Ys_data.mat']

shape_ind = 0

fname = fnames[shape_ind]

M1 = scio.loadmat(fname)
data = M1['data']

counter = 0
for i in range(len(data)):
    
    
    t = data[i,0]
    size = data[i,1]
    a = data[i,2]
    b = data[i,3]
    
    tm = time.time()
    #t = rnd.randint(0, 10)
    IF0 = Shapes.InputField(t, size, a, b, inputField, PWL)
    
    #angle = rnd.randint(-180, 180)
    #angle = 0
    #IF0 = rotate(IF, angle, reshape = False)
    
    #IF0[IF0 > 0.2] = 1.0
    #IF0[IF0 < 0.2] = 0.0
    
    Ex = VD.propFS(IF0, spacing, z, 0)
    #Ey = propFS(IF0, spacing, z, 1)
    
    #E = Ex + 1j*Ey
    
    img_x = np.abs(Ex[-1,:, :])
    #img_y = np.abs(Ey[-1,:, :])
    
    if doplot:
        
        plt.figure(figsize = [9,7])
        fig = plt.pcolormesh(X, Y, IF0, cmap = 'hot')
        plt.colorbar()
        lim = 0.5
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        
        plt.figure(figsize = [9, 7])
        fig = plt.pcolormesh(X, Y, np.abs(img_x)**2,  cmap = 'hot')
        plt.colorbar()
        plt.xlabel('x, in λ')
        plt.ylabel('y, in λ')
        lim = 5
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.clim(0.98, 1.02)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        
    
    fname = 'dp' + str(shape_ind) + '_' + str(i) + '.mat'
    

    ## save maps 1024x1024 for convenient resizing (512x512, or 256x256):

    scio.savemat(fname, {'Ex' : np.abs(img_x[2488:3512, 2488:3512])**2, 
                         'label' : t, 
                         'size' : size,
                         'a' : a,
                         'b' : b})
    
    elapsed = time.time() - tm
    print ('Time elapsed = ', elapsed, ' i = ', i)
    
    
print('Game over!')